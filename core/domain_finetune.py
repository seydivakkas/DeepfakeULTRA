"""
Domain Generalization Fine-Tuner — Harici veri setleri uzerinden model adaptasyonu.

Tam tri-branch pipeline (RGB + Freq + Mesh) ile domain fine-tuning.
Mevcut feedback fine-tuner'dan farki:
  - Tam pipeline (DWT+DCT+Phase+Mesh)
  - Kademeli katman acma (fusion+classifier → backbone)
  - CombinedLoss (Focal + Label Smoothing)
  - Validation AUC tracking + best model

Kullanim:
    from core.domain_finetune import DomainFineTuner
    tuner = DomainFineTuner()
    tuner.start(datasets=["celeb_df_v2", "dfdc"], epochs=8, lr=5e-5)
"""

import os
import random
import threading
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms

from config import DEVICE, model_cfg, paths

# ── Sabitler ──
EXTERNAL_TESTS_DIR = paths.BASE_DIR / "dataset" / "external_tests"
DOMAIN_FINETUNED_PATH = paths.MODEL_DIR / "domain_finetuned.pth"
DOMAIN_BACKUP_PATH = paths.MODEL_DIR / "domain_finetuned_backup.pth"
BEST_MODEL_GENERALIZED_PATH = paths.MODEL_DIR / "best_model_generalized.pth"

# Dataset metadata
KNOWN_DATASETS = {
    "celeb_df_v2": {"label": "Celeb-DF v2", "icon": "🎬"},
    "dfdc": {"label": "DFDC", "icon": "📹"},
    "deepfake20k": {"label": "Deepfake 20K", "icon": "🖼️"},
    "deepfakeface": {"label": "DeepFakeFace", "icon": "👤"},
    "faceforensics": {"label": "FaceForensics++", "icon": "🔬"},
}


# ── Thread-safe durum ──
_domain_ft_lock = threading.Lock()
_domain_ft_status = {
    "running": False,
    "phase": "",
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_auc": 0.0,
    "best_auc": 0.0,
    "progress": "",
    "completed": False,
    "error": None,
    "log": [],
}


def get_domain_ft_status() -> dict:
    """Mevcut domain fine-tune durumunu dondur."""
    with _domain_ft_lock:
        return dict(_domain_ft_status)


def _update_status(**kwargs):
    """Thread-safe durum guncelle."""
    with _domain_ft_lock:
        _domain_ft_status.update(kwargs)


def _log(msg: str):
    """Durum loguna mesaj ekle."""
    with _domain_ft_lock:
        _domain_ft_status["log"].append(msg)
        _domain_ft_status["progress"] = msg


class DomainAugmentation:
    """
    Kullanici tarafindan tanimlanan domain augmentation teknikleri:
    JPEG(Q=30-95) + Resize(0.25-0.75x) + Noise + Gamma + Color | %50 olasilikla
    """
    def __init__(self, size: int = 224, p: float = 0.5):
        self.size = size
        self.p = p

    def __call__(self, img):
        from PIL import Image, ImageEnhance
        import numpy as np
        import io

        # %50 olasilikla uygula
        if random.random() > self.p:
            return img

        # 1. Resize (0.25 - 0.75x) ve tekrar buyutme
        w, h = img.size
        scale = random.uniform(0.25, 0.75)
        sw, sh = int(w * scale), int(h * scale)
        if sw > 5 and sh > 5:
            img = img.resize((sw, sh), Image.BILINEAR)
            img = img.resize((w, h), Image.BILINEAR)

        # 2. JPEG compression (Q=30-95)
        quality = random.randint(30, 95)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # 3. Gaussian Noise
        img_np = np.array(img).astype(np.float32)
        noise_std = random.uniform(0.0, 15.0)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0.0, 255.0).astype(np.uint8)
            img = Image.fromarray(img_np)

        # 4. Gamma correction
        gamma = random.uniform(0.7, 1.5)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.power(img_np, gamma)
        img_np = np.clip(img_np * 255.0, 0.0, 255.0).astype(np.uint8)
        img = Image.fromarray(img_np)

        # 5. Color adjustments (brightness, contrast, saturation)
        img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))

        return img


def get_domain_train_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        DomainAugmentation(size=size, p=0.5),
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def discover_available_datasets() -> List[Dict]:
    """Kullanilabilir harici veri setlerini kesfet."""
    datasets = []
    if not EXTERNAL_TESTS_DIR.exists():
        return datasets

    for d in sorted(EXTERNAL_TESTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        real_dir = d / "real"
        fake_dir = d / "fake"
        if not real_dir.exists() or not fake_dir.exists():
            continue

        real_count = sum(1 for f in real_dir.rglob("*") if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        fake_count = sum(1 for f in fake_dir.rglob("*") if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"})

        if real_count < 5 or fake_count < 5:
            continue

        meta = KNOWN_DATASETS.get(d.name, {})
        datasets.append({
            "name": d.name,
            "path": str(d),
            "label": meta.get("label", d.name),
            "icon": meta.get("icon", "📁"),
            "real_count": real_count,
            "fake_count": fake_count,
            "balanced": min(real_count, fake_count),
        })

    return datasets


def _create_balanced_dataset(dataset_names: List[str], val_ratio: float = 0.2):
    """
    Secilen harici veri setlerini birlestir.
    50/50 dengeli, train/val split.
    DeepfakeDataset kullanarak tam pipeline (RGB+Freq+Mesh).
    """
    from core.data_pipeline import DeepfakeDataset

    train_datasets = []
    val_datasets = []

    for ds_name in dataset_names:
        ds_path = EXTERNAL_TESTS_DIR / ds_name
        if not ds_path.exists():
            _log(f"  ⚠️ {ds_name} bulunamadi, atlaniyor")
            continue

        real_dir = ds_path / "real"
        fake_dir = ds_path / "fake"
        if not real_dir.exists() or not fake_dir.exists():
            continue

        # Tam pipeline ile dataset olustur (ozel DomainAugmentation transformasyonu ile)
        train_ds = DeepfakeDataset(
            str(ds_path), split="train", source_tag=f"domain_{ds_name}",
            transform=get_domain_train_transform(model_cfg.IMG_SIZE)
        )

        # Val icin augmentation'siz
        val_ds = DeepfakeDataset(
            str(ds_path), split="val", source_tag=f"domain_{ds_name}"
        )

        if len(train_ds) < 10:
            _log(f"  ⚠️ {ds_name}: yetersiz veri ({len(train_ds)}), atlaniyor")
            continue

        # 50/50 dengeli ornekleme
        real_indices = [i for i, (_, label, *_) in enumerate(train_ds.samples) if label == 0]
        fake_indices = [i for i, (_, label, *_) in enumerate(train_ds.samples) if label == 1]
        balanced_n = min(len(real_indices), len(fake_indices))

        if balanced_n < 5:
            continue

        random.seed(42)
        sel_real = random.sample(real_indices, balanced_n)
        sel_fake = random.sample(fake_indices, balanced_n)
        all_indices = sel_real + sel_fake
        random.shuffle(all_indices)

        # Train / val split
        split_idx = int(len(all_indices) * (1 - val_ratio))
        train_idx = all_indices[:split_idx]
        val_idx = all_indices[split_idx:]

        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(val_ds, val_idx)

        train_datasets.append(train_subset)
        val_datasets.append(val_subset)

        meta = KNOWN_DATASETS.get(ds_name, {})
        _log(f"  ✅ {meta.get('icon', '📁')} {meta.get('label', ds_name)}: "
             f"{balanced_n}×2 = {balanced_n*2} (train={len(train_idx)}, val={len(val_idx)})")

    if not train_datasets:
        return None, None

    train_combined = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_combined = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    return train_combined, val_combined


def _get_external_eval_loaders(batch_size: int = 8, max_samples_per_class: int = 250) -> Dict[str, DataLoader]:
    """Her epoch sonunda harici AUC hesabi icin dengeli val loader'lari hazirla."""
    from core.data_pipeline import DeepfakeDataset
    loaders = {}
    datasets = discover_available_datasets()
    pin = DEVICE.type == "cuda"

    for ds in datasets:
        ds_name = ds["name"]
        ds_path = EXTERNAL_TESTS_DIR / ds_name

        try:
            val_ds = DeepfakeDataset(
                str(ds_path), split="val", source_tag=f"domain_{ds_name}"
            )

            real_indices = [i for i, (_, label, *_) in enumerate(val_ds.samples) if label == 0]
            fake_indices = [i for i, (_, label, *_) in enumerate(val_ds.samples) if label == 1]

            balanced_n = min(len(real_indices), len(fake_indices))
            if balanced_n < 5:
                continue

            n_samples = min(balanced_n, max_samples_per_class)

            random.seed(42)
            sel_real = random.sample(real_indices, n_samples)
            sel_fake = random.sample(fake_indices, n_samples)
            all_indices = sel_real + sel_fake

            subset = Subset(val_ds, all_indices)
            loader = DataLoader(
                subset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=pin
            )
            loaders[ds_name] = loader
            _log(f"🔍 Evaluator hazirlandi: {ds['label']} (N={len(all_indices)})")
        except Exception as e:
            _log(f"⚠️ Evaluator hazirlanirken hata ({ds_name}): {e}")

    return loaders


def _freeze_all(model):
    """Tum parametreleri dondur."""
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_fusion_classifier(model):
    """Sadece CrossBranchTransformer + classifier ac."""
    # cross_branch (CrossBranchTransformer)
    for param in model.cross_branch.parameters():
        param.requires_grad = True
    # classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    # mesh_proj (kucuk MLP — acmak faydali)
    for param in model.mesh_proj.parameters():
        param.requires_grad = True


def _unfreeze_backbone(model, backbone_lr_factor: float = 0.1):
    """Backbone'u da ac (discriminative LR icin parametreleri grupla)."""
    # RGB backbone — EfficientNet-B4 veya MobileNet
    if hasattr(model, 'rgb_backbone'):
        for param in model.rgb_backbone.parameters():
            param.requires_grad = True
    elif hasattr(model, 'rgb_features'):
        for param in model.rgb_features.parameters():
            param.requires_grad = True
    # RGB projeksiyon
    if hasattr(model, 'rgb_proj'):
        for param in model.rgb_proj.parameters():
            param.requires_grad = True
    # Frekans backbone
    for param in model.freq_features.parameters():
        param.requires_grad = True
    # mesh_mlp de ac
    for param in model.mesh_mlp.parameters():
        param.requires_grad = True


def _create_optimizer(model, lr: float, phase: int):
    """Faz bazli optimizer olustur."""
    if phase == 1:
        # Faz 1: sadece fusion + classifier
        trainable = [p for p in model.parameters() if p.requires_grad]
        return AdamW(trainable, lr=lr, weight_decay=1e-2)
    else:
        # Faz 2: discriminative LR
        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "rgb_backbone" in name or "rgb_features" in name or "freq_features" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        backbone_lr = lr * 0.1
        return AdamW([
            {"params": other_params, "lr": lr},
            {"params": backbone_params, "lr": backbone_lr},
        ], weight_decay=1e-2)


def _run_domain_finetune(
    model,
    original_state_dict: dict,
    dataset_names: List[str],
    epochs: int = 8,
    lr: float = 5e-5,
    batch_size: int = 8,
    phase1_epochs: int = 3,
):
    """Arka plan thread'inde domain fine-tune calistir."""
    try:
        _update_status(
            running=True, completed=False, error=None,
            epoch=0, total_epochs=epochs, best_auc=0.0,
            log=[], progress="Dataset hazirlaniyor..."
        )

        # Dataset olustur
        _log("📦 Veri setleri hazirlaniyor (tam pipeline: RGB+Freq+Mesh)...")
        train_dataset, val_dataset = _create_balanced_dataset(dataset_names)

        if train_dataset is None or len(train_dataset) < 10:
            _update_status(running=False, error="Yetersiz veri — fine-tune iptal.")
            return

        _log(f"📊 Toplam: train={len(train_dataset)}, val={len(val_dataset)}")

        # Harici test setleri evaluator'larini olustur
        eval_loaders = _get_external_eval_loaders(batch_size=batch_size)

        # DataLoader
        pin = DEVICE.type == "cuda"
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=pin, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=pin,
        )

        # Mevcut modeli yedekle
        if DOMAIN_FINETUNED_PATH.exists():
            shutil.copy2(str(DOMAIN_FINETUNED_PATH), str(DOMAIN_BACKUP_PATH))

        # Loss
        from core.loss_utils import CombinedLoss
        criterion = CombinedLoss()

        # Mixed precision
        use_amp = model_cfg.USE_MIXED_PRECISION and DEVICE.type == "cuda"
        scaler = GradScaler() if use_amp else None

        best_combined_score = 0.0
        patience = 0
        max_patience = 4

        for epoch in range(epochs):
            # ── Faz kontrolu ──
            if epoch < phase1_epochs:
                phase = 1
                if epoch == 0:
                    _freeze_all(model)
                    _unfreeze_fusion_classifier(model)
                    optimizer = _create_optimizer(model, lr, phase=1)
                    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
                    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
                    total_count = sum(1 for p in model.parameters())
                    _log(f"🧊 Faz 1: Fusion + Classifier ({trainable_count}/{total_count} parametre)")
            else:
                phase = 2
                if epoch == phase1_epochs:
                    _unfreeze_backbone(model)
                    optimizer = _create_optimizer(model, lr * 0.5, phase=2)
                    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - phase1_epochs, eta_min=lr * 0.01)
                    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
                    total_count = sum(1 for p in model.parameters())
                    _log(f"🔥 Faz 2: Backbone acildi ({trainable_count}/{total_count} parametre)")

            phase_label = "FUSION" if phase == 1 else "FULL"

            # ── Train ──
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                rgb, freq, mesh, labels, source_tags = batch
                rgb = rgb.to(DEVICE)
                freq = freq.to(DEVICE)
                mesh = mesh.to(DEVICE)
                labels = labels.to(DEVICE)

                if use_amp and scaler is not None:
                    with autocast(device_type="cuda"):
                        logits = model(rgb, freq, mesh)
                        losses = criterion(logits, labels)
                    loss = losses["total_loss"]
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    logits = model(rgb, freq, mesh)
                    losses = criterion(logits, labels)
                    loss = losses["total_loss"]
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            scheduler.step()
            avg_train_loss = train_loss / max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # ── Validation (Hold-out Domain) ──
            model.eval()
            val_probs = []
            val_labels_list = []

            with torch.no_grad():
                for batch in val_loader:
                    rgb, freq, mesh, labels, source_tags = batch
                    rgb = rgb.to(DEVICE)
                    freq = freq.to(DEVICE)
                    mesh = mesh.to(DEVICE)
                    labels = labels.to(DEVICE)

                    if use_amp:
                        with autocast(device_type="cuda"):
                            logits = model(rgb, freq, mesh)
                    else:
                        logits = model(rgb, freq, mesh)

                    probs = torch.softmax(logits, dim=1)[:, 1]
                    val_probs.extend(probs.cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())

            # AUC hesapla
            val_auc = 0.5
            try:
                from sklearn.metrics import roc_auc_score
                if len(set(val_labels_list)) > 1:
                    val_auc = roc_auc_score(val_labels_list, val_probs)
            except Exception:
                pass

            # ── Harici Test Setleri Degerlendirmesi (Epoch Sonu) ──
            external_aucs = {}
            from sklearn.metrics import roc_auc_score

            with torch.no_grad():
                for ext_name, loader in eval_loaders.items():
                    ext_probs = []
                    ext_labels_list = []

                    for batch in loader:
                        rgb, freq, mesh, labels, source_tags = batch
                        rgb = rgb.to(DEVICE)
                        freq = freq.to(DEVICE)
                        mesh = mesh.to(DEVICE)
                        labels = labels.to(DEVICE)

                        if use_amp:
                            with autocast(device_type="cuda"):
                                logits = model(rgb, freq, mesh)
                        else:
                            logits = model(rgb, freq, mesh)

                        probs = torch.softmax(logits, dim=1)[:, 1]
                        ext_probs.extend(probs.cpu().numpy())
                        ext_labels_list.extend(labels.cpu().numpy())

                    ext_auc = 0.5
                    if len(set(ext_labels_list)) > 1:
                        ext_auc = roc_auc_score(ext_labels_list, ext_probs)
                    external_aucs[ext_name] = ext_auc

            # Ortalama Harici AUC
            if external_aucs:
                ext_mean_auc = np.mean(list(external_aucs.values()))
            else:
                ext_mean_auc = 0.5

            # Combined Score Hesabi
            combined_score = 0.5 * val_auc + 0.5 * ext_mean_auc

            lr_current = optimizer.param_groups[0]["lr"]
            epoch_msg = (
                f"Epoch {epoch+1}/{epochs} [{phase_label}] | "
                f"Loss: {avg_train_loss:.4f} | Acc: {train_acc:.3f} | "
                f"Val AUC: {val_auc:.4f} | Ext Mean AUC: {ext_mean_auc:.4f} | "
                f"Combined Score: {combined_score:.4f} | LR: {lr_current:.2e}"
            )
            _log(epoch_msg)

            ext_details = ", ".join([f"{KNOWN_DATASETS.get(k, {}).get('label', k)}: {v:.4f}" for k, v in external_aucs.items()])
            if ext_details:
                _log(f"  📊 Harici Detaylar -> {ext_details}")

            _update_status(
                epoch=epoch + 1, train_loss=round(avg_train_loss, 4),
                val_auc=round(val_auc, 4), phase=phase_label,
            )

            # Combined Score ile model kaydet
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                patience = 0
                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_auc,
                    "ext_mean_auc": ext_mean_auc,
                    "combined_score": combined_score,
                    "domains": dataset_names,
                    "type": "domain_finetuned",
                }
                torch.save(checkpoint_data, str(BEST_MODEL_GENERALIZED_PATH))
                torch.save(checkpoint_data, str(DOMAIN_FINETUNED_PATH))
                _log(f"  💾 Best model kaydedildi — Combined Score: {combined_score:.4f} (Val AUC: {val_auc:.4f}, Ext AUC: {ext_mean_auc:.4f})")
                _update_status(best_auc=round(combined_score, 4))
            else:
                patience += 1

            # Early stopping
            if patience >= max_patience:
                _log(f"⏹️ Early stopping — {max_patience} epoch iyilesme yok")
                break

        # Tamamlandi
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        _log(f"\n✅ Domain fine-tune tamamlandi! Best Combined Score: {best_combined_score:.4f}")
        _update_status(running=False, completed=True)

    except Exception as e:
        import traceback
        # Hata durumunda orijinal modele geri don
        try:
            model.load_state_dict(original_state_dict)
            model.eval()
        except Exception:
            pass
        error_msg = f"{e}\n{traceback.format_exc()}"
        _log(f"❌ Hata: {error_msg}")
        _update_status(running=False, error=error_msg)


def start_domain_finetune(
    model,
    dataset_names: List[str],
    epochs: int = 8,
    lr: float = 5e-5,
    batch_size: int = 8,
    phase1_epochs: int = 3,
) -> str:
    """Domain fine-tune'u arka plan thread'inde baslat."""
    status = get_domain_ft_status()
    if status["running"]:
        return "⏳ Domain fine-tune zaten calisiyor..."

    if not dataset_names:
        return "⚠️ En az bir veri seti secin."

    # Orijinal state'i yedekle (rollback icin)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    thread = threading.Thread(
        target=_run_domain_finetune,
        args=(model, original_state, dataset_names, epochs, lr, batch_size, phase1_epochs),
        daemon=True,
    )
    thread.start()
    return f"🚀 Domain fine-tune baslatildi! {len(dataset_names)} veri seti, {epochs} epoch"


def rollback_domain_finetune(model) -> str:
    """Domain fine-tuned modeli geri al, orijinal checkpoint'e don."""
    original_path = paths.BEST_MODEL_PATH

    if not original_path.exists():
        return "❌ Orijinal model bulunamadi!"

    try:
        checkpoint = torch.load(str(original_path), map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()

        # Fine-tuned modelleri sil
        if DOMAIN_FINETUNED_PATH.exists():
            DOMAIN_FINETUNED_PATH.unlink()
        if BEST_MODEL_GENERALIZED_PATH.exists():
            BEST_MODEL_GENERALIZED_PATH.unlink()

        return "↩️ Orijinal modele geri donuldu."
    except Exception as e:
        return f"❌ Rollback hatasi: {e}"


def get_domain_ft_log() -> str:
    """Fine-tune logunu Markdown olarak dondur."""
    with _domain_ft_lock:
        log = _domain_ft_status.get("log", [])
    if not log:
        return "_Henuz log yok._"
    return "\n".join([f"- {line}" for line in log])
