"""
Domain Generalization Fine-Tune — DeepfakeULTRA V5

best_model.pth (AUC=0.9825) uzerine domain-robust fine-tuning.
Agresif domain augmentasyonu ile cross-dataset genelleme iyilestirmesi.

Strateji:
  - Domain augmentation %50 olasililikla her gorsele uygula
  - Dusuk LR (5e-5) ile mevcut performansi koruyarak genelleme ekle
  - Epoch 0-2: Backbone frozen (domain augmentation adapte)
  - Epoch 3-7: Backbone unfrozen, discriminative LR
  - Her epoch sonunda harici benchmark
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from config import model_cfg, paths, DEVICE, NUM_WORKERS
from core.dual_mobilenetv3 import DualPathDeepfakeDetector
from core.data_pipeline import get_dataloaders, DeepfakeDataset
from core.loss_utils import CombinedLoss
from core.domain_augmentation import DomainAugmentation

from sklearn.metrics import roc_auc_score, f1_score

# -- Sabitler --
FINETUNE_EPOCHS = 8
FINETUNE_LR = 5e-5       # Dusuk LR — mevcut performansi koru
BACKBONE_LR = 5e-6
UNFREEZE_EPOCH = 3
BATCH_SIZE = model_cfg.BATCH_SIZE
ACCUM_STEPS = model_cfg.GRADIENT_ACCUMULATION_STEPS
SAVE_PATH = paths.MODEL_DIR / "best_model_generalized.pth"

# Harici benchmark dataset'leri
EXT_BASE = paths.BASE_DIR / "dataset" / "external_tests"
BENCHMARK_DATASETS = ["celeb_df_v2", "faceforensics", "dfdc", "deepfake20k", "deepfakeface"]


def freeze_backbone(model):
    for p in model.rgb_features.parameters():
        p.requires_grad = False
    for p in model.freq_features.parameters():
        p.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f"  [FROZEN] Backbone donduruldu -- {frozen}/{total} parametre donuk")


def unfreeze_backbone(model):
    for p in model.rgb_features.parameters():
        p.requires_grad = True
    for p in model.freq_features.parameters():
        p.requires_grad = True
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f"  [UNFROZEN] Backbone acildi -- {trainable}/{total} parametre egitilebilir")


def create_optimizer(model, backbone_unfrozen=False):
    if not backbone_unfrozen:
        params = [p for p in model.parameters() if p.requires_grad]
        return AdamW(params, lr=FINETUNE_LR, weight_decay=1e-4)

    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "rgb_features" in name or "freq_features" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)

    return AdamW([
        {"params": other_params, "lr": FINETUNE_LR},
        {"params": backbone_params, "lr": BACKBONE_LR},
    ], weight_decay=1e-4)


def train_epoch(model, loader, criterion, optimizer, epoch, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = DEVICE.type == "cuda"

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [DG-Train]", leave=False)

    for step, batch in enumerate(pbar):
        rgb, freq, mesh, labels, _ = batch
        rgb, freq = rgb.to(DEVICE), freq.to(DEVICE)
        mesh, labels = mesh.to(DEVICE), labels.to(DEVICE)

        if use_amp and scaler:
            with autocast(device_type="cuda"):
                logits = model(rgb, freq, mesh)
                losses = criterion(logits, labels)
                loss = losses["total_loss"] / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(rgb, freq, mesh)
            losses = criterion(logits, labels)
            loss = losses["total_loss"] / ACCUM_STEPS
            loss.backward()
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += losses["total_loss"].item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{losses['total_loss'].item():.4f}", acc=f"{correct/total:.3f}")

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for batch in loader:
        rgb, freq, mesh, labels, _ = batch
        rgb, freq = rgb.to(DEVICE), freq.to(DEVICE)
        mesh, labels = mesh.to(DEVICE), labels.to(DEVICE)

        logits = model(rgb, freq, mesh)
        losses = criterion(logits, labels)

        total_loss += losses["total_loss"].item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        probs = torch.softmax(logits, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs_np = np.array(all_probs)
    auc = 0.5
    try:
        auc = roc_auc_score(all_labels, all_probs_np[:, 1])
    except Exception:
        pass

    macro_f1 = 0.0
    try:
        all_preds = np.argmax(all_probs_np, axis=1)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    except Exception:
        pass

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "auc": auc,
        "macro_f1": macro_f1,
    }


@torch.no_grad()
def quick_external_benchmark(model, dataset_name):
    """Tek harici dataset icin hizli AUC hesapla."""
    ds_path = EXT_BASE / dataset_name
    if not ds_path.exists():
        return None

    from torch.utils.data import DataLoader

    ds = DeepfakeDataset(str(ds_path), split="val", source_tag=dataset_name)
    if len(ds) == 0:
        return None

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers=False)

    all_probs, all_labels = [], []
    for batch in loader:
        rgb, freq, mesh, labels, _ = batch
        rgb, freq, mesh = rgb.to(DEVICE), freq.to(DEVICE), mesh.to(DEVICE)
        logits = model(rgb, freq, mesh)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    try:
        auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        return round(auc, 4)
    except Exception:
        return None


class _DomainAugWrapper:
    """Picklable wrapper for DomainAugmentation (Python 3.14 uyumlu)."""
    def __init__(self, domain_aug):
        self.domain_aug = domain_aug

    def __call__(self, img):
        return self.domain_aug(img)


def inject_domain_augmentation(dataset):
    """Dataset'in transform'una DomainAugmentation ekle."""
    domain_aug = DomainAugmentation(prob=0.5)
    wrapper = _DomainAugWrapper(domain_aug)

    if hasattr(dataset, 'datasets'):
        # ConcatDataset
        for ds in dataset.datasets:
            _inject_into_single_dataset(ds, wrapper)
    else:
        _inject_into_single_dataset(dataset, wrapper)


def _inject_into_single_dataset(ds, wrapper):
    """Tek bir DeepfakeDataset'e domain augmentation ekle."""
    if not hasattr(ds, 'transform') or ds.transform is None:
        return

    from torchvision import transforms

    # Mevcut transform listesinin basina DomainAugmentation ekle
    # (Resize'dan sonra, ToTensor'dan once)
    old_transforms = ds.transform.transforms
    new_transforms = []

    inserted = False
    for t in old_transforms:
        # Resize'dan sonra DomainAugmentation ekle
        new_transforms.append(t)
        if isinstance(t, transforms.Resize) and not inserted:
            new_transforms.append(transforms.RandomApply([wrapper], p=0.5))
            inserted = True

    ds.transform = transforms.Compose(new_transforms)


def main():
    print("=" * 60)
    print("  Domain Generalization Fine-Tune -- DeepfakeULTRA V5")
    print("=" * 60)

    # Model yukle
    model = DualPathDeepfakeDetector().to(DEVICE)
    ckpt_path = paths.BEST_MODEL_PATH
    if not ckpt_path.exists():
        print(f"[HATA] Checkpoint bulunamadi: {ckpt_path}")
        return

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    prev_auc = ckpt.get("val_auc", 0.0)
    print(f"  Checkpoint yuklendi: AUC={prev_auc:.4f}")

    # Backbone dondur
    freeze_backbone(model)

    # Data — mevcut train/val + domain augmentation
    print("\nVeri yukleniyor...")
    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    # Domain augmentation enjekte et
    print("  Domain augmentation enjekte ediliyor...")
    inject_domain_augmentation(train_loader.dataset)
    print(f"  Train: {len(train_loader.dataset):,} gorsel")
    print(f"  Val: {len(val_loader.dataset):,} gorsel")

    # Baslangic harici benchmark
    print("\n--- Baslangic Harici Benchmark ---")
    baseline_aucs = {}
    for ds_name in BENCHMARK_DATASETS:
        auc = quick_external_benchmark(model, ds_name)
        if auc is not None:
            baseline_aucs[ds_name] = auc
            print(f"  {ds_name}: AUC={auc:.4f}")

    # Loss + Optimizer
    from core.data_pipeline import compute_class_weights
    weights = compute_class_weights(train_loader.dataset)
    criterion = CombinedLoss(class_weights=weights)
    optimizer = create_optimizer(model, backbone_unfrozen=False)

    # AMP
    scaler = GradScaler() if DEVICE.type == "cuda" else None

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6)

    best_combined_score = 0.0
    backbone_unfrozen = False

    for epoch in range(FINETUNE_EPOCHS):
        # Unfreeze kontrolu
        if not backbone_unfrozen and epoch >= UNFREEZE_EPOCH:
            unfreeze_backbone(model)
            backbone_unfrozen = True
            optimizer = create_optimizer(model, backbone_unfrozen=True)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=FINETUNE_EPOCHS - epoch, eta_min=1e-6
            )
            if scaler:
                scaler = GradScaler()

        # Train
        train_m = train_epoch(model, train_loader, criterion, optimizer, epoch, scaler)
        val_m = validate_epoch(model, val_loader, criterion)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        phase = "[FROZEN]" if not backbone_unfrozen else "[UNFROZEN]"

        print(
            f"Epoch {epoch+1}/{FINETUNE_EPOCHS} {phase} | "
            f"Train Loss: {train_m['loss']:.4f} Acc: {train_m['accuracy']:.3f} | "
            f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.3f} "
            f"AUC: {val_m['auc']:.4f} F1: {val_m['macro_f1']:.4f} | LR: {lr:.2e}"
        )

        # Harici benchmark (her epoch)
        ext_aucs = {}
        for ds_name in BENCHMARK_DATASETS:
            auc = quick_external_benchmark(model, ds_name)
            if auc is not None:
                ext_aucs[ds_name] = auc

        # Harici AUC degisimlerini goster
        ext_summary = []
        for ds_name, auc in ext_aucs.items():
            base = baseline_aucs.get(ds_name, 0)
            diff = auc - base
            sign = "+" if diff > 0 else ""
            ext_summary.append(f"{ds_name}={auc:.3f}({sign}{diff:.3f})")
        if ext_summary:
            print(f"  Ext: {' | '.join(ext_summary)}")

        # Combined score: val_auc * 0.5 + ext_mean_auc * 0.5
        if ext_aucs:
            ext_mean = np.mean(list(ext_aucs.values()))
            combined = val_m["auc"] * 0.5 + ext_mean * 0.5
        else:
            combined = val_m["auc"]

        # Best model kaydet (combined score bazinda)
        if combined > best_combined_score:
            best_combined_score = combined
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_m["auc"],
                "val_acc": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"],
                "ext_aucs": ext_aucs,
                "combined_score": combined,
                "num_classes": model_cfg.NUM_CLASSES,
                "class_names": model_cfg.CLASS_NAMES,
                "run": "domain_generalization",
            }, str(SAVE_PATH))
            # Ana model olarak da kaydet
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_m["auc"],
                "num_classes": model_cfg.NUM_CLASSES,
                "class_names": model_cfg.CLASS_NAMES,
                "run": "domain_generalization",
            }, str(paths.BEST_MODEL_PATH))
            print(f"  [BEST] Combined={combined:.4f} (val={val_m['auc']:.4f} + ext_mean={ext_mean:.4f})")

    # Final
    print(f"\n{'='*60}")
    print(f"  Fine-tuning tamamlandi! Best Combined: {best_combined_score:.4f}")
    print(f"  Model: {SAVE_PATH}")
    print(f"{'='*60}")

    # Final benchmark
    print("\n--- Final Harici Benchmark ---")
    for ds_name in BENCHMARK_DATASETS:
        auc = quick_external_benchmark(model, ds_name)
        base = baseline_aucs.get(ds_name, 0)
        if auc is not None:
            diff = auc - base
            sign = "+" if diff > 0 else ""
            print(f"  {ds_name}: AUC={auc:.4f} ({sign}{diff:.4f})")


if __name__ == "__main__":
    main()
