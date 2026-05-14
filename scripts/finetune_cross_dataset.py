"""
Cross-Dataset Fine-Tuning — DeepfakeULTRA V5

best_model.pth (AUC=0.96) uzerine mixed fine-tuning.
Harici datasetlerden ornekler ekleyerek cross-dataset performansini iyilestirir.

Strateji:
  Epoch 0-1: Backbone frozen, LR=1e-4 (classifier adaptasyonu)
  Epoch 2-4: Backbone unfrozen, discriminative LR (backbone:1e-5, classifier:1e-4)
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
from core.data_pipeline import get_dataloaders
from core.loss_utils import CombinedLoss

# sklearn
from sklearn.metrics import roc_auc_score, f1_score

# ── Sabitler ──
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-4
BACKBONE_LR = 1e-5
UNFREEZE_EPOCH = 3
BATCH_SIZE = model_cfg.BATCH_SIZE
ACCUM_STEPS = model_cfg.GRADIENT_ACCUMULATION_STEPS
SAVE_PATH = paths.MODEL_DIR / "best_model_crossdataset.pth"


def freeze_backbone(model):
    for p in model.rgb_features.parameters():
        p.requires_grad = False
    for p in model.freq_features.parameters():
        p.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f"  🧊 Backbone donduruldu — {frozen}/{total} parametre donuk")


def unfreeze_backbone(model):
    for p in model.rgb_features.parameters():
        p.requires_grad = True
    for p in model.freq_features.parameters():
        p.requires_grad = True
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f"  🔥 Backbone acildi — {trainable}/{total} parametre egitilebilir")


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
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [FT-Train]", leave=False)

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


def main():
    print("=" * 60)
    print("  Cross-Dataset Fine-Tuning — DeepfakeULTRA V5")
    print("=" * 60)

    # Model yukle
    model = DualPathDeepfakeDetector().to(DEVICE)
    ckpt_path = paths.BEST_MODEL_PATH
    if not ckpt_path.exists():
        print(f"❌ Checkpoint bulunamadi: {ckpt_path}")
        return

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    prev_auc = ckpt.get("val_auc", 0.0)
    print(f"  📂 Checkpoint yuklendi: AUC={prev_auc:.4f}")

    # Backbone dondur
    freeze_backbone(model)

    # Data — mevcut train/val (harici ornekler zaten eklendi)
    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)
    print(f"  📊 Train: {len(train_loader.dataset):,} gorsel")
    print(f"  📊 Val: {len(val_loader.dataset):,} gorsel")

    # Loss + Optimizer
    from core.data_pipeline import compute_class_weights
    weights = compute_class_weights(train_loader.dataset)
    criterion = CombinedLoss(class_weights=weights)
    optimizer = create_optimizer(model, backbone_unfrozen=False)

    # AMP
    scaler = GradScaler() if DEVICE.type == "cuda" else None

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6)

    best_auc = 0.0
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
        phase = "🧊FROZEN" if not backbone_unfrozen else "🔥UNFROZEN"

        print(
            f"Epoch {epoch+1}/{FINETUNE_EPOCHS} [{phase}] | "
            f"Train Loss: {train_m['loss']:.4f} Acc: {train_m['accuracy']:.3f} | "
            f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.3f} "
            f"AUC: {val_m['auc']:.4f} F1: {val_m['macro_f1']:.4f} | LR: {lr:.2e}"
        )

        # Best model kaydet
        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": best_auc,
                "val_acc": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"],
                "num_classes": model_cfg.NUM_CLASSES,
                "class_names": model_cfg.CLASS_NAMES,
                "run": "crossdataset_finetune",
            }, str(SAVE_PATH))
            # Uyumluluk icin best_model.pth olarak da kaydet
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": best_auc,
                "num_classes": model_cfg.NUM_CLASSES,
                "class_names": model_cfg.CLASS_NAMES,
                "run": "crossdataset_finetune",
            }, str(paths.BEST_MODEL_PATH))
            print(f"  💾 Best model kaydedildi — AUC: {best_auc:.4f}")

    print(f"\n✅ Fine-tuning tamamlandi! Best AUC: {best_auc:.4f}")
    print(f"   Model: {SAVE_PATH}")

    # Otomatik benchmark
    print("\n" + "=" * 60)
    print("  Otomatik Cross-Dataset Benchmark")
    print("=" * 60)

    ext_base = paths.BASE_DIR / "dataset" / "external_tests"
    datasets = ["celeb_df_v2", "faceforensics", "dfdc", "deepfake20k", "deepfakeface"]

    for ds in datasets:
        ds_path = ext_base / ds
        if ds_path.exists():
            print(f"\n🔄 {ds} degerlendiriliyor...")
            os.system(
                f'python -u "{paths.BASE_DIR / "scripts" / "evaluate_model.py"}" '
                f'--external "{ds_path}"'
            )

    print("\n✅ Tum benchmarklar tamamlandi!")


if __name__ == "__main__":
    main()
