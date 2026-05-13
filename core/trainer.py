"""
Deepfake Detection System v4 — Eğitim Pipeline
AdamW + Warmup + CosineAnnealing + ReduceLROnPlateau + KD + GradClip + EarlyStopping + MLflow.
Kademeli Backbone Unfreeze + Mixup/CutMix + Per-class Metrik + Confusion Matrix.
Binary sınıflandırma: REAL (0) / FAKE (1)
RTX 4070 Laptop optimizasyonu: FP16 Mixed Precision + Gradient Accumulation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import time
from config import model_cfg, paths, DEVICE
# CUDA Performans Optimizasyonları — GPU MAX kullanım
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True         # Convolution için en hızlı algoritmayı seç
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 matmul (RTX 40xx Ada Lovelace)
    torch.backends.cudnn.allow_tf32 = True          # TF32 convolution
    torch.set_float32_matmul_precision("high")      # Yüksek hassasiyet + hız dengesi
from core.dual_mobilenetv3 import DualPathDeepfakeDetector
# Teacher: opsiyonel (KD_ALPHA > 0 ise yüklenir)
try:
    from core.efficientnet_teacher import EfficientNetTeacher, load_pretrained_teacher
    HAS_TEACHER = True
except ImportError:
    HAS_TEACHER = False
from core.loss_utils import CombinedLoss
from core.data_pipeline import (
    get_dataloaders, get_ffpp_dataloaders,
    cutmix_data, compute_class_weights,
)

# MLflow opsiyonel
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# sklearn opsiyonel (per-class metrik)
try:
    from sklearn.metrics import (
        roc_auc_score, classification_report,
        confusion_matrix as sk_confusion_matrix, f1_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════
# BACKBONE FREEZE/UNFREEZE
# ═══════════════════════════════════════════════════════════
def freeze_backbone(model):
    """Pretrained backbone katmanlarını dondur — sadece classifier öğrensin."""
    for param in model.rgb_features.parameters():
        param.requires_grad = False
    for param in model.freq_features.parameters():
        param.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f"  🧊 Backbone donduruldu — {frozen}/{total} parametre donuk")


def unfreeze_backbone(model):
    """Backbone'u aç — fine-tuning başlasın."""
    for param in model.rgb_features.parameters():
        param.requires_grad = True
    for param in model.freq_features.parameters():
        param.requires_grad = True
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f"  🔥 Backbone açıldı — {trainable}/{total} parametre eğitilebilir")


def create_discriminative_optimizer(model, base_lr):
    """Backbone için düşük LR, diğer katmanlar için yüksek LR."""
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "rgb_features" in name or "freq_features" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    backbone_lr = base_lr * model_cfg.BACKBONE_LR_FACTOR
    param_groups = [
        {"params": other_params, "lr": base_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ]
    optimizer = AdamW(param_groups, weight_decay=model_cfg.WEIGHT_DECAY)
    print(f"  📐 Discriminative LR: backbone={backbone_lr:.2e}, diğer={base_lr:.2e}")
    return optimizer


# ═══════════════════════════════════════════════════════════
# MIXUP DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════
def mixup_data(rgb, freq, mesh, labels, alpha=0.2):
    """Mixup: iki örneği karıştırarak sınıf sınırlarını yumuşatır."""
    if alpha <= 0:
        return rgb, freq, mesh, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(rgb.size(0), device=rgb.device)

    mixed_rgb = lam * rgb + (1 - lam) * rgb[idx]
    mixed_freq = lam * freq + (1 - lam) * freq[idx]
    mixed_mesh = lam * mesh + (1 - lam) * mesh[idx]

    return mixed_rgb, mixed_freq, mixed_mesh, labels, labels[idx], lam


def mixup_criterion(criterion, logits, labels_a, labels_b, lam):
    """Mixup için karışık loss hesaplama."""
    loss_a = criterion(logits, labels_a)
    loss_b = criterion(logits, labels_b)
    mixed = {}
    for key in loss_a:
        mixed[key] = lam * loss_a[key] + (1 - lam) * loss_b[key]
    return mixed


# ═══════════════════════════════════════════════════════════
# WARMUP SCHEDULER
# ═══════════════════════════════════════════════════════════
def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Linear warmup + Cosine annealing scheduler."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════
# CURRICULUM LEARNING YARDIMCILARI (G4)
# ═══════════════════════════════════════════════════════════
def _update_curriculum_ratio(loader, hard_real_ratio: float):
    """DataLoader'daki dataset(ler)in hard_real_aug.prob değerini güncelle."""
    from torch.utils.data import ConcatDataset
    dataset = loader.dataset
    datasets = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    for ds in datasets:
        if hasattr(ds, 'hard_real_aug') and ds.hard_real_aug is not None:
            ds.hard_real_aug.prob = hard_real_ratio


def _get_prev_ratio(epoch: int, cfg) -> float:
    """Bir önceki epoch'un curriculum ratio'sunu döndür (log duplikasyonu engelle)."""
    prev_epoch = epoch - 1
    for phase in cfg.CURRICULUM_PHASES:
        if phase["start"] <= prev_epoch <= phase["end"]:
            return phase["hard_real_ratio"]
    return -1.0


# ═══════════════════════════════════════════════════════════
# SINIF DAĞILIMI LOGLAMA
# ═══════════════════════════════════════════════════════════
def log_class_distribution(loader, split="train"):
    """DataLoader'daki sınıf dağılımını logla."""
    dataset = loader.dataset
    labels = []
    from torch.utils.data import ConcatDataset
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            if hasattr(ds, "samples"):
                labels.extend([l for _, l, *_ in ds.samples])
    elif hasattr(dataset, "samples"):
        labels.extend([l for _, l, *_ in dataset.samples])

    if not labels:
        return

    counts = Counter(labels)
    total = len(labels)
    print(f"  📊 {split} sınıf dağılımı:")
    for cls_id in sorted(counts.keys()):
        name = model_cfg.CLASS_NAMES[cls_id] if cls_id < len(model_cfg.CLASS_NAMES) else f"cls_{cls_id}"
        pct = counts[cls_id] / total * 100
        print(f"     {name}: {counts[cls_id]:,} ({pct:.1f}%)")


# ═══════════════════════════════════════════════════════════
# EĞİTİM EPOCH'U
# ═══════════════════════════════════════════════════════════
def train_epoch(model, teacher, loader, criterion, optimizer, epoch, scaler=None,
                warmup_scheduler=None, use_mixup=False, cutmix_ratio=0.6):
    """Tek epoch egitim — FP16 + Gradient Accumulation + Mixup/CutMix. Binary (REAL/FAKE)."""
    model.train()
    if teacher:
        teacher.eval()
    total_loss, correct, total = 0.0, 0, 0
    accum_steps = model_cfg.GRADIENT_ACCUMULATION_STEPS
    use_amp = model_cfg.USE_MIXED_PRECISION and DEVICE.type == "cuda"

    # Per-class sayaçlar
    class_correct = Counter()
    class_total = Counter()

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for step, batch in enumerate(pbar):
        # 5'li tuple: rgb, freq, mesh, labels, source_tags
        rgb, freq, mesh, labels, source_tags = batch
        rgb, freq = rgb.to(DEVICE), freq.to(DEVICE)
        mesh, labels = mesh.to(DEVICE), labels.to(DEVICE)
        # source_tags string tuple — GPU'ya gonderilmez, sadece loglama icin

        # CutMix / Mixup seçimi
        if use_mixup and model_cfg.USE_MIXUP:
            if np.random.random() < cutmix_ratio:
                # CutMix
                rgb, freq, mesh, labels_a, labels_b, lam = cutmix_data(
                    rgb, freq, mesh, labels, alpha=1.0
                )
            else:
                # Mixup
                rgb, freq, mesh, labels_a, labels_b, lam = mixup_data(
                    rgb, freq, mesh, labels, model_cfg.MIXUP_ALPHA
                )
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        # FGSM Adversarial Augmentation (hafif pertürbasyon)
        use_fgsm = (
            getattr(model_cfg, 'USE_FGSM_TRAINING', False)
            and epoch >= getattr(model_cfg, 'FGSM_START_EPOCH', 2)
            and (step % getattr(model_cfg, 'FGSM_EVERY_N_STEPS', 4)) == 0
        )
        if use_fgsm:
            eps_min = getattr(model_cfg, 'FGSM_EPSILON_MIN', 0.005)
            eps_max = getattr(model_cfg, 'FGSM_EPSILON_MAX', 0.03)
            eps = np.random.uniform(eps_min, eps_max)
            rgb_adv = rgb.detach().clone().requires_grad_(True)
            with autocast(device_type="cuda", enabled=use_amp):
                adv_logits = model(rgb_adv, freq, mesh)
                adv_loss = F.cross_entropy(adv_logits, labels_a)
            adv_loss.backward()
            if rgb_adv.grad is not None:
                rgb = (rgb + eps * rgb_adv.grad.sign()).detach()
            model.zero_grad()

        # Mixed Precision forward
        if use_amp and scaler is not None:
            with autocast(device_type="cuda"):
                # Contrastive: embedding + logits birlikte al
                if hasattr(model, 'forward_with_embeddings'):
                    student_logits, embeddings = model.forward_with_embeddings(rgb, freq, mesh)
                else:
                    student_logits = model(rgb, freq, mesh)
                    embeddings = None

                teacher_logits = None
                if teacher:
                    with torch.no_grad():
                        teacher_logits = teacher(rgb)

                if lam < 1.0:
                    losses = mixup_criterion(
                        lambda lg, lb: criterion(lg, lb, teacher_logits, embeddings),
                        student_logits, labels_a, labels_b, lam
                    )
                else:
                    losses = criterion(student_logits, labels, teacher_logits, embeddings)
                loss = losses["total_loss"] / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), model_cfg.GRADIENT_CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if warmup_scheduler:
                    warmup_scheduler.step()
        else:
            if hasattr(model, 'forward_with_embeddings'):
                student_logits, embeddings = model.forward_with_embeddings(rgb, freq, mesh)
            else:
                student_logits = model(rgb, freq, mesh)
                embeddings = None

            teacher_logits = None
            if teacher:
                with torch.no_grad():
                    teacher_logits = teacher(rgb)

            if lam < 1.0:
                losses = mixup_criterion(
                    lambda lg, lb: criterion(lg, lb, teacher_logits, embeddings),
                    student_logits, labels_a, labels_b, lam
                )
            else:
                losses = criterion(student_logits, labels, teacher_logits, embeddings)
            loss = losses["total_loss"] / accum_steps

            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), model_cfg.GRADIENT_CLIP_MAX_NORM)
                optimizer.step()
                optimizer.zero_grad()
                if warmup_scheduler:
                    warmup_scheduler.step()

        total_loss += losses["total_loss"].item() * labels_a.size(0)
        preds = student_logits.argmax(dim=1)
        correct += (preds == labels_a).sum().item()
        total += labels_a.size(0)

        # Per-class accuracy
        for cls_id in range(model_cfg.NUM_CLASSES):
            cls_mask = labels_a == cls_id
            if cls_mask.any():
                class_correct[cls_id] += (preds[cls_mask] == cls_id).sum().item()
                class_total[cls_id] += cls_mask.sum().item()

        pbar.set_postfix(
            loss=f"{losses['total_loss'].item():.4f}",
            acc=f"{correct/total:.3f}",
            amp="FP16" if use_amp else "FP32"
        )

    # Per-class accuracy raporu
    per_class = {}
    for cls_id in range(model_cfg.NUM_CLASSES):
        name = model_cfg.CLASS_NAMES[cls_id]
        if class_total[cls_id] > 0:
            per_class[name] = class_correct[cls_id] / class_total[cls_id]
        else:
            per_class[name] = 0.0

    return {"loss": total_loss / total, "accuracy": correct / total, "per_class_acc": per_class}


def validate_epoch(model, loader, criterion):
    """Validation epoch — AUC + per-class metrik + confusion matrix. Binary (REAL/FAKE)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for batch in loader:
            rgb, freq, mesh, labels, source_tags = batch
            rgb, freq = rgb.to(DEVICE), freq.to(DEVICE)
            mesh, labels = mesh.to(DEVICE), labels.to(DEVICE)

            logits = model(rgb, freq, mesh)
            losses = criterion(logits, labels)
            loss = losses["total_loss"]

            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # AUC ve metrikler
    auc = 0.5
    macro_f1 = 0.0
    per_class_metrics = {}

    if HAS_SKLEARN and len(set(all_labels)) > 1:
        all_probs_np = np.array(all_probs)

        # Binary AUC: FAKE olasılığı üzerinden
        try:
            auc = roc_auc_score(all_labels, all_probs_np[:, 1])
        except Exception:
            pass

        # F1
        try:
            labels_present = sorted(set(all_labels))
            macro_f1 = f1_score(all_labels, all_preds, average="macro",
                                labels=labels_present, zero_division=0)
        except Exception:
            pass

        # Per-class metrikleri
        try:
            labels_for_report = list(range(model_cfg.NUM_CLASSES))
            names_for_report = [model_cfg.CLASS_NAMES[i] for i in labels_for_report]
            report = classification_report(
                all_labels, all_preds,
                labels=labels_for_report,
                target_names=names_for_report,
                output_dict=True, zero_division=0
            )
            for name in names_for_report:
                if name in report:
                    per_class_metrics[name] = {
                        "precision": report[name]["precision"],
                        "recall": report[name]["recall"],
                        "f1": report[name]["f1-score"],
                    }
        except Exception:
            pass

        # Confusion matrix
        try:
            cm = sk_confusion_matrix(all_labels, all_preds, labels=[0, 1])
            log_dir = paths.BASE_DIR / "logs" / "run4"
            log_dir.mkdir(parents=True, exist_ok=True)
            np.save(log_dir / "confusion_matrix_latest.npy", cm)
        except Exception:
            pass

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "auc": auc,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
    }


# ═══════════════════════════════════════════════════════════
# ANA EĞİTİM FONKSİYONU
# ═══════════════════════════════════════════════════════════
def train_and_evaluate(epochs=None, batch_size=None, resume=None):
    """
    Ana eğitim fonksiyonu (Run #3).
    Warmup + Kademeli Unfreeze + Mixup/CutMix + ReduceLROnPlateau.
    Per-class metrik + Confusion matrix + Sınıf dağılımı loglama.
    RTX 4070 Laptop: FP16 + Gradient Accumulation.
    """
    epochs = epochs or model_cfg.EPOCHS
    batch_size = batch_size or model_cfg.BATCH_SIZE

    flags = []
    if model_cfg.USE_MIXED_PRECISION: flags.append("FP16")
    flags.append(f"AccumSteps={model_cfg.GRADIENT_ACCUMULATION_STEPS}")
    flags.append(f"Warmup={model_cfg.WARMUP_EPOCHS}ep")
    flags.append(f"Unfreeze@{model_cfg.UNFREEZE_EPOCH}")
    if model_cfg.USE_MIXUP: flags.append("Mixup+CutMix")
    if getattr(model_cfg, 'USE_HYBRID_FREQ', False): flags.append("HybridFreq18ch")
    if getattr(model_cfg, 'USE_CONTRASTIVE', False): flags.append("Triplet")
    if getattr(model_cfg, 'USE_FGSM_TRAINING', False): flags.append(f"FGSM(ε={model_cfg.FGSM_EPSILON_MIN}-{model_cfg.FGSM_EPSILON_MAX},every{model_cfg.FGSM_EVERY_N_STEPS})")
    flag_str = f" + {', '.join(flags)}" if flags else ""
    print(f"🚀 Eğitim başlıyor (Run #5 Forensic) — {epochs} epoch, batch={batch_size}, "
          f"efektif_batch={batch_size * model_cfg.GRADIENT_ACCUMULATION_STEPS}, "
          f"device={DEVICE}{flag_str}")
    print(f"   🎯 Sınıflar: {model_cfg.CLASS_NAMES} ({model_cfg.NUM_CLASSES} sınıf, binary)")
    print(f"   🔬 Loss: focal_w={getattr(model_cfg,'CONTRASTIVE_WEIGHT',0.0):.1f} "
          f"KD_α={model_cfg.KD_ALPHA} triplet_w={getattr(model_cfg,'CONTRASTIVE_WEIGHT',0.0):.1f}")
    paths.ensure_dirs()

    # Log dizini
    log_dir = paths.BASE_DIR / "logs" / "run4_binary"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Veri yükleme
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # Sınıf dağılımını logla
    log_class_distribution(train_loader, "train")

    # Dinamik sınıf ağırlıkları hesapla
    from core.data_pipeline import compute_class_weights
    dynamic_weights = compute_class_weights(train_loader.dataset)
    print(f"  ⚖️ Dinamik CLASS_WEIGHTS: {[f'{w:.2f}' for w in dynamic_weights]}")

    # Model
    model = DualPathDeepfakeDetector().to(DEVICE)

    # Teacher: sadece KD aktifse yukle (VRAM tasarrufu)
    teacher = None
    if model_cfg.KD_ALPHA > 0 and HAS_TEACHER:
        teacher = load_pretrained_teacher()
        print(f"  🎓 Teacher model yuklendi (KD_ALPHA={model_cfg.KD_ALPHA})")
    else:
        print(f"  ⚠️ KD devre disi (KD_ALPHA={model_cfg.KD_ALPHA}) — teacher yuklenmiyor")

    # Kademeli Unfreeze: başlangıçta backbone'u dondur
    freeze_backbone(model)

    # Loss, Optimizer, Scheduler
    criterion = CombinedLoss(class_weights=dynamic_weights)
    base_lr = model_cfg.LEARNING_RATE

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=base_lr, weight_decay=model_cfg.WEIGHT_DECAY)

    # Warmup + Cosine scheduler
    steps_per_epoch = len(train_loader) // model_cfg.GRADIENT_ACCUMULATION_STEPS
    warmup_scheduler = create_warmup_cosine_scheduler(
        optimizer, model_cfg.WARMUP_EPOCHS, epochs, steps_per_epoch
    )
    print(f"  📈 Warmup: {model_cfg.WARMUP_EPOCHS} epoch ({model_cfg.WARMUP_EPOCHS * steps_per_epoch} step)")

    # ReduceLROnPlateau (warmup bittikten sonra aktif)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=model_cfg.PLATEAU_FACTOR,
        patience=model_cfg.PLATEAU_PATIENCE, min_lr=model_cfg.COSINE_ETA_MIN
    )

    # Mixed Precision GradScaler
    scaler = None
    if model_cfg.USE_MIXED_PRECISION and DEVICE.type == "cuda":
        scaler = GradScaler()
        print("  ✅ FP16 Mixed Precision aktif — GradScaler etkin")

    # Checkpoint resume — kısmi yükleme (freq branch 18ch, diğerleri korunur)
    start_epoch = 0
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=DEVICE, weights_only=False)
        prev_state = ckpt.get("model_state_dict", ckpt)
        cur_state = model.state_dict()
        # Boyut uyuşanları yükle, freq branch ilk katmanını atla (12→18 ch değişti)
        loaded, skipped = [], []
        for k, v in prev_state.items():
            if k in cur_state and cur_state[k].shape == v.shape:
                cur_state[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(cur_state)
        start_epoch = ckpt.get("epoch", 0) + 1
        prev_auc = ckpt.get("val_auc", 0.0)
        print(f"📂 Kısmi resume: epoch {start_epoch}, AUC={prev_auc:.4f}")
        print(f"   Yüklenen: {len(loaded)} katman, Atlanan: {len(skipped)} katman")
        if skipped:
            print(f"   Atlanan katmanlar: {skipped[:5]}{'...' if len(skipped)>5 else ''}")

    # MLflow
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(f"file:{paths.MLRUNS_DIR}")
            mlflow.set_experiment("deepfake-v5-forensic")
            mlflow.start_run()
            mlflow.log_params({
                "lr": base_lr, "epochs": epochs,
                "batch_size": batch_size, "backbone": model_cfg.RGB_BACKBONE,
                "num_classes": model_cfg.NUM_CLASSES,
                "mixed_precision": model_cfg.USE_MIXED_PRECISION,
                "gradient_accumulation": model_cfg.GRADIENT_ACCUMULATION_STEPS,
                "warmup_epochs": model_cfg.WARMUP_EPOCHS,
                "unfreeze_epoch": model_cfg.UNFREEZE_EPOCH,
                "use_mixup": model_cfg.USE_MIXUP,
                "run_version": "run5_forensic",
                "freq_channels": model_cfg.DWT_CHANNELS,
                "use_hybrid_freq": getattr(model_cfg, 'USE_HYBRID_FREQ', False),
                "use_contrastive": getattr(model_cfg, 'USE_CONTRASTIVE', False),
                "contrastive_weight": getattr(model_cfg, 'CONTRASTIVE_WEIGHT', 0.0),
                "kd_alpha": model_cfg.KD_ALPHA,
                "label_smoothing": model_cfg.LABEL_SMOOTHING,
                "resume_from": resume or "scratch",
            })
        except Exception:
            pass

    # Early stopping
    best_auc = 0.0
    patience_counter = 0
    backbone_unfrozen = False

    for epoch in range(start_epoch, epochs):
        # Sınıf dağılımı loglama (her epoch başında)
        if epoch == 0:
            log_class_distribution(train_loader, f"train-epoch{epoch+1}")

        # Kademeli Unfreeze kontrolü
        if not backbone_unfrozen and epoch >= model_cfg.UNFREEZE_EPOCH:
            unfreeze_backbone(model)
            backbone_unfrozen = True
            optimizer = create_discriminative_optimizer(model, base_lr)
            warmup_scheduler = create_warmup_cosine_scheduler(
                optimizer, 0, epochs - epoch, steps_per_epoch
            )
            plateau_scheduler = ReduceLROnPlateau(
                optimizer, mode='max', factor=model_cfg.PLATEAU_FACTOR,
                patience=model_cfg.PLATEAU_PATIENCE, min_lr=model_cfg.COSINE_ETA_MIN
            )
            if scaler:
                scaler = GradScaler()

        # Mixup/CutMix: warmup bittikten sonra aktif
        use_mixup = epoch >= model_cfg.WARMUP_EPOCHS

        # Curriculum Learning (G4): Epoch'a göre hard_real_ratio güncelle
        if getattr(model_cfg, 'USE_CURRICULUM', False):
            hard_real_ratio = 0.0
            for phase in model_cfg.CURRICULUM_PHASES:
                if phase["start"] <= epoch <= phase["end"]:
                    hard_real_ratio = phase["hard_real_ratio"]
                    break
            # Dataset'teki HardRealAugmentation.prob'u güncelle
            _update_curriculum_ratio(train_loader, hard_real_ratio)
            if epoch == 0 or (epoch > 0 and hard_real_ratio != _get_prev_ratio(epoch, model_cfg)):
                print(f"  📚 Curriculum: epoch {epoch+1}, hard_real_ratio={hard_real_ratio:.2f}")

        # Eğitim
        train_metrics = train_epoch(
            model, teacher, train_loader, criterion, optimizer, epoch, scaler,
            warmup_scheduler=warmup_scheduler, use_mixup=use_mixup,
            cutmix_ratio=0.6,
        )
        val_metrics = validate_epoch(model, val_loader, criterion)

        # ReduceLROnPlateau: warmup bittikten sonra
        if epoch >= model_cfg.WARMUP_EPOCHS:
            plateau_scheduler.step(val_metrics["auc"])

        lr = optimizer.param_groups[0]["lr"]
        phase = "🧊FREEZE" if not backbone_unfrozen else "🔥UNFREEZE"
        mixup_str = "+MIX/CUT" if use_mixup else ""

        # Ana metrikler
        print(f"Epoch {epoch+1}/{epochs} [{phase}{mixup_str}] | "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.3f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f} "
              f"AUC: {val_metrics['auc']:.4f} F1: {val_metrics['macro_f1']:.4f} | LR: {lr:.2e}")

        # Per-class metrikleri yazdır
        if train_metrics.get("per_class_acc"):
            parts = [f"{k}:{v:.3f}" for k, v in train_metrics["per_class_acc"].items()]
            print(f"  📊 Train per-class acc: {', '.join(parts)}")

        if val_metrics.get("per_class"):
            for name, m in val_metrics["per_class"].items():
                print(f"  📊 Val {name}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

        # MLflow loglama
        if HAS_MLFLOW:
            try:
                log_data = {
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_auc": val_metrics["auc"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "lr": lr,
                }
                # Contrastive loss metriği logla
                if "triplet_loss" in train_metrics:
                    log_data["train_triplet_loss"] = train_metrics["triplet_loss"]
                for name, acc in train_metrics.get("per_class_acc", {}).items():
                    log_data[f"train_acc_{name}"] = acc
                for name, m in val_metrics.get("per_class", {}).items():
                    log_data[f"val_recall_{name}"] = m["recall"]
                    log_data[f"val_f1_{name}"] = m["f1"]
                mlflow.log_metrics(log_data, step=epoch)
            except Exception:
                pass

        # Best model kaydet
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            save_path = paths.MODEL_DIR / "best_run5_forensic.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_auc,
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "num_classes": model_cfg.NUM_CLASSES,
                "class_names": model_cfg.CLASS_NAMES,
                "run": "run5_forensic",
                "freq_channels": model_cfg.DWT_CHANNELS,
                "use_hybrid_freq": getattr(model_cfg, 'USE_HYBRID_FREQ', False),
            }, save_path)
            # Uyumluluk için best_model.pth olarak da kaydet
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": best_auc,
                "num_classes": model_cfg.NUM_CLASSES,
                "class_names": model_cfg.CLASS_NAMES,
                "run": "run5_forensic",
            }, paths.BEST_MODEL_PATH)
            print(f"  💾 Best model kaydedildi — AUC: {best_auc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= model_cfg.EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️ Early stopping — {model_cfg.EARLY_STOPPING_PATIENCE} epoch iyilesme yok")
            break

        # Periyodik checkpoint (her 5 epoch)
        if (epoch + 1) % 5 == 0:
            ckpt_path = paths.MODEL_DIR / f"checkpoint_epoch{epoch+1}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "run": "run5_forensic",
            }, ckpt_path)
            print(f"  💾 Checkpoint kaydedildi: {ckpt_path.name}")

    if HAS_MLFLOW:
        try:
            mlflow.end_run()
        except Exception:
            pass

    print(f"\n✅ Eğitim tamamlandı! En iyi AUC: {best_auc:.4f}")
    return model


# =================================================================
# AYRI EĞİTİM: FF++ (REAL/FAKE) ve CASIA (REAL/SPOOF)
# =================================================================
def _train_single_source(source_name, get_loaders_fn, branch_name, save_name,
                         epochs=None, batch_size=None):
    """
    Tek kaynak için eğitim fonksiyonu.
    
    Args:
        source_name: "FF++" veya "CASIA" (log için)
        get_loaders_fn: get_ffpp_dataloaders veya get_casia_dataloaders
        branch_name: "digital" veya "physical" (hangi head kullanılacak)
        save_name: Checkpoint dosya adı (örn: "best_ffpp.pth")
    """
    epochs = epochs or model_cfg.EPOCHS
    batch_size = batch_size or model_cfg.BATCH_SIZE

    print(f"\n{'='*60}")
    print(f"  {source_name} Eğitimi Başlıyor ({branch_name} branch)")
    print(f"  {epochs} epoch, batch={batch_size}, "
          f"efektif={batch_size * model_cfg.GRADIENT_ACCUMULATION_STEPS}")
    print(f"{'='*60}")
    paths.ensure_dirs()

    # Veri yükleme
    train_loader, val_loader, _ = get_loaders_fn(batch_size=batch_size)
    if train_loader is None:
        print(f"  ❌ {source_name} verisi bulunamadı!")
        return None

    # Sınıf dağılımı
    log_class_distribution(train_loader, f"{source_name}-train")

    # Dinamik ağırlıklar
    dynamic_weights = compute_class_weights(train_loader.dataset)
    print(f"  ⚖️ Dinamik CLASS_WEIGHTS: {[f'{w:.2f}' for w in dynamic_weights]}")

    # Kaynak bazlı aktif sınıflar (loss maskeleme)
    if branch_name == "digital":
        active_classes = [0, 1]  # FF++: REAL, FAKE
    elif branch_name == "physical":
        active_classes = [0, 2]  # CASIA: REAL, SPOOF
    else:
        active_classes = None    # Tüm sınıflar

    print(f"  🎯 Aktif sınıflar: {active_classes}")

    # Model (3 sınıf çıktılı, ama loss sadece aktif sınıflar üzerinden)
    model = DualPathDeepfakeDetector().to(DEVICE)
    teacher = load_pretrained_teacher()

    # Backbone dondur
    freeze_backbone(model)

    # Loss, Optimizer
    criterion = CombinedLoss(class_weights=dynamic_weights)
    base_lr = model_cfg.LEARNING_RATE
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=base_lr, weight_decay=model_cfg.WEIGHT_DECAY)

    # Scheduler
    steps_per_epoch = max(1, len(train_loader) // model_cfg.GRADIENT_ACCUMULATION_STEPS)
    warmup_scheduler = create_warmup_cosine_scheduler(
        optimizer, model_cfg.WARMUP_EPOCHS, epochs, steps_per_epoch
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=model_cfg.PLATEAU_FACTOR,
        patience=model_cfg.PLATEAU_PATIENCE, min_lr=model_cfg.COSINE_ETA_MIN
    )

    # AMP
    scaler = None
    if model_cfg.USE_MIXED_PRECISION and DEVICE.type == "cuda":
        scaler = GradScaler()
        print(f"  ✅ FP16 aktif")

    # MLflow
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(f"file:{paths.MLRUNS_DIR}")
            mlflow.set_experiment(f"deepfake-v3-{source_name.lower()}")
            mlflow.start_run()
            mlflow.log_params({
                "source": source_name, "branch": branch_name,
                "epochs": epochs, "batch_size": batch_size,
                "lr": base_lr,
            })
        except Exception:
            pass

    best_auc = 0.0
    patience_counter = 0
    backbone_unfrozen = False

    for epoch in range(epochs):
        # Unfreeze
        if not backbone_unfrozen and epoch >= model_cfg.UNFREEZE_EPOCH:
            unfreeze_backbone(model)
            backbone_unfrozen = True
            optimizer = create_discriminative_optimizer(model, base_lr)
            warmup_scheduler = create_warmup_cosine_scheduler(
                optimizer, 0, epochs - epoch, steps_per_epoch
            )
            plateau_scheduler = ReduceLROnPlateau(
                optimizer, mode='max', factor=model_cfg.PLATEAU_FACTOR,
                patience=model_cfg.PLATEAU_PATIENCE, min_lr=model_cfg.COSINE_ETA_MIN
            )
            if scaler:
                scaler = GradScaler()

        use_mixup = epoch >= model_cfg.WARMUP_EPOCHS

        # Eğitim
        train_metrics = train_epoch(
            model, teacher, train_loader, criterion, optimizer, epoch, scaler,
            warmup_scheduler=warmup_scheduler, use_mixup=use_mixup,
            cutmix_ratio=model_cfg.CUTMIX_RATIO,
            active_classes=active_classes,
        )

        # Validation
        if val_loader:
            val_metrics = validate_epoch(model, val_loader, criterion,
                                         active_classes=active_classes)
        else:
            val_metrics = {"loss": 0, "accuracy": 0, "auc": 0.5, "macro_f1": 0, "per_class": {}}

        if epoch >= model_cfg.WARMUP_EPOCHS:
            plateau_scheduler.step(val_metrics["auc"])

        lr = optimizer.param_groups[0]["lr"]
        phase = "FREEZE" if not backbone_unfrozen else "UNFREEZE"

        print(f"[{source_name}] Epoch {epoch+1}/{epochs} [{phase}] | "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.3f} | "
              f"Val Acc: {val_metrics['accuracy']:.3f} AUC: {val_metrics['auc']:.4f} | LR: {lr:.2e}")

        # Per-class
        if train_metrics.get("per_class_acc"):
            parts = [f"{k}:{v:.3f}" for k, v in train_metrics["per_class_acc"].items()]
            print(f"  Per-class: {', '.join(parts)}")
        if val_metrics.get("per_class"):
            for name, m in val_metrics["per_class"].items():
                print(f"  Val {name}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

        # MLflow
        if HAS_MLFLOW:
            try:
                mlflow.log_metrics({
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_acc": val_metrics["accuracy"],
                    "val_auc": val_metrics["auc"],
                    "lr": lr,
                }, step=epoch)
            except Exception:
                pass

        # Best model
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            save_path = paths.MODEL_DIR / save_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": best_auc,
                "val_acc": val_metrics["accuracy"],
                "source": source_name,
                "branch": branch_name,
                "num_classes": model_cfg.NUM_CLASSES,
                "class_names": model_cfg.CLASS_NAMES,
                "run": "run3",
            }, save_path)
            print(f"  Best {source_name} model: AUC={best_auc:.4f} -> {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= model_cfg.EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping ({source_name})")
            break

    if HAS_MLFLOW:
        try: mlflow.end_run()
        except Exception: pass

    print(f"\n✅ {source_name} eğitimi tamamlandı! En iyi AUC: {best_auc:.4f}")
    return model


def train_ffpp(epochs=None, batch_size=None):
    """
    FF++ eğitimi: Sadece REAL/FAKE sınıfları.
    Browser extension / fotoğraf yükleme kaynağı.
    """
    return _train_single_source(
        source_name="FF++",
        get_loaders_fn=get_ffpp_dataloaders,
        branch_name="digital",
        save_name="best_ffpp.pth",
        epochs=epochs, batch_size=batch_size,
    )


def train_casia(epochs=None, batch_size=None):
    """
    CASIA eğitimi: Sadece REAL/SPOOF sınıfları.
    Webcam / canlı kamera kaynağı.
    """
    return _train_single_source(
        source_name="CASIA",
        get_loaders_fn=get_casia_dataloaders,
        branch_name="physical",
        save_name="best_casia.pth",
        epochs=epochs, batch_size=batch_size,
    )


def train_both_sequential(epochs=None, batch_size=None):
    """
    Sıralı eğitim: Önce FF++ (REAL/FAKE), sonra CASIA (REAL/SPOOF).
    Her iki model de ayrı checkpoint olarak kaydedilir.
    """
    print("\n" + "=" * 60)
    print("  SIRAYLA EĞİTİM: FF++ (REAL/FAKE) -> CASIA (REAL/SPOOF)")
    print("=" * 60)

    # Aşama 1: FF++
    ffpp_model = train_ffpp(epochs=epochs, batch_size=batch_size)

    # CUDA bellek temizleme — Windows shared memory sorunu önleme
    import gc
    del ffpp_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("\n🧹 CUDA bellek temizlendi, CASIA eğitimine geçiliyor...\n")

    # Aşama 2: CASIA
    casia_model = train_casia(epochs=epochs, batch_size=batch_size)

    print("\n" + "=" * 60)
    print("  ✅ Her iki eğitim tamamlandı!")
    print("  - FF++ model: models/best_ffpp.pth")
    print("  - CASIA model: models/best_casia.pth")
    print("=" * 60)

    return None, casia_model


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "ffpp":
            train_ffpp()
        elif mode == "casia":
            train_casia()
        elif mode == "both":
            train_both_sequential()
        elif mode == "unified":
            train_and_evaluate()
        else:
            print(f"Kullanım: python -m core.trainer [ffpp|casia|both|unified]")
    else:
        # Varsayılan: sıralı eğitim
        train_both_sequential()
