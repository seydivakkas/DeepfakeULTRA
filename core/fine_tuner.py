"""
Active Learning Fine-Tuner — Geri bildirim havuzundan model guncelleme.

Guvenli yaklasim: Sadece classifier head fine-tune edilir.
Backbone (RGB/Freq/Mesh) ve Fusion katmanlari dondurulur.
"""
import threading
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from config import DEVICE, PathConfig

paths = PathConfig()

# ── Sabitler ──
FEEDBACK_DIR = paths.BASE_DIR / "feedback_images"
MIN_SAMPLES = 10
FINETUNE_EPOCHS = 5
FINETUNE_LR = 1e-4
FINETUNE_BATCH = 4
FINETUNED_PATH = paths.MODEL_DIR / "finetuned_model.pth"
BACKUP_PATH = paths.MODEL_DIR / "finetuned_backup.pth"


class FeedbackDataset(Dataset):
    """feedback_images/ dizininden REAL/FAKE dataset olusturur."""

    def __init__(self, image_paths: list, labels: list, img_size: int = 224):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = self.transform(img)
        label = self.labels[idx]  # 0=REAL, 1=FAKE
        return img_tensor, label


def check_readiness() -> dict:
    """Feedback havuzunun fine-tune icin hazir olup olmadigini kontrol et."""
    real_dir = FEEDBACK_DIR / "REAL"
    fake_dir = FEEDBACK_DIR / "FAKE"

    real_count = len(list(real_dir.glob("*.jpg"))) if real_dir.exists() else 0
    fake_count = len(list(fake_dir.glob("*.jpg"))) if fake_dir.exists() else 0
    # png de say
    real_count += len(list(real_dir.glob("*.png"))) if real_dir.exists() else 0
    fake_count += len(list(fake_dir.glob("*.png"))) if fake_dir.exists() else 0

    total = real_count + fake_count
    ready = total >= MIN_SAMPLES and real_count >= 2 and fake_count >= 2

    return {
        "ready": ready,
        "total": total,
        "real_count": real_count,
        "fake_count": fake_count,
        "min_required": MIN_SAMPLES,
        "message": (
            f"Havuz: {real_count} REAL + {fake_count} FAKE = {total} gorsel"
            if total > 0 else "Henuz geri bildirim yok"
        ),
        "has_finetuned": FINETUNED_PATH.exists(),
    }


def save_feedback_image(image, label: str) -> str:
    """Gorseli feedback_images/REAL veya /FAKE dizinine kaydet."""
    target_dir = FEEDBACK_DIR / label.upper()
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label.lower()}_{timestamp}.jpg"
    filepath = target_dir / filename

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        return ""

    image.convert("RGB").save(str(filepath), quality=95)
    return str(filepath)


def _collect_dataset() -> tuple:
    """feedback_images/ dizininden yol+etiket listesi topla."""
    image_paths = []
    labels = []

    for label_name, label_id in [("REAL", 0), ("FAKE", 1)]:
        label_dir = FEEDBACK_DIR / label_name
        if not label_dir.exists():
            continue
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for fpath in label_dir.glob(ext):
                image_paths.append(str(fpath))
                labels.append(label_id)

    return image_paths, labels


# ── Fine-Tune State (thread-safe) ──
_finetune_lock = threading.Lock()
_finetune_status = {
    "running": False,
    "progress": "",
    "epoch": 0,
    "total_epochs": FINETUNE_EPOCHS,
    "loss": 0.0,
    "completed": False,
    "error": None,
}


def get_finetune_status() -> dict:
    """Mevcut fine-tune durumunu dondur."""
    with _finetune_lock:
        return dict(_finetune_status)


def _run_finetune_thread(model, original_state_dict):
    """Arka plan thread'inde fine-tune calistir."""
    global _finetune_status

    try:
        with _finetune_lock:
            _finetune_status["running"] = True
            _finetune_status["completed"] = False
            _finetune_status["error"] = None
            _finetune_status["progress"] = "Dataset hazirlaniyor..."

        # Dataset topla
        image_paths, labels = _collect_dataset()
        if len(image_paths) < MIN_SAMPLES:
            with _finetune_lock:
                _finetune_status["error"] = f"Yetersiz veri: {len(image_paths)}/{MIN_SAMPLES}"
                _finetune_status["running"] = False
            return

        dataset = FeedbackDataset(image_paths, labels)
        loader = DataLoader(dataset, batch_size=FINETUNE_BATCH, shuffle=True,
                            num_workers=0, pin_memory=True)

        # Mevcut modeli yedekle
        if FINETUNED_PATH.exists():
            shutil.copy2(str(FINETUNED_PATH), str(BACKUP_PATH))

        # Backbone dondur — sadece classifier egitilecek
        for param in model.parameters():
            param.requires_grad = False

        # Classifier head'i ac
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Optimizer ve loss
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=FINETUNE_LR, weight_decay=1e-2
        )
        criterion = nn.CrossEntropyLoss()

        model.train()
        best_loss = float("inf")

        for epoch in range(FINETUNE_EPOCHS):
            epoch_loss = 0.0
            n_batches = 0

            for batch_imgs, batch_labels in loader:
                batch_imgs = batch_imgs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                # Basitlestirilmis forward: sadece RGB path + classifier
                # (Feedback gorsellerinde DWT/Mesh hesaplamak cok pahali)
                with torch.no_grad():
                    rgb_feat = model.rgb_pool(model.rgb_features(batch_imgs)).flatten(1)
                    rgb_feat = model.rgb_proj(rgb_feat)

                # Mesh ve freq icin sifir tensoru (frozen oldugu icin etki yok)
                b = rgb_feat.shape[0]
                freq_feat = torch.zeros_like(rgb_feat)
                mesh_feat = torch.zeros(b, rgb_feat.shape[1]).to(DEVICE)

                # Fusion (frozen)
                with torch.no_grad():
                    fused = model.fusion(rgb_feat, freq_feat, mesh_feat)
                    temporal_in = fused.unsqueeze(1)
                    lstm_out = model.temporal_lstm(temporal_in)
                    attended = model.temporal_attention(lstm_out)

                # Classifier (trainable)
                logits = model.classifier(attended)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            with _finetune_lock:
                _finetune_status["epoch"] = epoch + 1
                _finetune_status["loss"] = round(avg_loss, 4)
                _finetune_status["progress"] = (
                    f"Epoch {epoch+1}/{FINETUNE_EPOCHS} | Loss: {avg_loss:.4f}"
                )

            # Early stopping: loss artiyorsa dur
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Checkpoint kaydet
                torch.save(model.state_dict(), str(FINETUNED_PATH))
            elif epoch > 2 and avg_loss > best_loss * 1.5:
                with _finetune_lock:
                    _finetune_status["progress"] += " | Early stopping"
                break

        # Tum parametreleri tekrar ac (inference icin)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        with _finetune_lock:
            _finetune_status["running"] = False
            _finetune_status["completed"] = True
            _finetune_status["progress"] = (
                f"Tamamlandi! Final Loss: {best_loss:.4f} | "
                f"Model kaydedildi: finetuned_model.pth"
            )

    except Exception as e:
        # Hata durumunda orijinal modele geri don
        model.load_state_dict(original_state_dict)
        model.eval()
        with _finetune_lock:
            _finetune_status["running"] = False
            _finetune_status["error"] = str(e)
            _finetune_status["progress"] = f"Hata: {e}"


def start_finetune(model) -> str:
    """Fine-tune'u arka plan thread'inde baslat."""
    status = get_finetune_status()
    if status["running"]:
        return "Fine-tune zaten calisiyor..."

    readiness = check_readiness()
    if not readiness["ready"]:
        return f"Yetersiz veri: {readiness['message']}"

    # Orijinal state'i yedekle (rollback icin)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    thread = threading.Thread(
        target=_run_finetune_thread,
        args=(model, original_state),
        daemon=True,
    )
    thread.start()
    return "Fine-tune baslatildi! Arka planda calisiyor..."


def rollback_model(model) -> str:
    """Fine-tuned modeli geri al, orijinale don."""
    original_path = paths.BEST_MODEL_PATH

    if not original_path.exists():
        return "Orijinal model bulunamadi!"

    try:
        checkpoint = torch.load(str(original_path), map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()

        # Fine-tuned modeli sil
        if FINETUNED_PATH.exists():
            FINETUNED_PATH.unlink()

        return "Orijinal modele geri donuldu."
    except Exception as e:
        return f"Rollback hatasi: {e}"
