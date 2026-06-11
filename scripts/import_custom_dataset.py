"""
DeepfakeULTRA — Kişisel Veri Seti ile Fine-Tune Aracı
======================================================
Kullanım:
    python scripts/import_custom_dataset.py --real C:/path/to/real --fake C:/path/to/fake

Klasör yapısı da kabul edilir:
    python scripts/import_custom_dataset.py --dir C:/MyDataset
    (MyDataset/real/ ve MyDataset/fake/ ya da MyDataset/REAL/ ve MyDataset/FAKE/ beklenir)

Ne yapar:
    1. Görselleri feedback_images/REAL ve feedback_images/FAKE altına kopyalar
    2. Model üzerinde fine-tune başlatır (sadece classifier head)
    3. Sonuçları raporlar
"""

import sys
import os
import shutil
import argparse
from pathlib import Path

# Proje kök dizinini Python yoluna ekle
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Sabitler ──
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
FEEDBACK_DIR = ROOT / "feedback_images"


def find_images(directory: Path) -> list[Path]:
    """Bir dizindeki tüm desteklenen görsel dosyalarını bul."""
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def copy_images_to_feedback(source_dir: Path, label: str) -> int:
    """Görselleri feedback_images/<LABEL>/ dizinine kopyala."""
    target_dir = FEEDBACK_DIR / label.upper()
    target_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(source_dir)
    if not images:
        print(f"  ⚠️  {source_dir} içinde görsel bulunamadı!")
        return 0

    print(f"  📂 {label.upper()}: {len(images)} görsel kopyalanıyor → {target_dir}")
    copied = 0
    for img_path in images:
        dest = target_dir / img_path.name
        # Çakışma varsa timestamp ekle
        if dest.exists():
            stem = img_path.stem
            suffix = img_path.suffix
            import time
            dest = target_dir / f"{stem}_{int(time.time() * 1000)}{suffix}"
        shutil.copy2(str(img_path), str(dest))
        copied += 1

    print(f"  ✅ {copied} görsel kopyalandı.")
    return copied


def run_finetune(epochs: int = 10, lr: float = 5e-5):
    """Modeli fine-tune et."""
    print("\n" + "═" * 60)
    print("  🧠 Fine-Tune Başlatılıyor")
    print("═" * 60)

    try:
        import torch
        from config import DEVICE, paths
        from core.dual_mobilenetv3 import DualPathDeepfakeDetector
        from core.fine_tuner import FeedbackDataset, FEEDBACK_DIR as FB_DIR
        from torch.utils.data import DataLoader
        from torchvision import transforms

        # Mevcut modeli yükle
        model_path = paths.BEST_MODEL_PATH
        if not model_path.exists():
            # Alternatif model dosyaları ara
            for candidate in ["best_run5_forensic.pth", "finetuned_model.pth"]:
                p = paths.MODEL_DIR / candidate
                if p.exists():
                    model_path = p
                    break

        if not model_path.exists():
            print("  ❌ Model dosyası bulunamadı! Önce modeli eğitin.")
            return

        print(f"  📂 Model yükleniyor: {model_path.name}")
        model = DualPathDeepfakeDetector().to(DEVICE)
        ckpt = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"  ✅ Model yüklendi → {DEVICE}")

        # Dataset hazırla
        image_paths, labels = [], []
        for label_name, label_id in [("REAL", 0), ("FAKE", 1)]:
            label_dir = FB_DIR / label_name
            if not label_dir.exists():
                continue
            for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.webp"]:
                for fpath in label_dir.glob(ext):
                    image_paths.append(str(fpath))
                    labels.append(label_id)

        total = len(image_paths)
        real_count = labels.count(0)
        fake_count = labels.count(1)
        print(f"\n  📊 Dataset: {real_count} REAL + {fake_count} FAKE = {total} görsel")

        if total < 4:
            print("  ❌ Çok az görsel! En az 2 REAL + 2 FAKE gerekli.")
            return

        # Transform + DataLoader
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        dataset = FeedbackDataset(image_paths, labels)
        dataset.transform = transform

        # %80 train / %20 val split
        val_size = max(2, int(total * 0.2))
        train_size = total - val_size

        import torch.utils.data as tud
        train_ds, val_ds = tud.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        batch_size = min(8, train_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"  ✂️  Train: {train_size} | Val: {val_size} | Batch: {batch_size}")

        # ── Fine-Tune: Sadece classifier head ──
        # Tüm parametreleri dondur
        for param in model.parameters():
            param.requires_grad = False

        # Classifier'ı aç
        for param in model.classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  🔓 Eğitilecek parametre: {trainable:,} / {total_params:,} "
              f"({trainable/total_params*100:.1f}%)")

        import torch.nn as nn
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        criterion = nn.CrossEntropyLoss()

        print(f"\n  📈 Eğitim: {epochs} epoch, LR={lr:.0e}")
        print("  " + "─" * 50)

        best_val_acc = 0.0
        best_model_state = None

        model.train()
        for epoch in range(epochs):
            # ── Train ──
            train_loss, train_correct, train_total = 0.0, 0, 0
            model.train()

            for batch_imgs, batch_labels in train_loader:
                batch_imgs = batch_imgs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                with torch.no_grad():
                    rgb_feat = model.rgb_pool(model.rgb_features(batch_imgs)).flatten(1)
                    rgb_feat = model.rgb_proj(rgb_feat)

                b = rgb_feat.shape[0]
                freq_feat = torch.zeros_like(rgb_feat)
                mesh_feat = torch.zeros(b, rgb_feat.shape[1]).to(DEVICE)

                with torch.no_grad():
                    fused = model.fusion(rgb_feat, freq_feat, mesh_feat)
                    temporal_in = fused.unsqueeze(1)
                    lstm_out = model.temporal_lstm(temporal_in)
                    attended = model.temporal_attention(lstm_out)

                logits = model.classifier(attended)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = logits.argmax(dim=1)
                train_correct += (preds == batch_labels).sum().item()
                train_total += batch_labels.size(0)

            # ── Val ──
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch_imgs, batch_labels in val_loader:
                    batch_imgs = batch_imgs.to(DEVICE)
                    batch_labels = batch_labels.to(DEVICE)

                    rgb_feat = model.rgb_pool(model.rgb_features(batch_imgs)).flatten(1)
                    rgb_feat = model.rgb_proj(rgb_feat)

                    b = rgb_feat.shape[0]
                    freq_feat = torch.zeros_like(rgb_feat)
                    mesh_feat = torch.zeros(b, rgb_feat.shape[1]).to(DEVICE)

                    fused = model.fusion(rgb_feat, freq_feat, mesh_feat)
                    temporal_in = fused.unsqueeze(1)
                    lstm_out = model.temporal_lstm(temporal_in)
                    attended = model.temporal_attention(lstm_out)

                    logits = model.classifier(attended)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == batch_labels).sum().item()
                    val_total += batch_labels.size(0)

            scheduler.step()

            train_acc = train_correct / max(train_total, 1)
            val_acc = val_correct / max(val_total, 1)
            avg_loss = train_loss / max(len(train_loader), 1)
            lr_cur = optimizer.param_groups[0]["lr"]

            print(f"  Epoch {epoch+1:2d}/{epochs} │ "
                  f"Loss: {avg_loss:.4f} │ "
                  f"Train Acc: {train_acc*100:.1f}% │ "
                  f"Val Acc: {val_acc*100:.1f}% │ "
                  f"LR: {lr_cur:.1e}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # En iyi modeli kaydet
        print("\n  " + "─" * 50)
        if best_model_state:
            save_path = paths.MODEL_DIR / "finetuned_model.pth"
            torch.save(best_model_state, str(save_path))
            print(f"  💾 En iyi model kaydedildi: {save_path.name}")
            print(f"  🏆 En iyi Val Accuracy: {best_val_acc*100:.1f}%")

            # best_model.pth olarak da güncelle
            ckpt_data = {
                "model_state_dict": best_model_state,
                "val_acc": best_val_acc,
                "num_classes": 2,
                "class_names": ["REAL", "FAKE"],
                "run": "custom_finetune",
                "custom_dataset": f"{real_count}real_{fake_count}fake",
            }
            torch.save(ckpt_data, str(paths.BEST_MODEL_PATH))
            print(f"  ✅ best_model.pth güncellendi — sistem artık yeni modeli kullanacak!")

        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    except Exception as e:
        import traceback
        print(f"\n  ❌ Fine-tune hatası: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="DeepfakeULTRA — Kişisel Veri Seti ile Fine-Tune",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Ayrı klasörler:
  python scripts/import_custom_dataset.py --real C:/Data/real --fake C:/Data/fake

  # Tek üst klasör (içinde real/ ve fake/ olmalı):
  python scripts/import_custom_dataset.py --dir C:/MyDataset

  # Epoch ve LR ayarla:
  python scripts/import_custom_dataset.py --real ./real --fake ./fake --epochs 15 --lr 1e-4

  # Sadece kopyala, eğitme:
  python scripts/import_custom_dataset.py --real ./real --fake ./fake --no-train
        """
    )
    parser.add_argument("--real", type=str, default=None,
                        help="REAL görsel klasörü yolu")
    parser.add_argument("--fake", type=str, default=None,
                        help="FAKE görsel klasörü yolu")
    parser.add_argument("--dir", type=str, default=None,
                        help="Üst klasör (içinde real/ ve fake/ olmalı)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Fine-tune epoch sayısı (varsayılan: 10)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (varsayılan: 5e-5)")
    parser.add_argument("--no-train", action="store_true",
                        help="Sadece görselleri kopyala, eğitme")
    parser.add_argument("--clear-existing", action="store_true",
                        help="Mevcut feedback havuzunu temizle")

    args = parser.parse_args()

    print("═" * 60)
    print("  🎯 DeepfakeULTRA — Kişisel Dataset Fine-Tune")
    print("═" * 60)

    # Mevcut havuzu temizle (opsiyonel)
    if args.clear_existing and FEEDBACK_DIR.exists():
        print(f"\n  🗑️  Mevcut feedback havuzu temizleniyor: {FEEDBACK_DIR}")
        shutil.rmtree(str(FEEDBACK_DIR))
        print("  ✅ Temizlendi.")

    # Klasörleri belirle
    real_dir = None
    fake_dir = None

    if args.dir:
        base = Path(args.dir)
        # real/ veya REAL/ klasörünü bul
        for name in ["real", "REAL", "Real"]:
            if (base / name).exists():
                real_dir = base / name
                break
        for name in ["fake", "FAKE", "Fake"]:
            if (base / name).exists():
                fake_dir = base / name
                break

        if not real_dir or not fake_dir:
            print(f"  ❌ {args.dir} içinde real/ ve fake/ klasörleri bulunamadı!")
            print("  💡 Klasör yapısı: MyDataset/real/ ve MyDataset/fake/")
            sys.exit(1)

    if args.real:
        real_dir = Path(args.real)
    if args.fake:
        fake_dir = Path(args.fake)

    if not real_dir or not fake_dir:
        parser.print_help()
        sys.exit(1)

    # Klasör kontrolü
    if not real_dir.exists():
        print(f"  ❌ REAL klasörü bulunamadı: {real_dir}")
        sys.exit(1)
    if not fake_dir.exists():
        print(f"  ❌ FAKE klasörü bulunamadı: {fake_dir}")
        sys.exit(1)

    print(f"\n  📁 REAL kaynak: {real_dir}")
    print(f"  📁 FAKE kaynak: {fake_dir}")

    # Görselleri kopyala
    print("\n  📋 Görseller kopyalanıyor...")
    real_copied = copy_images_to_feedback(real_dir, "REAL")
    fake_copied = copy_images_to_feedback(fake_dir, "FAKE")
    total = real_copied + fake_copied

    print(f"\n  📊 Toplam kopyalanan: {real_copied} REAL + {fake_copied} FAKE = {total} görsel")

    if total == 0:
        print("  ❌ Hiç görsel kopyalanamadı!")
        sys.exit(1)

    # Fine-tune
    if not args.no_train:
        run_finetune(epochs=args.epochs, lr=args.lr)
    else:
        print("\n  ℹ️  --no-train seçildi. Görseller kopyalandı ama eğitim yapılmadı.")
        print(f"  💡 Eğitmek için: python scripts/import_custom_dataset.py "
              f"--real {real_dir} --fake {fake_dir}")

    print("\n" + "═" * 60)
    print("  ✨ İşlem tamamlandı!")
    print("═" * 60)


if __name__ == "__main__":
    main()
