"""Egitim ilerleme monitor — canli izleme icin."""
import sys, time, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_progress():
    models_dir = PROJECT_ROOT / "models"
    
    print("=" * 60)
    print("  EGITIM ILERLEME RAPORU")
    print(f"  Zaman: {time.strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Checkpoint dosyalarindan ilerleme
    import torch
    checkpoints = sorted(models_dir.glob("*.pth"), key=lambda f: f.stat().st_mtime)
    
    latest_epoch = -1
    latest_auc = 0
    
    for f in checkpoints:
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            epoch = ckpt.get("epoch", -1)
            auc = ckpt.get("val_auc", ckpt.get("best_auc", 0))
            acc = ckpt.get("val_acc", 0)
            run = ckpt.get("run", "?")
            mtime = time.strftime("%H:%M", time.localtime(f.stat().st_mtime))
            
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_auc = auc
            
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  {f.name:35s} | epoch={epoch:>2} | AUC={auc:.4f} | acc={acc:.3f} | {mtime} | {size_mb:.0f}MB")
        except Exception as e:
            print(f"  {f.name}: hata - {e}")
    
    # Tahmin hesapla
    if latest_epoch >= 0:
        # best_model.pth'in degisim zamanlarina bakarak epoch hizi hesapla
        best = models_dir / "best_model.pth"
        if best.exists():
            elapsed_since_start = time.time() - checkpoints[0].stat().st_mtime if checkpoints else 0
            if latest_epoch > 0 and elapsed_since_start > 0:
                sec_per_epoch = elapsed_since_start / (latest_epoch + 1)
                remaining = (29 - latest_epoch) * sec_per_epoch
                hours = remaining / 3600
                print(f"\n  Mevcut epoch: {latest_epoch + 1}/30")
                print(f"  Epoch hizi: ~{sec_per_epoch/60:.0f} dk/epoch")
                print(f"  Tahmini kalan: ~{hours:.1f} saat")
    
    # MLflow log varsa oku
    log_dir = PROJECT_ROOT / "logs" / "run4_binary"
    if log_dir.exists():
        cm_file = PROJECT_ROOT / "logs" / "run4" / "confusion_matrix_latest.npy"
        if cm_file.exists():
            import numpy as np
            cm = np.load(cm_file)
            print(f"\n  Son Confusion Matrix:")
            print(f"    Pred REAL  Pred FAKE")
            print(f"    REAL  {cm[0][0]:>6}  {cm[0][1]:>6}")
            print(f"    FAKE  {cm[1][0]:>6}  {cm[1][1]:>6}")


if __name__ == "__main__":
    check_progress()
