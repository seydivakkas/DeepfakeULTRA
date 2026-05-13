"""
DeepfakeULTRA — Canlı Eğitim İzleme Scripti
PowerShell'den çalıştır: python monitor_training.py
Ctrl+C ile çık.
"""
import torch
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent
BEST_MODEL = PROJECT / "models" / "best_run5_forensic.pth"
BEST_COPY = PROJECT / "models" / "best_model.pth"
CM_FILE = PROJECT / "logs" / "run4" / "confusion_matrix_latest.npy"
CLASS_NAMES = ["REAL", "FAKE"]

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def load_checkpoint():
    if not BEST_MODEL.exists():
        return None
    try:
        ckpt = torch.load(BEST_MODEL, map_location="cpu", weights_only=False)
        return ckpt
    except Exception:
        return None

def load_cm():
    if not CM_FILE.exists():
        return None
    try:
        return np.load(CM_FILE)
    except Exception:
        return None

def fmt_time(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}s {m}dk {s}sn"

def main():
    print("DeepfakeULTRA Egitim Izleyici baslatildi...")
    print("Ctrl+C ile cik.\n")

    prev_epoch = -1
    prev_cm_time = None
    start_time = time.time()

    while True:
        try:
            clear()
            now = datetime.now().strftime("%H:%M:%S")
            elapsed = fmt_time(time.time() - start_time)

            print("=" * 60)
            print(f"  DeepfakeULTRA V5 — Canli Egitim Izleyici")
            print(f"  Saat: {now} | Izleme suresi: {elapsed}")
            print("=" * 60)

            # Checkpoint bilgisi
            ckpt = load_checkpoint()
            if ckpt:
                epoch = ckpt.get("epoch", "?")
                auc = ckpt.get("val_auc", 0)
                acc = ckpt.get("val_acc", 0)
                f1 = ckpt.get("val_macro_f1", 0)
                mod_time = datetime.fromtimestamp(BEST_MODEL.stat().st_mtime)

                status = "YENi EPOCH!" if epoch != prev_epoch and prev_epoch >= 0 else ""
                prev_epoch = epoch

                print(f"\n  Son Best Checkpoint:")
                print(f"    Epoch:     {epoch + 1} / 20")
                print(f"    AUC:       {auc:.4f}")
                print(f"    Accuracy:  {acc:.4f}" if acc else "")
                print(f"    Macro F1:  {f1:.4f}" if f1 else "")
                print(f"    Guncelleme:{mod_time.strftime('%H:%M:%S')}")
                if status:
                    print(f"    >>> {status} <<<")

                # Kalan epoch tahmini
                remaining = 20 - (epoch + 1)
                if remaining > 0:
                    print(f"\n  Kalan Epoch: {remaining}")
            else:
                print("\n  Checkpoint bulunamadi...")

            # Confusion Matrix
            cm = load_cm()
            if cm is not None and cm.shape == (2, 2):
                cm_time = datetime.fromtimestamp(CM_FILE.stat().st_mtime)
                cm_age = (datetime.now() - cm_time).total_seconds()

                if prev_cm_time and cm_time != prev_cm_time:
                    epoch_signal = " (EPOCH TAMAMLANDI!)"
                else:
                    epoch_signal = ""
                prev_cm_time = cm_time

                total = cm.sum()
                real_recall = cm[0, 0] / max(cm[0].sum(), 1) * 100
                fake_recall = cm[1, 1] / max(cm[1].sum(), 1) * 100
                overall_acc = (cm[0, 0] + cm[1, 1]) / max(total, 1) * 100

                print(f"\n  Confusion Matrix (son validasyon):{epoch_signal}")
                print(f"    Guncelleme: {cm_time.strftime('%H:%M:%S')} ({int(cm_age)}sn once)")
                print(f"    {'':>12} Pred REAL  Pred FAKE")
                print(f"    {'Gercek REAL':>12} {cm[0,0]:>8,}  {cm[0,1]:>8,}")
                print(f"    {'Gercek FAKE':>12} {cm[1,0]:>8,}  {cm[1,1]:>8,}")
                print(f"\n    REAL Recall:  {real_recall:.1f}%")
                print(f"    FAKE Recall:  {fake_recall:.1f}%")
                print(f"    Overall Acc:  {overall_acc:.1f}%")

            # GPU durumu
            try:
                import subprocess
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if r.returncode == 0:
                    parts = r.stdout.strip().split(", ")
                    if len(parts) == 4:
                        print(f"\n  GPU Durumu:")
                        print(f"    Kullanim:  {parts[0]}%")
                        print(f"    VRAM:      {parts[1]} / {parts[2]} MB")
                        print(f"    Sicaklik:  {parts[3]}°C")
            except Exception:
                pass

            # Aktif Python sureci
            try:
                import subprocess
                r = subprocess.run(
                    ["powershell", "-Command",
                     "Get-Process python -ErrorAction SilentlyContinue | "
                     "Where-Object {$_.WorkingSet64 -gt 500MB} | "
                     "Select-Object Id, @{N='RAM_MB';E={[math]::Round($_.WorkingSet64/1MB)}} | "
                     "Format-Table -HideTableHeaders"],
                    capture_output=True, text=True, timeout=5
                )
                if r.stdout.strip():
                    print(f"\n  Egitim Sureci: AKTIF")
                    for line in r.stdout.strip().split("\n"):
                        line = line.strip()
                        if line:
                            print(f"    {line}")
                else:
                    print(f"\n  !! EGITIM SURECI BULUNAMADI !!")
            except Exception:
                pass

            print(f"\n{'=' * 60}")
            print(f"  Sonraki kontrol: 30 saniye sonra...")

            time.sleep(30)

        except KeyboardInterrupt:
            print("\n\nIzleyici durduruldu.")
            break
        except Exception as e:
            print(f"\nHata: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
