"""
DeepfakeULTRA V5 — Tam Eğitim Pipeline Orkestratörü
Sıfırdan production-ready modele kadar tüm adımları yönetir.

Kullanım:
    # Tüm pipeline (veri hazırlığı + eğitim + kalibrasyon)
    python run_training.py

    # Sadece eğitim (veri zaten hazırsa)
    python run_training.py --skip-data-prep

    # Sadece veri hazırlığı
    python run_training.py --data-only

    # Sadece değerlendirme (eğitim sonrası)
    python run_training.py --eval-only

    # Mevcut checkpoint'ten DEVAM (sıfırdan değil)
    python run_training.py --resume models/best_model.pth --skip-data-prep
"""
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================
# YARDIMCI FONKSİYONLAR
# ===========================================================
def header(step: int, title: str, emoji: str = "🔹"):
    print(f"\n{'=' * 65}")
    print(f"  {emoji}  ADIM {step}: {title}")
    print(f"{'=' * 65}")


def run_script(script_path: str, args: list = None, check: bool = True) -> bool:
    """Script çalıştır, hata olursa raporla."""
    cmd = [sys.executable, str(PROJECT_ROOT / script_path)] + (args or [])
    print(f"  ▶ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  ❌ HATA: {script_path} başarısız (kod={result.returncode})")
        if check:
            sys.exit(result.returncode)
        return False
    return True


def check_dataset_exists() -> bool:
    """Train/val split mevcut mu?"""
    train_real = PROJECT_ROOT / "dataset" / "faces_split" / "train" / "real"
    train_fake = PROJECT_ROOT / "dataset" / "faces_split" / "train" / "fake"
    return train_real.exists() and train_fake.exists()


def count_images(directory: Path) -> int:
    """Dizindeki toplam görsel sayısı."""
    if not directory.exists():
        return 0
    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        count += len(list(directory.rglob(ext)))
    return count


def print_dataset_summary():
    """Veri seti özet istatistikleri."""
    base = PROJECT_ROOT / "dataset"
    splits = ["train", "val", "test"]
    print("\n  📊 Veri Seti Özeti:")
    total = 0
    for split in splits:
        for label in ["real", "fake"]:
            d = base / "faces_split" / split / label
            n = count_images(d)
            total += n
            print(f"     {split}/{label}: {n:,} görsel")

    jury_real = count_images(base / "jury_test" / "real")
    jury_fake = count_images(base / "jury_test" / "fake")
    print(f"     jury/real: {jury_real:,} görsel")
    print(f"     jury/fake: {jury_fake:,} görsel")
    print(f"     ---------------------")
    print(f"     TOPLAM: {total:,} görsel (jury hariç)")


# ===========================================================
# FAZ 0: VERİ HAZIRLIĞI
# ===========================================================
def phase_data_prep():
    """Tüm veri hazırlama adımlarını çalıştır."""
    header(0, "VERİ HAZIRLIĞI", "🗂️")

    # Adım 0.1: Hard-real üret
    header(1, "Hard-Real Üretimi (7000 görsel)", "🎨")
    print("  beauty_filter: 1500 | hdr_edited: 1500 | low_quality: 1500")
    print("  heavy_makeup: 1000  | profile_angle: 500 | screen_recapture: 1000")
    run_script("scripts/generate_hard_real.py")

    # Adım 0.2: Smart Split
    header(2, "Kalite-Bilinçli Veri Bölme (70/15/15)", "✂️")
    run_script("scripts/06_smart_split.py")

    # Adım 0.3: Jury genişletme
    header(3, "Jury Setini Genişlet (→ 5000, identity-safe, stratified)", "🧪")
    run_script("scripts/extend_jury.py")

    # Adım 0.4: Leakage kontrolü
    header(4, "Leakage Kontrolü (pHash + MD5)", "🔐")
    run_script(
        "scripts/leakage_checker.py",
        args=["--check", "dataset/faces_split/train", "dataset/jury_test",
              "--no-embedding"],   # Hızlı mod: FaceNet embedding atla
        check=False,  # Leakage varsa uyar ama durma
    )

    print_dataset_summary()
    print("\n  ✅ Veri hazırlığı tamamlandı.")


# ===========================================================
# FAZ 1: EĞİTİM
# ===========================================================
def phase_train(resume: str = None):
    """
    Sıfırdan eğitim başlat.
    resume=None → mevcut ağırlıkları yok say, epoch 0'dan başla.
    """
    header(5, "MODEL EĞİTİMİ (20 Epoch, Curriculum Learning)", "🚀")

    from config import model_cfg, DEVICE
    print(f"\n  ⚙️  Konfigürasyon:")
    print(f"     Cihaz:           {DEVICE}")
    print(f"     Batch Size:      {model_cfg.BATCH_SIZE}")
    print(f"     Epoch:           {model_cfg.EPOCHS}")
    print(f"     LR:              {model_cfg.LEARNING_RATE}")
    print(f"     Accumulation:    {model_cfg.GRADIENT_ACCUMULATION_STEPS} step")
    print(f"     Efektif Batch:   {model_cfg.BATCH_SIZE * model_cfg.GRADIENT_ACCUMULATION_STEPS}")
    print(f"     FP16:            {model_cfg.USE_MIXED_PRECISION}")
    print(f"     Curriculum:      {model_cfg.USE_CURRICULUM}")
    print(f"     Focal Alpha:     {model_cfg.FOCAL_ALPHA}")
    print(f"     Unfreeze Epoch:  {model_cfg.UNFREEZE_EPOCH}")
    print(f"\n  📚 Curriculum Takvimi:")
    for phase in model_cfg.CURRICULUM_PHASES[:4]:
        print(f"     Epoch {phase['start']:2d}-{min(phase['end'], model_cfg.EPOCHS):2d}: "
              f"hard_real_ratio = {phase['hard_real_ratio']:.0%}")

    if resume:
        print(f"\n  ⚠️  NOT: --resume verildi → {resume} dosyasından devam edilecek")
    else:
        print(f"\n  ✅ Sıfırdan eğitim — mevcut checkpoint KULLANILMIYOR")

    print(f"\n  ⏳ Eğitim başlıyor... (tahmini ~4-5 saat)\n")

    from core.trainer import train_and_evaluate
    train_and_evaluate(resume=resume)


# ===========================================================
# FAZ 2: DEĞERLENDİRME & KALİBRASYON
# ===========================================================
def phase_evaluate():
    """Eğitim sonrası tam değerlendirme ve kalibrasyon."""
    header(6, "DEĞERLENDİRME & KALİBRASYON", "📊")

    # Adım 6.1: Çok katmanlı metrik değerlendirme
    print("\n  [6.1] AUC / EER / ECE / Brier / Latency / ONNX...")
    run_script("scripts/evaluate_model.py")

    # Adım 6.2: Youden J-statistic eşik optimizasyonu
    print("\n  [6.2] Optimal eşik hesaplama (Youden J-statistic)...")
    threshold_script = PROJECT_ROOT / "scripts" / "find_threshold.py"
    if threshold_script.exists():
        run_script("scripts/find_threshold.py", check=False)
    else:
        print("  ⚠️  find_threshold.py bulunamadı, atlanıyor")

    # Adım 6.3: FP/FN arşivi + hard-negative mining
    print("\n  [6.3] Hata analizi ve hard-negative mining...")
    run_script("scripts/error_analysis.py", check=False)

    # Adım 6.4: Haftalık izleme kurulumu
    print("\n  [6.4] Haftalık izleme raporu...")
    run_script("scripts/weekly_scheduler.py", args=["--report"], check=False)

    # Sonuçları göster
    eval_dir = PROJECT_ROOT / "evaluation"
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.exists():
        import json
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        print(f"\n  {'-' * 50}")
        print(f"  📈 SONUÇLAR:")
        print(f"     ROC-AUC:       {m.get('roc_auc', 'N/A')}")
        print(f"     EER:           {m.get('eer', 'N/A')}")
        print(f"     ECE:           {m.get('ece', 'N/A')}")
        if 'brier_score' in m:
            print(f"     Brier Score:   {m['brier_score']}")
        if 'calibrated_ece' in m:
            ece = m['calibrated_ece']
            status = "✅ < 5%" if ece < 0.05 else "⚠️ > 5%"
            print(f"     Calibrated ECE:{ece:.4f} ({status})")
        if 'temperature' in m:
            print(f"     Temperature:   {m['temperature']}")
        if 'latency' in m:
            print(f"     Latency:       {m['latency'].get('mean_ms', 'N/A'):.1f}ms")
        print(f"     FP / FN:       {m.get('fp_total', 0)} / {m.get('fn_total', 0)}")
        print(f"  {'-' * 50}")


# ===========================================================
# ANA AKIŞ
# ===========================================================
def main():
    parser = argparse.ArgumentParser(
        description="DeepfakeULTRA V5 — Tam Eğitim Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Veri hazırlığını atla (veri zaten hazırsa kullan)")
    parser.add_argument("--data-only", action="store_true",
                        help="Sadece veri hazırlığı yap")
    parser.add_argument("--eval-only", action="store_true",
                        help="Sadece değerlendirme yap (eğitim sonrası)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint yolu (verilirse o noktadan devam eder)")
    args = parser.parse_args()

    start = datetime.now()
    print(f"\n{'=' * 65}")
    print(f"  🔍 DeepfakeULTRA V5 — Tam Eğitim Pipeline")
    print(f"  Başlangıç: {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mod: {'sadece veri' if args.data_only else 'sadece eval' if args.eval_only else 'tam pipeline'}")
    if args.resume:
        print(f"  Resume: {args.resume}")
    else:
        print(f"  Eğitim: SIFIRDAN (mevcut ağırlıklar kullanılmıyor)")
    print(f"{'=' * 65}")

    # Eval-only modu
    if args.eval_only:
        phase_evaluate()
        _print_summary(start)
        return

    # Veri hazırlığı
    if not args.skip_data_prep:
        if check_dataset_exists():
            print("\n  ℹ️  Mevcut veri seti bulundu.")
            print("       Yeniden oluşturmak için --skip-data-prep KULLANMAYIN.")
            print("       Devam edilecek...\n")
        phase_data_prep()
    else:
        print("\n  ⏭️  Veri hazırlığı atlandı (--skip-data-prep)")
        print_dataset_summary()

    if args.data_only:
        print("\n  ✅ --data-only: Sadece veri hazırlığı tamamlandı.")
        _print_summary(start)
        return

    # Eğitim (sıfırdan)
    phase_train(resume=args.resume)

    # Değerlendirme
    phase_evaluate()

    _print_summary(start)


def _print_summary(start: datetime):
    elapsed = (datetime.now() - start).total_seconds()
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n{'=' * 65}")
    print(f"  ✅ Pipeline tamamlandı!")
    print(f"  ⏱️  Toplam süre: {h}s {m}dk {s}sn")
    print(f"\n  📂 Çıktılar:")
    print(f"     models/best_model.pth              → Checkpoint")
    print(f"     models/calibration_weights.json    → Temperature T")
    print(f"     models/optimal_threshold.txt       → Youden eşiği")
    print(f"     evaluation/metrics.json            → Tüm metrikler")
    print(f"     evaluation/reliability_diagram.png → Kalibrasyon")
    print(f"     metadata/hard_negatives.json       → Replay buffer")
    print(f"     reports/                           → Haftalık raporlar")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
