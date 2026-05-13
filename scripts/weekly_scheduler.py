"""
GÖREV 6: Haftalık Otomatik Değerlendirme ve Raporlama
Windows Task Scheduler veya manuel çalıştırma ile kullanılır.

Akış:
    1. evaluate_model.py çalıştır (metrikler)
    2. error_analysis.py çalıştır (hata arşivi + hard-negative mining)
    3. Haftalık özet rapor oluştur

Kullanım:
    python scripts/weekly_scheduler.py              # Tüm pipeline
    python scripts/weekly_scheduler.py --report     # Sadece rapor
    python scripts/weekly_scheduler.py --setup-task # Windows Task Scheduler kurulumu
"""
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

SCRIPTS_DIR = Path(__file__).parent
PROJECT_DIR = Path(__file__).parent.parent


def run_evaluation():
    """evaluate_model.py çalıştır."""
    print("\n" + "=" * 60)
    print("ADIM 1: Model Değerlendirme")
    print("=" * 60)

    script = SCRIPTS_DIR / "evaluate_model.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠️ evaluate_model.py hata: {result.stderr}")
        return False
    return True


def run_error_analysis():
    """error_analysis.py çalıştır (hard-negative mining dahil)."""
    print("\n" + "=" * 60)
    print("ADIM 2: Hata Analizi & Hard-Negative Mining")
    print("=" * 60)

    script = SCRIPTS_DIR / "error_analysis.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠️ error_analysis.py hata: {result.stderr}")
        return False
    return True


def run_weekly_report():
    """Haftalık özet rapor oluştur."""
    print("\n" + "=" * 60)
    print("ADIM 3: Haftalık Rapor")
    print("=" * 60)

    script = SCRIPTS_DIR / "error_analysis.py"
    result = subprocess.run(
        [sys.executable, str(script), "--weekly-report"],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    return result.returncode == 0


def setup_windows_task():
    """Windows Task Scheduler'da haftalık görev oluştur."""
    print("\n🔧 Windows Task Scheduler kurulumu")

    task_name = "DeepfakeULTRA_Weekly_Report"
    script_path = Path(__file__).resolve()
    python_path = sys.executable

    # schtasks komutu
    cmd = (
        f'schtasks /create /tn "{task_name}" '
        f'/tr "\"{python_path}\" \"{script_path}\"" '
        f'/sc WEEKLY /d MON /st 09:00 '
        f'/f'
    )

    print(f"  Komut: {cmd}")
    print(f"\n  Bu komutu yönetici olarak çalıştırın:")
    print(f"  PowerShell (Admin): {cmd}")
    print(f"\n  Silmek için: schtasks /delete /tn \"{task_name}\" /f")


def full_pipeline():
    """Tam haftalık pipeline."""
    start = datetime.now()
    print(f"🚀 DeepfakeULTRA Haftalık Pipeline — {start.strftime('%Y-%m-%d %H:%M')}")

    # Adım 1: Değerlendirme
    eval_ok = run_evaluation()

    # Adım 2: Hata analizi (değerlendirme başarılıysa)
    if eval_ok:
        run_error_analysis()

    # Adım 3: Haftalık rapor
    run_weekly_report()

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n✅ Pipeline tamamlandı ({elapsed:.0f}s)")


def main():
    parser = argparse.ArgumentParser(description="Haftalık Otomatik Değerlendirme")
    parser.add_argument("--report", action="store_true", help="Sadece rapor oluştur")
    parser.add_argument("--setup-task", action="store_true",
                        help="Windows Task Scheduler kurulumu")
    args = parser.parse_args()

    if args.setup_task:
        setup_windows_task()
    elif args.report:
        run_weekly_report()
    else:
        full_pipeline()


if __name__ == "__main__":
    main()
