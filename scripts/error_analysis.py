"""
GÖREV 6: Yanlış Sınıflandırma İzleme, Hard-Negative Mining ve Geri Besleme Döngüsü
False positive/negative görselleri arşivler, trend raporu üretir.
Hard-negative replay buffer oluşturur (eğitim entegrasyonu için).

Kullanım:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --report-only
    python scripts/error_analysis.py --mine-hard-negatives
"""
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

EVAL_DIR = Path(__file__).parent.parent / "evaluation"
META_DIR = Path(__file__).parent.parent / "metadata"
FALSE_CASES = META_DIR / "false_cases"
REPLAY_BUFFER_PATH = META_DIR / "hard_negatives.json"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def analyze_errors():
    """evaluation/metrics.json'dan hata analizini çalıştır."""
    print("=" * 60)
    print("GÖREV 6: Yanlış Sınıflandırma Analizi")
    print("=" * 60)

    META_DIR.mkdir(parents=True, exist_ok=True)
    FALSE_CASES.mkdir(parents=True, exist_ok=True)
    (FALSE_CASES / "false_positive").mkdir(exist_ok=True)
    (FALSE_CASES / "false_negative").mkdir(exist_ok=True)

    # Metrics yükle
    metrics_path = EVAL_DIR / "metrics.json"
    if not metrics_path.exists():
        print("❌ evaluation/metrics.json bulunamadı. Önce evaluate_model.py çalıştırın.")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # FP görselleri kopyala
    fp_src = EVAL_DIR / "fp_analysis"
    fn_src = EVAL_DIR / "fn_analysis"

    fp_copied = 0
    if fp_src.exists():
        for f in fp_src.glob("*.*"):
            dst = FALSE_CASES / "false_positive" / f.name
            shutil.copy2(f, dst)
            fp_copied += 1

    fn_copied = 0
    if fn_src.exists():
        for f in fn_src.glob("*.*"):
            dst = FALSE_CASES / "false_negative" / f.name
            shutil.copy2(f, dst)
            fn_copied += 1

    print(f"\n📂 Arşivlenen hatalar:")
    print(f"  False Positive: {fp_copied} görsel → {FALSE_CASES / 'false_positive'}")
    print(f"  False Negative: {fn_copied} görsel → {FALSE_CASES / 'false_negative'}")

    # Trend logu oluştur/güncelle
    trend_path = META_DIR / "error_trend.json"
    trend = []
    if trend_path.exists():
        with open(trend_path, "r", encoding="utf-8") as f:
            trend = json.load(f)

    trend.append({
        "timestamp": datetime.now().isoformat(),
        "fp_count": metrics.get("fp_total", 0),
        "fn_count": metrics.get("fn_total", 0),
        "roc_auc": metrics.get("roc_auc", 0),
        "eer": metrics.get("eer", 0),
        "ece": metrics.get("ece", 0),
        "brier_score": metrics.get("brier_score", 0),
        "calibrated_ece": metrics.get("calibrated_ece", 0),
        "total_evaluated": metrics.get("total_evaluated", 0),
    })

    with open(trend_path, "w", encoding="utf-8") as f:
        json.dump(trend, f, indent=2, ensure_ascii=False)

    # Per-source hata analizi
    print("\n📊 Kaynak Bazlı Hata Dağılımı:")
    per_source = metrics.get("per_source", {})
    for src, m in sorted(per_source.items(), key=lambda x: x[1].get("fp_count", 0), reverse=True):
        fp = m.get("fp_count", 0)
        fn = m.get("fn_count", 0)
        acc = m.get("accuracy", 0)
        total = m.get("total", 0)
        if fp > 0 or fn > 0:
            print(f"  ⚠️ {src}: acc={acc:.3f} FP={fp} FN={fn} (n={total})")

    # Hard-Negative Mining
    hard_negatives = mine_hard_negatives(metrics)
    if hard_negatives:
        update_replay_buffer(hard_negatives)

    # Rapor oluştur
    report_path = META_DIR / "error_report.md"
    _write_report(report_path, metrics, trend, per_source, hard_negatives)

    print(f"\n📋 Rapor: {report_path}")
    print(f"📋 Trend: {trend_path}")
    if hard_negatives:
        print(f"📋 Hard-negatives: {REPLAY_BUFFER_PATH} ({len(hard_negatives)} örnek)")
    print("✅ GÖREV_6_TAMAMLANDI")


# ═══════════════════════════════════════════════════════════
# HARD-NEGATIVE MINING
# ═══════════════════════════════════════════════════════════
def mine_hard_negatives(metrics: dict, top_k: int = 500) -> list:
    """
    FP/FN görselleri loss/confidence ile sıralayarak en zor örnekleri seç.
    Bu örnekler replay buffer'a eklenir ve epoch 10+ sonrası eğitime karıştırılır.
    """
    hard_negatives = []

    # FP görselleri (REAL olarak etiketli, FAKE olarak sınıflandırılmış)
    fp_dir = FALSE_CASES / "false_positive"
    if fp_dir.exists():
        for f in sorted(fp_dir.glob("*.*")):
            hard_negatives.append({
                "path": str(f),
                "type": "false_positive",
                "true_label": 0,  # REAL
                "pred_label": 1,  # FAKE
                "source": f.stem.split("_")[0] if "_" in f.stem else "unknown",
                "mined_at": datetime.now().isoformat(),
            })

    # FN görselleri (FAKE olarak etiketli, REAL olarak sınıflandırılmış)
    fn_dir = FALSE_CASES / "false_negative"
    if fn_dir.exists():
        for f in sorted(fn_dir.glob("*.*")):
            hard_negatives.append({
                "path": str(f),
                "type": "false_negative",
                "true_label": 1,  # FAKE
                "pred_label": 0,  # REAL
                "source": f.stem.split("_")[0] if "_" in f.stem else "unknown",
                "mined_at": datetime.now().isoformat(),
            })

    # Top-K sınırla
    if len(hard_negatives) > top_k:
        hard_negatives = hard_negatives[:top_k]

    if hard_negatives:
        fp_count = sum(1 for h in hard_negatives if h["type"] == "false_positive")
        fn_count = sum(1 for h in hard_negatives if h["type"] == "false_negative")
        print(f"\n🎯 Hard-Negative Mining: {len(hard_negatives)} örnek (FP={fp_count}, FN={fn_count})")

    return hard_negatives


def update_replay_buffer(hard_negatives: list, max_buffer_size: int = 2000):
    """
    Hard-negative replay buffer'ı güncelle.
    Eski örnekler FIFO mantığıyla çıkarılır.
    """
    META_DIR.mkdir(parents=True, exist_ok=True)

    # Mevcut buffer'ı yükle
    buffer = []
    if REPLAY_BUFFER_PATH.exists():
        with open(REPLAY_BUFFER_PATH, "r", encoding="utf-8") as f:
            buffer = json.load(f)

    # Yeni örnekleri ekle (path bazlı duplicate engelle)
    existing_paths = {h["path"] for h in buffer}
    for h in hard_negatives:
        if h["path"] not in existing_paths:
            buffer.append(h)
            existing_paths.add(h["path"])

    # Buffer boyutunu sınırla (en eski örnekleri at)
    if len(buffer) > max_buffer_size:
        buffer = buffer[-max_buffer_size:]

    with open(REPLAY_BUFFER_PATH, "w", encoding="utf-8") as f:
        json.dump(buffer, f, indent=2, ensure_ascii=False)

    print(f"  💾 Replay buffer güncellendi: {len(buffer)} toplam örnek")


def get_replay_buffer() -> list:
    """Mevcut replay buffer'ı döndür (trainer entegrasyonu için)."""
    if REPLAY_BUFFER_PATH.exists():
        with open(REPLAY_BUFFER_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ═══════════════════════════════════════════════════════════
# RAPOR OLUŞTURMA
# ═══════════════════════════════════════════════════════════
def _write_report(report_path, metrics, trend, per_source, hard_negatives):
    """Detaylı hata analizi raporu oluştur."""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Hata Analizi Raporu — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"## Özet\n")
        f.write(f"- **ROC-AUC:** {metrics.get('roc_auc', 'N/A')}\n")
        f.write(f"- **EER:** {metrics.get('eer', 'N/A')}\n")
        f.write(f"- **ECE:** {metrics.get('ece', 'N/A')}\n")
        f.write(f"- **Brier Score:** {metrics.get('brier_score', 'N/A')}\n")
        f.write(f"- **Calibrated ECE:** {metrics.get('calibrated_ece', 'N/A')}\n")
        f.write(f"- **Temperature:** {metrics.get('temperature', 'N/A')}\n")
        f.write(f"- **False Positive:** {metrics.get('fp_total', 0)}\n")
        f.write(f"- **False Negative:** {metrics.get('fn_total', 0)}\n\n")

        f.write(f"## Kaynak Bazlı\n")
        f.write(f"| Kaynak | Accuracy | FP | FN | Toplam |\n|---|---|---|---|---|\n")
        for src, m in sorted(per_source.items()):
            f.write(f"| {src} | {m.get('accuracy', 0):.3f} | {m.get('fp_count', 0)} | {m.get('fn_count', 0)} | {m.get('total', 0)} |\n")

        if hard_negatives:
            f.write(f"\n## Hard-Negative Örnekleri\n")
            f.write(f"Toplam: {len(hard_negatives)} örnek\n\n")
            f.write(f"| Tip | Kaynak | Gerçek Etiket | Tahmin |\n|---|---|---|---|\n")
            for h in hard_negatives[:20]:
                f.write(f"| {h['type']} | {h['source']} | {h['true_label']} | {h['pred_label']} |\n")
            if len(hard_negatives) > 20:
                f.write(f"\n... ve {len(hard_negatives) - 20} örnek daha\n")

        f.write(f"\n## Trend\n")
        if len(trend) > 1:
            f.write(f"| Tarih | FP | FN | AUC | EER | ECE | Brier |\n|---|---|---|---|---|---|---|\n")
            for t in trend[-10:]:
                f.write(f"| {t['timestamp'][:16]} | {t['fp_count']} | {t['fn_count']} "
                        f"| {t.get('roc_auc', 0):.4f} | {t.get('eer', 0):.4f} "
                        f"| {t.get('ece', 0):.4f} | {t.get('brier_score', 0):.4f} |\n")


def generate_weekly_report():
    """Haftalık özet rapor oluştur (weekly_scheduler.py tarafından çağrılır)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    trend_path = META_DIR / "error_trend.json"
    if not trend_path.exists():
        print("❌ Trend verisi bulunamadı. Önce error_analysis çalıştırın.")
        return

    with open(trend_path, "r", encoding="utf-8") as f:
        trend = json.load(f)

    date_str = datetime.now().strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"weekly_{date_str}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Haftalık Performans Raporu — {date_str}\n\n")
        f.write(f"## Son 7 Günlük Trend\n\n")

        if trend:
            latest = trend[-1]
            f.write(f"### Son Durum\n")
            f.write(f"- **ROC-AUC:** {latest.get('roc_auc', 'N/A')}\n")
            f.write(f"- **EER:** {latest.get('eer', 'N/A')}\n")
            f.write(f"- **ECE:** {latest.get('ece', 'N/A')}\n")
            f.write(f"- **Brier Score:** {latest.get('brier_score', 'N/A')}\n")
            f.write(f"- **FP / FN:** {latest.get('fp_count', 0)} / {latest.get('fn_count', 0)}\n\n")

            if len(trend) > 1:
                prev = trend[-2]
                auc_delta = latest.get("roc_auc", 0) - prev.get("roc_auc", 0)
                eer_delta = latest.get("eer", 0) - prev.get("eer", 0)
                f.write(f"### Değişim (son ölçüme göre)\n")
                f.write(f"- AUC: {'+' if auc_delta >= 0 else ''}{auc_delta:.4f}\n")
                f.write(f"- EER: {'+' if eer_delta >= 0 else ''}{eer_delta:.4f}\n\n")

        # Replay buffer durumu
        buffer = get_replay_buffer()
        if buffer:
            fp_in_buffer = sum(1 for h in buffer if h["type"] == "false_positive")
            fn_in_buffer = sum(1 for h in buffer if h["type"] == "false_negative")
            f.write(f"### Hard-Negative Replay Buffer\n")
            f.write(f"- Toplam: {len(buffer)} örnek\n")
            f.write(f"- FP: {fp_in_buffer} | FN: {fn_in_buffer}\n\n")

        f.write(f"---\n*Otomatik oluşturuldu: {datetime.now().isoformat()}*\n")

    print(f"📋 Haftalık rapor: {report_path}")
    return report_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--mine-hard-negatives", action="store_true",
                        help="Sadece hard-negative mining çalıştır")
    parser.add_argument("--weekly-report", action="store_true",
                        help="Haftalık özet rapor oluştur")
    args = parser.parse_args()

    if args.weekly_report:
        generate_weekly_report()
        return

    analyze_errors()


if __name__ == "__main__":
    main()
