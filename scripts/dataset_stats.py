"""
Veri Seti Istatistik Raporu — Pandas Tabanlı Analiz

faces_split_v2 uzerinde kaynak cesitliligi, dosya boyutu dagilimi,
uzanti analizi ve denge kontrolleri yapar.

Kullanim:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --source faces_split
    python scripts/dataset_stats.py --output reports/dataset_stats.csv
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import paths

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("pandas yuklu degil: pip install pandas")

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def detect_source(filename: str) -> str:
    """Dosya adindan kaynak veri setini tahmin et."""
    name = filename.lower()
    if name.startswith("df40_"):
        parts = name.split("_")
        return "df40_" + parts[1] if len(parts) >= 2 else "df40_other"
    elif name.startswith("celeba_"):
        return "celeba_hq"
    elif name.startswith("ffpp_") or name.startswith("faceforensics"):
        return "ffpp"
    elif name.startswith("ffhq"):
        return "ffhq"
    elif name.startswith("utk") or "utkface" in name:
        return "utkface"
    elif name.startswith("sidset") or name.startswith("sid_"):
        return "sidset"
    elif name.startswith("sbi_"):
        return "sbi_generated"
    elif name.startswith("vggface2") or name.startswith("vgg_"):
        return "vggface2"
    elif name.startswith("genimage") or name.startswith("gen_"):
        return "genimage"
    elif "custom" in name or "team" in name:
        return "custom_team"
    else:
        return "diger"


def scan_split_directory(base_dir: Path) -> list:
    """Split dizinindeki tum gorselleri tara ve metadata topla."""
    records = []
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            cls_dir = base_dir / split / cls
            if not cls_dir.exists():
                continue
            for f in cls_dir.rglob("*"):
                if not f.is_file() or f.suffix.lower() not in SUPPORTED:
                    continue
                if ".cache" in str(f):
                    continue
                try:
                    size_kb = f.stat().st_size / 1024
                except OSError:
                    size_kb = 0
                records.append({
                    "split": split,
                    "label": cls,
                    "source": detect_source(f.name),
                    "size_kb": round(size_kb, 1),
                    "ext": f.suffix.lower(),
                    "filename": f.name,
                })
    return records


def generate_report(df: pd.DataFrame):
    """DataFrame uzerinde kapsamli istatistik raporu olustur."""
    print("\n" + "=" * 70)
    print("  VERI SETI ISTATISTIK RAPORU")
    print("=" * 70)

    # 1. Genel ozet
    print("\n📊 GENEL OZET")
    print(f"  Toplam gorsel: {len(df):,}")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        real_n = len(sub[sub["label"] == "real"])
        fake_n = len(sub[sub["label"] == "fake"])
        total_n = len(sub)
        balance = real_n / total_n * 100 if total_n > 0 else 0
        print(f"  {split:6s}: {real_n:>8,} REAL + {fake_n:>8,} FAKE = "
              f"{total_n:>8,} ({balance:.1f}% real)")

    # 2. Kaynak cesitliligi
    print("\n📋 KAYNAK CESITLILIGI (split × label × source)")
    pivot = (df.groupby(["split", "label", "source"])
             .size()
             .reset_index(name="count")
             .pivot_table(index="source", columns=["split", "label"],
                          values="count", fill_value=0, aggfunc="sum"))
    # Kolon isimlerini duzelt
    pivot.columns = [f"{s}/{l}" for s, l in pivot.columns]
    pivot["TOPLAM"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("TOPLAM", ascending=False)
    print(pivot.to_string())

    # 3. Dosya boyutu dagilimi
    print("\n📏 DOSYA BOYUTU DAGILIMI (KB)")
    size_stats = df.groupby("label")["size_kb"].describe().round(1)
    print(size_stats.to_string())

    # Cok kucuk dosyalar (muhtemel bozuk)
    tiny = df[df["size_kb"] < 2]
    if len(tiny) > 0:
        print(f"\n  ⚠️ {len(tiny)} dosya < 2KB (muhtemel bozuk):")
        for _, row in tiny.head(10).iterrows():
            print(f"    {row['split']}/{row['label']}: {row['filename']} ({row['size_kb']:.1f} KB)")

    # 4. Uzanti dagilimi
    print("\n📎 UZANTI DAGILIMI")
    ext_counts = df.groupby(["label", "ext"]).size().reset_index(name="count")
    print(ext_counts.to_string(index=False))

    # 5. Train/Val/Test orani
    print("\n⚖️ SPLIT ORANLARI")
    split_counts = df.groupby("split").size()
    total = split_counts.sum()
    for split in ["train", "val", "test"]:
        if split in split_counts.index:
            n = split_counts[split]
            print(f"  {split}: {n:,} ({n/total*100:.1f}%)")

    # 6. Label dengesizligi uyarisi
    print("\n🔍 DENGE KONTROLU")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        real_n = len(sub[sub["label"] == "real"])
        fake_n = len(sub[sub["label"] == "fake"])
        if real_n == 0 or fake_n == 0:
            print(f"  🔴 {split}: Bir sinif TAMAMEN EKSIK!")
        else:
            ratio = max(real_n, fake_n) / min(real_n, fake_n)
            if ratio > 1.1:
                print(f"  ⚠️ {split}: Dengesiz (oran={ratio:.2f})")
            else:
                print(f"  ✅ {split}: Dengeli (oran={ratio:.2f})")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Veri seti istatistik raporu")
    parser.add_argument("--source", type=str, default="faces_split_v2",
                        help="Split dizini (varsayilan: faces_split_v2)")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV olarak kaydet")
    args = parser.parse_args()

    if not HAS_PANDAS:
        print("pandas yuklu degil!")
        return

    base_dir = paths.DATASET_DIR / args.source
    if not base_dir.exists():
        print(f"Dizin bulunamadi: {base_dir}")
        return

    print(f"Taranıyor: {base_dir}")
    records = scan_split_directory(base_dir)

    if not records:
        print("Gorsel bulunamadi!")
        return

    df = pd.DataFrame(records)
    generate_report(df)

    if args.output:
        output_path = paths.BASE_DIR / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\n💾 CSV kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
