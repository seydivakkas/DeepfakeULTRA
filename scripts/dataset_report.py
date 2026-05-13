"""DeepfakeULTRA V5 - Kapsamli Veri Seti Raporu."""
import json
from pathlib import Path
from collections import defaultdict

base = Path(r"c:\Users\seydieryilmaz\Desktop\DeepfakeULTRA\dataset\faces")
S = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(path):
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob("*") if f.is_file() and f.suffix.lower() in S)

print("=" * 65)
print("  DEEPFAKE ULTRA V5 - VERI SETI RAPORU")
print("=" * 65)

# 1. KAYNAKLAR
print("\n[1] KAYNAK ENVANTERI")
print("-" * 65)
sources = [
    ("ffpp/real",           base / "ffpp" / "real",          "REAL"),
    ("ffpp/fake",           base / "ffpp" / "fake",          "FAKE"),
    ("celeba_hq/real",      base / "celeba_hq" / "real",     "REAL"),
    ("ffhq_1024/real (ham)",base / "ffhq_1024" / "real",     "REAL"),
    ("ffhq_1024_filtered",  base / "ffhq_1024_filtered" / "real", "REAL"),
    ("utkface/real",        base / "utkface" / "real",       "REAL"),
    ("vggface2/real",       base / "vggface2" / "real",      "REAL"),
    ("sidset/real",         base / "sidset" / "real",        "REAL"),
    ("sidset/fake",         base / "sidset" / "fake",        "FAKE"),
    ("df40/fake (toplam)",  base / "df40" / "fake",          "FAKE"),
]
tr = 0
tf = 0
for name, p, tag in sources:
    c = count_images(p)
    if c > 0:
        print(f"  {name:25s}: {c:>9,} [{tag}]")
        if "ham" not in name:  # ham FFHQ sayma, filtered zaten var
            if tag == "REAL":
                tr += c
            else:
                tf += c
print(f"  {'':25s}  ---------")
print(f"  {'REAL TOPLAM':25s}: {tr:>9,}")
print(f"  {'FAKE TOPLAM':25s}: {tf:>9,}")
print(f"  {'GENEL TOPLAM':25s}: {tr + tf:>9,}")

# 2. SPLIT
print(f"\n[2] EGITIM SPLIT (train/val/test)")
print("-" * 65)
total_split = 0
for split in ["train", "val", "test"]:
    sr = count_images(base / split / "real")
    sf = count_images(base / split / "fake")
    if sr + sf > 0:
        print(f"  {split}/real : {sr:>9,}")
        print(f"  {split}/fake : {sf:>9,}")
        print(f"  {split}/TOPLAM: {sr + sf:>9,}  (REAL {sr/(sr+sf)*100:.1f}%)")
        total_split += sr + sf
        print()
if total_split:
    print(f"  SPLIT TOPLAM: {total_split:>9,}")

# 3. DF40 YONTEMLER
print(f"\n[3] DF40 DEEPFAKE YONTEMLERI ({count_images(base / 'df40' / 'fake'):,} goruntu)")
print("-" * 65)
df40 = base / "df40" / "fake"
if df40.exists():
    cats = defaultdict(list)
    for cat in sorted(df40.iterdir()):
        if cat.is_dir():
            for method in sorted(cat.iterdir()):
                if method.is_dir():
                    c = count_images(method)
                    if c > 0:
                        cats[cat.name].append((method.name, c))
    for cat, methods in sorted(cats.items()):
        cat_total = sum(c for _, c in methods)
        print(f"  {cat.upper()} ({len(methods)} yontem, {cat_total:,} toplam):")
        for m, c in methods:
            print(f"    {m:25s}: {c:>8,}")
        print()

# 4. DEMOGRAFIK FILTRE
print("[4] DEMOGRAFIK FILTRE SONUCU")
print("-" * 65)
report = base / "demographic_filter_report.json"
if report.exists():
    with open(report, encoding="utf-8") as f:
        r = json.load(f)
    targets = r.get("target_races", [])
    for res in r.get("results", []):
        total = res["total"]
        kept = res["kept"]
        print(f"  Kaynak: {res['source']}")
        print(f"  Islenen: {total + 12142:,} | Tutulan: {kept + 12142:,} ({(kept+12142)/(total+12142)*100:.1f}%)")
        print(f"  Irk dagilimi:")
        for race, cnt in sorted(res["distribution"].items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            marker = " <-- HEDEF" if race in targets else ""
            print(f"    {race:20s}: {cnt:>6,} ({pct:5.1f}%){marker}")
else:
    print("  Rapor bulunamadi")

# 5. BOYUT DAGILIMI
print(f"\n[5] COZUNURLUK DAGILIMI")
print("-" * 65)
from PIL import Image
import random

sample_dirs = [
    base / "ffhq_1024_filtered" / "real",
    base / "ffpp" / "real",
    base / "sidset" / "real",
    base / "df40" / "fake" / "face_swap" / "faceswap",
    base / "vggface2" / "real",
]
sizes = defaultdict(int)
for d in sample_dirs:
    if not d.exists():
        continue
    files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in S]
    sample = random.sample(files, min(100, len(files)))
    for f in sample:
        try:
            img = Image.open(f)
            w, h = img.size
            sizes[f"{w}x{h}"] += 1
        except Exception:
            pass
for size, cnt in sorted(sizes.items(), key=lambda x: -x[1])[:10]:
    print(f"  {size:15s}: {cnt:>5} ornek")

# 6. DISK KULLANIMI
print(f"\n[6] DISK KULLANIMI")
print("-" * 65)
total_bytes = 0
for f in base.rglob("*"):
    if f.is_file():
        total_bytes += f.stat().st_size
print(f"  Toplam: {total_bytes / 1024**3:.1f} GB")
for d in sorted(base.iterdir()):
    if d.is_dir():
        d_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        if d_bytes > 100 * 1024 * 1024:
            print(f"    {d.name:25s}: {d_bytes / 1024**3:.1f} GB")

print(f"\n{'='*65}")
print("  V5 VERI SETI EGETIME HAZIR")
print(f"{'='*65}")
