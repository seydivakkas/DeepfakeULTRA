"""
DF40 veri setinden egitimde kullanilmayan 50 real + 50 fake resmi
GAN ve Diffusion olarak etiketleyerek test klasorune cikartir.
"""
import os
import sys
import shutil
import random
import json
from pathlib import Path
from datetime import datetime

# Windows konsol encoding sorununu coz
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

random.seed(42)

# Yollar
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DF40_FAKE_DIR = PROJECT_ROOT / "dataset" / "faces" / "df40" / "fake"
TRAIN_FAKE_DIR = PROJECT_ROOT / "dataset" / "faces_split_v2" / "train" / "fake"
TRAIN_REAL_DIR = PROJECT_ROOT / "dataset" / "faces_split_v2" / "train" / "real"
TEST_FAKE_DIR = PROJECT_ROOT / "dataset" / "faces_split_v2" / "test" / "fake"
TEST_REAL_DIR = PROJECT_ROOT / "dataset" / "faces_split_v2" / "test" / "real"
VAL_FAKE_DIR = PROJECT_ROOT / "dataset" / "faces_split_v2" / "val" / "fake"
VAL_REAL_DIR = PROJECT_ROOT / "dataset" / "faces_split_v2" / "val" / "real"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "df40_test_samples"

# Metot -> Teknoloji eslemesi
DIFFUSION_METHODS = {"dit", "ddim", "midjourney", "collabdiff"}
GAN_METHODS = {
    "stylegan3", "danet",
    "starganv2", "styleclip",
    "blendface", "e4s", "facedancer", "faceswap",
    "fsgan", "inswap", "mobileswap",
    "facevid2vid", "fomm", "heygen", "hyperreenact",
    "lia", "mcnet", "mraa", "one_shot_free", "pirender",
    "sadtalker",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_used_stems():
    """Egitim/test/val'de daha once kullanilan dosya stem'lerini topla (set-based, hizli)."""
    used = set()
    for d in [TRAIN_FAKE_DIR, TRAIN_REAL_DIR, TEST_FAKE_DIR, TEST_REAL_DIR, VAL_FAKE_DIR, VAL_REAL_DIR]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    used.add(f.stem)
    return used


def collect_df40_fake_images():
    """DF40 fake dizininden tum resimleri metot ve teknolojiyle birlikte topla."""
    images = []
    
    for category_dir in sorted(DF40_FAKE_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        
        for method_dir in sorted(category_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            method_lower = method.lower()
            
            if method_lower in DIFFUSION_METHODS:
                tech = "diffusion"
            elif method_lower in GAN_METHODS:
                tech = "gan"
            else:
                print(f"  [UYARI] Bilinmeyen metot: {method} ({category}), atlaniyor.")
                continue
            
            count = 0
            for img in method_dir.iterdir():
                if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                    images.append((img, method, tech, category))
                    count += 1
            print(f"    {category}/{method} ({tech}): {count} resim")
    
    return images


def main():
    print("=" * 60)
    print("DF40 Test Ornekleri Cikartma Scripti")
    print("=" * 60)
    
    # 1. Daha once kullanilan dosyalari topla (stem bazli set)
    print("\n[1/5] Egitim/test/val'de kullanilan dosyalar taraniyor...")
    used_stems = get_used_stems()
    print(f"  -> Toplam {len(used_stems)} dosya daha once kullanilmis.")
    
    # 2. DF40 fake resimlerini topla
    print("\n[2/5] DF40 fake resimleri taraniyor...")
    all_fake_images = collect_df40_fake_images()
    print(f"  -> Toplam {len(all_fake_images)} fake resim bulundu.")
    
    # Egitimde kullanilan dosyalari filtrele (stem set lookup - O(1))
    available_fake = []
    for img_path, method, tech, category in all_fake_images:
        if img_path.stem not in used_stems:
            available_fake.append((img_path, method, tech, category))
    
    print(f"  -> Egitimde kullanilmayan: {len(available_fake)} fake resim.")
    
    # 3. Teknolojiye gore ayir
    gan_images = [x for x in available_fake if x[2] == "gan"]
    diff_images = [x for x in available_fake if x[2] == "diffusion"]
    print(f"\n[3/5] GAN: {len(gan_images)}, Diffusion: {len(diff_images)}")
    
    # 4. 50 fake sec (25 GAN + 25 Diffusion dengeli dagilim)
    print("\n[4/5] Ornekler seciliyor...")
    n_gan = min(25, len(gan_images))
    n_diff = min(25, len(diff_images))
    
    if n_gan < 25 and n_diff >= 25:
        n_diff = min(50 - n_gan, len(diff_images))
    elif n_diff < 25 and n_gan >= 25:
        n_gan = min(50 - n_diff, len(gan_images))
    
    def stratified_sample(images, n):
        by_method = {}
        for item in images:
            by_method.setdefault(item[1], []).append(item)
        
        selected = []
        methods = list(by_method.keys())
        random.shuffle(methods)
        
        per_method = max(1, n // len(methods)) if methods else 0
        remainder = n
        
        for method in methods:
            pool = by_method[method]
            random.shuffle(pool)
            take = min(per_method, len(pool), remainder)
            selected.extend(pool[:take])
            remainder -= take
        
        if remainder > 0:
            selected_set = set(id(x) for x in selected)
            remaining_pool = [x for x in images if id(x) not in selected_set]
            random.shuffle(remaining_pool)
            selected.extend(remaining_pool[:remainder])
        
        return selected[:n]
    
    selected_gan = stratified_sample(gan_images, n_gan)
    selected_diff = stratified_sample(diff_images, n_diff)
    selected_fake = selected_gan + selected_diff
    
    print(f"  -> Secilen: {len(selected_gan)} GAN + {len(selected_diff)} Diffusion = {len(selected_fake)} fake")
    
    # 5. 50 real resim - egitimde kullanilmayan
    real_sources = [
        PROJECT_ROOT / "dataset" / "faces" / "hard_real",
        PROJECT_ROOT / "dataset" / "faces" / "sidset" / "real",
        PROJECT_ROOT / "dataset" / "faces" / "custom_team" / "real",
        PROJECT_ROOT / "dataset" / "faces" / "celeba_hq",
        PROJECT_ROOT / "dataset" / "faces" / "ffhq_256",
        PROJECT_ROOT / "dataset" / "faces" / "vggface2",
        PROJECT_ROOT / "dataset" / "faces" / "utkface",
    ]
    
    available_real = []
    for source_dir in real_sources:
        if not source_dir.exists():
            print(f"  [BILGI] {source_dir.name} bulunamadi, atlaniyor.")
            continue
        count = 0
        # Recursive tarama (alt dizinlerdeki resimleri de bul)
        for img in source_dir.rglob("*"):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                if img.stem not in used_stems:
                    available_real.append((img, source_dir.name))
                    count += 1
        print(f"  {source_dir.name}: {count} kullanilmamis real resim")
    
    print(f"  -> Toplam kullanilmamis real resim havuzu: {len(available_real)}")
    
    random.shuffle(available_real)
    selected_real = available_real[:50]
    print(f"  -> Secilen: {len(selected_real)} real resim")
    
    # 6. Cikti dizini olustur ve kopyala
    print(f"\n[5/5] Dosyalar kopyalaniyor -> {OUTPUT_DIR}")
    
    out_real = OUTPUT_DIR / "real"
    out_gan = OUTPUT_DIR / "fake_gan"
    out_diff = OUTPUT_DIR / "fake_diffusion"
    
    for d in [out_real, out_gan, out_diff]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Real kopyala
    real_manifest = []
    for i, (img, source_name) in enumerate(selected_real):
        dest = out_real / f"real_{i:03d}_{img.name}"
        shutil.copy2(img, dest)
        real_manifest.append({
            "index": i,
            "filename": dest.name,
            "source": str(img),
            "label": "real",
            "source_dataset": source_name
        })
    
    # Fake GAN kopyala
    gan_manifest = []
    for i, (img, method, tech, category) in enumerate(selected_gan):
        dest = out_gan / f"gan_{i:03d}_{method}_{img.name}"
        shutil.copy2(img, dest)
        gan_manifest.append({
            "index": i,
            "filename": dest.name,
            "source": str(img),
            "label": "fake",
            "tech": "GAN",
            "method": method,
            "category": category
        })
    
    # Fake Diffusion kopyala
    diff_manifest = []
    for i, (img, method, tech, category) in enumerate(selected_diff):
        dest = out_diff / f"diff_{i:03d}_{method}_{img.name}"
        shutil.copy2(img, dest)
        diff_manifest.append({
            "index": i,
            "filename": dest.name,
            "source": str(img),
            "label": "fake",
            "tech": "Diffusion",
            "method": method,
            "category": category
        })
    
    # Manifest dosyasi olustur
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "description": "DF40 test ornekleri - egitimde kullanilmamis",
        "summary": {
            "total_real": len(real_manifest),
            "total_fake_gan": len(gan_manifest),
            "total_fake_diffusion": len(diff_manifest),
            "total": len(real_manifest) + len(gan_manifest) + len(diff_manifest),
            "gan_methods": sorted(set(x["method"] for x in gan_manifest)),
            "diffusion_methods": sorted(set(x["method"] for x in diff_manifest)),
        },
        "real": real_manifest,
        "fake_gan": gan_manifest,
        "fake_diffusion": diff_manifest,
    }
    
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    # Ozet yazdir
    print("\n" + "=" * 60)
    print("TAMAMLANDI!")
    print("=" * 60)
    print(f"\nCikti dizini: {OUTPUT_DIR}")
    print(f"\nOzet:")
    print(f"  +-- real/           -> {len(real_manifest)} gercek resim")
    print(f"  +-- fake_gan/       -> {len(gan_manifest)} GAN fake resim")
    if gan_manifest:
        print(f"  |   Metotlar: {', '.join(sorted(set(x['method'] for x in gan_manifest)))}")
    print(f"  +-- fake_diffusion/ -> {len(diff_manifest)} Diffusion fake resim")
    if diff_manifest:
        print(f"  |   Metotlar: {', '.join(sorted(set(x['method'] for x in diff_manifest)))}")
    print(f"  +-- manifest.json   -> Detayli kaynak bilgisi")
    print(f"\n  Toplam: {manifest['summary']['total']} resim")


if __name__ == "__main__":
    main()
