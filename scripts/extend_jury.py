"""
GÖREV 2: Jury Setinin Genişletilmesi (800 → 3000)
Eğitimde kullanılmayan kaynaklardan leakage-free, kaynak dengeli jury seti.

Kullanım:
    python scripts/extend_jury.py
    python scripts/extend_jury.py --check-leakage
"""
import os
import sys
import hashlib
import random
import csv
import shutil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

# 3-Katmanlı Leakage Kontrolü (G1)
try:
    from scripts.leakage_checker import (
        build_reference_index, check_leakage as check_leakage_3layer,
        generate_leakage_report, ReferenceIndex,
    )
    HAS_LEAKAGE_CHECKER = True
except ImportError:
    HAS_LEAKAGE_CHECKER = False
    print("⚠️ leakage_checker modülü yüklenemedi, MD5-only mod aktif")

BASE = Path(__file__).parent.parent / "dataset"
FACES = BASE / "faces"
SPLIT = BASE / "faces_split"
JURY = BASE / "jury_test"
JURY_BACKUP = BASE / "jury_test_backup_v1"

# Hedef dağılım (G2: 4000-5000 ideal)
TARGET_TOTAL = 5000
TARGET_REAL = 2500
TARGET_FAKE = 2500

# FAKE stratified sampling kategorileri (G2)
FAKE_CATEGORIES = {
    "gan": ["df40"],               # ProGAN/StyleGAN
    "diffusion": ["genimage"],      # Stable Diffusion, DALL-E
    "faceswap": ["ffpp"],           # DeepFaceLab, FaceSwap
    "audio_driven": ["custom_team"], # Audio-driven face reenactment
    "hybrid": ["sidset"],           # Tampered/composite
}


def file_hash(path: Path) -> str:
    """MD5 hash hesapla."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_training_hashes_or_index():
    """
    3-katmanlı index (varsa) veya MD5 hash seti döndür.
    Returns: (ref_index: ReferenceIndex | None, train_hashes: set)
    """
    if HAS_LEAKAGE_CHECKER:
        print("🔐 3-Katmanlı Eğitim Seti Index'i oluşturuluyor...")
        index_dir = BASE / ".leakage_index"
        if index_dir.exists() and (index_dir / "md5_hashes.json").exists():
            from scripts.leakage_checker import load_index
            ref_index = load_index(index_dir)
            return ref_index, ref_index.md5_hashes
        else:
            ref_index = build_reference_index(
                SPLIT / "train", use_phash=True, use_embedding=False
            )
            return ref_index, ref_index.md5_hashes
    else:
        return None, collect_training_hashes()


def collect_training_hashes() -> set:
    """Eğitim split'indeki tüm dosyaların hash'lerini topla (leakage engeli)."""
    print("🔐 Eğitim seti hash'leri hesaplanıyor...")
    train_hashes = set()

    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            split_dir = SPLIT / split / label
            if not split_dir.exists():
                continue
            files = list(split_dir.glob("*.*"))
            count = 0
            for f in files:
                try:
                    train_hashes.add(file_hash(f))
                    count += 1
                except Exception:
                    pass
            print(f"  {split}/{label}: {count} hash")

    print(f"  📋 Toplam eğitim hash: {len(train_hashes)}")
    return train_hashes


def collect_source_files(label: str, sources: list) -> list:
    """Belirli kaynaklardan dosya topla."""
    files = []
    for src_name in sources:
        src_dir = FACES / src_name / label
        if not src_dir.exists():
            continue
        found = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        for f in found:
            files.append({"path": f, "source": src_name, "label": label})
    random.shuffle(files)
    return files


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Jury seti genişletme")
    parser.add_argument("--check-leakage", action="store_true")
    args = parser.parse_args()

    if args.check_leakage:
        check_leakage()
        return

    print("=" * 60)
    print("GÖREV 2: Jury Setinin Genişletilmesi (800 → 5000)")
    print("=" * 60)

    # 1. 3-katmanlı eğitim index'i oluştur
    ref_index, train_hashes = collect_training_hashes_or_index()

    # 2. Mevcut jury'yi yedekle
    if not JURY_BACKUP.exists() and JURY.exists():
        print(f"\n💾 Mevcut jury yedekleniyor → {JURY_BACKUP}")
        shutil.copytree(JURY, JURY_BACKUP)

    # 3. Mevcut jury hash'lerini al (bunları koruyacağız)
    existing_jury_hashes = set()
    existing_count = {"real": 0, "fake": 0}
    for label in ["real", "fake"]:
        jury_label_dir = JURY / label
        if jury_label_dir.exists():
            for f in jury_label_dir.glob("*.*"):
                existing_jury_hashes.add(file_hash(f))
                existing_count[label] += 1
    print(f"\n📂 Mevcut jury: REAL={existing_count['real']}, FAKE={existing_count['fake']}")

    # 4. Ek REAL görseller topla
    needed_real = TARGET_REAL - existing_count["real"]
    needed_fake = TARGET_FAKE - existing_count["fake"]

    leakage_report = []
    added = {"real": 0, "fake": 0}

    # REAL kaynaklar (eğitimde olmayan örnekler)
    if needed_real > 0:
        print(f"\n🟢 {needed_real} ek REAL görsel ekleniyor...")
        real_sources = collect_source_files("real", ["vggface2", "ffhq_256", "celeba_hq", "utkface"])

        # Hard-real ekle (varsa)
        hard_real_dir = FACES / "hard_real"
        if hard_real_dir.exists():
            for cat_dir in hard_real_dir.iterdir():
                if cat_dir.is_dir():
                    for f in cat_dir.glob("*.jpg"):
                        real_sources.append({"path": f, "source": f"hard_real/{cat_dir.name}", "label": "real"})

        random.shuffle(real_sources)

        # Kaynak dengeli dağılım hedefi
        source_quotas = defaultdict(int)
        per_source = max(50, needed_real // max(1, len(set(s["source"] for s in real_sources))))

        for item in real_sources:
            if added["real"] >= needed_real:
                break

            h = file_hash(item["path"])

            # 3-katmanlı leakage kontrolü
            if HAS_LEAKAGE_CHECKER and ref_index is not None:
                result = check_leakage_3layer(item["path"], ref_index, check_identity=True)
                if result.is_leaked:
                    leakage_report.append({
                        "file": str(item["path"].name),
                        "source": item["source"],
                        "label": "real",
                        "status": f"BLOCKED ({result.leakage_type})",
                        "leakage_type": result.leakage_type,
                    })
                    continue
            elif h in train_hashes:
                leakage_report.append({
                    "file": str(item["path"].name),
                    "source": item["source"],
                    "label": "real",
                    "status": "BLOCKED (leakage)",
                })
                continue

            if h in existing_jury_hashes:
                continue

            # Kaynak dengesi
            if source_quotas[item["source"]] >= per_source * 2:
                continue

            # Kopyala
            dst = JURY / "real" / f"jury_ext_real_{added['real']:04d}_{item['source'].replace('/', '_')}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(item["path"], dst)
                existing_jury_hashes.add(h)
                source_quotas[item["source"]] += 1
                added["real"] += 1
            except Exception as e:
                leakage_report.append({
                    "file": str(item["path"].name),
                    "source": item["source"],
                    "label": "real",
                    "status": f"ERROR: {e}",
                })

        print(f"  ✅ {added['real']} REAL eklendi")

    # FAKE kaynaklar
    if needed_fake > 0:
        print(f"\n🔴 {needed_fake} ek FAKE görsel ekleniyor (stratified)...")

        # Stratified sampling: her kategoriden eşit sayıda (G2)
        per_category = max(50, needed_fake // len(FAKE_CATEGORIES))
        print(f"  🎯 Kategori başına hedef: {per_category}")

        for cat_name, cat_sources in FAKE_CATEGORIES.items():
            cat_files = collect_source_files("fake", cat_sources)
            random.shuffle(cat_files)
            cat_added = 0

            for item in cat_files:
                if added["fake"] >= needed_fake:
                    break
                if cat_added >= per_category:
                    break

                h = file_hash(item["path"])

                # 3-katmanlı leakage kontrolü
                if HAS_LEAKAGE_CHECKER and ref_index is not None:
                    result = check_leakage_3layer(item["path"], ref_index)
                    if result.is_leaked:
                        leakage_report.append({
                            "file": str(item["path"].name),
                            "source": item["source"],
                            "label": "fake",
                            "status": f"BLOCKED ({result.leakage_type})",
                            "leakage_type": result.leakage_type,
                        })
                        continue
                elif h in train_hashes:
                    leakage_report.append({
                        "file": str(item["path"].name),
                        "source": item["source"],
                        "label": "fake",
                        "status": "BLOCKED (md5)",
                        "leakage_type": "md5",
                    })
                    continue

                if h in existing_jury_hashes:
                    continue

                dst = JURY / "fake" / f"jury_ext_fake_{added['fake']:04d}_{cat_name}_{item['source']}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(item["path"], dst)
                    existing_jury_hashes.add(h)
                    added["fake"] += 1
                    cat_added += 1
                except Exception as e:
                    leakage_report.append({
                        "file": str(item["path"].name),
                        "source": item["source"],
                        "label": "fake",
                        "status": f"ERROR: {e}",
                        "leakage_type": "error",
                    })

            print(f"    {cat_name}: {cat_added} eklendi")

        print(f"  ✅ {added['fake']} FAKE eklendi (stratified)")

    # 5. Leakage raporu kaydet (leakage_type sütunu dahil)
    report_path = BASE / "jury_leakage_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "source", "label", "status", "leakage_type"])
        writer.writeheader()
        writer.writerows(leakage_report)

    # 6. Final rapor
    final_real = len(list((JURY / "real").glob("*.*"))) if (JURY / "real").exists() else 0
    final_fake = len(list((JURY / "fake").glob("*.*"))) if (JURY / "fake").exists() else 0

    print(f"\n{'=' * 60}")
    print(f"📊 SONUÇ:")
    print(f"  REAL: {existing_count['real']} → {final_real} (+{added['real']})")
    print(f"  FAKE: {existing_count['fake']} → {final_fake} (+{added['fake']})")
    print(f"  Toplam: {final_real + final_fake}")
    print(f"  Leakage engellenen: {sum(1 for r in leakage_report if 'BLOCKED' in r['status'])}")
    print(f"  Rapor: {report_path}")
    print(f"{'=' * 60}")
    print("✅ GÖREV_2_TAMAMLANDI")


def check_leakage():
    """Mevcut jury setinde leakage kontrolü."""
    print("🔐 Jury leakage kontrolü...")

    train_hashes = collect_training_hashes()

    leaked = 0
    total = 0
    for label in ["real", "fake"]:
        jury_dir = JURY / label
        if not jury_dir.exists():
            continue
        for f in jury_dir.glob("*.*"):
            total += 1
            h = file_hash(f)
            if h in train_hashes:
                leaked += 1
                print(f"  ⚠️ LEAKAGE: {label}/{f.name}")

    if leaked == 0:
        print(f"\n✅ Leakage yok ({total} görsel kontrol edildi)")
    else:
        print(f"\n❌ {leaked}/{total} görsel eğitim setiyle çakışıyor!")


if __name__ == "__main__":
    main()
