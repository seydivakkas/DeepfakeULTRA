"""
3-Katmanlı Leakage Kontrol Modülü
Katman 1: MD5 Hash (byte-level exact match)
Katman 2: pHash (perceptual hash, Hamming distance < 10)
Katman 3: FaceNet Embedding (cosine similarity > 0.85)

Identity-level separation: cosine similarity > 0.70 → aynı kişi

Kullanım:
    python scripts/leakage_checker.py --check dataset/faces_split/train dataset/jury_test
    python scripts/leakage_checker.py --build-index dataset/faces_split/train
"""
import sys
import hashlib
import csv
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# Opsiyonel bağımlılıklar
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("⚠️ imagehash yüklü değil. pHash devre dışı. → pip install imagehash")

try:
    from facenet_pytorch import InceptionResnetV1
    import torch
    HAS_FACENET = True
except ImportError:
    HAS_FACENET = False
    print("⚠️ facenet-pytorch yüklü değil. Embedding devre dışı. → pip install facenet-pytorch")

try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Leakage eşik değerleri
PHASH_THRESHOLD = 10        # Hamming distance < 10 → near-duplicate
EMBEDDING_THRESHOLD = 0.85  # Cosine similarity > 0.85 → perceptual duplicate
IDENTITY_THRESHOLD = 0.70   # Cosine similarity > 0.70 → aynı kişi


@dataclass
class LeakageResult:
    """Tek bir dosya için leakage kontrol sonucu."""
    file: str
    source: str = ""
    label: str = ""
    is_leaked: bool = False
    leakage_type: str = "clean"  # clean | md5 | phash | embedding | identity
    matched_file: str = ""
    similarity: float = 0.0
    details: str = ""


@dataclass
class ReferenceIndex:
    """Eğitim seti referans index'i."""
    md5_hashes: Set[str] = field(default_factory=set)
    phash_map: Dict[str, str] = field(default_factory=dict)  # hash_str → dosya adı
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)  # dosya adı → embedding
    file_count: int = 0


# ═══════════════════════════════════════════════════════════
# KATMAN 1: MD5 HASH
# ═══════════════════════════════════════════════════════════
def compute_md5(path: Path) -> str:
    """Dosyanın MD5 hash'ini hesapla."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════
# KATMAN 2: PERCEPTUAL HASH (pHash)
# ═══════════════════════════════════════════════════════════
def compute_phash(path: Path, hash_size: int = 16) -> Optional[str]:
    """pHash hesapla — görsel benzerlik için."""
    if not HAS_IMAGEHASH:
        return None
    try:
        img = Image.open(path).convert("RGB")
        h = imagehash.phash(img, hash_size=hash_size)
        return str(h)
    except Exception:
        return None


def phash_distance(hash1: str, hash2: str) -> int:
    """İki pHash arasındaki Hamming distance."""
    if not HAS_IMAGEHASH:
        return 999
    try:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2
    except Exception:
        return 999


# ═══════════════════════════════════════════════════════════
# KATMAN 3: FACENET EMBEDDING
# ═══════════════════════════════════════════════════════════
_facenet_model = None
_facenet_transform = None


def _get_facenet():
    """Singleton FaceNet model (lazy load)."""
    global _facenet_model, _facenet_transform
    if _facenet_model is None and HAS_FACENET and HAS_TORCHVISION:
        _facenet_model = InceptionResnetV1(pretrained="vggface2").eval()
        _facenet_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return _facenet_model, _facenet_transform


def compute_facenet_embedding(path: Path) -> Optional[np.ndarray]:
    """FaceNet 512-d embedding hesapla."""
    model, transform = _get_facenet()
    if model is None:
        return None
    try:
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = model(tensor).squeeze().numpy()
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    except Exception:
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """İki embedding arasındaki cosine benzerlik."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ═══════════════════════════════════════════════════════════
# REFERANS INDEX OLUŞTURMA
# ═══════════════════════════════════════════════════════════
def build_reference_index(
    directory: Path,
    use_phash: bool = True,
    use_embedding: bool = True,
    max_files: int = None,
) -> ReferenceIndex:
    """
    Eğitim seti için 3-katmanlı referans index oluştur.

    Args:
        directory: Taranacak kök dizin (recursive)
        use_phash: pHash hesapla
        use_embedding: FaceNet embedding hesapla (yavaş, GPU önerilir)
        max_files: Maksimum dosya sayısı (test amaçlı)
    """
    index = ReferenceIndex()
    files = []

    for ext in IMAGE_EXTS:
        files.extend(directory.rglob(f"*{ext}"))

    # .cache dizinlerini atla
    files = [f for f in files if ".cache" not in str(f)]

    if max_files:
        files = files[:max_files]

    total = len(files)
    print(f"  📂 {total} dosya indeksleniyor ({directory.name})...")

    for i, f in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"    [{i+1}/{total}] işleniyor...")

        # Katman 1: MD5
        try:
            md5 = compute_md5(f)
            index.md5_hashes.add(md5)
        except Exception:
            continue

        # Katman 2: pHash
        if use_phash and HAS_IMAGEHASH:
            ph = compute_phash(f)
            if ph:
                index.phash_map[ph] = f.name

        # Katman 3: FaceNet (opsiyonel, yavaş)
        if use_embedding and HAS_FACENET:
            emb = compute_facenet_embedding(f)
            if emb is not None:
                index.embeddings[f.name] = emb

        index.file_count += 1

    print(f"  ✅ Index oluşturuldu: MD5={len(index.md5_hashes)}, "
          f"pHash={len(index.phash_map)}, Embedding={len(index.embeddings)}")
    return index


def save_index(index: ReferenceIndex, path: Path):
    """Index'i diske kaydet (embedding'ler .npy olarak)."""
    path.mkdir(parents=True, exist_ok=True)

    # MD5
    with open(path / "md5_hashes.json", "w") as f:
        json.dump(list(index.md5_hashes), f)

    # pHash
    with open(path / "phash_map.json", "w") as f:
        json.dump(index.phash_map, f)

    # Embeddings
    if index.embeddings:
        names = list(index.embeddings.keys())
        vectors = np.stack([index.embeddings[n] for n in names])
        np.save(path / "embeddings.npy", vectors)
        with open(path / "embedding_names.json", "w") as f:
            json.dump(names, f)

    print(f"  💾 Index kaydedildi: {path}")


def load_index(path: Path) -> ReferenceIndex:
    """Diske kaydedilmiş index'i yükle."""
    index = ReferenceIndex()

    md5_path = path / "md5_hashes.json"
    if md5_path.exists():
        with open(md5_path) as f:
            index.md5_hashes = set(json.load(f))

    phash_path = path / "phash_map.json"
    if phash_path.exists():
        with open(phash_path) as f:
            index.phash_map = json.load(f)

    emb_path = path / "embeddings.npy"
    names_path = path / "embedding_names.json"
    if emb_path.exists() and names_path.exists():
        vectors = np.load(emb_path)
        with open(names_path) as f:
            names = json.load(f)
        for name, vec in zip(names, vectors):
            index.embeddings[name] = vec

    index.file_count = len(index.md5_hashes)
    print(f"  📂 Index yüklendi: MD5={len(index.md5_hashes)}, "
          f"pHash={len(index.phash_map)}, Embedding={len(index.embeddings)}")
    return index


# ═══════════════════════════════════════════════════════════
# LEAKAGE KONTROL
# ═══════════════════════════════════════════════════════════
def check_leakage(
    candidate_path: Path,
    ref_index: ReferenceIndex,
    check_identity: bool = False,
) -> LeakageResult:
    """
    Tek dosya için 3-katmanlı leakage kontrolü.

    Args:
        candidate_path: Kontrol edilecek dosya
        ref_index: Referans index (eğitim seti)
        check_identity: True ise identity-level kontrol (cosine > 0.70)
    """
    result = LeakageResult(file=candidate_path.name)

    # Katman 1: MD5
    try:
        md5 = compute_md5(candidate_path)
        if md5 in ref_index.md5_hashes:
            result.is_leaked = True
            result.leakage_type = "md5"
            result.similarity = 1.0
            result.details = "Byte-level exact match"
            return result
    except Exception:
        pass

    # Katman 2: pHash
    if HAS_IMAGEHASH and ref_index.phash_map:
        candidate_phash = compute_phash(candidate_path)
        if candidate_phash:
            for ref_hash, ref_file in ref_index.phash_map.items():
                dist = phash_distance(candidate_phash, ref_hash)
                if dist < PHASH_THRESHOLD:
                    result.is_leaked = True
                    result.leakage_type = "phash"
                    result.matched_file = ref_file
                    result.similarity = 1.0 - (dist / 64.0)
                    result.details = f"Hamming distance={dist}"
                    return result

    # Katman 3: FaceNet Embedding
    if HAS_FACENET and ref_index.embeddings:
        candidate_emb = compute_facenet_embedding(candidate_path)
        if candidate_emb is not None:
            threshold = IDENTITY_THRESHOLD if check_identity else EMBEDDING_THRESHOLD
            leak_type = "identity" if check_identity else "embedding"

            for ref_file, ref_emb in ref_index.embeddings.items():
                sim = cosine_similarity(candidate_emb, ref_emb)
                if sim > threshold:
                    result.is_leaked = True
                    result.leakage_type = leak_type
                    result.matched_file = ref_file
                    result.similarity = sim
                    result.details = f"Cosine similarity={sim:.4f}"
                    return result

    return result


def check_batch_leakage(
    candidate_dir: Path,
    ref_index: ReferenceIndex,
    check_identity: bool = False,
    label: str = "",
) -> List[LeakageResult]:
    """Bir dizindeki tüm dosyaları kontrol et."""
    results = []
    files = []
    for ext in IMAGE_EXTS:
        files.extend(candidate_dir.rglob(f"*{ext}"))
    files = [f for f in files if ".cache" not in str(f)]

    total = len(files)
    print(f"  🔍 {total} dosya kontrol ediliyor ({candidate_dir.name})...")

    for i, f in enumerate(files):
        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{total}]...")

        result = check_leakage(f, ref_index, check_identity=check_identity)
        result.label = label
        result.source = f.parent.name
        results.append(result)

    leaked = sum(1 for r in results if r.is_leaked)
    print(f"  {'❌' if leaked > 0 else '✅'} Sonuç: {leaked}/{total} leakage tespit edildi")
    return results


# ═══════════════════════════════════════════════════════════
# RAPOR OLUŞTURMA
# ═══════════════════════════════════════════════════════════
def generate_leakage_report(results: List[LeakageResult], output_path: Path):
    """Leakage raporunu CSV olarak kaydet."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "source", "label", "is_leaked",
            "leakage_type", "matched_file", "similarity", "details"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "file": r.file,
                "source": r.source,
                "label": r.label,
                "is_leaked": r.is_leaked,
                "leakage_type": r.leakage_type,
                "matched_file": r.matched_file,
                "similarity": f"{r.similarity:.4f}",
                "details": r.details,
            })

    leaked = sum(1 for r in results if r.is_leaked)
    by_type = {}
    for r in results:
        if r.is_leaked:
            by_type[r.leakage_type] = by_type.get(r.leakage_type, 0) + 1

    print(f"\n📋 Leakage Raporu: {output_path}")
    print(f"  Toplam kontrol: {len(results)}")
    print(f"  Leakage: {leaked}")
    for lt, count in sorted(by_type.items()):
        print(f"    {lt}: {count}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="3-Katmanlı Leakage Kontrol")
    parser.add_argument("--build-index", type=str, help="Index oluştur (eğitim dizini)")
    parser.add_argument("--check", nargs=2, metavar=("REF_DIR", "CANDIDATE_DIR"),
                        help="Leakage kontrolü (referans, aday)")
    parser.add_argument("--identity", action="store_true",
                        help="Identity-level kontrol (cosine > 0.70)")
    parser.add_argument("--no-embedding", action="store_true",
                        help="FaceNet embedding atla (hızlı mod)")
    parser.add_argument("--report", type=str, default="leakage_report.csv",
                        help="Rapor dosya adı")

    args = parser.parse_args()
    base = Path(__file__).parent.parent / "dataset"
    index_dir = base / ".leakage_index"

    if args.build_index:
        ref_dir = Path(args.build_index)
        if not ref_dir.is_absolute():
            ref_dir = base / args.build_index
        index = build_reference_index(
            ref_dir,
            use_embedding=not args.no_embedding,
        )
        save_index(index, index_dir)
        return

    if args.check:
        ref_path, candidate_path = args.check
        ref_dir = Path(ref_path)
        cand_dir = Path(candidate_path)
        if not ref_dir.is_absolute():
            ref_dir = base / ref_path
        if not cand_dir.is_absolute():
            cand_dir = base / candidate_path

        # Kayıtlı index var mı?
        if index_dir.exists() and (index_dir / "md5_hashes.json").exists():
            print("📂 Kayıtlı index yükleniyor...")
            index = load_index(index_dir)
        else:
            print("📂 Index oluşturuluyor...")
            index = build_reference_index(ref_dir, use_embedding=not args.no_embedding)
            save_index(index, index_dir)

        # Kontrol
        results = []
        for label in ["real", "fake"]:
            label_dir = cand_dir / label
            if label_dir.exists():
                r = check_batch_leakage(label_dir, index,
                                        check_identity=args.identity,
                                        label=label)
                results.extend(r)

        # Rapor
        report_path = base / args.report
        generate_leakage_report(results, report_path)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
