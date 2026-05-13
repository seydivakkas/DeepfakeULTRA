"""
SBI (Self-Blended Images) fiziksel veri üretici.
REAL görselleri alıp SBI augmentation ile sahte FAKE görselleri oluşturur.
Bu görseller eğitim setine fiziksel olarak eklenir.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from config import paths
from core.sbi_augmentation import SBITransform

def generate_sbi_fakes(
    source_dir: str,
    output_dir: str,
    count: int = 5000,
    seed: int = 42,
):
    """
    REAL görselden SBI ile FAKE görseller üret.
    
    Args:
        source_dir: REAL görsellerin bulunduğu dizin
        output_dir: Üretilen SBI görsellerin kaydedileceği dizin
        count: Üretilecek görsel sayısı
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Kaynak REAL görselleri bul
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    real_files = [f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in extensions]
    
    if not real_files:
        print(f"❌ Kaynak dizinde görsel bulunamadı: {src}")
        return 0
    
    print(f"📂 Kaynak: {len(real_files)} REAL görsel")
    print(f"📂 Hedef:  {out}")
    print(f"🎯 Üretilecek: {count} SBI fake")
    
    sbi = SBITransform()
    generated = 0
    failed = 0
    
    # Görselleri karıştır ve döngüsel kullan
    random.shuffle(real_files)
    
    for i in tqdm(range(count), desc="SBI üretiliyor"):
        img_path = real_files[i % len(real_files)]
        try:
            img = Image.open(str(img_path)).convert("RGB")
            img = img.resize((224, 224))
            
            # SBI uygula
            blended = sbi(img)
            
            if blended is not None and isinstance(blended, Image.Image):
                save_path = out / f"sbi_gen_{generated:05d}.jpg"
                blended.save(str(save_path), quality=92)
                generated += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            continue
    
    print(f"\n✅ Üretilen: {generated}")
    print(f"❌ Başarısız: {failed}")
    return generated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000, help="Üretilecek SBI fake sayısı")
    parser.add_argument("--source", type=str, default=None, help="REAL kaynak dizin")
    parser.add_argument("--output", type=str, default=None, help="Çıktı dizini")
    args = parser.parse_args()
    
    # Varsayılan kaynaklar
    source = args.source or str(Path("dataset/faces_split/train/REAL"))
    output = args.output or str(Path("dataset/faces_split/train/FAKE/sbi_generated"))
    
    generated = generate_sbi_fakes(source, output, count=args.count)
    
    if generated > 0:
        print(f"\n📊 Eğitim setine {generated} SBI fake eklendi!")
        print(f"   Konum: {output}")
        print(f"   Sonraki adım: Eğitimi yeniden başlat (Run 6)")
