"""
Jüri demo klasörünü ünlü yüzleri ile oluşturur.
Kaynaklar:
  - CelebDF v2 (ünlü deepfake/real çiftleri)
  - Deepfake20K / DeepFakeFace / DFDC / FaceForensics++ (çeşitlilik)
Yüz doğrulaması yapılır.
"""
import os
import cv2
import shutil
import random

random.seed(42)

# Haarcascade
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    cv2_data = os.path.join(os.path.dirname(cv2.__file__), "data")
    shutil.copy2(os.path.join(cv2_data, "haarcascade_frontalface_default.xml"), CASCADE_PATH)

cascade = cv2.CascadeClassifier(CASCADE_PATH)

def has_face(fpath):
    """Görselde yüz var mı kontrol eder."""
    img = cv2.imread(fpath)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    return len(faces) > 0

# Hedef klasör
DST = "dataset/jury_demo_100"

# Temizle ve yeniden oluştur
if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(os.path.join(DST, "real"), exist_ok=True)
os.makedirs(os.path.join(DST, "fake"), exist_ok=True)

# Kaynak datasetler ve ağırlıkları (CelebDF ağırlıklı - ünlü yüzler)
EXT = "dataset/external_tests"
sources = {
    "celeb_df_v2": 50,      # 50 ünlü yüz - ana kaynak
    "deepfake20k": 15,      # 15 çeşitli
    "deepfakeface": 15,     # 15 çeşitli
    "dfdc": 10,             # 10 DFDC
    "faceforensics": 10,    # 10 FF++
}

total_real = 0
total_fake = 0

for dataset, count in sources.items():
    real_dir = os.path.join(EXT, dataset, "real")
    fake_dir = os.path.join(EXT, dataset, "fake")

    # Real görselleri topla (yüzlü olanlar)
    real_files = [f for f in os.listdir(real_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(real_files)

    added_real = 0
    for fname in real_files:
        if added_real >= count:
            break
        fpath = os.path.join(real_dir, fname)
        if has_face(fpath):
            dst_name = f"{dataset}_real_{fname}"
            shutil.copy2(fpath, os.path.join(DST, "real", dst_name))
            added_real += 1

    # Fake görselleri topla (yüzlü olanlar)
    fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(fake_files)

    added_fake = 0
    for fname in fake_files:
        if added_fake >= count:
            break
        fpath = os.path.join(fake_dir, fname)
        if has_face(fpath):
            dst_name = f"{dataset}_fake_{fname}"
            shutil.copy2(fpath, os.path.join(DST, "fake", dst_name))
            added_fake += 1

    print(f"  {dataset}: {added_real} real + {added_fake} fake eklendi")
    total_real += added_real
    total_fake += added_fake

print(f"\nFinal: {total_real} real + {total_fake} fake = {total_real + total_fake} toplam")
print(f"Konum: {os.path.abspath(DST)}")
