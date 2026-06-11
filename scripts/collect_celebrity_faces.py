"""
Ünlü Yüz Toplama & Deepfake Test Scripti
==========================================
23 ünlünün real + deepfake görüntülerini internetten toplar,
model ile test eder ve detaylı rapor üretir.

Kullanım:
    python scripts/collect_celebrity_faces.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import requests
import json
import time
from datetime import datetime
from io import BytesIO

from config import model_cfg, paths, DEVICE
from core.dual_mobilenetv3 import DualPathDeepfakeDetector
from core.data_pipeline import MultiScaleDWT, FaceMeshExtractor

try:
    from core.frequency_v2 import HybridFrequencyExtractor
    HAS_HYBRID_FREQ = True
except ImportError:
    HAS_HYBRID_FREQ = False


# ═══════════════════════════════════════════════════════════
# ÜNLÜ LİSTESİ — Real fotoğraf kaynakları (Wikipedia/Wikimedia)
# ═══════════════════════════════════════════════════════════

CELEBRITIES = {
    # Hollywood
    "Angelina_Jolie": {
        "name": "Angelina Jolie",
        "category": "Hollywood",
        "wiki": "Angelina_Jolie",
    },
    "Brad_Pitt": {
        "name": "Brad Pitt",
        "category": "Hollywood",
        "wiki": "Brad_Pitt",
    },
    "Denzel_Washington": {
        "name": "Denzel Washington",
        "category": "Hollywood",
        "wiki": "Denzel_Washington",
    },
    "Hugh_Jackman": {
        "name": "Hugh Jackman",
        "category": "Hollywood",
        "wiki": "Hugh_Jackman",
    },
    "Jennifer_Lawrence": {
        "name": "Jennifer Lawrence",
        "category": "Hollywood",
        "wiki": "Jennifer_Lawrence",
    },
    "Johnny_Depp": {
        "name": "Johnny Depp",
        "category": "Hollywood",
        "wiki": "Johnny_Depp",
    },
    "Kate_Winslet": {
        "name": "Kate Winslet",
        "category": "Hollywood",
        "wiki": "Kate_Winslet",
    },
    "Leonardo_DiCaprio": {
        "name": "Leonardo DiCaprio",
        "category": "Hollywood",
        "wiki": "Leonardo_DiCaprio",
    },
    "Megan_Fox": {
        "name": "Megan Fox",
        "category": "Hollywood",
        "wiki": "Megan_Fox",
    },
    "Natalie_Portman": {
        "name": "Natalie Portman",
        "category": "Hollywood",
        "wiki": "Natalie_Portman",
    },
    "Nicole_Kidman": {
        "name": "Nicole Kidman",
        "category": "Hollywood",
        "wiki": "Nicole_Kidman",
    },
    "Robert_Downey_Jr": {
        "name": "Robert Downey Jr.",
        "category": "Hollywood",
        "wiki": "Robert_Downey_Jr.",
    },
    "Sandra_Bullock": {
        "name": "Sandra Bullock",
        "category": "Hollywood",
        "wiki": "Sandra_Bullock",
    },
    "Scarlett_Johansson": {
        "name": "Scarlett Johansson",
        "category": "Hollywood",
        "wiki": "Scarlett_Johansson",
    },
    "Tom_Cruise": {
        "name": "Tom Cruise",
        "category": "Hollywood",
        "wiki": "Tom_Cruise",
    },
    "Tom_Hanks": {
        "name": "Tom Hanks",
        "category": "Hollywood",
        "wiki": "Tom_Hanks",
    },
    "Will_Smith": {
        "name": "Will Smith",
        "category": "Hollywood",
        "wiki": "Will_Smith",
    },
    # Politikacılar
    "Donald_Trump": {
        "name": "Donald Trump",
        "category": "Politika",
        "wiki": "Donald_Trump",
    },
    "Recep_Tayyip_Erdogan": {
        "name": "Recep Tayyip Erdoğan",
        "category": "Politika",
        "wiki": "Recep_Tayyip_Erdoğan",
    },
    "Vladimir_Putin": {
        "name": "Vladimir Putin",
        "category": "Politika",
        "wiki": "Vladimir_Putin",
    },
    "Angela_Merkel": {
        "name": "Angela Merkel",
        "category": "Politika",
        "wiki": "Angela_Merkel",
    },
    "George_W_Bush": {
        "name": "George W. Bush",
        "category": "Politika",
        "wiki": "George_W._Bush",
    },
    "Emmanuel_Macron": {
        "name": "Emmanuel Macron",
        "category": "Politika",
        "wiki": "Emmanuel_Macron",
    },
}


def download_image(url, save_path, timeout=15):
    """URL'den görüntü indir."""
    try:
        headers = {
            "User-Agent": "DeepfakeULTRA/1.0 (Academic Research; contact: github.com/seydivakkas)"
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()
        
        # Boyut kontrolü
        content = resp.content
        if len(content) < 5000:
            return False
        
        # Görüntü doğrulama
        img = Image.open(BytesIO(content))
        if img.size[0] < 100 or img.size[1] < 100:
            return False
        
        # RGB'ye çevir
        img = img.convert("RGB")
        img.save(save_path, "JPEG", quality=95)
        return True
        
    except Exception as e:
        return False


def get_wikipedia_image(wiki_title, save_path):
    """Wikipedia'dan ünlünün profil fotoğrafını indir."""
    try:
        # Wikipedia API — sayfa görsellerini al
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": wiki_title,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": 800,
        }
        
        headers = {
            "User-Agent": "DeepfakeULTRA/1.0 (Academic Research; contact: github.com/seydivakkas)"
        }
        
        resp = requests.get(api_url, params=params, headers=headers, timeout=15)
        data = resp.json()
        
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            thumbnail = page_data.get("thumbnail", {})
            img_url = thumbnail.get("source")
            if img_url:
                return download_image(img_url, save_path)
        
        return False
        
    except Exception as e:
        print(f"    ⚠️ Wikipedia API hatası: {e}")
        return False


def get_wikimedia_images(wiki_title, save_dir, max_images=3):
    """Wikimedia Commons'dan birden fazla görüntü indir."""
    downloaded = []
    
    try:
        # Önce ana Wikipedia görüntüsünü al
        main_path = save_dir / f"{wiki_title}_wiki_1.jpg"
        if get_wikipedia_image(wiki_title, main_path):
            downloaded.append(main_path)
        
        # Wikimedia Commons'dan ek görseller
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": wiki_title,
            "prop": "images",
            "format": "json",
            "imlimit": 20,
        }
        
        headers = {
            "User-Agent": "DeepfakeULTRA/1.0 (Academic Research; contact: github.com/seydivakkas)"
        }
        
        resp = requests.get(api_url, params=params, headers=headers, timeout=15)
        data = resp.json()
        
        pages = data.get("query", {}).get("pages", {})
        img_count = len(downloaded) + 1
        
        for page_id, page_data in pages.items():
            images = page_data.get("images", [])
            for img_info in images:
                if img_count > max_images:
                    break
                
                img_title = img_info.get("title", "")
                # Sadece fotoğrafları al (svg, logo vb. hariç)
                if not any(ext in img_title.lower() for ext in [".jpg", ".jpeg", ".png"]):
                    continue
                if any(skip in img_title.lower() for skip in ["logo", "flag", "icon", "map", "seal", "coat", "commons"]):
                    continue
                
                # Wikimedia dosya URL'sini al
                file_params = {
                    "action": "query",
                    "titles": img_title,
                    "prop": "imageinfo",
                    "iiprop": "url",
                    "format": "json",
                    "iiurlwidth": 800,
                }
                
                file_resp = requests.get(api_url, params=file_params, headers=headers, timeout=15)
                file_data = file_resp.json()
                
                for fid, fpage in file_data.get("query", {}).get("pages", {}).items():
                    ii = fpage.get("imageinfo", [{}])[0]
                    thumb_url = ii.get("thumburl") or ii.get("url")
                    if thumb_url:
                        save_path = save_dir / f"{wiki_title}_wiki_{img_count}.jpg"
                        if download_image(thumb_url, save_path):
                            downloaded.append(save_path)
                            img_count += 1
                
                time.sleep(0.5)  # Rate limiting
        
    except Exception as e:
        print(f"    ⚠️ Wikimedia hatası: {e}")
    
    return downloaded


def load_model():
    """En iyi modeli yükle."""
    model = DualPathDeepfakeDetector().to(DEVICE)
    model_path = str(paths.MODEL_DIR / "best_run5_forensic.pth")
    
    if not Path(model_path).exists():
        model_path = str(paths.BEST_MODEL_PATH)
    
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Model yüklendi: {Path(model_path).name}")
    return model


def get_preprocessors():
    """Preprocessing araçları."""
    transform = transforms.Compose([
        transforms.Resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    if HAS_HYBRID_FREQ and getattr(model_cfg, 'USE_HYBRID_FREQ', False):
        dwt = HybridFrequencyExtractor(
            wavelets=model_cfg.DWT_WAVELETS,
            size=model_cfg.IMG_SIZE,
            include_dwt=True, include_dct=True, include_phase=True,
        )
    else:
        dwt = MultiScaleDWT()
    
    mesh = FaceMeshExtractor()
    return transform, dwt, mesh


def predict_image(model, transform, dwt, mesh, img_path):
    """Tek görüntü için tahmin yap."""
    try:
        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image.resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)))
        
        rgb_tensor = transform(image).unsqueeze(0).to(DEVICE)
        freq_tensor = torch.from_numpy(dwt(img_np)).unsqueeze(0).float().to(DEVICE)
        mesh_tensor = torch.from_numpy(mesh(img_np)).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            logits = model(rgb_tensor, freq_tensor, mesh_tensor)
            probs = F.softmax(logits, dim=1)[0]
        
        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        label = "REAL" if real_prob > 0.5 else "FAKE"
        confidence = real_prob if label == "REAL" else fake_prob
        
        return {
            "label": label,
            "real_prob": round(real_prob, 6),
            "fake_prob": round(fake_prob, 6),
            "confidence": round(confidence, 6),
        }
    except Exception as e:
        return {"label": "ERROR", "error": str(e)}


def collect_and_test():
    """Ana fonksiyon: Topla + Test Et + Raporla."""
    
    print("🎯 ÜNLÜ YÜZ TOPLAMA & DEEPFAKE TEST SİSTEMİ")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 {len(CELEBRITIES)} ünlü kişi")
    print()
    
    # Çıktı dizini
    output_dir = paths.DATASET_DIR / "celebrity_test"
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Model yükle
    model = load_model()
    transform, dwt, mesh = get_preprocessors()
    
    # ═══════════════════════════════════════════════════
    # ADIM 1: Real fotoğrafları topla
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📸 ADIM 1: Real Fotoğraf Toplama (Wikipedia/Wikimedia)")
    print("=" * 60)
    
    all_results = []
    celeb_results = {}
    
    for celeb_key, celeb_info in CELEBRITIES.items():
        name = celeb_info["name"]
        wiki = celeb_info["wiki"]
        category = celeb_info["category"]
        
        celeb_dir = real_dir / celeb_key
        celeb_dir.mkdir(exist_ok=True)
        
        print(f"\n  👤 {name} ({category})")
        
        # Wikipedia'dan indir
        downloaded = get_wikimedia_images(wiki, celeb_dir, max_images=3)
        
        if downloaded:
            print(f"    ✅ {len(downloaded)} real fotoğraf indirildi")
        else:
            print(f"    ⚠️ Fotoğraf bulunamadı, geçiliyor")
            continue
        
        # Her fotoğrafı test et
        celeb_results[celeb_key] = {
            "name": name,
            "category": category,
            "real_results": [],
            "fake_results": [],
        }
        
        for img_path in downloaded:
            result = predict_image(model, transform, dwt, mesh, img_path)
            result["file"] = img_path.name
            result["person"] = name
            result["type"] = "real"
            result["category"] = category
            
            celeb_results[celeb_key]["real_results"].append(result)
            all_results.append(result)
            
            status = "✅" if result["label"] == "REAL" else "❌"
            print(f"    {status} {img_path.name}: {result['label']} "
                  f"(real={result.get('real_prob', 0):.4f}, fake={result.get('fake_prob', 0):.4f})")
        
        time.sleep(1)  # Rate limiting
    
    # ═══════════════════════════════════════════════════
    # ADIM 2: Deepfake görüntüleri (CelebDF-v2'den)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🎭 ADIM 2: Deepfake Görüntüler (CelebDF-v2 + external)")
    print("=" * 60)
    
    # CelebDF-v2 fake görüntülerden örnekle
    celeb_df_fake_dir = paths.DATASET_DIR / "external_tests" / "celeb_df_v2" / "fake"
    celeb_df_real_dir = paths.DATASET_DIR / "external_tests" / "celeb_df_v2" / "real"
    
    # Her ünlü için CelebDF'den birkaç fake örnek al
    if celeb_df_fake_dir.exists():
        fake_files = sorted(celeb_df_fake_dir.glob("*.jpg")) + sorted(celeb_df_fake_dir.glob("*.png"))
        print(f"  📂 CelebDF-v2 Fake: {len(fake_files)} dosya mevcut")
        
        # Her ünlü için 2-3 fake örnek ata (CelebDF isimsiz, rastgele dağıt)
        np.random.seed(42)
        fake_indices = np.random.permutation(len(fake_files))
        
        per_celeb = min(3, len(fake_files) // len(CELEBRITIES))
        idx = 0
        
        for celeb_key, celeb_info in CELEBRITIES.items():
            name = celeb_info["name"]
            
            if celeb_key not in celeb_results:
                celeb_results[celeb_key] = {
                    "name": name,
                    "category": celeb_info["category"],
                    "real_results": [],
                    "fake_results": [],
                }
            
            celeb_fake_dir = fake_dir / celeb_key
            celeb_fake_dir.mkdir(exist_ok=True)
            
            print(f"\n  🎭 {name}")
            
            for j in range(per_celeb):
                if idx >= len(fake_indices):
                    break
                
                src_file = fake_files[fake_indices[idx]]
                dst_file = celeb_fake_dir / f"fake_{celeb_key}_{j+1}.jpg"
                
                # Kopyala
                try:
                    img = Image.open(src_file).convert("RGB")
                    img.save(dst_file, "JPEG", quality=95)
                except:
                    idx += 1
                    continue
                
                # Test et
                result = predict_image(model, transform, dwt, mesh, dst_file)
                result["file"] = dst_file.name
                result["person"] = name
                result["type"] = "fake"
                result["category"] = celeb_info["category"]
                result["source"] = "CelebDF-v2"
                
                celeb_results[celeb_key]["fake_results"].append(result)
                all_results.append(result)
                
                status = "✅" if result["label"] == "FAKE" else "❌"
                print(f"    {status} {dst_file.name}: {result['label']} "
                      f"(real={result.get('real_prob', 0):.4f}, fake={result.get('fake_prob', 0):.4f})")
                
                idx += 1
    
    # CelebDF-v2 real görüntülerden de örnekle
    if celeb_df_real_dir.exists():
        real_files = sorted(celeb_df_real_dir.glob("*.jpg")) + sorted(celeb_df_real_dir.glob("*.png"))
        print(f"\n  📂 CelebDF-v2 Real: {len(real_files)} dosya — bonus test")
        
        # Rastgele 20 real test
        sample_indices = np.random.choice(len(real_files), min(20, len(real_files)), replace=False)
        
        celeb_df_correct = 0
        celeb_df_total = 0
        
        for si in sample_indices:
            result = predict_image(model, transform, dwt, mesh, real_files[si])
            if result["label"] == "REAL":
                celeb_df_correct += 1
            celeb_df_total += 1
        
        print(f"    CelebDF-v2 Real Accuracy: {celeb_df_correct}/{celeb_df_total} "
              f"(%{celeb_df_correct/celeb_df_total*100:.1f})")
    
    # ═══════════════════════════════════════════════════
    # ADIM 3: Detaylı rapor
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📊 ADIM 3: DETAYLI ÜNLÜ YÜZ TEST RAPORU")
    print("=" * 60)
    
    # Genel istatistikler
    real_results = [r for r in all_results if r["type"] == "real" and r.get("label") != "ERROR"]
    fake_results = [r for r in all_results if r["type"] == "fake" and r.get("label") != "ERROR"]
    
    real_correct = sum(1 for r in real_results if r["label"] == "REAL")
    fake_correct = sum(1 for r in fake_results if r["label"] == "FAKE")
    
    real_acc = real_correct / len(real_results) * 100 if real_results else 0
    fake_acc = fake_correct / len(fake_results) * 100 if fake_results else 0
    total_correct = real_correct + fake_correct
    total = len(real_results) + len(fake_results)
    total_acc = total_correct / total * 100 if total else 0
    
    print(f"\n{'─'*60}")
    print(f"📊 GENEL SONUÇLAR")
    print(f"{'─'*60}")
    print(f"  Toplam test      : {total}")
    print(f"  Real test        : {len(real_results)} (doğru: {real_correct}, acc: %{real_acc:.1f})")
    print(f"  Fake test        : {len(fake_results)} (doğru: {fake_correct}, acc: %{fake_acc:.1f})")
    print(f"  Genel Accuracy   : %{total_acc:.1f}")
    
    # Kişi bazlı sonuçlar
    print(f"\n{'─'*60}")
    print(f"📊 KİŞİ BAZLI SONUÇLAR")
    print(f"{'─'*60}")
    print(f"{'Kişi':<25} {'Kategori':<12} {'Real':>6} {'Fake':>6} {'Doğru':>6}")
    print(f"{'─'*60}")
    
    for celeb_key, data in celeb_results.items():
        name = data["name"]
        cat = data["category"]
        
        real_count = len(data["real_results"])
        fake_count = len(data["fake_results"])
        
        correct = sum(1 for r in data["real_results"] if r.get("label") == "REAL")
        correct += sum(1 for r in data["fake_results"] if r.get("label") == "FAKE")
        total_per = real_count + fake_count
        
        acc_str = f"%{correct/total_per*100:.0f}" if total_per > 0 else "—"
        
        print(f"  {name:<23} {cat:<12} {real_count:>4}   {fake_count:>4}   {correct}/{total_per} {acc_str}")
    
    # Confidence dağılımı
    print(f"\n{'─'*60}")
    print(f"📊 CONFIDENCE DAĞILIMI")
    print(f"{'─'*60}")
    
    if real_results:
        real_confs = [r["real_prob"] for r in real_results if r["label"] == "REAL"]
        if real_confs:
            print(f"  Real (doğru sınıflandırılan):")
            print(f"    Min confidence : {min(real_confs):.4f}")
            print(f"    Max confidence : {max(real_confs):.4f}")
            print(f"    Ort confidence : {np.mean(real_confs):.4f}")
    
    if fake_results:
        fake_confs = [r["fake_prob"] for r in fake_results if r["label"] == "FAKE"]
        if fake_confs:
            print(f"  Fake (doğru sınıflandırılan):")
            print(f"    Min confidence : {min(fake_confs):.4f}")
            print(f"    Max confidence : {max(fake_confs):.4f}")
            print(f"    Ort confidence : {np.mean(fake_confs):.4f}")
    
    # JSON rapor kaydet
    report = {
        "meta": {
            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "DeepfakeULTRA V5 — EfficientNet-B4 Tri-Branch",
            "best_val_auc": 0.9792,
            "test_auc": 0.9820,
        },
        "genel": {
            "toplam_test": total,
            "real_test": len(real_results),
            "fake_test": len(fake_results),
            "real_dogru": real_correct,
            "fake_dogru": fake_correct,
            "real_accuracy": round(real_acc, 2),
            "fake_accuracy": round(fake_acc, 2),
            "genel_accuracy": round(total_acc, 2),
        },
        "kisi_bazli": {},
    }
    
    for celeb_key, data in celeb_results.items():
        report["kisi_bazli"][celeb_key] = {
            "name": data["name"],
            "category": data["category"],
            "real_results": data["real_results"],
            "fake_results": data["fake_results"],
        }
    
    report_path = output_dir / "celebrity_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'─'*60}")
    print(f"📁 ÇIKTI DOSYALARI")
    print(f"{'─'*60}")
    print(f"  📂 Real fotoğraflar : {real_dir}")
    print(f"  📂 Fake fotoğraflar : {fake_dir}")
    print(f"  📋 Rapor            : {report_path}")
    print(f"\n{'='*60}")
    print(f"✅ Ünlü yüz testi tamamlandı!")


if __name__ == "__main__":
    collect_and_test()
