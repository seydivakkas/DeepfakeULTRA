"""
CelebDF-v2 Domain Fine-tuning + Ünlü Deepfake Re-test
======================================================
1) CelebDF-v2 üzerinde domain fine-tuning (8 epoch)
2) Fine-tuned model ile ünlü deepfake'leri tekrar test et
3) Karşılaştırmalı rapor üret

Kullanım:
    python scripts/run_domain_finetune.py
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
import json
from datetime import datetime

from config import model_cfg, paths, DEVICE
from core.dual_mobilenetv3 import DualPathDeepfakeDetector
from core.data_pipeline import MultiScaleDWT, FaceMeshExtractor
from core.domain_finetune import (
    _run_domain_finetune, get_domain_ft_status,
    BEST_MODEL_GENERALIZED_PATH, DOMAIN_FINETUNED_PATH
)

try:
    from core.frequency_v2 import HybridFrequencyExtractor
    HAS_HYBRID_FREQ = True
except ImportError:
    HAS_HYBRID_FREQ = False


def load_model(model_path=None):
    """Model yükle."""
    model = DualPathDeepfakeDetector().to(DEVICE)
    model_path = model_path or str(paths.MODEL_DIR / "best_run5_forensic.pth")
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
            wavelets=model_cfg.DWT_WAVELETS, size=model_cfg.IMG_SIZE,
            include_dwt=True, include_dct=True, include_phase=True,
        )
    else:
        dwt = MultiScaleDWT()
    mesh = FaceMeshExtractor()
    return transform, dwt, mesh


def predict_image(model, transform, dwt, mesh, img_path):
    """Tek görüntü tahmin."""
    try:
        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image.resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)))
        rgb = transform(image).unsqueeze(0).to(DEVICE)
        freq = torch.from_numpy(dwt(img_np)).unsqueeze(0).float().to(DEVICE)
        mesh_t = torch.from_numpy(mesh(img_np)).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            logits = model(rgb, freq, mesh_t)
            probs = F.softmax(logits, dim=1)[0]
        
        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        label = "REAL" if real_prob > 0.5 else "FAKE"
        
        return {
            "label": label,
            "real_prob": round(real_prob, 6),
            "fake_prob": round(fake_prob, 6),
            "confidence": round(max(real_prob, fake_prob), 6),
        }
    except Exception as e:
        return {"label": "ERROR", "error": str(e)}


def test_celebrity_fakes(model, transform, dwt, mesh, label=""):
    """Ünlü fake görüntüleri test et."""
    celeb_test_dir = paths.DATASET_DIR / "celebrity_test"
    fake_dir = celeb_test_dir / "fake"
    real_dir = celeb_test_dir / "real"
    
    results = {"real": {}, "fake": {}}
    
    # Real test
    if real_dir.exists():
        for person_dir in sorted(real_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            person = person_dir.name
            results["real"][person] = []
            for img_path in sorted(person_dir.glob("*.jpg")):
                r = predict_image(model, transform, dwt, mesh, img_path)
                r["file"] = img_path.name
                results["real"][person].append(r)
    
    # Fake test
    if fake_dir.exists():
        for person_dir in sorted(fake_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            person = person_dir.name
            results["fake"][person] = []
            for img_path in sorted(person_dir.glob("*.jpg")):
                r = predict_image(model, transform, dwt, mesh, img_path)
                r["file"] = img_path.name
                results["fake"][person].append(r)
    
    # Özet
    real_total = sum(len(v) for v in results["real"].values())
    real_correct = sum(1 for v in results["real"].values() for r in v if r["label"] == "REAL")
    fake_total = sum(len(v) for v in results["fake"].values())
    fake_correct = sum(1 for v in results["fake"].values() for r in v if r["label"] == "FAKE")
    
    total = real_total + fake_total
    correct = real_correct + fake_correct
    
    print(f"\n  📊 {label} Sonuçlar:")
    print(f"    Real: {real_correct}/{real_total} (%{real_correct/real_total*100:.1f})" if real_total else "    Real: 0")
    print(f"    Fake: {fake_correct}/{fake_total} (%{fake_correct/fake_total*100:.1f})" if fake_total else "    Fake: 0")
    print(f"    Genel: {correct}/{total} (%{correct/total*100:.1f})" if total else "    Genel: 0")
    
    return results, {
        "real_total": real_total, "real_correct": real_correct,
        "fake_total": fake_total, "fake_correct": fake_correct,
        "total": total, "correct": correct,
    }


def test_celebdf_dataset(model, transform, dwt, mesh, label="", max_per_class=200):
    """CelebDF-v2 veri setini test et."""
    celeb_df_dir = paths.DATASET_DIR / "external_tests" / "celeb_df_v2"
    real_dir = celeb_df_dir / "real"
    fake_dir = celeb_df_dir / "fake"
    
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    
    if real_dir.exists():
        real_files = sorted(real_dir.glob("*.jpg")) + sorted(real_dir.glob("*.png"))
        np.random.seed(42)
        indices = np.random.choice(len(real_files), min(max_per_class, len(real_files)), replace=False)
        for i in indices:
            r = predict_image(model, transform, dwt, mesh, real_files[i])
            if r["label"] == "REAL":
                real_correct += 1
            real_total += 1
    
    if fake_dir.exists():
        fake_files = sorted(fake_dir.glob("*.jpg")) + sorted(fake_dir.glob("*.png"))
        np.random.seed(42)
        indices = np.random.choice(len(fake_files), min(max_per_class, len(fake_files)), replace=False)
        for i in indices:
            r = predict_image(model, transform, dwt, mesh, fake_files[i])
            if r["label"] == "FAKE":
                fake_correct += 1
            fake_total += 1
    
    total = real_total + fake_total
    correct = real_correct + fake_correct
    
    print(f"\n  📊 CelebDF-v2 {label}:")
    print(f"    Real: {real_correct}/{real_total} (%{real_correct/real_total*100:.1f})" if real_total else "")
    print(f"    Fake: {fake_correct}/{fake_total} (%{fake_correct/fake_total*100:.1f})" if fake_total else "")
    print(f"    Genel: {correct}/{total} (%{correct/total*100:.1f})" if total else "")
    
    return {
        "real_total": real_total, "real_correct": real_correct,
        "fake_total": fake_total, "fake_correct": fake_correct,
    }


def main():
    print("=" * 60)
    print("🎯 DOMAIN FINE-TUNING + ÜNLÜ DEEPFAKE TEST")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ═══════════════════════════════════════════════════
    # ADIM 1: Orijinal model ile baseline test
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📊 ADIM 1: ORİJİNAL MODEL — BASELINE TEST")
    print("=" * 60)
    
    model = load_model()
    transform, dwt, mesh = get_preprocessors()
    
    print("\n  🔍 CelebDF-v2 baseline testi...")
    baseline_celebdf = test_celebdf_dataset(model, transform, dwt, mesh, label="(Orijinal Model)")
    
    print("\n  🔍 Ünlü deepfake baseline testi...")
    baseline_celeb_results, baseline_celeb_stats = test_celebrity_fakes(
        model, transform, dwt, mesh, label="Orijinal Model"
    )
    
    # ═══════════════════════════════════════════════════
    # ADIM 2: CelebDF-v2 Domain Fine-tuning
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🔧 ADIM 2: CelebDF-v2 DOMAIN FINE-TUNING")
    print("=" * 60)
    
    # Orijinal state'i yedekle
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Fine-tuning (synchronous)
    print("\n  🚀 Fine-tuning başlatılıyor...")
    print("  📦 Veri setleri: celeb_df_v2, dfdc, deepfake20k")
    
    datasets_to_use = []
    external_dir = paths.DATASET_DIR / "external_tests"
    for ds_name in ["celeb_df_v2", "dfdc", "deepfake20k", "faceforensics"]:
        ds_path = external_dir / ds_name
        if ds_path.exists() and (ds_path / "real").exists() and (ds_path / "fake").exists():
            datasets_to_use.append(ds_name)
            print(f"    ✅ {ds_name} mevcut")
    
    if not datasets_to_use:
        print("  ❌ Harici veri seti bulunamadı!")
        return
    
    # Doğrudan çalıştır (thread yerine)
    _run_domain_finetune(
        model=model,
        original_state_dict=original_state,
        dataset_names=datasets_to_use,
        epochs=10,
        lr=5e-5,
        batch_size=16,
        phase1_epochs=3,
    )
    
    # Status kontrol
    status = get_domain_ft_status()
    if status.get("error"):
        print(f"\n  ❌ Fine-tuning hatası: {status['error']}")
        # Orijinal modele geri dön
        model.load_state_dict(original_state)
        model.eval()
        return
    
    # Fine-tuned modeli yükle
    if BEST_MODEL_GENERALIZED_PATH.exists():
        ckpt = torch.load(str(BEST_MODEL_GENERALIZED_PATH), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"\n  ✅ Fine-tuned model yüklendi")
        print(f"     Combined Score: {ckpt.get('combined_score', '?')}")
        print(f"     Val AUC: {ckpt.get('val_auc', '?')}")
        print(f"     Ext Mean AUC: {ckpt.get('ext_mean_auc', '?')}")
    
    # ═══════════════════════════════════════════════════
    # ADIM 3: Fine-tuned model ile re-test
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📊 ADIM 3: FINE-TUNED MODEL — RE-TEST")
    print("=" * 60)
    
    print("\n  🔍 CelebDF-v2 fine-tuned testi...")
    finetuned_celebdf = test_celebdf_dataset(model, transform, dwt, mesh, label="(Fine-tuned)")
    
    print("\n  🔍 Ünlü deepfake fine-tuned testi...")
    finetuned_celeb_results, finetuned_celeb_stats = test_celebrity_fakes(
        model, transform, dwt, mesh, label="Fine-tuned Model"
    )
    
    # ═══════════════════════════════════════════════════
    # ADIM 4: Karşılaştırmalı rapor
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📊 ADIM 4: KARŞILAŞTIRMALI RAPOR")
    print("=" * 60)
    
    print(f"\n{'─'*60}")
    print(f"{'Metrik':<35} {'Orijinal':>12} {'Fine-tuned':>12}")
    print(f"{'─'*60}")
    
    # CelebDF-v2
    b_real = baseline_celebdf
    f_real = finetuned_celebdf
    
    b_real_acc = b_real['real_correct']/b_real['real_total']*100 if b_real['real_total'] else 0
    f_real_acc = f_real['real_correct']/f_real['real_total']*100 if f_real['real_total'] else 0
    b_fake_acc = b_real['fake_correct']/b_real['fake_total']*100 if b_real['fake_total'] else 0
    f_fake_acc = f_real['fake_correct']/f_real['fake_total']*100 if f_real['fake_total'] else 0
    
    print(f"  CelebDF Real Acc      {b_real_acc:>10.1f}%  {f_real_acc:>10.1f}%")
    print(f"  CelebDF Fake Acc      {b_fake_acc:>10.1f}%  {f_fake_acc:>10.1f}%")
    
    # Ünlü
    b_s = baseline_celeb_stats
    f_s = finetuned_celeb_stats
    
    b_celeb_real = b_s['real_correct']/b_s['real_total']*100 if b_s['real_total'] else 0
    f_celeb_real = f_s['real_correct']/f_s['real_total']*100 if f_s['real_total'] else 0
    b_celeb_fake = b_s['fake_correct']/b_s['fake_total']*100 if b_s['fake_total'] else 0
    f_celeb_fake = f_s['fake_correct']/f_s['fake_total']*100 if f_s['fake_total'] else 0
    
    print(f"  Ünlü Real Acc         {b_celeb_real:>10.1f}%  {f_celeb_real:>10.1f}%")
    print(f"  Ünlü Fake Acc         {b_celeb_fake:>10.1f}%  {f_celeb_fake:>10.1f}%")
    print(f"{'─'*60}")
    
    # Kişi bazlı karşılaştırma
    print(f"\n{'─'*60}")
    print(f"📊 KİŞİ BAZLI FAKE TESPIT KARŞILAŞTIRMASI")
    print(f"{'─'*60}")
    print(f"{'Kişi':<25} {'Orijinal':>10} {'Fine-tuned':>12}")
    print(f"{'─'*60}")
    
    for person in sorted(set(list(baseline_celeb_results.get("fake", {}).keys()) + list(finetuned_celeb_results.get("fake", {}).keys()))):
        b_fakes = baseline_celeb_results.get("fake", {}).get(person, [])
        f_fakes = finetuned_celeb_results.get("fake", {}).get(person, [])
        
        b_correct = sum(1 for r in b_fakes if r["label"] == "FAKE")
        f_correct = sum(1 for r in f_fakes if r["label"] == "FAKE")
        b_total = len(b_fakes)
        f_total = len(f_fakes)
        
        b_str = f"{b_correct}/{b_total}" if b_total else "—"
        f_str = f"{f_correct}/{f_total}" if f_total else "—"
        
        name = person.replace("_", " ")
        print(f"  {name:<23} {b_str:>10} {f_str:>12}")
    
    print(f"{'─'*60}")
    
    # JSON rapor kaydet
    report = {
        "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": {
            "celebdf": baseline_celebdf,
            "celebrity_stats": baseline_celeb_stats,
        },
        "finetuned": {
            "celebdf": finetuned_celebdf,
            "celebrity_stats": finetuned_celeb_stats,
        },
    }
    
    report_path = paths.DATASET_DIR / "celebrity_test" / "finetune_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 Rapor: {report_path}")
    print(f"\n{'='*60}")
    print(f"✅ Domain fine-tuning + ünlü re-test tamamlandı!")


if __name__ == "__main__":
    main()
