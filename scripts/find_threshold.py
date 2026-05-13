"""
Optimal karar eşiği (threshold) bulma scripti.
Validation setinden optimal threshold hesaplar, sonra jury'ye uygular.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, accuracy_score, f1_score, roc_auc_score
from config import model_cfg, DEVICE, paths
from core.dual_mobilenetv3 import DualPathDeepfakeDetector
from core.data_pipeline import FaceMeshExtractor

# Frekans extractor
if getattr(model_cfg, 'USE_HYBRID_FREQ', False):
    from core.frequency_v2 import HybridFrequencyExtractor
    freq_ext = HybridFrequencyExtractor(wavelets=model_cfg.DWT_WAVELETS, size=model_cfg.IMG_SIZE)
else:
    from core.data_pipeline import MultiScaleDWT
    freq_ext = MultiScaleDWT()

mesh_ext = FaceMeshExtractor()
transform = transforms.Compose([
    transforms.Resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def load_model(path):
    model = DualPathDeepfakeDetector()
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    cur = model.state_dict()
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
    model.load_state_dict(cur)
    model.to(DEVICE).eval()
    return model

def predict_prob(model, img_path):
    """Tek görsel için FAKE olasılığı döndür."""
    try:
        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image.resize((224, 224)))
        freq = torch.from_numpy(freq_ext(img_np)).float().unsqueeze(0).to(DEVICE)
        mesh = torch.from_numpy(mesh_ext(img_np)).float().unsqueeze(0).to(DEVICE)
        rgb = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(rgb, freq, mesh)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return float(probs[1])  # p_fake
    except:
        return None

def collect_predictions(model, data_dir):
    """Bir dizinden tüm görselleri okuyup (label, p_fake) döndür."""
    labels, probs = [], []
    for label_name, label_id in [("REAL", 0), ("FAKE", 1)]:
        label_dir = Path(data_dir) / label_name
        if not label_dir.exists():
            continue
        files = [f for f in label_dir.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED]
        print(f"  {label_name}: {len(files)} dosya işleniyor...")
        for i, f in enumerate(files):
            p = predict_prob(model, str(f))
            if p is not None:
                labels.append(label_id)
                probs.append(p)
            if (i+1) % 5000 == 0:
                print(f"    {i+1}/{len(files)}")
    return np.array(labels), np.array(probs)

def find_optimal_threshold(labels, probs):
    """Youden's J statistic ile optimal threshold bul."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    # Youden's J = TPR - FPR → maximize
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    
    # F1 bazlı threshold
    best_f1, best_f1_thresh = 0, 0.5
    for t in np.arange(0.3, 0.7, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = t
    
    return {
        "youden_threshold": float(best_thresh),
        "youden_j": float(j_scores[best_idx]),
        "f1_threshold": float(best_f1_thresh),
        "f1_score": float(best_f1),
    }

def evaluate_with_threshold(labels, probs, threshold):
    """Verilen threshold ile metrikler hesapla."""
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(labels, probs)
    return {"accuracy": acc, "f1": f1, "auc": auc, "threshold": threshold}

if __name__ == "__main__":
    model_path = "models/best_run5_forensic.pth"
    print(f"Model: {model_path}")
    model = load_model(model_path)
    
    # 1. Validation setinden optimal threshold bul
    val_dir = Path("dataset/faces_split/val")
    print(f"\n=== Validation setinden threshold hesaplama ===")
    print(f"Dizin: {val_dir}")
    
    val_labels, val_probs = collect_predictions(model, val_dir)
    print(f"\nToplam: {len(val_labels)} görsel")
    
    thresholds = find_optimal_threshold(val_labels, val_probs)
    print(f"\n📊 Optimal Threshold Sonuçları:")
    print(f"  Youden's J:  threshold={thresholds['youden_threshold']:.4f} (J={thresholds['youden_j']:.4f})")
    print(f"  F1-optimal:  threshold={thresholds['f1_threshold']:.4f} (F1={thresholds['f1_score']:.4f})")
    
    # 2. Varsayılan 0.5 vs optimal karşılaştırma
    print(f"\n=== Threshold Karşılaştırma (Val Set) ===")
    for t in [0.5, thresholds['youden_threshold'], thresholds['f1_threshold']]:
        r = evaluate_with_threshold(val_labels, val_probs, t)
        print(f"  t={t:.4f} → Acc={r['accuracy']:.4f} F1={r['f1']:.4f} AUC={r['auc']:.4f}")
    
    # 3. Jury set'e uygula
    jury_dir = paths.BASE_DIR / "dataset" / "jury_test"
    print(f"\n=== Jury Set Değerlendirmesi ===")
    
    jury_labels, jury_probs = [], []
    source_data = {}
    for label_name, label_id in [("real", 0), ("fake", 1)]:
        label_dir = jury_dir / label_name
        if not label_dir.exists():
            continue
        for source_dir in sorted(label_dir.iterdir()):
            if not source_dir.is_dir() or source_dir.name == ".cache":
                continue
            source_key = f"{label_name}/{source_dir.name}"
            files = [f for f in source_dir.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED]
            print(f"  {source_key}: {len(files)} dosya...")
            src_labels, src_probs = [], []
            for f in files:
                p = predict_prob(model, str(f))
                if p is not None:
                    jury_labels.append(label_id)
                    jury_probs.append(p)
                    src_labels.append(label_id)
                    src_probs.append(p)
            source_data[source_key] = (src_labels, src_probs)
    
    jury_labels = np.array(jury_labels)
    jury_probs = np.array(jury_probs)
    
    best_t = thresholds['youden_threshold']
    print(f"\n📊 Jury Sonuçları — Threshold={best_t:.4f} vs 0.5:")
    for t_name, t_val in [("t=0.50", 0.5), (f"t={best_t:.2f}", best_t)]:
        r = evaluate_with_threshold(jury_labels, jury_probs, t_val)
        print(f"  {t_name} → Acc={r['accuracy']:.4f} F1={r['f1']:.4f} AUC={r['auc']:.4f}")
    
    print(f"\n📋 Kaynak Bazlı (t={best_t:.4f}):")
    for source_key, (sl, sp) in sorted(source_data.items()):
        sl, sp = np.array(sl), np.array(sp)
        preds = (sp >= best_t).astype(int)
        acc = accuracy_score(sl, preds)
        emoji = "🟢" if source_key.startswith("real") else "🔴"
        print(f"  {emoji} {source_key:30s} | Acc={acc:.3f} (n={len(sl)})")
    
    # Threshold'u config'e kaydet
    thresh_path = paths.MODEL_DIR / "optimal_threshold.txt"
    with open(thresh_path, "w") as f:
        f.write(f"{best_t:.6f}\n")
    print(f"\n💾 Optimal threshold kaydedildi: {thresh_path}")
