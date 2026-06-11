"""
Hızlı Domain Fine-tuning — CelebDF-v2 + DFDC
==============================================
Doğrudan DataLoader ile fine-tuning, sonra ünlü re-test.

Kullanım:
    python scripts/quick_domain_finetune.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json, random, time
from datetime import datetime

from config import model_cfg, paths, DEVICE
from core.dual_mobilenetv3 import DualPathDeepfakeDetector
from core.data_pipeline import MultiScaleDWT, FaceMeshExtractor

try:
    from core.frequency_v2 import HybridFrequencyExtractor
    HAS_HYBRID_FREQ = True
except ImportError:
    HAS_HYBRID_FREQ = False


# ═══════════════════════════════════════════════════════════
# Basit Dataset — sadece real/fake klasörlerini oku
# ═══════════════════════════════════════════════════════════
class SimpleDeepfakeDataset(Dataset):
    """Basit deepfake veri seti — real/fake klasörlerinden yükler."""
    
    def __init__(self, root_dir, transform=None, max_per_class=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        root = Path(root_dir)
        real_dir = root / "real"
        fake_dir = root / "fake"
        
        self.samples = []
        
        # Real görüntüler (label=0)
        if real_dir.exists():
            real_files = [f for f in real_dir.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if max_per_class and len(real_files) > max_per_class:
                random.seed(42)
                real_files = random.sample(real_files, max_per_class)
            self.samples.extend([(str(f), 0) for f in real_files])
        
        # Fake görüntüler (label=1)
        if fake_dir.exists():
            fake_files = [f for f in fake_dir.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if max_per_class and len(fake_files) > max_per_class:
                random.seed(42)
                fake_files = random.sample(fake_files, max_per_class)
            self.samples.extend([(str(f), 1) for f in fake_files])
        
        # Frekans ve mesh ekstraktörleri
        if HAS_HYBRID_FREQ and getattr(model_cfg, 'USE_HYBRID_FREQ', False):
            self.dwt = HybridFrequencyExtractor(
                wavelets=model_cfg.DWT_WAVELETS, size=model_cfg.IMG_SIZE,
                include_dwt=True, include_dct=True, include_phase=True,
            )
        else:
            self.dwt = MultiScaleDWT()
        
        self.mesh = FaceMeshExtractor()
        
        random.shuffle(self.samples)
        
        real_count = sum(1 for _, l in self.samples if l == 0)
        fake_count = sum(1 for _, l in self.samples if l == 1)
        print(f"    Dataset: {len(self.samples)} toplam (Real={real_count}, Fake={fake_count})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img.resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)))
            
            rgb = self.transform(img)
            freq = torch.from_numpy(self.dwt(img_np)).float()
            mesh = torch.from_numpy(self.mesh(img_np)).float()
            
            return rgb, freq, mesh, label
        except Exception:
            # Hatalı görüntülerde dummy döndür
            rgb = torch.zeros(3, model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)
            freq = torch.zeros(model_cfg.DWT_CHANNELS, model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)
            mesh = torch.zeros(1404)
            return rgb, freq, mesh, label


def load_model():
    """Model yükle."""
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


def predict_image(model, img_path, transform, dwt, mesh):
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
        return {"label": "REAL" if probs[0] > 0.5 else "FAKE",
                "real_prob": float(probs[0]), "fake_prob": float(probs[1])}
    except:
        return {"label": "ERROR", "real_prob": 0, "fake_prob": 0}


def evaluate_celebdf(model, transform, dwt, mesh, label="", n=200):
    """CelebDF-v2 hızlı test."""
    celeb_dir = paths.DATASET_DIR / "external_tests" / "celeb_df_v2"
    
    results = {"real_correct": 0, "real_total": 0, "fake_correct": 0, "fake_total": 0}
    
    for cls, lbl in [("real", "REAL"), ("fake", "FAKE")]:
        d = celeb_dir / cls
        if not d.exists(): continue
        files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
        np.random.seed(42)
        idx = np.random.choice(len(files), min(n, len(files)), replace=False)
        for i in idx:
            r = predict_image(model, files[i], transform, dwt, mesh)
            results[f"{cls}_total"] += 1
            if r["label"] == lbl:
                results[f"{cls}_correct"] += 1
    
    ra = results['real_correct']/max(results['real_total'],1)*100
    fa = results['fake_correct']/max(results['fake_total'],1)*100
    print(f"  📊 CelebDF-v2 {label}: Real=%{ra:.1f}, Fake=%{fa:.1f}, "
          f"Genel=%{(results['real_correct']+results['fake_correct'])/max(results['real_total']+results['fake_total'],1)*100:.1f}")
    return results


def evaluate_celebrities(model, transform, dwt, mesh, label=""):
    """Ünlü deepfake testi."""
    celeb_dir = paths.DATASET_DIR / "celebrity_test"
    real_c, real_t, fake_c, fake_t = 0, 0, 0, 0
    
    for cls, lbl in [("real", "REAL"), ("fake", "FAKE")]:
        d = celeb_dir / cls
        if not d.exists(): continue
        for person_dir in sorted(d.iterdir()):
            if not person_dir.is_dir(): continue
            for img in person_dir.glob("*.jpg"):
                r = predict_image(model, img, transform, dwt, mesh)
                if cls == "real":
                    real_t += 1
                    if r["label"] == "REAL": real_c += 1
                else:
                    fake_t += 1
                    if r["label"] == "FAKE": fake_c += 1
    
    print(f"  📊 Ünlü {label}: Real={real_c}/{real_t}(%{real_c/max(real_t,1)*100:.1f}), "
          f"Fake={fake_c}/{fake_t}(%{fake_c/max(fake_t,1)*100:.1f})")
    return {"real_correct": real_c, "real_total": real_t, "fake_correct": fake_c, "fake_total": fake_t}


def main():
    print("=" * 60)
    print("🎯 HIZLI DOMAIN FINE-TUNING + ÜNLÜ DEEPFAKE TEST")
    print("=" * 60)
    
    # Model yükle
    model = load_model()
    
    # Preprocessors
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
    
    # ═══════════════════════════════════════════
    # ADIM 1: Baseline
    # ═══════════════════════════════════════════
    print("\n📊 ADIM 1: BASELINE (Orijinal Model)")
    baseline_cdf = evaluate_celebdf(model, transform, dwt, mesh, "(Orijinal)")
    baseline_cel = evaluate_celebrities(model, transform, dwt, mesh, "(Orijinal)")
    
    # ═══════════════════════════════════════════
    # ADIM 2: Dataset hazırla
    # ═══════════════════════════════════════════
    print("\n🔧 ADIM 2: DATASET HAZIRLAMA")
    
    ext_dir = paths.DATASET_DIR / "external_tests"
    train_datasets = []
    
    # Augmentasyonlu transform
    train_transform = transforms.Compose([
        transforms.Resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    for ds_name in ["celeb_df_v2", "dfdc", "deepfake20k", "faceforensics"]:
        ds_path = ext_dir / ds_name
        if ds_path.exists() and (ds_path / "real").exists() and (ds_path / "fake").exists():
            print(f"  📦 {ds_name} yükleniyor...")
            ds = SimpleDeepfakeDataset(str(ds_path), transform=train_transform, max_per_class=1000)
            if len(ds) > 10:
                train_datasets.append(ds)
    
    if not train_datasets:
        print("  ❌ Veri seti bulunamadı!")
        return
    
    # Birleştir
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(train_datasets)
    
    # Train/val split
    total = len(combined)
    val_size = int(total * 0.15)
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(combined, [train_size, val_size],
                                                      generator=torch.Generator().manual_seed(42))
    
    print(f"\n  📊 Toplam: {total} | Train: {train_size} | Val: {val_size}")
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0,
                            pin_memory=True)
    
    # ═══════════════════════════════════════════
    # ADIM 3: Fine-tuning
    # ═══════════════════════════════════════════
    print("\n🔥 ADIM 3: DOMAIN FINE-TUNING (10 epoch)")
    
    # Orijinal state yedekle
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Faz 1 (epoch 1-3): Sadece fusion + classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.cross_branch.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.mesh_proj.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Faz 1: {trainable:,}/{total_params:,} parametre eğitilecek")
    
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler() if DEVICE.type == "cuda" else None
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(10):
        # Faz 2: Epoch 3'te backbone aç
        if epoch == 3:
            print("  🔥 Faz 2: Backbone açılıyor...")
            if hasattr(model, 'rgb_backbone'):
                for param in model.rgb_backbone.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'rgb_features'):
                for param in model.rgb_features.parameters():
                    param.requires_grad = True
            if hasattr(model, 'rgb_proj'):
                for param in model.rgb_proj.parameters():
                    param.requires_grad = True
            for param in model.freq_features.parameters():
                param.requires_grad = True
            for param in model.mesh_mlp.parameters():
                param.requires_grad = True
            
            # Yeni optimizer (discriminative LR)
            backbone_params = []
            other_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad: continue
                if "rgb_backbone" in name or "rgb_features" in name or "freq_features" in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
            
            optimizer = AdamW([
                {"params": other_params, "lr": 2.5e-5},
                {"params": backbone_params, "lr": 2.5e-6},
            ], weight_decay=1e-2)
            scheduler = CosineAnnealingLR(optimizer, T_max=7, eta_min=5e-7)
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Faz 2: {trainable:,}/{total_params:,} parametre eğitilecek")
        
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch_idx, (rgb, freq, mesh_t, labels) in enumerate(train_loader):
            rgb = rgb.to(DEVICE)
            freq = freq.to(DEVICE)
            mesh_t = mesh_t.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if scaler:
                with autocast(device_type="cuda"):
                    logits = model(rgb, freq, mesh_t)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                logits = model(rgb, freq, mesh_t)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for rgb, freq, mesh_t, labels in val_loader:
                rgb, freq, mesh_t, labels = rgb.to(DEVICE), freq.to(DEVICE), mesh_t.to(DEVICE), labels.to(DEVICE)
                if scaler:
                    with autocast(device_type="cuda"):
                        logits = model(rgb, freq, mesh_t)
                else:
                    logits = model(rgb, freq, mesh_t)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / max(train_total, 1)
        lr = optimizer.param_groups[0]["lr"]
        
        phase = "FUSION" if epoch < 3 else "FULL"
        print(f"  Epoch {epoch+1}/10 [{phase}] | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | LR: {lr:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    💾 Best model! Val Acc: {val_acc:.3f}")
    
    # Best model yükle
    if best_state:
        model.load_state_dict(best_state)
        # Kaydet
        save_path = paths.MODEL_DIR / "best_model_generalized.pth"
        torch.save({
            "model_state_dict": best_state,
            "val_acc": best_val_acc,
            "type": "domain_finetuned",
            "datasets": ["celeb_df_v2", "dfdc", "deepfake20k", "faceforensics"],
        }, str(save_path))
        print(f"\n  ✅ Best model kaydedildi: {save_path.name} (Val Acc: {best_val_acc:.3f})")
    
    model.eval()
    
    # ═══════════════════════════════════════════
    # ADIM 4: Fine-tuned model ile re-test
    # ═══════════════════════════════════════════
    print("\n📊 ADIM 4: FINE-TUNED MODEL RE-TEST")
    finetuned_cdf = evaluate_celebdf(model, transform, dwt, mesh, "(Fine-tuned)")
    finetuned_cel = evaluate_celebrities(model, transform, dwt, mesh, "(Fine-tuned)")
    
    # ═══════════════════════════════════════════
    # ADIM 5: Karşılaştırma
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📊 KARŞILAŞTIRMALI RAPOR")
    print("=" * 60)
    
    def acc(d, k): return d[f'{k}_correct']/max(d[f'{k}_total'],1)*100
    
    print(f"\n{'Metrik':<30} {'Orijinal':>10} {'Fine-tuned':>12} {'Fark':>8}")
    print("─" * 62)
    print(f"  CelebDF Real Acc       {acc(baseline_cdf,'real'):>8.1f}%  {acc(finetuned_cdf,'real'):>10.1f}%  {acc(finetuned_cdf,'real')-acc(baseline_cdf,'real'):>+7.1f}")
    print(f"  CelebDF Fake Acc       {acc(baseline_cdf,'fake'):>8.1f}%  {acc(finetuned_cdf,'fake'):>10.1f}%  {acc(finetuned_cdf,'fake')-acc(baseline_cdf,'fake'):>+7.1f}")
    print(f"  Ünlü Real Acc          {acc(baseline_cel,'real'):>8.1f}%  {acc(finetuned_cel,'real'):>10.1f}%  {acc(finetuned_cel,'real')-acc(baseline_cel,'real'):>+7.1f}")
    print(f"  Ünlü Fake Acc          {acc(baseline_cel,'fake'):>8.1f}%  {acc(finetuned_cel,'fake'):>10.1f}%  {acc(finetuned_cel,'fake')-acc(baseline_cel,'fake'):>+7.1f}")
    print("─" * 62)
    
    # JSON rapor
    report = {
        "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": {"celebdf": baseline_cdf, "celebrity": baseline_cel},
        "finetuned": {"celebdf": finetuned_cdf, "celebrity": finetuned_cel},
        "best_val_acc": best_val_acc,
    }
    report_path = paths.DATASET_DIR / "celebrity_test" / "finetune_comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 Rapor: {report_path}")
    print(f"\n✅ Tamamlandı!")


if __name__ == "__main__":
    main()
