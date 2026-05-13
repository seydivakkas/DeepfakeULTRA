"""
Deepfake Detection System v3.0 — Adversarial Robustness Testing
FGSM, PGD, C&W saldırıları + Epsilon Sweep.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional
from config import model_cfg, DEVICE


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL Image → normalized RGB tensor (1, 3, H, W)."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Normalized tensor → PIL Image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    img = tensor * std + mean
    img = img.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def _prepare_inputs(image: Image.Image, model: nn.Module):
    """Görüntüyü model inputlarına dönüştür (rgb, freq, mesh)."""
    from core.data_pipeline import FaceMeshExtractor
    try:
        from core.frequency_v2 import HybridFrequencyExtractor
        freq_ext = HybridFrequencyExtractor()  # 18 kanal (DWT+DCT+Phase)
    except ImportError:
        from core.data_pipeline import MultiScaleDWT
        freq_ext = MultiScaleDWT()  # fallback 12 kanal
    img_np = np.array(image.resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)))
    rgb = _image_to_tensor(image).to(DEVICE)
    mesh_ext = FaceMeshExtractor()
    freq = torch.from_numpy(freq_ext(img_np)).unsqueeze(0).float().to(DEVICE)
    mesh = torch.from_numpy(mesh_ext(img_np)).unsqueeze(0).float().to(DEVICE)
    return rgb, freq, mesh


def _get_prediction(model: nn.Module, rgb, freq, mesh) -> tuple:
    """Model tahmini → (verdict, fake_prob)."""
    model.eval()
    with torch.no_grad():
        logits = model(rgb, freq, mesh)
        probs = F.softmax(logits, dim=1)[0]
    fake_prob = float(probs[1])
    verdict = "FAKE" if fake_prob > 0.5 else "REAL"
    return verdict, fake_prob


# ================================================================
# FGSM — Fast Gradient Sign Method
# ================================================================
def fgsm_attack(model, rgb, freq, mesh, epsilon, target_class=1):
    """Tek adımda gradient sign pertürbasyon."""
    model.eval()
    rgb_adv = rgb.clone().detach().requires_grad_(True)
    logits = model(rgb_adv, freq, mesh)
    loss = F.cross_entropy(logits, torch.tensor([target_class]).to(DEVICE))
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * rgb_adv.grad.sign()
    perturbed = (rgb + perturbation).detach()
    return perturbed


# ================================================================
# PGD — Projected Gradient Descent
# ================================================================
def pgd_attack(model, rgb, freq, mesh, epsilon, alpha=None, steps=20, target_class=1):
    """İteratif projected gradient descent saldırısı."""
    if alpha is None:
        alpha = epsilon / 4
    model.eval()
    perturbed = rgb.clone().detach()

    for _ in range(steps):
        perturbed.requires_grad_(True)
        logits = model(perturbed, freq, mesh)
        loss = F.cross_entropy(logits, torch.tensor([target_class]).to(DEVICE))
        model.zero_grad()
        loss.backward()
        grad_sign = perturbed.grad.sign()
        perturbed = (perturbed.detach() + alpha * grad_sign)
        # Epsilon topu içinde projection
        delta = torch.clamp(perturbed - rgb, -epsilon, epsilon)
        perturbed = (rgb + delta).detach()

    return perturbed


# ================================================================
# C&W — Carlini & Wagner (Basitleştirilmiş L2)
# ================================================================
def cw_attack(model, rgb, freq, mesh, epsilon, steps=50, lr=0.01, c=1.0, target_class=1):
    """Carlini & Wagner L2 optimizasyon saldırısı."""
    model.eval()
    perturbation = torch.zeros_like(rgb, requires_grad=True, device=DEVICE)
    optimizer = torch.optim.Adam([perturbation], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        perturbed = rgb + perturbation
        logits = model(perturbed, freq, mesh)

        # Hedef sınıf logit'ini artır
        target_logit = logits[0, target_class]
        other_logit = logits[0, 1 - target_class]
        # f(x') = max(Z_other - Z_target, 0) → minimize
        attack_loss = torch.clamp(other_logit - target_logit, min=0)

        # L2 norm cezası
        l2_loss = torch.norm(perturbation)

        # Epsilon sınırı cezası
        eps_penalty = torch.clamp(perturbation.abs() - epsilon, min=0).sum()

        loss = attack_loss + c * l2_loss + 10.0 * eps_penalty
        loss.backward()
        optimizer.step()

    perturbed = (rgb + perturbation).detach()
    return perturbed


# ================================================================
# ANA FONKSİYON — run_adversarial_attack
# ================================================================
def run_adversarial_attack(
    image: Image.Image,
    attack_type: str,
    epsilon: float,
    model: nn.Module,
) -> dict:
    """
    Adversarial saldırı çalıştır.

    Args:
        image: PIL Image (RGB)
        attack_type: 'FGSM', 'PGD', 'CW'
        epsilon: Pertürbasyon büyüklüğü (0.001-0.3)
        model: DualPathDeepfakeDetector

    Returns:
        dict: original/perturbed görseller, kararlar, olasılıklar, başarı durumu
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    rgb, freq, mesh = _prepare_inputs(image, model)
    orig_verdict, orig_prob = _get_prediction(model, rgb, freq, mesh)

    # Saldırı uygula
    attack_fn = {
        "FGSM": lambda: fgsm_attack(model, rgb, freq, mesh, epsilon),
        "PGD": lambda: pgd_attack(model, rgb, freq, mesh, epsilon),
        "CW": lambda: cw_attack(model, rgb, freq, mesh, epsilon),
    }

    if attack_type not in attack_fn:
        return {"error": f"Bilinmeyen saldırı tipi: {attack_type}"}

    perturbed_rgb = attack_fn[attack_type]()
    pert_verdict, pert_prob = _get_prediction(model, perturbed_rgb, freq, mesh)
    attack_success = orig_verdict != pert_verdict

    # Özet metin
    summary = (
        f"Saldırı: {attack_type} | ε={epsilon:.4f}\n"
        f"Orijinal: {orig_verdict} (fake={orig_prob:.4f})\n"
        f"Pertürbe: {pert_verdict} (fake={pert_prob:.4f})\n"
        f"Sonuç: {'✅ Karar değişti!' if attack_success else '❌ Karar değişmedi'}"
    )

    return {
        "original_image": _tensor_to_image(rgb),
        "perturbed_image": _tensor_to_image(perturbed_rgb),
        "original_verdict": orig_verdict,
        "perturbed_verdict": pert_verdict,
        "original_prob": orig_prob,
        "perturbed_prob": pert_prob,
        "attack_success": attack_success,
        "summary_text": summary,
    }


# ================================================================
# EPSILON SWEEP
# ================================================================
def epsilon_sweep(
    image: Image.Image,
    attack_type: str,
    epsilon_range: tuple = (0.001, 0.3),
    steps: int = 20,
    model: nn.Module = None,
) -> dict:
    """
    Farklı epsilon değerleri için saldırı sonuçlarını hesapla.

    Returns:
        dict: epsilons, fake_probs, decision_flip_epsilon
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    rgb, freq, mesh = _prepare_inputs(image, model)
    orig_verdict, _ = _get_prediction(model, rgb, freq, mesh)

    epsilons = np.linspace(epsilon_range[0], epsilon_range[1], steps).tolist()
    fake_probs = []
    decision_flip = None

    attack_map = {"FGSM": fgsm_attack, "PGD": pgd_attack, "CW": cw_attack}
    attack_fn = attack_map.get(attack_type, fgsm_attack)

    for eps in epsilons:
        perturbed = attack_fn(model, rgb, freq, mesh, eps)
        verdict, prob = _get_prediction(model, perturbed, freq, mesh)
        fake_probs.append(prob)

        if decision_flip is None and verdict != orig_verdict:
            decision_flip = eps

    return {
        "epsilons": epsilons,
        "fake_probs": fake_probs,
        "decision_flip_epsilon": decision_flip,
        "original_verdict": orig_verdict,
    }


# ================================================================
# BRANCH KNOCKOUT TESTİ — Hangi dal ne kadar kritik?
# ================================================================
def branch_knockout_test(
    image: Image.Image,
    model: nn.Module,
) -> dict:
    """
    Her branch'i sırayla sıfırlayıp modelin tepkisini ölçer.
    CrossBranchTransformer'ın branch bağımlılığını ortaya çıkarır.

    Returns:
        dict: Her branch kombinasyonu için (verdict, fake_prob)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    rgb, freq, mesh = _prepare_inputs(image, model)
    model.eval()

    results = {}

    # Tam model (baseline)
    with torch.no_grad():
        logits = model(rgb, freq, mesh)
        probs = F.softmax(logits, dim=1)[0]
    results["all"] = {
        "verdict": "FAKE" if probs[1] > 0.5 else "REAL",
        "fake_prob": float(probs[1]),
        "label": "🔬 Tüm Dallar (Baseline)",
    }

    # Branch knockout kombinasyonları
    knockouts = [
        ("no_rgb", torch.zeros_like(rgb), freq, mesh,
         "🔴 RGB Kapalı (Freq + Mesh)"),
        ("no_freq", rgb, torch.zeros_like(freq), mesh,
         "🔵 Frekans Kapalı (RGB + Mesh)"),
        ("no_mesh", rgb, freq, torch.zeros_like(mesh),
         "🟡 Mesh Kapalı (RGB + Freq)"),
        ("only_rgb", rgb, torch.zeros_like(freq), torch.zeros_like(mesh),
         "🟢 Sadece RGB"),
        ("only_freq", torch.zeros_like(rgb), freq, torch.zeros_like(mesh),
         "🔵 Sadece Frekans"),
        ("only_mesh", torch.zeros_like(rgb), torch.zeros_like(freq), mesh,
         "🟡 Sadece Mesh"),
    ]

    for key, r, f, m, label in knockouts:
        with torch.no_grad():
            logits = model(r, f, m)
            probs = F.softmax(logits, dim=1)[0]
        results[key] = {
            "verdict": "FAKE" if probs[1] > 0.5 else "REAL",
            "fake_prob": float(probs[1]),
            "label": label,
        }

    # En kritik branch'i belirle
    baseline_prob = results["all"]["fake_prob"]
    max_shift = 0
    critical_branch = "bilinmiyor"
    for key in ["no_rgb", "no_freq", "no_mesh"]:
        shift = abs(results[key]["fake_prob"] - baseline_prob)
        if shift > max_shift:
            max_shift = shift
            branch_names = {"no_rgb": "RGB", "no_freq": "Frekans", "no_mesh": "Mesh"}
            critical_branch = branch_names[key]

    results["_critical_branch"] = critical_branch
    results["_max_shift"] = max_shift

    return results


# ================================================================
# FREKANS BAND ABLASYONU — 18 kanaldan hangisi belirleyici?
# ================================================================
def frequency_band_ablation(
    image: Image.Image,
    model: nn.Module,
) -> dict:
    """
    DWT (12ch), DCT (3ch), Phase (3ch) gruplarını ayrı ayrı
    maskeleyerek hangi frekans bilgisinin kararı en çok etkilediğini gösterir.

    Returns:
        dict: Her frekans grubu için (verdict, fake_prob)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    rgb, freq, mesh = _prepare_inputs(image, model)
    model.eval()

    # Baseline
    with torch.no_grad():
        logits = model(rgb, freq, mesh)
        probs = F.softmax(logits, dim=1)[0]
    baseline_prob = float(probs[1])

    results = {
        "baseline": {"fake_prob": baseline_prob, "label": "Tüm Frekans (18ch)"},
    }

    # Frekans band grupları (18 kanal: 0-11=DWT, 12-14=DCT, 15-17=Phase)
    band_groups = {
        "no_dwt":   {"mask_range": (0, 12),  "label": "DWT Kapalı (DCT+Phase)"},
        "no_dct":   {"mask_range": (12, 15), "label": "DCT Kapalı (DWT+Phase)"},
        "no_phase": {"mask_range": (15, 18), "label": "Phase Kapalı (DWT+DCT)"},
        "only_dwt": {"keep_range": (0, 12),  "label": "Sadece DWT (12ch)"},
        "only_dct": {"keep_range": (12, 15), "label": "Sadece DCT (3ch)"},
        "only_phase": {"keep_range": (15, 18), "label": "Sadece Phase (3ch)"},
    }

    for key, cfg in band_groups.items():
        masked_freq = freq.clone()
        if "mask_range" in cfg:
            s, e = cfg["mask_range"]
            masked_freq[:, s:e, :, :] = 0
        elif "keep_range" in cfg:
            s, e = cfg["keep_range"]
            mask = torch.zeros_like(masked_freq)
            mask[:, s:e, :, :] = masked_freq[:, s:e, :, :]
            masked_freq = mask

        with torch.no_grad():
            logits = model(rgb, masked_freq, mesh)
            probs = F.softmax(logits, dim=1)[0]

        results[key] = {
            "fake_prob": float(probs[1]),
            "shift": float(probs[1]) - baseline_prob,
            "label": cfg["label"],
        }

    return results


# ================================================================
# ÇÖZÜNÜRLÜK DAYANIKLILIĞI — Düşük kalitede ne olur?
# ================================================================
def resolution_robustness(
    image: Image.Image,
    model: nn.Module,
    resolutions: list = None,
) -> dict:
    """
    Görseli farklı çözünürlüklere düşürüp tekrar 224'e büyüterek
    modelin kalite kaybına tepkisini ölçer.

    Returns:
        dict: resolutions, fake_probs, verdicts
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    if resolutions is None:
        resolutions = [32, 48, 64, 96, 128, 160, 192, 224]

    model.eval()
    target_size = model_cfg.IMG_SIZE
    fake_probs = []
    verdicts = []

    for res in resolutions:
        # Çözünürlüğü düşür ve tekrar büyüt
        degraded = image.resize((res, res), Image.BILINEAR)
        restored = degraded.resize((target_size, target_size), Image.BILINEAR)

        rgb, freq, mesh = _prepare_inputs(restored, model)
        with torch.no_grad():
            logits = model(rgb, freq, mesh)
            probs = F.softmax(logits, dim=1)[0]

        fp = float(probs[1])
        fake_probs.append(fp)
        verdicts.append("FAKE" if fp > 0.5 else "REAL")

    # Karar değişim noktası
    baseline_verdict = verdicts[-1]  # 224px = orijinal
    flip_res = None
    for i, (res, v) in enumerate(zip(resolutions, verdicts)):
        if v != baseline_verdict and flip_res is None:
            flip_res = res

    return {
        "resolutions": resolutions,
        "fake_probs": fake_probs,
        "verdicts": verdicts,
        "decision_flip_resolution": flip_res,
        "baseline_verdict": baseline_verdict,
    }


# ================================================================
# ÇİFT SIKIŞTIRMA TESTİ — Gerçek dünya zinciri simülasyonu
# ================================================================
def double_compression_test(
    image: Image.Image,
    model: nn.Module,
) -> dict:
    """
    Gerçek dünya senaryosu: bir görsel farklı platformlardan
    ardışık sıkıştırmaya tabi tutulur.

    Zincirler:
        1. Orijinal → TikTok → Twitter
        2. Orijinal → Twitter → TikTok
        3. Orijinal → TikTok → Screenshot(Q=50) → Twitter

    Returns:
        dict: Her zincir için sonuçlar
    """
    import io

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    model.eval()

    def jpeg_compress(img, quality):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def resize_compress(img, max_dim, quality):
        w, h = img.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.BILINEAR)
        return jpeg_compress(img, quality)

    # Orijinal
    rgb, freq, mesh = _prepare_inputs(image, model)
    with torch.no_grad():
        logits = model(rgb, freq, mesh)
        probs = F.softmax(logits, dim=1)[0]
    orig_prob = float(probs[1])

    chains = {
        "original": {
            "label": "📷 Orijinal",
            "steps": [],
            "image": image,
        },
        "tiktok_twitter": {
            "label": "🎵 TikTok → Twitter",
            "steps": [
                ("TikTok", 1080, 72),
                ("Twitter", 4096, 80),
            ],
        },
        "twitter_tiktok": {
            "label": "🐦 Twitter → TikTok",
            "steps": [
                ("Twitter", 4096, 80),
                ("TikTok", 1080, 72),
            ],
        },
        "tiktok_screenshot_twitter": {
            "label": "🎵 TikTok → Screenshot → Twitter",
            "steps": [
                ("TikTok", 1080, 72),
                ("Screenshot", 1920, 50),
                ("Twitter", 4096, 80),
            ],
        },
    }

    results = {}
    for key, chain in chains.items():
        if key == "original":
            results[key] = {
                "label": chain["label"],
                "fake_prob": orig_prob,
                "verdict": "FAKE" if orig_prob > 0.5 else "REAL",
                "steps": "Orijinal",
            }
            continue

        img = image.copy()
        step_names = []
        for step_name, max_dim, quality in chain["steps"]:
            img = resize_compress(img, max_dim, quality)
            step_names.append(f"{step_name}(Q={quality})")

        rgb, freq, mesh = _prepare_inputs(img, model)
        with torch.no_grad():
            logits = model(rgb, freq, mesh)
            probs = F.softmax(logits, dim=1)[0]

        fp = float(probs[1])
        results[key] = {
            "label": chain["label"],
            "fake_prob": fp,
            "verdict": "FAKE" if fp > 0.5 else "REAL",
            "shift": fp - orig_prob,
            "steps": " → ".join(step_names),
        }

    return results

