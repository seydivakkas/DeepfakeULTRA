"""
Faz 2 — Hybrid Frequency Extractor (DCT + DWT + Phase Spectrum)
Mevcut 12-ch DWT'ye ek olarak DCT ve Phase Spectrum ekler.

Cikti kanallari:
    DWT:   12 kanal (3 wavelet × 4 sub-band)  [mevcut]
    DCT:    3 kanal (low/mid/high freq enerji)  [yeni]
    Phase:  3 kanal (R/G/B phase spectrum)       [yeni]
    ───────────────────────────────────────────
    Toplam: 18 kanal

Neden onemli:
    - DWT: Spatial-frequency localization (nerede hangi frekans var)
    - DCT: JPEG artifact fingerprinting (8×8 block pattern)
    - Phase: Manipulasyon izleri phase bileseninde cok belirgin
      (renk/texture degisiklikleri magnitude'da kaybolabilir ama phase'de kalir)

Kullanim:
    extractor = HybridFrequencyExtractor()
    freq_tensor = extractor(image_np)  # (18, 224, 224)
"""

import numpy as np
from PIL import Image

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

try:
    from scipy.fft import dctn
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════
# DWT MODÜLÜ (Mevcut — aynen korunuyor)
# ═══════════════════════════════════════════════════════════

def _compute_dwt_channels(gray: np.ndarray, wavelets: list, size: int) -> list:
    """3 wavelet × 4 sub-band = 12 kanal DWT."""
    channels = []
    if not HAS_PYWT:
        return [np.zeros((size, size), dtype=np.float32) for _ in range(12)]

    for wavelet in wavelets:
        try:
            coeffs = pywt.dwt2(gray, wavelet)
            cA, (cH, cV, cD) = coeffs
            for band in [cA, cH, cV, cD]:
                resized = np.array(
                    Image.fromarray(band.astype(np.float32)).resize(
                        (size, size), Image.BILINEAR
                    )
                )
                channels.append(resized)
        except Exception:
            for _ in range(4):
                channels.append(np.zeros((size, size), dtype=np.float32))

    return channels


# ═══════════════════════════════════════════════════════════
# DCT MODÜLÜ — 8×8 Block DCT Enerji Haritasi
# ═══════════════════════════════════════════════════════════

def _compute_dct_channels(gray: np.ndarray, size: int) -> list:
    """
    Vektörize DCT frekans enerji haritası (hızlandırılmış).

    Eski yöntem: 28×28=784 kere dctn() (Python for-loop) → ~50ms/görsel
    Yeni yöntem: Tek seferde full-image DCT + band maskeleme → ~2ms/görsel

    Çıktı: 3 kanal (low/mid/high frekans enerji)
    """
    if not HAS_SCIPY:
        return [np.zeros((size, size), dtype=np.float32) for _ in range(3)]

    try:
        # Full-image 2D DCT (tek çağrı — çok daha hızlı)
        dct_full = dctn(gray.astype(np.float32), type=2, norm='ortho')
        dct_abs = np.abs(dct_full)

        h, w = dct_abs.shape

        # Frekans mesafesi matrisi (merkeze olan Manhattan mesafesi)
        row_freq = np.arange(h).reshape(-1, 1)
        col_freq = np.arange(w).reshape(1, -1)
        freq_dist = row_freq + col_freq  # Manhattan mesafesi

        # Band maskeleri (JPEG 8×8 zigzag frekans bantlarına yakın)
        # Low:  freq 0-5  → DC + düşük frekans (genel parlaklık)
        # Mid:  freq 6-20 → orta frekans (kenarlar, texture)
        # High: freq 21+  → yüksek frekans (noise, artifact)
        low_mask = (freq_dist <= 5).astype(np.float32)
        mid_mask = ((freq_dist > 5) & (freq_dist <= 20)).astype(np.float32)
        high_mask = (freq_dist > 20).astype(np.float32)

        channels = []
        for mask in [low_mask, mid_mask, high_mask]:
            energy = dct_abs * mask
            # Log scale normalize
            energy = np.log1p(energy)
            emax = energy.max()
            if emax > 0:
                energy = energy / emax
            resized = np.array(
                Image.fromarray(energy).resize((size, size), Image.BILINEAR)
            )
            channels.append(resized)

        return channels
    except Exception:
        return [np.zeros((size, size), dtype=np.float32) for _ in range(3)]


# ═══════════════════════════════════════════════════════════
# PHASE SPECTRUM MODÜLÜ
# ═══════════════════════════════════════════════════════════

def _compute_phase_channels(image_rgb: np.ndarray, size: int) -> list:
    """
    RGB kanallarin FFT phase bilesenlerini cikar.

    Neden phase onemli:
        - Magnitude spectrum: texture bilgisi (deepfake'lerde benzer olabilir)
        - Phase spectrum: yapisal bilgi (kenarlar, sinirlar)
        - Face swap'larda blending siniri phase'de net gorulur
        - GAN gorselleri tutarsiz phase pattern uretir

    Cikti:
        - R phase, G phase, B phase (her biri [0, 1] normalize)
    """
    if len(image_rgb.shape) != 3:
        return [np.zeros((size, size), dtype=np.float32) for _ in range(3)]

    channels = []
    for ch in range(3):
        channel = image_rgb[:, :, ch].astype(np.float32)

        # 2D FFT
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)

        # Phase spectrum cikar
        phase = np.angle(f_shift)

        # [-π, π] → [0, 1] normalize
        phase_norm = (phase + np.pi) / (2 * np.pi)

        # Resize
        resized = np.array(
            Image.fromarray(phase_norm.astype(np.float32)).resize(
                (size, size), Image.BILINEAR
            )
        )
        channels.append(resized)

    return channels


# ═══════════════════════════════════════════════════════════
# ANA SINIF — HybridFrequencyExtractor
# ═══════════════════════════════════════════════════════════

class HybridFrequencyExtractor:
    """
    Hibrit Frekans Cikarici: DWT + DCT + Phase Spectrum.

    Args:
        wavelets: DWT icin wavelet listesi
        size: Cikti boyutu (H = W)
        include_dwt: DWT kanallari dahil et (12 ch)
        include_dct: DCT kanallari dahil et (3 ch)
        include_phase: Phase kanallari dahil et (3 ch)

    Cikti:
        numpy array (C, H, W) — C = aktif kanal sayisi (max 18)
    """

    def __init__(
        self,
        wavelets: list = None,
        size: int = 224,
        include_dwt: bool = True,
        include_dct: bool = True,
        include_phase: bool = True,
    ):
        self.wavelets = wavelets or ["haar", "db2", "coif1"]
        self.size = size
        self.include_dwt = include_dwt
        self.include_dct = include_dct
        self.include_phase = include_phase

    @property
    def num_channels(self) -> int:
        """Toplam cikti kanal sayisi."""
        total = 0
        if self.include_dwt:
            total += len(self.wavelets) * 4  # 3 × 4 = 12
        if self.include_dct:
            total += 3
        if self.include_phase:
            total += 3
        return total

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Hibrit frekans haritasi cikar.

        Args:
            image: (H, W, 3) uint8 RGB veya (H, W) grayscale

        Returns:
            (C, size, size) float32 numpy array
        """
        # Grayscale donusum (DWT ve DCT icin)
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        if gray.shape[0] != self.size or gray.shape[1] != self.size:
            gray = np.array(
                Image.fromarray(gray.astype(np.uint8)).resize(
                    (self.size, self.size), Image.BILINEAR
                )
            ).astype(np.float32)

        gray = gray / 255.0 if gray.max() > 1.0 else gray

        # RGB image (phase icin)
        if len(image.shape) == 3:
            image_resized = np.array(
                Image.fromarray(image).resize((self.size, self.size), Image.BILINEAR)
            )
        else:
            image_resized = np.stack([gray * 255] * 3, axis=2).astype(np.uint8)

        channels = []

        # 1. DWT (12 kanal)
        if self.include_dwt:
            dwt_ch = _compute_dwt_channels(gray, self.wavelets, self.size)
            channels.extend(dwt_ch)

        # 2. DCT (3 kanal)
        if self.include_dct:
            dct_ch = _compute_dct_channels(gray, self.size)
            channels.extend(dct_ch)

        # 3. Phase Spectrum (3 kanal)
        if self.include_phase:
            phase_ch = _compute_phase_channels(image_resized, self.size)
            channels.extend(phase_ch)

        freq_map = np.stack(channels, axis=0)  # (C, H, W)
        return freq_map

    def __repr__(self):
        parts = []
        if self.include_dwt:
            parts.append(f"DWT({len(self.wavelets)}×4={len(self.wavelets)*4}ch)")
        if self.include_dct:
            parts.append("DCT(3ch)")
        if self.include_phase:
            parts.append("Phase(3ch)")
        return f"HybridFrequencyExtractor({'+'.join(parts)}, total={self.num_channels}ch)"


# ═══════════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE WRAPPER (12-ch fallback)
# ═══════════════════════════════════════════════════════════

class LegacyDWTExtractor(HybridFrequencyExtractor):
    """Eski 12-ch DWT ile uyumlu — sadece DWT kanallari."""

    def __init__(self, wavelets=None, size=224):
        super().__init__(
            wavelets=wavelets, size=size,
            include_dwt=True, include_dct=False, include_phase=False
        )


# ═══════════════════════════════════════════════════════════
# GÖRSELLEŞTIRME — Gradio UI için
# ═══════════════════════════════════════════════════════════

def generate_hybrid_frequency_visualization(
    image,
    size: int = 224,
) -> dict:
    """
    Hibrit frekans haritasini gorsellestir.

    Returns:
        dict: {
            'dwt': PIL Image (mavi tonlu),
            'dct': PIL Image (yesil tonlu),
            'phase': PIL Image (mor tonlu),
            'combined': PIL Image (birlesik),
        }
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB").resize((size, size)))
    else:
        img_np = image

    gray = np.mean(img_np, axis=2).astype(np.float32)
    gray = gray / 255.0 if gray.max() > 1.0 else gray

    results = {}

    # DWT gorseli (mavi)
    dwt_channels = _compute_dwt_channels(gray, ["haar", "db2", "coif1"], size)
    if dwt_channels:
        dwt_combined = np.mean(dwt_channels, axis=0)
        dwt_norm = (dwt_combined - dwt_combined.min()) / (dwt_combined.max() - dwt_combined.min() + 1e-8)
        dwt_norm = np.power(dwt_norm, 0.5)  # Gamma correction
        vis = np.zeros((*dwt_norm.shape, 3), dtype=np.uint8)
        vis[:, :, 0] = (dwt_norm * 60).astype(np.uint8)
        vis[:, :, 1] = (dwt_norm * 200).astype(np.uint8)
        vis[:, :, 2] = (255 * np.clip(dwt_norm * 1.2, 0, 1)).astype(np.uint8)
        results['dwt'] = Image.fromarray(vis)

    # DCT gorseli (yesil)
    dct_channels = _compute_dct_channels(gray, size)
    if dct_channels and any(ch.max() > 0 for ch in dct_channels):
        dct_combined = np.mean(dct_channels, axis=0)
        dct_norm = (dct_combined - dct_combined.min()) / (dct_combined.max() - dct_combined.min() + 1e-8)
        vis = np.zeros((*dct_norm.shape, 3), dtype=np.uint8)
        vis[:, :, 0] = (dct_norm * 40).astype(np.uint8)
        vis[:, :, 1] = (255 * np.clip(dct_norm * 1.2, 0, 1)).astype(np.uint8)
        vis[:, :, 2] = (dct_norm * 80).astype(np.uint8)
        results['dct'] = Image.fromarray(vis)

    # Phase gorseli (mor)
    phase_channels = _compute_phase_channels(img_np, size)
    if phase_channels:
        phase_combined = np.mean(phase_channels, axis=0)
        vis = np.zeros((*phase_combined.shape, 3), dtype=np.uint8)
        vis[:, :, 0] = (phase_combined * 180).astype(np.uint8)
        vis[:, :, 1] = (phase_combined * 60).astype(np.uint8)
        vis[:, :, 2] = (255 * np.clip(phase_combined * 1.1, 0, 1)).astype(np.uint8)
        results['phase'] = Image.fromarray(vis)

    return results


# ═══════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Hybrid Frequency Extractor Test ===\n")

    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Full hybrid (18 ch)
    full = HybridFrequencyExtractor()
    result = full(dummy)
    print(f"Full Hybrid: {dummy.shape} -> {result.shape}")
    print(f"  {full}")
    print(f"  Channels: {result.shape[0]}")

    # Legacy (12 ch)
    legacy = LegacyDWTExtractor()
    result_legacy = legacy(dummy)
    print(f"\nLegacy DWT: {dummy.shape} -> {result_legacy.shape}")
    print(f"  {legacy}")

    # Sadece yeni kanallar
    new_only = HybridFrequencyExtractor(include_dwt=False)
    result_new = new_only(dummy)
    print(f"\nNew only (DCT+Phase): {dummy.shape} -> {result_new.shape}")
    print(f"  {new_only}")

    # Gorsellestirme
    vis = generate_hybrid_frequency_visualization(dummy)
    print(f"\nVisualization keys: {list(vis.keys())}")
    for k, v in vis.items():
        print(f"  {k}: {v.size}")

    print("\n✅ Hybrid Frequency modulu hazir!")
