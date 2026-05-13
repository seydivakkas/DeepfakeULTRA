"""
Deepfake Detection System v4 — Hiyerarşik Alt-Tip Sınıflandırıcı
FAKE tespiti sonrası alt-tip belirleme: Digital / Physical / AI-Generated.

Yöntemler:
    1. Kaynak ipucu (source_hint) → doğrudan eşleme
    2. Frekans analizi (DWT) → dijital manipülasyon artifact'ları
    3. Texture analizi (LBP) → fiziksel saldırı desenleri
"""

import numpy as np
from typing import Optional

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ═══════════════════════════════════════════════════════════
# FREKANS ANALİZÖRÜ
# ═══════════════════════════════════════════════════════════
class FrequencyAnalyzer:
    """
    DWT tabanlı frekans profili çıkarma.
    Dijital manipülasyonlar yüksek frekans bandında belirgin artifact bırakır.
    AI-sentezli görüntüler ise düşük frekans baskınlığı gösterir.
    """

    # Eşik değerleri — calibrasyon sonrası ayarlanabilir
    HIGH_FREQ_THRESHOLD = 0.15
    LOW_FREQ_RATIO_THRESHOLD = 3.0

    def analyze(self, image: np.ndarray) -> dict:
        """
        Frekans profilini çıkar.

        Args:
            image: (H, W, 3) uint8 RGB görüntü

        Returns:
            dict: {
                "high_freq_energy": float,  — yüksek frekans enerjisi
                "low_freq_energy": float,   — düşük frekans enerjisi
                "freq_ratio": float,        — düşük/yüksek oranı
                "is_digital": bool,         — dijital manipülasyon artifact'ı var mı
                "is_ai_generated": bool,    — AI sentezi özelliği var mı
            }
        """
        if not HAS_PYWT:
            return self._default_result()

        # Gri tonlamaya çevir
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        gray = gray / 255.0 if gray.max() > 1.0 else gray

        try:
            coeffs = pywt.dwt2(gray, "haar")
            cA, (cH, cV, cD) = coeffs

            # Enerji hesaplama
            low_freq_energy = float(np.mean(cA ** 2))
            high_freq_energy = float(np.mean(cH ** 2) + np.mean(cV ** 2) + np.mean(cD ** 2))

            # Oran
            freq_ratio = low_freq_energy / max(high_freq_energy, 1e-8)

            return {
                "high_freq_energy": high_freq_energy,
                "low_freq_energy": low_freq_energy,
                "freq_ratio": freq_ratio,
                "is_digital": high_freq_energy > self.HIGH_FREQ_THRESHOLD,
                "is_ai_generated": freq_ratio > self.LOW_FREQ_RATIO_THRESHOLD,
            }
        except Exception:
            return self._default_result()

    def _default_result(self) -> dict:
        return {
            "high_freq_energy": 0.0,
            "low_freq_energy": 0.0,
            "freq_ratio": 1.0,
            "is_digital": False,
            "is_ai_generated": False,
        }


# ═══════════════════════════════════════════════════════════
# TEXTURE ANALİZÖRÜ
# ═══════════════════════════════════════════════════════════
class TextureAnalyzer:
    """
    LBP (Local Binary Pattern) + Moiré pattern tespiti.
    Fiziksel saldırılar (baskı/ekran) düzenli doku desenleri üretir.
    """

    MOIRE_THRESHOLD = 0.12
    TEXTURE_UNIFORMITY_THRESHOLD = 0.25

    def analyze(self, image: np.ndarray) -> dict:
        """
        Texture profilini çıkar.

        Args:
            image: (H, W, 3) uint8 RGB

        Returns:
            dict: {
                "moire_score": float,           — Moiré desen skoru
                "texture_uniformity": float,    — doku düzenliliği
                "is_physical": bool,            — fiziksel saldırı ipucu
            }
        """
        if not HAS_CV2:
            return self._default_result()

        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Moiré tespiti: frekans alanında periyodik desen arama
            moire_score = self._detect_moire(gray)

            # Texture uniformity: LBP benzeri basit histogram analizi
            texture_uniformity = self._compute_texture_uniformity(gray)

            is_physical = (
                moire_score > self.MOIRE_THRESHOLD
                or texture_uniformity > self.TEXTURE_UNIFORMITY_THRESHOLD
            )

            return {
                "moire_score": float(moire_score),
                "texture_uniformity": float(texture_uniformity),
                "is_physical": is_physical,
            }
        except Exception:
            return self._default_result()

    def _detect_moire(self, gray: np.ndarray) -> float:
        """FFT tabanlı Moiré desen tespiti."""
        h, w = gray.shape
        # FFT uygula
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Merkez bölgeyi maskele (DC bileşen)
        center_h, center_w = h // 2, w // 2
        mask_radius = min(h, w) // 10
        magnitude[
            center_h - mask_radius:center_h + mask_radius,
            center_w - mask_radius:center_w + mask_radius
        ] = 0

        # Orta-yüksek frekans bandındaki pik oranı
        total_energy = np.sum(magnitude)
        if total_energy < 1e-8:
            return 0.0

        # Periyodik pikleri tespit: üst %5'teki piklerin oranı
        threshold = np.percentile(magnitude, 95)
        peak_energy = np.sum(magnitude[magnitude > threshold])
        moire_score = peak_energy / total_energy

        return moire_score

    def _compute_texture_uniformity(self, gray: np.ndarray) -> float:
        """Basit Laplacian varyansı tabanlı texture ölçümü."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.var(laplacian)
        # Düşük varyans = düz/pürüzsüz doku (baskı veya AI üretimi)
        # Yüksek varyans = doğal doku
        # Normalize: fiziksel saldırılarda orta düzey varyans görülür
        normalized = 1.0 / (1.0 + variance / 1000.0)
        return normalized

    def _default_result(self) -> dict:
        return {
            "moire_score": 0.0,
            "texture_uniformity": 0.0,
            "is_physical": False,
        }


# ═══════════════════════════════════════════════════════════
# HİYERARŞİK ALT-TİP SINIFLANDIRICI
# ═══════════════════════════════════════════════════════════
class SubtypeClassifier:
    """
    FAKE olarak tespit edilen görüntülerin alt-tipini belirler.

    Karar hiyerarşisi:
        1. source_hint varsa → doğrudan eşleme (en güvenilir)
        2. Texture analizi → fiziksel saldırı tespiti (Moiré/baskı)
        3. Frekans analizi → dijital vs. AI-sentez ayrımı
    """

    # Kaynak ipucu eşlemeleri
    SOURCE_HINT_MAP = {
        "webcam": "physical",
        "live": "physical",
        "upload": "digital",
        "browser_extension": "digital",
    }

    def __init__(self):
        self.freq_analyzer = FrequencyAnalyzer()
        self.texture_analyzer = TextureAnalyzer()

    def classify(
        self,
        image: np.ndarray,
        source_hint: Optional[str] = None,
    ) -> dict:
        """
        FAKE görüntünün alt-tipini belirle.

        Args:
            image: (H, W, 3) uint8 RGB
            source_hint: "webcam", "upload", "live", "browser_extension" vb.

        Returns:
            dict: {
                "subtype": str,        — "digital" | "physical" | "ai_generated"
                "confidence": float,   — alt-tip güven skoru (0-1)
                "method": str,         — karar yöntemi
                "freq_analysis": dict, — frekans detayları
                "texture_analysis": dict, — texture detayları
            }
        """
        # Yöntem 1: Kaynak ipucu → doğrudan eşleme
        if source_hint and source_hint in self.SOURCE_HINT_MAP:
            return {
                "subtype": self.SOURCE_HINT_MAP[source_hint],
                "confidence": 0.95,
                "method": "source_hint",
                "freq_analysis": {},
                "texture_analysis": {},
            }

        # Analizleri çalıştır
        freq_result = self.freq_analyzer.analyze(image)
        texture_result = self.texture_analyzer.analyze(image)

        # Yöntem 2: Fiziksel saldırı tespiti
        if texture_result["is_physical"]:
            confidence = min(0.85, 0.5 + texture_result["moire_score"] * 3)
            return {
                "subtype": "physical",
                "confidence": confidence,
                "method": "texture_analysis",
                "freq_analysis": freq_result,
                "texture_analysis": texture_result,
            }

        # Yöntem 3: Dijital manipülasyon tespiti
        if freq_result["is_digital"]:
            confidence = min(0.80, 0.5 + freq_result["high_freq_energy"] * 2)
            return {
                "subtype": "digital",
                "confidence": confidence,
                "method": "frequency_analysis",
                "freq_analysis": freq_result,
                "texture_analysis": texture_result,
            }

        # Yöntem 4: AI sentezi (düşük artifact profili)
        if freq_result["is_ai_generated"]:
            confidence = min(0.75, 0.4 + (1.0 / (1.0 + freq_result["freq_ratio"])) * 2)
            return {
                "subtype": "ai_generated",
                "confidence": confidence,
                "method": "frequency_ratio",
                "freq_analysis": freq_result,
                "texture_analysis": texture_result,
            }

        # Fallback: belirsiz → digital (en yaygın)
        return {
            "subtype": "digital",
            "confidence": 0.40,
            "method": "fallback",
            "freq_analysis": freq_result,
            "texture_analysis": texture_result,
        }


if __name__ == "__main__":
    print("SubtypeClassifier testi...")
    clf = SubtypeClassifier()

    # Rastgele görüntü testi
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Source hint ile
    result = clf.classify(dummy, source_hint="webcam")
    print(f"  webcam hint: {result['subtype']} (conf={result['confidence']:.2f})")

    # Source hint olmadan
    result = clf.classify(dummy)
    print(f"  Analiz sonucu: {result['subtype']} (conf={result['confidence']:.2f}, method={result['method']})")
