"""
Deepfake Detection System v3 — EfficientNet-B5 Öğretmen Model
Knowledge Distillation için soft logits üreten büyük model.

Kullanım:
    teacher = EfficientNetTeacher().to(device)
    soft_logits = teacher(rgb_images)  # → (batch, 2) soft targets
"""

import torch
import torch.nn as nn
from torchvision import models
from config import model_cfg, DEVICE


class EfficientNetTeacher(nn.Module):
    """
    EfficientNet-B5 tabanlı öğretmen model.
    Knowledge Distillation sırasında öğrenci modele (DualPathDeepfakeDetector)
    soft target sağlar.

    Parametre sayısı: ~30M (öğrencinin ~5-6x'i)
    Giriş: RGB görüntü (batch, 3, 224, 224)
    Çıkış: Logits (batch, 2) — KD loss için softmax uygulanmadan
    """

    def __init__(
        self,
        num_classes: int = model_cfg.NUM_CLASSES,
        pretrained: bool = True,
    ):
        super().__init__()

        # EfficientNet-B5 backbone
        weights = models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b5(weights=weights)

        # Feature extractor (son FC hariç)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Orijinal classifier boyutunu al
        in_features = backbone.classifier[1].in_features  # 2048

        # Yeni sınıflandırıcı head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        # Ağırlık başlatma
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 224, 224) — RGB görüntü
        Returns:
            logits: (batch, 2) — softmax uygulanmamış logits
        """
        features = self.features(x)
        pooled = self.avgpool(features)
        flat = pooled.flatten(1)
        logits = self.classifier(flat)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Özellik vektörü çıkar (KD ara katman eşleştirme için)."""
        features = self.features(x)
        return self.avgpool(features).flatten(1)


def load_pretrained_teacher(
    checkpoint_path: str = None,
    device: torch.device = DEVICE,
) -> EfficientNetTeacher:
    """
    Önceden eğitilmiş öğretmen modeli yükle.
    Checkpoint yoksa pretrained ImageNet ağırlıklarla döndür.
    """
    teacher = EfficientNetTeacher(pretrained=True).to(device)

    if checkpoint_path:
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            teacher.load_state_dict(state_dict)
            print(f"✅ Öğretmen checkpoint yüklendi: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Öğretmen checkpoint yüklenemedi: {e}. ImageNet ağırlıkları kullanılıyor.")

    teacher.eval()
    return teacher


if __name__ == "__main__":
    teacher = EfficientNetTeacher().to(DEVICE)

    # Parametre sayısı
    total = sum(p.numel() for p in teacher.parameters())
    print(f"📊 EfficientNet-B5 Öğretmen Model")
    print(f"   Parametre: {total:,}")
    print(f"   Cihaz: {DEVICE}")

    # Forward pass testi
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
    teacher.eval()
    with torch.no_grad():
        logits = teacher(dummy)
        feats = teacher.extract_features(dummy)

    print(f"\n✅ Forward pass başarılı")
    print(f"   Logits: {logits.shape}")
    print(f"   Features: {feats.shape}")
    print(f"   Soft probs: {torch.softmax(logits, dim=1).cpu().numpy()}")
