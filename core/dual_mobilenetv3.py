"""
Deepfake Detection System v5 — DualPathDeepfakeDetector
Tri-Branch (RGB + Frekans + Face Mesh) + CrossBranchTransformer füzyon mimarisi.

Mimari:
    RGB Branch  → MobileNetV3-Large (pretrained) → 960-dim
    Freq Branch → MobileNetV3-Large (18-ch input) → 960-dim
    Mesh Branch → FaceMeshMLP (468×3 → 128 → 960)
         ↓
    CrossBranchTransformer (2-layer, 4-head Self-Attention)
         ↓
    Mean Pool → 960-dim
         ↓
    Classifier (960 → 256 → 2)

Çıktı: 2 sınıf — REAL (0) / FAKE (1)  [binary]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import model_cfg, DEVICE
import math


# ═══════════════════════════════════════════════════════════
# FACE MESH MLP — 468 Landmark İşleme
# ═══════════════════════════════════════════════════════════
class FaceMeshMLP(nn.Module):
    """
    MediaPipe Face Mesh 468 3D landmark noktasını
    yoğun bir özellik vektörüne dönüştürür.
    """

    def __init__(
        self,
        input_dim: int = model_cfg.MESH_INPUT_DIM,
        hidden_dim: int = model_cfg.MESH_HIDDEN_DIM,
        output_dim: int = model_cfg.MESH_OUTPUT_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1404) — 468 × 3 landmark koordinatları
        Returns:
            (batch, output_dim) — özellik vektörü
        """
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# CROSS-BRANCH TRANSFORMER — Dallar Arası Dikkat Mekanizması
# ═══════════════════════════════════════════════════════════
class CrossBranchTransformer(nn.Module):
    """
    3 branch çıktısını (RGB, Freq, Mesh) Transformer Self-Attention
    ile birleştirir. BiLSTM + LearnableFusion yerine — tek görsel
    analizine optimize edilmiş cross-modal dikkat mekanizması.

    Her branch çıktısı bir token olarak ele alınır (seq_len=3).
    Self-Attention, hangi branch'in hangi durumda daha bilgilendirici
    olduğunu öğrenir.

    Neden BiLSTM değil:
        - BiLSTM temporal (sıralı) veri için tasarlanmıştır
        - Tek görsel analizinde seq_len=1 → BiLSTM boş çalışır
        - Transformer, 3 branch arasındaki cross-modal ilişkileri
          paralel olarak öğrenir — daha verimli ve anlamlı
    """

    def __init__(
        self,
        feature_dim: int = model_cfg.FUSION_DIM,
        num_heads: int = model_cfg.XBRANCH_HEADS,
        num_layers: int = model_cfg.XBRANCH_LAYERS,
        dropout: float = model_cfg.XBRANCH_DROPOUT,
        ff_mult: int = model_cfg.XBRANCH_FF_MULT,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Öğrenilebilir branch tipi gömmesi (positional encoding yerine)
        # RGB=0, Freq=1, Mesh=2 → her branch'e kimlik kazandırır
        self.branch_embed = nn.Parameter(torch.randn(3, feature_dim) * 0.02)

        # Pre-LayerNorm Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm — daha stabil eğitim
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.final_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        freq_feat: torch.Tensor,
        mesh_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rgb_feat:  (batch, feature_dim)
            freq_feat: (batch, feature_dim)
            mesh_feat: (batch, feature_dim)
        Returns:
            (batch, feature_dim) — cross-attention sonrası birleşik vektör
        """
        # 3 branch'i token sekansı olarak yığınla: (batch, 3, feature_dim)
        x = torch.stack([rgb_feat, freq_feat, mesh_feat], dim=1)

        # Branch embedding ekle (her token kendi modalitesini bilir)
        x = x + self.branch_embed.unsqueeze(0)

        # Transformer Self-Attention: branch'ler arası çapraz dikkat
        x = self.transformer(x)  # (batch, 3, feature_dim)
        x = self.final_norm(x)

        # Mean pooling: tüm branch'lerin ağırlıklı ortalaması
        return x.mean(dim=1)  # (batch, feature_dim)


# ═══════════════════════════════════════════════════════════
# ANA MODEL — DualPathDeepfakeDetector
# ═══════════════════════════════════════════════════════════
class DualPathDeepfakeDetector(nn.Module):
    """
    Tri-Branch Deepfake Detector — Ana model.

    Üç branch (RGB, Frekans, Face Mesh) → CrossBranchTransformer →
    Classifier.
    """

    def __init__(self, num_classes: int = model_cfg.NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # ── RGB Branch: MobileNetV3-Large (pretrained) ──
        rgb_backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.rgb_features = rgb_backbone.features
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)
        rgb_out_dim = 960  # MobileNetV3-Large son katman

        # ── Frekans Branch: MobileNetV3-Large (DWT 18-ch input) ──
        freq_backbone = models.mobilenet_v3_large(weights=None)
        # İlk conv katmanını 18 kanala uyarla
        original_conv = freq_backbone.features[0][0]
        freq_backbone.features[0][0] = nn.Conv2d(
            model_cfg.DWT_CHANNELS, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        self.freq_features = freq_backbone.features
        self.freq_pool = nn.AdaptiveAvgPool2d(1)
        freq_out_dim = 960

        # ── Mesh Branch: FaceMeshMLP ──
        self.mesh_mlp = FaceMeshMLP()
        mesh_out_dim = model_cfg.MESH_OUTPUT_DIM  # 128

        # ── Branch boyutlarını hizala ──
        fusion_dim = model_cfg.FUSION_DIM  # 960 (ortak boyut)
        self.rgb_proj = nn.Identity()  # Zaten 960
        self.freq_proj = nn.Identity()  # Zaten 960
        self.mesh_proj = nn.Sequential(
            nn.Linear(mesh_out_dim, fusion_dim),
            nn.ReLU(inplace=True),
        )

        # ── CrossBranchTransformer (BiLSTM + Fusion yerine) ──
        self.cross_branch = CrossBranchTransformer(
            feature_dim=fusion_dim,
            num_heads=model_cfg.XBRANCH_HEADS,
            num_layers=model_cfg.XBRANCH_LAYERS,
            dropout=model_cfg.XBRANCH_DROPOUT,
            ff_mult=model_cfg.XBRANCH_FF_MULT,
        )

        # ── Sınıflandırıcı — binary head (REAL/FAKE) ──
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(model_cfg.CLASSIFIER_DROPOUT),
            nn.Linear(256, num_classes),
        )

        # Ağırlık başlatma
        self._init_weights()

    def _init_weights(self):
        """Xavier/He ağırlık başlatma (pretrained dışındaki katmanlar)."""
        for name, module in self.named_modules():
            if "rgb_features" in name or "freq_features" in name:
                continue
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def extract_features(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        mesh: torch.Tensor,
    ) -> tuple:
        """
        Her branch'ten özellik vektörü çıkar.
        Returns: rgb_feat, freq_feat, mesh_feat — her biri (batch, fusion_dim)
        """
        rgb_feat = self.rgb_features(rgb)
        rgb_feat = self.rgb_pool(rgb_feat).flatten(1)
        rgb_feat = self.rgb_proj(rgb_feat)

        freq_feat = self.freq_features(freq)
        freq_feat = self.freq_pool(freq_feat).flatten(1)
        freq_feat = self.freq_proj(freq_feat)

        mesh_feat = self.mesh_mlp(mesh)
        mesh_feat = self.mesh_proj(mesh_feat)

        return rgb_feat, freq_feat, mesh_feat

    def _get_attended_features(self, rgb, freq, mesh):
        """Ortak özellik çıkarım + cross-branch attention."""
        rgb_feat, freq_feat, mesh_feat = self.extract_features(rgb, freq, mesh)
        attended = self.cross_branch(rgb_feat, freq_feat, mesh_feat)
        return attended

    def forward(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        mesh: torch.Tensor,
        source_tags=None,  # Geriye uyumluluk — kullanılmaz
    ) -> torch.Tensor:
        """
        Ana forward pass (binary sınıflandırma).

        Args:
            rgb:  (batch, 3, 224, 224)
            freq: (batch, 18, 224, 224)  — DWT+DCT+Phase
            mesh: (batch, 1404)
            source_tags: kullanılmıyor (geriye uyumluluk)

        Returns:
            logits: (batch, 2) — [real, fake]
        """
        attended = self._get_attended_features(rgb, freq, mesh)
        return self.classifier(attended)

    def forward_with_embeddings(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        mesh: torch.Tensor,
    ) -> tuple:
        """
        Contrastive learning için logits + embedding döndür.

        Returns:
            (logits, embeddings): logits (batch, 2), embeddings (batch, 256)
        """
        attended = self._get_attended_features(rgb, freq, mesh)
        # classifier[0] = Linear(960, 256), classifier[1] = ReLU
        embedding = self.classifier[1](self.classifier[0](attended))  # (batch, 256)
        logits = self.classifier(attended)  # (batch, 2)
        return logits, embedding

    def forward_branches(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        mesh: torch.Tensor,
    ) -> dict:
        """
        Her branch'in ayrı çıktısını döndürür (analiz/debug amaçlı).

        Returns:
            Dict: rgb_logits, freq_logits, mesh_logits, fused_logits
        """
        rgb_feat, freq_feat, mesh_feat = self.extract_features(rgb, freq, mesh)

        # Her branch için basit sınıflandırıcı
        if not hasattr(self, "_branch_heads"):
            dim = model_cfg.FUSION_DIM
            self._branch_heads = nn.ModuleDict({
                "rgb": nn.Linear(dim, self.num_classes),
                "freq": nn.Linear(dim, self.num_classes),
                "mesh": nn.Linear(dim, self.num_classes),
            }).to(rgb.device)

        return {
            "rgb_logits": self._branch_heads["rgb"](rgb_feat),
            "freq_logits": self._branch_heads["freq"](freq_feat),
            "mesh_logits": self._branch_heads["mesh"](mesh_feat),
            "fused_logits": self.forward(rgb, freq, mesh),
        }


# ═══════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════
def count_parameters(model: nn.Module) -> dict:
    """Model parametre sayısını hesapla."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def get_model_summary(model: nn.Module) -> str:
    """Model özetini döndür."""
    params = count_parameters(model)
    return (
        f"📊 Model: DualPathDeepfakeDetector\n"
        f"   Toplam Parametre: {params['total']:,}\n"
        f"   Eğitilebilir: {params['trainable']:,}\n"
        f"   Donmuş: {params['frozen']:,}\n"
        f"   CrossBranch: {model_cfg.XBRANCH_LAYERS} layer, "
        f"{model_cfg.XBRANCH_HEADS} head\n"
        f"   Fusion Dim: {model_cfg.FUSION_DIM}"
    )


if __name__ == "__main__":
    # Hızlı model testi
    model = DualPathDeepfakeDetector().to(DEVICE)
    print(get_model_summary(model))

    # Tek frame testi
    batch = 2
    rgb = torch.randn(batch, 3, 224, 224).to(DEVICE)
    freq = torch.randn(batch, model_cfg.DWT_CHANNELS, 224, 224).to(DEVICE)
    mesh = torch.randn(batch, 1404).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(rgb, freq, mesh)
    print(f"\n✅ Tek frame: giriş → logits {logits.shape}")
    print(f"   Olasılıklar: {torch.softmax(logits, dim=1).cpu().numpy()}")

    # Embedding testi
    with torch.no_grad():
        logits, emb = model.forward_with_embeddings(rgb, freq, mesh)
    print(f"✅ Embedding: {emb.shape}")
