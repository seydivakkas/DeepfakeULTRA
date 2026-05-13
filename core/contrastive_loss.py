"""
Faz 2 — Contrastive Learning: Triplet Loss + Hard Negative Mining
REAL/FAKE embedding ayrımını keskinlestiren loss fonksiyonu.

Mantik:
    - Anchor:   Mevcut ornegin embedding vektoru
    - Positive: Ayni siniftan (REAL<->REAL veya FAKE<->FAKE)
    - Negative: Karsi siniftan (REAL<->FAKE)
    - Hard Negative Mining: En yakin yanlis sinif ornegini sec

    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    → REAL ve FAKE embedding'lerini birbirinden uzaklastirir
    → Sinif icindeki ornekleri yakinlastirir
    → Daha ayirt edici embedding uzayi olusturur

Kullanim:
    contrastive_loss = TripletContrastiveLoss(margin=1.0)
    loss = contrastive_loss(embeddings, labels)

    # Combined loss ile:
    combined = CombinedLossV2(focal_weight=0.7, triplet_weight=0.3)
    loss_dict = combined(logits, labels, embeddings=embeddings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletContrastiveLoss(nn.Module):
    """
    Triplet Loss ile REAL/FAKE embedding ayrimi.

    Hard Negative Mining:
        - Her anchor icin, en YAKIN karsi-sinif ornegini negative olarak sec
        - Bu, modeli "zor" orneklere odaklanmaya zorlar
        - Kolay negative'ler sifir loss uretir (zaten ayristirilmis)

    Args:
        margin: Triplet loss margin (varsayilan: 1.0)
        distance: 'euclidean' veya 'cosine'
        mining_strategy: 'hard' (en zor), 'semi-hard', 'all'
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance: str = "euclidean",
        mining_strategy: str = "hard",
    ):
        super().__init__()
        self.margin = margin
        self.distance = distance
        self.mining_strategy = mining_strategy

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Batch icindeki tum ciftler arasi mesafe matrisi.

        Args:
            embeddings: (N, D) — normalize edilmis embedding'ler

        Returns:
            (N, N) mesafe matrisi
        """
        if self.distance == "cosine":
            # Cosine distance = 1 - cosine_similarity
            normed = F.normalize(embeddings, p=2, dim=1)
            sim = torch.mm(normed, normed.t())
            return 1 - sim
        else:
            # Euclidean distance
            dot = torch.mm(embeddings, embeddings.t())
            sq_norms = torch.diag(dot)
            distances = sq_norms.unsqueeze(0) - 2 * dot + sq_norms.unsqueeze(1)
            distances = torch.clamp(distances, min=0.0)
            return torch.sqrt(distances + 1e-8)

    def _mine_triplets_hard(
        self,
        dist_mat: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """Hard Negative Mining — en zor triplet'leri sec.

        Her anchor icin:
            - Positive: Ayni siniftan EN UZAK ornek
            - Negative: Karsi siniftan EN YAKIN ornek
        """
        batch_size = labels.size(0)
        device = labels.device

        # Mask matrisleri
        # same_class[i,j] = True eger labels[i] == labels[j]
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        diff_class = ~same_class

        # Kendi kendine eslesmeyi engelle
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Hard positive: ayni siniftan en uzak
        pos_mask = same_class & ~eye_mask
        pos_dist = dist_mat.clone()
        pos_dist[~pos_mask] = 0.0  # Gecersizleri sifirla
        hardest_pos, _ = pos_dist.max(dim=1)  # (N,)

        # Hard negative: karsi siniftan en yakin
        neg_dist = dist_mat.clone()
        neg_dist[~diff_class] = float('inf')  # Ayni sinifi sonsuza gonder
        hardest_neg, _ = neg_dist.min(dim=1)  # (N,)

        return hardest_pos, hardest_neg

    def _mine_triplets_semihard(
        self,
        dist_mat: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """Semi-Hard Mining — d(a,p) < d(a,n) < d(a,p) + margin."""
        batch_size = labels.size(0)
        device = labels.device

        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff_class = ~same_class
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Positive
        pos_mask = same_class & ~eye_mask
        pos_dist = dist_mat.clone()
        pos_dist[~pos_mask] = 0.0
        hardest_pos, _ = pos_dist.max(dim=1)

        # Semi-hard negative: d(a,p) < d(a,n) < d(a,p) + margin
        neg_dist = dist_mat.clone()
        neg_dist[~diff_class] = float('inf')

        # Semi-hard filtre
        semi_hard_mask = (neg_dist > hardest_pos.unsqueeze(1)) & \
                         (neg_dist < hardest_pos.unsqueeze(1) + self.margin)

        neg_dist_sh = dist_mat.clone()
        neg_dist_sh[~(diff_class & semi_hard_mask)] = float('inf')
        hardest_neg, _ = neg_dist_sh.min(dim=1)

        # Semi-hard bulunamayanlar icin hard negative'e fallback
        no_semihard = hardest_neg == float('inf')
        if no_semihard.any():
            neg_dist_fallback = dist_mat.clone()
            neg_dist_fallback[~diff_class] = float('inf')
            fallback_neg, _ = neg_dist_fallback.min(dim=1)
            hardest_neg[no_semihard] = fallback_neg[no_semihard]

        return hardest_pos, hardest_neg

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triplet loss hesapla.

        Args:
            embeddings: (N, D) — model embedding ciktisi
            labels: (N,) — sinif etiketleri (0=REAL, 1=FAKE)

        Returns:
            scalar loss tensor
        """
        # En az 2 sinif olmali
        unique_labels = labels.unique()
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Pairwise mesafe matrisi
        dist_mat = self._pairwise_distances(embeddings)

        # Triplet mining
        if self.mining_strategy == "hard":
            pos_dist, neg_dist = self._mine_triplets_hard(dist_mat, labels)
        elif self.mining_strategy == "semi-hard":
            pos_dist, neg_dist = self._mine_triplets_semihard(dist_mat, labels)
        else:
            # All triplets
            pos_dist, neg_dist = self._mine_triplets_hard(dist_mat, labels)

        # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        losses = F.relu(pos_dist - neg_dist + self.margin)

        # Gecerli triplet'lerin ortalamasi (sifir olmayanlar)
        valid_triplets = losses > 0
        if valid_triplets.sum() > 0:
            loss = losses[valid_triplets].mean()
        else:
            loss = losses.mean()

        return loss

    def __repr__(self):
        return (f"TripletContrastiveLoss("
                f"margin={self.margin}, "
                f"distance={self.distance}, "
                f"mining={self.mining_strategy})")


class CombinedLossV2(nn.Module):
    """
    Focal Loss + Triplet Contrastive Loss birlesimi.

    L_total = α × L_focal + β × L_triplet

    Args:
        focal_weight: Focal loss agirligi (α)
        triplet_weight: Triplet loss agirligi (β)
        triplet_margin: Triplet loss margin
        focal_gamma: Focal loss gamma
        label_smoothing: Label smoothing orani
        class_weights: Sinif agirliklari
    """

    def __init__(
        self,
        focal_weight: float = 0.7,
        triplet_weight: float = 0.3,
        triplet_margin: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        class_weights: list = None,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.triplet_weight = triplet_weight

        # Mevcut FocalLoss'u import etmek yerine burada yeniden tanimla
        # (aktif egitim dosyalarina bagimlilik olmamasi icin)
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        if class_weights is None:
            class_weights = [1.0, 1.0]
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

        self.triplet_loss = TripletContrastiveLoss(
            margin=triplet_margin,
            distance="cosine",
            mining_strategy="hard",
        )

    def _focal_loss(self, logits, targets):
        """Inline Focal Loss (label smoothing dahil)."""
        weights = self.class_weights.to(logits.device)
        nc = logits.size(1)

        with torch.no_grad():
            smooth_val = self.label_smoothing / max(nc - 1, 1)
            st = torch.full_like(logits, smooth_val)
            st.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)
        fw = (1 - p) ** self.focal_gamma
        aw = weights[targets].unsqueeze(1).expand_as(logits)
        per_class_loss = -aw * fw * st * log_p
        return per_class_loss.sum(dim=1).mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        embeddings: torch.Tensor = None,
        teacher_logits: torch.Tensor = None,
    ) -> dict:
        """
        Combined loss hesapla.

        Args:
            logits: (N, C) — model sinif logit'leri
            targets: (N,) — sinif etiketleri
            embeddings: (N, D) — model embedding ciktisi (contrastive icin)
            teacher_logits: Kullanilmiyor (uyumluluk icin)

        Returns:
            dict: {total_loss, focal_loss, triplet_loss, kd_loss}
        """
        focal = self._focal_loss(logits, targets)

        triplet = torch.tensor(0.0, device=logits.device)
        if embeddings is not None and self.triplet_weight > 0:
            triplet = self.triplet_loss(embeddings, targets)

        total = self.focal_weight * focal + self.triplet_weight * triplet

        return {
            "total_loss": total,
            "focal_loss": focal,
            "triplet_loss": triplet,
            "kd_loss": torch.tensor(0.0, device=logits.device),
        }

    def __repr__(self):
        return (f"CombinedLossV2("
                f"focal_w={self.focal_weight}, "
                f"triplet_w={self.triplet_weight}, "
                f"gamma={self.focal_gamma}, "
                f"smooth={self.label_smoothing})")


# ═══════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Contrastive Loss Test ===\n")

    batch_size = 16
    embed_dim = 256
    num_classes = 2

    # Dummy data
    embeddings = torch.randn(batch_size, embed_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    logits = torch.randn(batch_size, num_classes)

    # Triplet Loss
    triplet = TripletContrastiveLoss(margin=1.0, distance="euclidean")
    t_loss = triplet(embeddings, labels)
    print(f"Triplet Loss (euclidean): {t_loss.item():.4f}")

    triplet_cos = TripletContrastiveLoss(margin=0.5, distance="cosine")
    tc_loss = triplet_cos(embeddings, labels)
    print(f"Triplet Loss (cosine):    {tc_loss.item():.4f}")

    # Semi-hard mining
    triplet_sh = TripletContrastiveLoss(margin=1.0, mining_strategy="semi-hard")
    sh_loss = triplet_sh(embeddings, labels)
    print(f"Triplet Loss (semi-hard): {sh_loss.item():.4f}")

    # CombinedLossV2
    combined = CombinedLossV2(focal_weight=0.7, triplet_weight=0.3)
    result = combined(logits, labels, embeddings=embeddings)
    print(f"\nCombined Loss V2:")
    print(f"  Total:   {result['total_loss'].item():.4f}")
    print(f"  Focal:   {result['focal_loss'].item():.4f}")
    print(f"  Triplet: {result['triplet_loss'].item():.4f}")

    # Gradient check
    embeddings.requires_grad_(True)
    t_loss = triplet(embeddings, labels)
    t_loss.backward()
    print(f"\n✅ Gradient OK — grad norm: {embeddings.grad.norm().item():.4f}")

    print(f"\n✅ Contrastive Loss modulu hazir!")
