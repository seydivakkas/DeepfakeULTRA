"""
Deepfake Detection System v5 — Loss Fonksiyonları
Binary FocalLoss (REAL/FAKE), Knowledge Distillation Loss, Triplet Contrastive Loss.
Run 5: L = focal_w × L_focal + kd_w × L_KD + contrastive_w × L_triplet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import model_cfg


class FocalLoss(nn.Module):
    """
    Focal Loss — zor örneklere odaklanma. FL(p) = -α(1-p)^γ log(p)
    Binary sınıflandırma: REAL (0) / FAKE (1)
    """
    def __init__(self, gamma=model_cfg.FOCAL_GAMMA,
                 class_weights=None,
                 label_smoothing=model_cfg.LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is None:
            class_weights = model_cfg.CLASS_WEIGHTS
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

    def forward(self, logits, targets):
        weights = self.class_weights.to(logits.device)

        nc = logits.size(1)
        with torch.no_grad():
            smooth_val = self.label_smoothing / max(nc - 1, 1)
            st = torch.full_like(logits, smooth_val)
            st.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)
        fw = (1 - p) ** self.gamma
        aw = weights[targets].unsqueeze(1).expand_as(logits)

        per_class_loss = -aw * fw * st * log_p
        return per_class_loss.sum(dim=1).mean()


class KnowledgeDistillationLoss(nn.Module):
    """KD Loss — T² × KL(softmax(z_s/T) || softmax(z_t/T))"""
    def __init__(self, temperature=model_cfg.KD_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        T = self.temperature
        s_soft = F.log_softmax(student_logits / T, dim=1)
        t_soft = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(s_soft, t_soft, reduction="batchmean") * (T * T)


class CombinedLoss(nn.Module):
    """
    Focal + KD + Triplet Contrastive birleşimi (Run 5).

    Ağırlık dağılımı (USE_CONTRASTIVE=True, KD_ALPHA=0.1):
        focal_weight    = 0.70
        kd_weight       = 0.10
        contrastive_w   = 0.20
    """
    def __init__(self, alpha=model_cfg.KD_ALPHA, gamma=model_cfg.FOCAL_GAMMA,
                 temperature=model_cfg.KD_TEMPERATURE, class_weights=None):
        super().__init__()
        self.alpha = alpha

        self.use_contrastive = getattr(model_cfg, 'USE_CONTRASTIVE', False)
        self.contrastive_weight = getattr(model_cfg, 'CONTRASTIVE_WEIGHT', 0.0) if self.use_contrastive else 0.0

        kd_w = alpha if alpha > 0 else 0.0
        self.focal_weight = max(0.0, 1.0 - kd_w - self.contrastive_weight)

        self.focal_loss = FocalLoss(gamma=gamma, class_weights=class_weights)
        self.kd_loss = KnowledgeDistillationLoss(temperature=temperature)

        self.triplet_loss = None
        if self.use_contrastive:
            try:
                from core.contrastive_loss import TripletContrastiveLoss
                self.triplet_loss = TripletContrastiveLoss(
                    margin=getattr(model_cfg, 'CONTRASTIVE_MARGIN', 1.0),
                    distance=getattr(model_cfg, 'CONTRASTIVE_DISTANCE', 'cosine'),
                    mining_strategy=getattr(model_cfg, 'CONTRASTIVE_MINING', 'hard'),
                )
            except ImportError:
                self.use_contrastive = False

    def forward(self, student_logits, targets, teacher_logits=None, embeddings=None):
        focal = self.focal_loss(student_logits, targets)

        kd = torch.tensor(0.0, device=student_logits.device)
        if teacher_logits is not None and self.alpha > 0:
            kd = self.kd_loss(student_logits, teacher_logits)

        triplet = torch.tensor(0.0, device=student_logits.device)
        if self.use_contrastive and embeddings is not None and self.triplet_loss is not None:
            try:
                triplet = self.triplet_loss(embeddings, targets)
            except Exception:
                pass

        total = (self.focal_weight * focal
                 + self.alpha * kd
                 + self.contrastive_weight * triplet)

        return {
            "total_loss": total,
            "focal_loss": focal,
            "kd_loss": kd,
            "triplet_loss": triplet,
        }
