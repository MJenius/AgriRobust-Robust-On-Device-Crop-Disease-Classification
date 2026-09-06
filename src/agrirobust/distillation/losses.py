"""Knowledge distillation loss functions for AgriRobust Phase 4."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResponseKDLoss(nn.Module):
    """Response-based (logit) knowledge distillation loss using KL divergence.

    Formulation:
      p_t = softmax(z_t / T)
      log_p_s = log_softmax(z_s / T)
      L_KD = KLDiv(log_p_s, p_t, reduction='batchmean') * (T^2)
      L_total = (1 - alpha) * L_CE + alpha * L_KD
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Standard Cross-Entropy loss on ground truth
        ce = self.ce_loss(student_logits, targets)

        # Log-softmax for student, Softmax for teacher
        log_p_s = F.log_softmax(student_logits / self.temperature, dim=-1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=-1)

        kd = self.kl_div(log_p_s, p_t) * (self.temperature**2)
        total_loss = (1.0 - self.alpha) * ce + self.alpha * kd
        return total_loss, ce, kd


class FeatureHintLoss(nn.Module):
    """Feature-level distillation loss aligning projected student features to teacher features.

    Formulation:
      L_feat = MSE(W_proj(f_s), f_t)
      L_total = L_CE + beta * L_feat
    """

    def __init__(self, student_dim: int = 576, teacher_dim: int = 768, beta: float = 0.5):
        super().__init__()
        self.beta = beta
        self.projection = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.BatchNorm1d(teacher_dim),
            nn.ReLU(inplace=True),
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce = self.ce_loss(student_logits, targets)
        proj_s = self.projection(student_features)
        feat_loss = self.mse_loss(proj_s, teacher_features)
        total_loss = ce + self.beta * feat_loss
        return total_loss, ce, feat_loss


class CombinedKDLoss(nn.Module):
    """Combined logit and feature distillation loss."""

    def __init__(
        self,
        student_dim: int = 576,
        teacher_dim: int = 768,
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        super().__init__()
        self.response_kd = ResponseKDLoss(temperature=temperature, alpha=alpha)
        self.feature_kd = FeatureHintLoss(student_dim=student_dim, teacher_dim=teacher_dim, beta=beta)

    def forward(
        self,
        student_logits: torch.Tensor,
        student_features: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        resp_loss, ce, kd = self.response_kd(student_logits, teacher_logits, targets)
        proj_s = self.feature_kd.projection(student_features)
        feat_loss = self.feature_kd.mse_loss(proj_s, teacher_features)
        total_loss = resp_loss + self.feature_kd.beta * feat_loss
        return total_loss, ce, kd, feat_loss
