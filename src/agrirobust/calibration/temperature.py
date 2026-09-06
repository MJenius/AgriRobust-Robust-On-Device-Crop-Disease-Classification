"""Post-hoc temperature scaling calibration for neural network logits.

Rules:
- Temperature parameter T > 0 is optimized strictly on the validation set.
- Temperature scaling preserves raw top-1 class predictions and accuracy exactly.
- Learned temperature is frozen and transferred unchanged to clean, corrupted, and field domains.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


class TemperatureScaler:
    """Post-hoc temperature scaling fitted via maximum likelihood on validation logits."""

    def __init__(self, initial_temperature: float = 1.0):
        self.temperature: float = float(initial_temperature)
        self.is_fitted: bool = False
        self.fit_loss_before: Optional[float] = None
        self.fit_loss_after: Optional[float] = None

    def fit(
        self,
        val_logits: Union[np.ndarray, torch.Tensor],
        val_targets: Union[np.ndarray, torch.Tensor],
        bounds: Tuple[float, float] = (0.05, 10.0),
    ) -> "TemperatureScaler":
        """Optimize single scalar temperature T by minimizing negative log-likelihood on validation data.

        Args:
            val_logits: Array or Tensor of shape (N, K) containing unnormalized logits.
            val_targets: Array or Tensor of shape (N,) containing integer class targets.
            bounds: Permissible bounds for the temperature scalar (min_T, max_T).

        Returns:
            self
        """
        if isinstance(val_logits, torch.Tensor):
            val_logits = val_logits.detach().cpu().numpy()
        if isinstance(val_targets, torch.Tensor):
            val_targets = val_targets.detach().cpu().numpy()

        val_logits = val_logits.astype(np.float64)
        val_targets = val_targets.astype(np.int64)

        n_samples = len(val_logits)
        if n_samples == 0:
            raise ValueError("Validation set cannot be empty.")

        def nll_objective(t_arr: np.ndarray) -> float:
            t = float(t_arr[0])
            scaled_logits = val_logits / t
            # Stable log-sum-exp
            max_logits = np.max(scaled_logits, axis=1, keepdims=True)
            log_sum_exp = max_logits + np.log(np.sum(np.exp(scaled_logits - max_logits), axis=1, keepdims=True))
            log_probs = scaled_logits - log_sum_exp
            loss = -np.mean(log_probs[np.arange(n_samples), val_targets])
            return float(loss)

        self.fit_loss_before = nll_objective(np.array([1.0]))

        res = minimize(
            nll_objective,
            x0=np.array([1.0]),
            method="L-BFGS-B",
            bounds=[bounds],
            options={"maxiter": 100, "ftol": 1e-9},
        )

        self.temperature = float(res.x[0])
        self.fit_loss_after = float(res.fun)
        self.is_fitted = True
        return self

    def scale_logits(self, logits: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply the frozen learned temperature to scale logits."""
        if isinstance(logits, torch.Tensor):
            return logits / self.temperature
        return logits / self.temperature

    def predict_probabilities(self, logits: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Scale logits by temperature and compute softmax probabilities."""
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        scaled = logits.astype(np.float64) / self.temperature
        max_logits = np.max(scaled, axis=1, keepdims=True)
        exp_logits = np.exp(scaled - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs.astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize temperature scaler metadata."""
        return {
            "temperature": round(self.temperature, 4),
            "is_fitted": self.is_fitted,
            "fit_loss_before": round(self.fit_loss_before, 4) if self.fit_loss_before is not None else None,
            "fit_loss_after": round(self.fit_loss_after, 4) if self.fit_loss_after is not None else None,
        }
