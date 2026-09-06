"""Pruning utilities for AgriRobust deployment compression.

Implements magnitude-based unstructured pruning and sparsity calculations.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)


def apply_unstructured_pruning(
    model: nn.Module,
    amount: float,
    target_layer_types: Optional[Tuple[Type[nn.Module], ...]] = None,
    make_permanent: bool = True,
) -> nn.Module:
    """Apply L1 unstructured pruning to target layers (e.g. classifier nn.Linear).

    Args:
        model: Frozen PyTorch model.
        amount: Fraction of weights to prune in [0.0, 1.0).
        target_layer_types: Module types to prune (default: (nn.Linear,)).
        make_permanent: Whether to remove pruning reparameterization and make zeros permanent.

    Returns:
        Pruned PyTorch model copy.
    """
    if target_layer_types is None:
        target_layer_types = (nn.Linear,)

    model_copy = copy.deepcopy(model)
    model_copy.eval()

    if amount <= 0.0:
        return model_copy

    for name, module in model_copy.named_modules():
        if isinstance(module, target_layer_types) and hasattr(module, "weight"):
            prune.l1_unstructured(module, name="weight", amount=amount)
            if make_permanent:
                prune.remove(module, "weight")

    return model_copy


def calculate_sparsity(model: nn.Module) -> Dict[str, float]:
    """Calculate exact layer-wise and global zero-weight sparsity percentages.

    Returns:
        Dictionary mapping layer name to sparsity percentage, plus 'global_sparsity'.
    """
    total_zeros = 0
    total_elements = 0
    layer_sparsity: Dict[str, float] = {}

    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            zeros = (param == 0).sum().item()
            numel = param.numel()
            total_zeros += zeros
            total_elements += numel
            layer_sparsity[name] = float(zeros / numel * 100.0) if numel > 0 else 0.0

    global_pct = float(total_zeros / total_elements * 100.0) if total_elements > 0 else 0.0
    layer_sparsity["global_sparsity"] = global_pct
    layer_sparsity["total_zeros"] = total_zeros
    layer_sparsity["total_elements"] = total_elements
    return layer_sparsity
