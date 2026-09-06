"""Quantization utilities for AgriRobust deployment compression.

Supports dynamic INT8 quantization and environment compatibility probing for static INT8.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_dynamic_quantization(
    model: nn.Module,
    qconfig_spec: Optional[set[Type[nn.Module]]] = None,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """Apply dynamic INT8 quantization to supported layers (e.g. nn.Linear).

    Args:
        model: Frozen PyTorch model in FP32.
        qconfig_spec: Set of module types to quantize. Defaults to {nn.Linear}.
        dtype: Quantized integer datatype (default: torch.qint8).

    Returns:
        Quantized PyTorch model copy.
    """
    if qconfig_spec is None:
        qconfig_spec = {nn.Linear}

    model_copy = copy.deepcopy(model)
    model_copy.eval()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_copy,
        qconfig_spec=qconfig_spec,
        dtype=dtype,
    )
    return quantized_model


def probe_static_quantization(
    model: nn.Module,
    calibration_loader: Optional[Any] = None,
) -> Dict[str, Any]:
    """Probe eager static post-training quantization on current host environment.

    Records PyTorch version, available engines, and exact compatibility status/error
    without raising unhandled exceptions.

    Returns:
        Dictionary detailing backend compatibility report.
    """
    pytorch_version = torch.__version__
    supported_engines = list(torch.backends.quantized.supported_engines)
    current_engine = torch.backends.quantized.engine

    report: Dict[str, Any] = {
        "pytorch_version": pytorch_version,
        "supported_engines": supported_engines,
        "current_engine": current_engine,
        "supported": False,
        "status": "NOT_SUPPORTED",
        "error_message": None,
    }

    try:
        model_copy = copy.deepcopy(model)
        model_copy.eval()

        # Attempt standard eager static quantization config
        if "onednn" in supported_engines:
            backend = "onednn"
        elif "qnnpack" in supported_engines:
            backend = "qnnpack"
        elif "fbgemm" in supported_engines:
            backend = "fbgemm"
        else:
            backend = current_engine

        qconfig = torch.ao.quantization.get_default_qconfig(backend)
        model_copy.qconfig = qconfig
        prepared_model = torch.ao.quantization.prepare(model_copy, inplace=False)

        # Feed sample calibration batch if provided, otherwise dummy batch
        if calibration_loader is not None:
            with torch.no_grad():
                for i, (images, _) in enumerate(calibration_loader):
                    prepared_model(images)
                    if i >= 4:
                        break
        else:
            dummy_input = torch.randn(1, 3, 224, 224)
            prepared_model(dummy_input)

        converted_model = torch.ao.quantization.convert(prepared_model, inplace=False)

        # Test forward pass
        dummy_test = torch.randn(1, 3, 224, 224)
        out = converted_model(dummy_test)
        if out.shape[-1] == 38:
            report["supported"] = True
            report["status"] = "SUPPORTED"
            report["converted_model"] = converted_model
            return report

    except Exception as e:
        report["error_message"] = f"{type(e).__name__}: {str(e)}"
        report["status"] = f"NOT_SUPPORTED ({report['error_message']})"
        logger.warning(
            "Static PTQ compatibility probe reported unsupported backend: %s",
            report["error_message"],
        )

    return report


def save_quantized_model(model: nn.Module, save_path: Path | str) -> Path:
    """Serialize quantized model state dict or full module."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, str(save_path))
    return save_path


def load_quantized_model(load_path: Path | str) -> nn.Module:
    """Load serialized quantized model."""
    load_path = Path(load_path)
    model = torch.load(str(load_path), map_location="cpu", weights_only=False)
    model.eval()
    return model
