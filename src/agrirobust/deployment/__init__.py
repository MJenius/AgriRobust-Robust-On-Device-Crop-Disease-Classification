"""Deployment package for AgriRobust Phase 8.

Provides model export, preprocessing parity, verification, and benchmarking utilities
for mobile on-device deployment.
"""

from agrirobust.deployment.export import export_deployment_bundle, export_torchscript_model
from agrirobust.deployment.preprocessing import (
    ANDROID_PREPROCESSING_SPEC,
    preprocess_image_for_deployment,
    verify_preprocessing_parity,
)
from agrirobust.deployment.validate_export import evaluate_export_parity
from agrirobust.deployment.mobile_benchmark import benchmark_mobile_runtime

__all__ = [
    "export_torchscript_model",
    "export_deployment_bundle",
    "ANDROID_PREPROCESSING_SPEC",
    "preprocess_image_for_deployment",
    "verify_preprocessing_parity",
    "evaluate_export_parity",
    "benchmark_mobile_runtime",
]
