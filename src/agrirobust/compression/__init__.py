"""AgriRobust Compression Modules.

Provides quantization, pruning, and deployment benchmark utilities.
"""

from agrirobust.compression.quantization import (
    apply_dynamic_quantization,
    probe_static_quantization,
    save_quantized_model,
    load_quantized_model,
)
from agrirobust.compression.pruning import (
    apply_unstructured_pruning,
    calculate_sparsity,
)
from agrirobust.compression.benchmark import (
    benchmark_cpu_latency,
    measure_footprint,
)

__all__ = [
    "apply_dynamic_quantization",
    "probe_static_quantization",
    "save_quantized_model",
    "load_quantized_model",
    "apply_unstructured_pruning",
    "calculate_sparsity",
    "benchmark_cpu_latency",
    "measure_footprint",
]
