"""Device archive entity resolution pipeline.

End-to-end, globally-consistent entity resolution for device archives
coming from heterogeneous source systems:

1. native tsfresh time-series feature extraction from per-archive telemetry
2. pairwise match probabilities from a pre-fitted pairwise model
3. globally minimum-expected-risk consistent partition (exact MILP
   correlation clustering with must-link / cannot-link constraints and
   asymmetric false-merge / missed-merge costs) -- never threshold +
   connected components
4. master records: entity id = min member id; attributes = first non-NULL
   value by source priority, preserving 0 / FALSE / ''
5. actual rebinding of existing repair work orders (disappeared source
   archive -> NULL), all other business values untouched
6. atomic, fully re-openable generation persistence; any failure in
   extraction / model / optimization / save keeps the previous generation.
"""

from .errors import (
    PipelineError,
    InputError,
    ExtractionError,
    ModelError,
    OptimizationError,
    SaveError,
)
from .pipeline import PipelineConfig, run_pipeline

__version__ = "1.0.0"

__all__ = [
    "PipelineConfig",
    "run_pipeline",
    "PipelineError",
    "InputError",
    "ExtractionError",
    "ModelError",
    "OptimizationError",
    "SaveError",
    "__version__",
]
