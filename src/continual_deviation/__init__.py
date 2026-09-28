"""Research scaffold for deviation-corrected continual RL updates."""

from .benchmarks import build_checkpoint_schedule, continuing_swimmer_project
from .config import (
    BenchmarkConfig,
    CorrectionConfig,
    PPOConfig,
    ProjectConfig,
    RepresentationConfig,
    RuntimeConfig,
    load_project_config,
)
from .representation import (
    RepresentationComparison,
    capture_layer_outputs,
    cosine_drift,
    fit_ridge_probe,
    linear_cka,
    ridge_probe_r2,
)
from .runtime import (
    autocast_context,
    configure_torch_runtime,
    prepare_model_for_runtime,
    resolve_device,
    resolve_dtype,
)
from .update import (
    CorrectionResult,
    DeviationCandidate,
    corrected_policy_loss,
    positive_regret,
    select_reference_deviation,
)

__all__ = [
    "BenchmarkConfig",
    "CorrectionConfig",
    "CorrectionResult",
    "DeviationCandidate",
    "PPOConfig",
    "ProjectConfig",
    "RepresentationComparison",
    "RepresentationConfig",
    "RuntimeConfig",
    "build_checkpoint_schedule",
    "capture_layer_outputs",
    "continuing_swimmer_project",
    "corrected_policy_loss",
    "cosine_drift",
    "configure_torch_runtime",
    "fit_ridge_probe",
    "linear_cka",
    "load_project_config",
    "positive_regret",
    "prepare_model_for_runtime",
    "resolve_device",
    "resolve_dtype",
    "ridge_probe_r2",
    "select_reference_deviation",
    "autocast_context",
]
