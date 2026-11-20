"""Failure domains of the pipeline.

Every step that is allowed to abort a run (keeping the previous
generation untouched) raises a subclass of :class:`PipelineError`
carrying its failure domain.
"""


class PipelineError(Exception):
    """Base class for all pipeline failures."""

    domain = "pipeline"

    def __init__(self, message, *, detail=None):
        super().__init__(message)
        self.detail = dict(detail or {})

    def as_dict(self):
        return {"domain": self.domain, "message": str(self), "detail": self.detail}


class InputError(PipelineError):
    """Business inputs (archives, costs, constraints, ...) are invalid."""

    domain = "input"


class ExtractionError(PipelineError):
    """Native tsfresh feature extraction failed."""

    domain = "extraction"


class ModelError(PipelineError):
    """Pre-fitted pairwise model missing, corrupt or unusable."""

    domain = "model"


class OptimizationError(PipelineError):
    """Global risk minimisation failed or constraints are infeasible."""

    domain = "optimization"


class SaveError(PipelineError):
    """Persisting / promoting the generation failed; old generation kept."""

    domain = "save"
