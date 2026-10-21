"""Fine-tune and calibrate system-one decision models with a scikit-learn style fit / predict interface."""

from .architectures import Architecture, register_architecture
from .base import NotFittedError, SystemOneModel
from .evaluation import baselines, compare
from .models import DecoderModel, EncoderModel
from .version import __version__

__all__ = [
    "Architecture",
    "DecoderModel",
    "EncoderModel",
    "NotFittedError",
    "SystemOneModel",
    "__version__",
    "baselines",
    "compare",
    "register_architecture",
]
