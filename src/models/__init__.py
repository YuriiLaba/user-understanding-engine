import platform
from .base_model import BaseVisionModel
from .openai_model import OpenAIVisionModel

__all__ = ["BaseVisionModel", "OpenAIVisionModel"]

if platform.system() == "Darwin":
    from .mlx_model import MLXVisionModel
    __all__.append("MLXVisionModel")
