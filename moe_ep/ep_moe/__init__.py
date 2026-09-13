"""Toy MoE Expert Parallelism demo for a single node of 4x Tesla V100.

Forward inference only.  See ``moe_ep/README.md``.

The package is deliberately dependency-light: ``torch`` + ``torch.distributed``
and the standard library.  Nothing here imports a training framework.
"""

from .config import ModelConfig, PRESETS, RunConfig
from .dist_utils import EPLayout, DistContext, init_distributed

__all__ = [
    "ModelConfig",
    "RunConfig",
    "PRESETS",
    "EPLayout",
    "DistContext",
    "init_distributed",
]
