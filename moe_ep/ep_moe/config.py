"""Model / run configuration and the built-in presets.

Everything that changes the shape of the computation lives here so the
benchmark scripts stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import torch

# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
# ``ffn`` is the SwiGLU intermediate width.  Each expert therefore holds
# 3 * hidden * ffn parameters (w_gate, w_up, w_down).
PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {"hidden": 1024, "ffn": 4096, "num_experts": 8, "top_k": 2},
    "medium": {"hidden": 2048, "ffn": 8192, "num_experts": 8, "top_k": 2},
    "wide": {"hidden": 2048, "ffn": 8192, "num_experts": 32, "top_k": 2},
}

TORCH_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

#: Dtypes that are numerically unsafe to benchmark on Volta (sm_70).
#: V100 has fp16 tensor cores but *no* bf16 path; cuBLAS either errors out or
#: silently falls back to slow emulation, which makes any timing meaningless.
DTYPES_UNSUPPORTED_ON_VOLTA = ("bf16",)


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the toy MoE.

    The model is a plain stack of MoE layers operating on ``[T, hidden]``
    activations.  There is no attention and no LM head by default -- both would
    only add replicated compute noise to an EP measurement.  ``vocab`` exists
    for the optional embedding/head path enabled by ``--io-layers``.
    """

    hidden: int = 1024
    ffn: int = 4096
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 2
    vocab: int = 32000
    dtype: str = "fp16"
    seed: int = 20250913

    # -- derived ----------------------------------------------------------
    @property
    def torch_dtype(self) -> torch.dtype:
        return TORCH_DTYPES[self.dtype]

    @property
    def dtype_bytes(self) -> int:
        return torch.tensor([], dtype=self.torch_dtype).element_size()

    @property
    def params_per_expert(self) -> int:
        """SwiGLU: three ``hidden x ffn`` matrices."""
        return 3 * self.hidden * self.ffn

    def expert_bytes(self, num_experts: Optional[int] = None) -> int:
        n = self.num_experts if num_experts is None else num_experts
        return n * self.params_per_expert * self.dtype_bytes

    def replicated_bytes(self, io_layers: bool = False) -> int:
        """Parameters every rank holds in full (router, plus optional io)."""
        router = self.num_layers * self.hidden * self.num_experts
        io = 0
        if io_layers:
            io = 2 * self.vocab * self.hidden  # embedding + head
        return (router + io) * self.dtype_bytes

    # -- construction -----------------------------------------------------
    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> "ModelConfig":
        if name not in PRESETS:
            raise KeyError(
                f"unknown preset {name!r}; available: {sorted(PRESETS)}"
            )
        base = dict(PRESETS[name])
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)

    def with_overrides(self, **overrides: Any) -> "ModelConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)

    # -- validation -------------------------------------------------------
    def validate(self, world_size: int, allow_bf16: bool = False) -> None:
        if self.hidden <= 0 or self.ffn <= 0:
            raise ValueError("hidden and ffn must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k={self.top_k} exceeds num_experts={self.num_experts}"
            )
        if self.num_experts % world_size != 0:
            raise ValueError(
                f"num_experts={self.num_experts} is not divisible by ep_size="
                f"{world_size}. EP needs every rank to own the same number of "
                f"experts. Try --num-experts {self._nearest_divisible(world_size)}."
            )
        if self.dtype not in TORCH_DTYPES:
            raise ValueError(
                f"unknown dtype {self.dtype!r}; expected one of "
                f"{sorted(TORCH_DTYPES)}"
            )
        if self.dtype in DTYPES_UNSUPPORTED_ON_VOLTA and not allow_bf16:
            raise ValueError(
                f"dtype={self.dtype!r} is not usable on this hardware class. "
                "V100 is Volta (sm_70): it has fp16 tensor cores but no bf16 "
                "path, so cuBLAS either errors or falls back to slow "
                "emulation and any timing is meaningless. Use --dtype fp16 "
                "(default) or --dtype fp32, or pass --allow-bf16 to override."
            )

    def _nearest_divisible(self, world_size: int) -> int:
        """Smallest multiple of ``world_size`` that is >= num_experts."""
        return max(world_size, -(-self.num_experts // world_size) * world_size)

    def summary(self) -> str:
        return (
            f"hidden={self.hidden} ffn={self.ffn} E={self.num_experts} "
            f"k={self.top_k} layers={self.num_layers} dtype={self.dtype}"
        )


@dataclass
class RunConfig:
    """Everything a benchmark entrypoint needs besides the model shape."""

    ep_size: int = 4
    tokens: int = 4096          # GLOBAL token count, split across ranks
    tokens_per_rank: Optional[int] = None
    skew: float = 0.0
    skew_hot: int = 4
    dtype: Optional[str] = None  # overrides ModelConfig.dtype when set
    warmup: int = 5
    iters: int = 20
    nvtx: bool = False
    fuse_dispatch: bool = False
    grouped_gemm: bool = False
    io_layers: bool = False
    identity_experts: bool = False
    deterministic: bool = False
    tag: str = ""
    out_dir: str = "out"

    def local_tokens(self, world_size: int) -> int:
        """Tokens this rank owns.

        Token counts are per rank, not per model: with EP=4 each rank sees a
        quarter of the global batch, which is what makes the EP=1/2/4 runs
        comparable at fixed ``--tokens``.
        """
        if self.tokens_per_rank is not None:
            return self.tokens_per_rank
        if self.tokens % world_size != 0:
            raise ValueError(
                f"--tokens {self.tokens} is not divisible by ep_size="
                f"{world_size}. Use --tokens-per-rank to bypass, or pick one "
                f"of {[self.tokens - self.tokens % world_size + i * world_size for i in range(1, 4)]}."
            )
        return self.tokens // world_size
