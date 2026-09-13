"""Deterministic, layout-independent parameter initialisation.

The single most important property here: **expert ``e`` has byte-identical
weights no matter where it lives.**

* on rank ``r`` of an EP=4 run it sits at local index ``e - r*epr``;
* in the dense reference bank it sits at index ``e``.

Both paths call :func:`expert_weight` with the same ``(seed, expert_id)``, so
the weights agree by construction rather than by luck.  That is what makes the
correctness comparison in ``verify_ep.py`` meaningful -- and it is why the
router weight is seeded the same on every rank: a rank-dependent seed would
send tokens to the wrong expert, and the symptom would be "numbers are slightly
off" rather than a crash.

Tensors are drawn on CPU and moved afterwards, so the result does not depend on
the device, the stream, or the current CUDA RNG state.
"""

from __future__ import annotations

from typing import Sequence

import torch

_TAG_EXPERT = 1_000_000
_TAG_ROUTER = 2_000_000
_TAG_IO = 3_000_000

_MODULUS = 2**63 - 1


def _generator(seed: int, tag: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed((int(seed) * 1_000_003 + int(tag)) % _MODULUS)
    return g


def _normal(seed: int, tag: int, shape: Sequence[int], std: float) -> torch.Tensor:
    t = torch.empty(tuple(shape), dtype=torch.float32)
    t.normal_(0.0, std, generator=_generator(seed, tag))
    return t


def expert_weight(
    seed: int,
    expert_id: int,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    std: float = 0.02,
) -> torch.Tensor:
    """Weights for global expert ``expert_id`` (shape ``hidden x ffn`` etc.)."""
    t = _normal(seed, _TAG_EXPERT + expert_id, shape, std)
    return t.to(device=device, dtype=dtype)


def router_weight(
    seed: int,
    layer_idx: int,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    std: float = 0.02,
) -> torch.Tensor:
    """Router weights -- identical on every rank by construction."""
    t = _normal(seed, _TAG_ROUTER + layer_idx, shape, std)
    return t.to(device=device, dtype=dtype)


def io_weight(
    seed: int,
    tag: int,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    std: float = 0.02,
) -> torch.Tensor:
    t = _normal(seed, _TAG_IO + tag, shape, std)
    return t.to(device=device, dtype=dtype)


def make_global_tokens(
    seed: int,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    std: float = 1.0,
) -> torch.Tensor:
    """The full batch, generated identically on every rank.

    Inputs are generated globally and then *sliced* per rank rather than each
    rank drawing its own tokens.  That keeps EP=1/2/4 comparable at a fixed
    ``--tokens``: rank 0 always sees the same rows, and the EP=4 run is the
    same problem partitioned four ways.
    """
    t = _normal(seed, 777, (num_tokens, hidden), std)
    return t.to(device=device, dtype=dtype)


def slice_for_rank(
    x_global: torch.Tensor, rank: int, world: int, tokens_per_rank: int
) -> torch.Tensor:
    start = rank * tokens_per_rank
    return x_global.narrow(0, start, tokens_per_rank).contiguous()
