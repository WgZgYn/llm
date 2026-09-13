"""Process-group setup and the EP rank layout.

The layout is deliberately the simplest possible: **pure EP**, meaning
``world_size == ep_size`` and rank ``r`` owns experts
``[r*epr, (r+1)*epr)`` where ``epr = num_experts // world_size``.

We do *not* build sub-groups for ``ep_size < world_size``.  The sweep runs one
``torchrun --nproc_per_node=$EP`` per EP size instead, so ``world_size`` and
``ep_size`` can never disagree.  That removes an entire class of
partial-group bugs, which matter a lot here because the code is written on a
machine that cannot run it.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EPLayout:
    """Maps global expert ids to the rank that owns them."""

    rank: int
    world: int
    num_experts: int
    top_k: int

    def __post_init__(self) -> None:
        if self.num_experts % self.world != 0:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by "
                f"world_size={self.world}"
            )

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.world

    @property
    def expert_start(self) -> int:
        return self.rank * self.experts_per_rank

    @property
    def expert_end(self) -> int:
        return self.expert_start + self.experts_per_rank

    def expert_to_rank(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Global expert id -> owning rank.  Vectorised, stays on device."""
        return torch.div(expert_ids, self.experts_per_rank, rounding_mode="floor")

    def is_local(self, expert_id: int) -> bool:
        return self.expert_start <= expert_id < self.expert_end

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "world": self.world,
            "num_experts": self.num_experts,
            "experts_per_rank": self.experts_per_rank,
            "expert_start": self.expert_start,
            "expert_end": self.expert_end,
            "top_k": self.top_k,
        }


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------
@dataclass
class DistContext:
    rank: int
    local_rank: int
    world: int
    device: torch.device
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def layout(self, num_experts: int, top_k: int) -> EPLayout:
        return EPLayout(self.rank, self.world, num_experts, top_k)


def init_distributed(
    backend: str = "nccl",
    timeout_minutes: int = 3,
    require_cuda: bool = True,
) -> DistContext:
    """Set up the process group with a *short* timeout and hard-fail on misuse.

    Two deliberate deviations from the usual boilerplate, both aimed at the
    fact that this code is first executed on a machine nobody has debugged it
    on:

    * ``timeout`` is 3 minutes, not the 30-minute default.  A mismatched
      ``all_to_all`` split hangs the collective; with the default timeout that
      is a 30-minute wait before you learn anything.  Pair this with
      ``TORCH_NCCL_ASYNC_ERROR_HANDLING=1`` (see ``scripts/env.example.sh``) to
      get a Python traceback naming the failing rank instead of a hang.
    * ``LOCAL_RANK`` is *required*.  Several scripts in this repo default it to
      0, which is fine for a single-process benchmark but silently puts four
      processes on GPU 0 when you forget ``torchrun`` -- and then every memory
      and bandwidth number is garbage rather than an error.
    """
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "LOCAL_RANK is not set. This script must be launched with torchrun, "
            "e.g.\n"
            "    torchrun --standalone --nproc_per_node=4 scripts/check_env.py\n"
            "Running it with plain `python` would put every process on GPU 0 "
            "and produce meaningless numbers."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))

    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This demo targets 4x V100 with NCCL; "
                "there is intentionally no CPU/gloo fallback."
            )
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) "
                f"are visible. Check CUDA_VISIBLE_DEVICES -- with "
                f"--nproc_per_node=4 it should normally be unset."
            )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )

    backend_name = dist.get_backend()
    return DistContext(rank, local_rank, world, device, str(backend_name))


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def nccl_version() -> str:
    """``torch.cuda.nccl.version()`` is not portable across builds.

    The Windows wheels lack ``torch._C._nccl_version`` entirely and raise
    ``AttributeError``; other versions return an int, a tuple, or a packed int.
    Normalise defensively -- a version *string* must never be able to crash a
    diagnostic script.
    """
    try:
        v = torch.cuda.nccl.version()
    except Exception as exc:  # pragma: no cover - platform dependent
        return f"n/a ({type(exc).__name__})"
    if isinstance(v, (tuple, list)):
        return ".".join(str(x) for x in v)
    return str(v)


# --------------------------------------------------------------------------
# Cross-rank reduction helpers
# --------------------------------------------------------------------------
def all_gather_object(obj, world: int, device: torch.device):
    """Gather one picklable object per rank onto every rank."""
    bucket = [None] * world
    dist.all_gather_object(bucket, obj)
    return bucket


def all_gather_vector(t: torch.Tensor, world: int) -> torch.Tensor:
    """Gather a 1-D float tensor of length ``world`` -> ``[world, world]``.

    Row ``r`` is rank ``r``'s vector.  Requires every rank to pass the same
    length; that is guaranteed because the vector is always length ``world``.
    """
    if not t.is_cuda:
        t = t.to(torch.cuda.current_device())
    t = t.contiguous()
    out = torch.empty((world, t.numel()), dtype=t.dtype, device=t.device)
    dist.all_gather_into_tensor(out, t)
    return out


def reduce_max_int(value: int) -> int:
    """All-reduce a python int across ranks (used for failure flags)."""
    t = torch.tensor([int(value)], dtype=torch.int32, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def max_across_ranks(value: float) -> float:
    t = torch.tensor([float(value)], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())
