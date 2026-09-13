"""Model stacks, parameter fingerprints, and GPU memory accounting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .config import ModelConfig
from .dist_utils import EPLayout
from .moe_layer import DenseMoELayer, EPMoELayer
from .timer import PhaseTimer


# ----------------------------------------------------------------------
# Forward bookkeeping
# ----------------------------------------------------------------------
@dataclass
class ForwardStats:
    """Optional side outputs of a forward pass, for the reporting tables.

    ``record_dispatch`` costs a host sync (it materialises the split lists), so
    it only runs when ``keep_dispatches`` is set -- and callers that set it must
    keep it out of the timed region.
    """

    expert_load: Optional[torch.Tensor] = None
    keep_dispatches: bool = False
    dispatches: List[dict] = field(default_factory=list)

    def record_load(self, counts: torch.Tensor) -> None:
        self.expert_load = (
            counts if self.expert_load is None else self.expert_load + counts
        )

    def record_dispatch(self, ctx) -> None:
        if self.keep_dispatches:
            self.dispatches.append(ctx.as_dict())


# ----------------------------------------------------------------------
# Stacks
# ----------------------------------------------------------------------
class EPMoEStack(nn.Module):
    """A stack of expert-parallel MoE layers."""

    def __init__(
        self,
        cfg: ModelConfig,
        layout: EPLayout,
        group=None,
        device: torch.device = torch.device("cpu"),
        **layer_kwargs,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layout = layout
        self.layers = nn.ModuleList(
            [
                EPMoELayer(cfg, layout, i, group, device=device, **layer_kwargs)
                for i in range(cfg.num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        timer: Optional[PhaseTimer] = None,
        stats: Optional[ForwardStats] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, timer, stats)
        return x

    @property
    def expert_param_bytes(self) -> int:
        return sum(l.experts.param_bytes for l in self.layers)

    @property
    def replicated_param_bytes(self) -> int:
        return sum(
            p.numel() * p.element_size()
            for l in self.layers
            for p in l.router.parameters()
        )


class DenseMoEStack(nn.Module):
    """Reference stack: every expert on every rank, no communication."""

    def __init__(
        self,
        cfg: ModelConfig,
        device: torch.device = torch.device("cpu"),
        **layer_kwargs,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [
                DenseMoELayer(cfg, i, device=device, **layer_kwargs)
                for i in range(cfg.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def expert_param_bytes(self) -> int:
        return sum(l.experts.param_bytes for l in self.layers)


# ----------------------------------------------------------------------
# Fingerprints
# ----------------------------------------------------------------------
def _hash_tensor(t: torch.Tensor, sample: int = 64) -> str:
    flat = t.detach().float().cpu().reshape(-1)
    n = min(sample, flat.numel())
    return hashlib.sha256(flat[:n].numpy().tobytes()).hexdigest()[:16]


def router_fingerprint(stack) -> str:
    """Hash of every router weight.  **Must be identical on every rank.**

    A rank-dependent seed here is the worst failure mode in the project: the
    routers disagree, tokens go to the wrong expert, and nothing crashes -- the
    numbers are just quietly wrong.  ``check_env.py`` all-gathers this string and
    asserts all ranks agree.
    """
    h = hashlib.sha256()
    for i, layer in enumerate(stack.layers):
        h.update(f"L{i}:".encode())
        h.update(_hash_tensor(layer.router.weight).encode())
    return h.hexdigest()[:16]


def expert_fingerprint_map(bank) -> Dict[int, str]:
    """``{global_expert_id: weight_hash}`` for the experts this rank holds.

    Gathered across ranks this must be a partition of ``0..E-1``: every expert
    present exactly once, with no id claimed by two ranks.
    """
    out: Dict[int, str] = {}
    for e in range(bank.count):
        h = hashlib.sha256()
        for name in ("w_gate", "w_up", "w_down"):
            h.update(_hash_tensor(getattr(bank, name)[e]).encode())
        out[bank.expert_start + e] = h.hexdigest()[:16]
    return out


def check_expert_partition(maps: List[Dict[int, str]], num_experts: int) -> List[str]:
    """Validate that the gathered per-rank expert maps tile ``0..E-1``."""
    problems: List[str] = []
    seen: Dict[int, int] = {}
    for rank, m in enumerate(maps):
        for gid in m:
            if gid in seen:
                problems.append(
                    f"expert {gid} claimed by both rank {seen[gid]} and rank {rank}"
                )
            seen[gid] = rank
    missing = sorted(set(range(num_experts)) - set(seen))
    if missing:
        problems.append(f"experts never initialised anywhere: {missing}")
    if len(seen) != num_experts:
        problems.append(
            f"expected {num_experts} distinct experts, found {len(seen)}"
        )
    return problems


# ----------------------------------------------------------------------
# Memory accounting
# ----------------------------------------------------------------------
def memory_report() -> Dict[str, float]:
    """Snapshot GPU memory.

    ``max_memory_reserved`` is reported alongside ``max_memory_allocated``
    because the reserved figure is the honest footprint: NCCL's own
    communication buffers are never visible to the torch allocator, so a table
    built only from ``max_memory_allocated`` understates EP's real cost -- which
    matters, since "where do EP's memory savings come from" is a headline
    question for this demo.
    """
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_MB": torch.cuda.memory_allocated() / 2**20,
        "reserved_MB": torch.cuda.memory_reserved() / 2**20,
        "max_allocated_MB": torch.cuda.max_memory_allocated() / 2**20,
        "max_reserved_MB": torch.cuda.max_memory_reserved() / 2**20,
        "device_free_MB": free / 2**20,
        "device_total_MB": total / 2**20,
    }


def param_bytes(module: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def estimate_model_bytes(cfg: ModelConfig, ep_size: int) -> Dict[str, float]:
    """Analytic memory model -- what the measurements are compared against."""
    per_rank_experts = cfg.num_experts // ep_size
    expert = cfg.expert_bytes(per_rank_experts) * cfg.num_layers
    replicated = cfg.replicated_bytes() * cfg.num_layers
    return {
        "expert_weights_MB": expert / 2**20,
        "replicated_weights_MB": replicated / 2**20,
        "naive_ep1_expert_MB": cfg.expert_bytes() * cfg.num_layers / 2**20,
        "saving_vs_ep1_MB": (cfg.expert_bytes() - cfg.expert_bytes(per_rank_experts))
        * cfg.num_layers
        / 2**20,
    }
