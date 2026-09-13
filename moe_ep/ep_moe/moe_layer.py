"""The MoE layer, in an expert-parallel and a dense (reference) flavour.

``EPMoELayer`` is the thing under study: router -> dispatch -> local experts ->
combine.  ``DenseMoELayer`` is the same maths with all experts on every rank and
no communication at all; it exists purely so ``verify_ep.py`` has something to
compare against, and it shares the router and expert weight initialisation with
the EP path so that any difference is a *communication* difference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .comm import AllToAllDispatcher
from .config import ModelConfig
from .dist_utils import EPLayout
from .experts import ExpertBank
from .router import TopKRouter, expert_load
from .timer import PhaseTimer, timed


class EPMoELayer(nn.Module):
    """One MoE layer with its experts sharded across the EP group."""

    def __init__(
        self,
        cfg: ModelConfig,
        layout: EPLayout,
        layer_idx: int,
        group=None,
        device: torch.device = torch.device("cpu"),
        skew: float = 0.0,
        skew_hot: int = 0,
        balance: str = "natural",
        fuse: bool = False,
        grouped: bool = False,
        identity: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layout = layout
        self.layer_idx = layer_idx
        self.prefix = f"L{layer_idx}."
        self.grouped = grouped

        self.router = TopKRouter(
            cfg, layer_idx, skew, skew_hot, balance, device=device
        )
        self.experts = ExpertBank(
            cfg, layout.expert_start, layout.experts_per_rank, device, identity
        )
        self.dispatcher = AllToAllDispatcher(
            layout, group, fuse=fuse, prefix=self.prefix
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        timer: Optional[PhaseTimer] = None,
        stats=None,
    ) -> torch.Tensor:
        with timed(timer, self.prefix + "router"):
            topk_idx, topk_w = self.router(x)

        if stats is not None:
            stats.record_load(self.expert_load(topk_idx))

        sorted_x, ctx = self.dispatcher.dispatch(x, topk_idx, topk_w, timer)

        if stats is not None:
            stats.record_dispatch(ctx)

        with timed(timer, self.prefix + "expert"):
            out_sorted = self.experts.forward(
                sorted_x, ctx, grouped=self.grouped, timer=timer
            )

        out = self.dispatcher.combine(out_sorted, ctx, timer)
        return out

    # ------------------------------------------------------------------
    def expert_load(self, topk_idx: torch.Tensor) -> torch.Tensor:
        """Token count for every *global* expert, this rank's contribution."""
        return expert_load(topk_idx, self.cfg.num_experts)

    @property
    def param_bytes(self) -> int:
        return self.experts.param_bytes


class DenseMoELayer(nn.Module):
    """Reference implementation: every expert on every rank, zero comms.

    Deliberately naive -- a python loop over ``E`` experts with a boolean mask --
    because it only has to be obviously correct, and because it must not share
    any code path with the EP version that could hide a bug.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        device: torch.device = torch.device("cpu"),
        skew: float = 0.0,
        skew_hot: int = 0,
        balance: str = "natural",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.router = TopKRouter(
            cfg, layer_idx, skew, skew_hot, balance, device=device
        )
        self.experts = ExpertBank(cfg, 0, cfg.num_experts, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_idx, topk_w = self.router(x)                    # [T,k]
        out = torch.zeros_like(x)
        for e in range(self.cfg.num_experts):
            mask = topk_idx == e                             # [T,k] bool
            if not bool(mask.any()):
                continue
            tok, slot = mask.nonzero(as_tuple=True)          # [n], [n]
            h = self.experts.forward_expert(x.index_select(0, tok), e)
            out.index_add_(0, tok, h * topk_w[tok, slot].unsqueeze(1))
        return out

    @property
    def param_bytes(self) -> int:
        return self.experts.param_bytes
