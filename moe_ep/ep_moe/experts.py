"""Local expert FFN bank and its two compute strategies.

Weights are stored as one stacked ``[E_local, H, F]`` parameter per projection
rather than a ``ModuleList`` of experts: it keeps the parameter accounting
trivial (``sum(p.numel())`` is the honest number the memory table reports) and
makes the padded grouped-GEMM path a single ``bmm``.

Both paths must produce identical results; ``--impl both`` in
``scripts/verify_ep.py`` asserts that.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .comm import DispatchContext
from .config import ModelConfig
from .init_utils import expert_weight
from .timer import PhaseTimer


def _stacked(cfg: ModelConfig, start: int, count: int, device: torch.device):
    """Stack ``count`` experts starting at global id ``start``."""
    gate = torch.stack(
        [
            expert_weight(cfg.seed, start + i, (cfg.hidden, cfg.ffn), cfg.torch_dtype, device)
            for i in range(count)
        ]
    )
    up = torch.stack(
        [
            expert_weight(cfg.seed, start + i, (cfg.hidden, cfg.ffn), cfg.torch_dtype, device)
            for i in range(count)
        ]
    )
    down = torch.stack(
        [
            expert_weight(cfg.seed, start + i, (cfg.ffn, cfg.hidden), cfg.torch_dtype, device)
            for i in range(count)
        ]
    )
    return gate, up, down


class ExpertBank(nn.Module):
    """A contiguous range of global experts, held on one rank.

    ``expert_start`` is the *global* id of local index 0.  The weights depend
    only on ``(cfg.seed, global_id)``, so the same expert carries identical
    weights on every rank and in the dense reference -- the property the whole
    correctness comparison rests on.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        expert_start: int,
        count: int,
        device: torch.device,
        identity: bool = False,
    ) -> None:
        super().__init__()
        self.expert_start = expert_start
        self.count = count
        self.identity = identity
        self.cfg = cfg

        if identity:
            # Diagnostic mode: the "expert" is the identity function.  Running
            # the full dispatch/combine with no maths at all isolates any
            # remaining comm bug, which is exactly what check_env.py needs.
            self.register_buffer("_dummy", torch.zeros(1, dtype=cfg.torch_dtype))
            return

        gate, up, down = _stacked(cfg, expert_start, count, device)
        self.w_gate = nn.Parameter(gate)
        self.w_up = nn.Parameter(up)
        self.w_down = nn.Parameter(down)

    # ------------------------------------------------------------------
    @property
    def param_bytes(self) -> int:
        if self.identity:
            return 0
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def forward_expert(self, x: torch.Tensor, e: int) -> torch.Tensor:
        """Explicit per-expert forward (used by the dense reference too)."""
        if self.identity:
            return x
        h = F.silu(x @ self.w_gate[e]) * (x @ self.w_up[e])
        return h @ self.w_down[e]

    # ------------------------------------------------------------------
    # Strategy A: per-expert loop
    # ------------------------------------------------------------------
    def forward_loop(
        self, sorted_x: torch.Tensor, ctx: DispatchContext
    ) -> torch.Tensor:
        """One GEMM group per local expert, sliced from the grouped input.

        This is the default because the kernel count maps directly onto expert
        load: a starved expert contributes nothing and a hot one dominates, so
        routing imbalance shows up in the timing instead of being averaged away.
        Empty experts are skipped -- a local skip, never a collective skip.
        """
        out = torch.empty_like(sorted_x)
        slices = list(ctx.expert_slices())
        # ``out`` is uninitialised, so a row missed by every slice would silently
        # become NaN garbage rather than an error.  The slices are a partition of
        # [0, recv_n) by construction (counts come from a bincount of the same
        # tensor), so this costs a python-level sum and catches any future
        # refactor that breaks the invariant.
        covered = sum(hi - lo for _, lo, hi in slices)
        if covered != ctx.recv_n:
            raise RuntimeError(
                f"expert slices cover {covered} rows but recv_n={ctx.recv_n}; "
                f"counts={ctx.counts.tolist()}"
            )
        for e, lo, hi in slices:
            out[lo:hi] = self.forward_expert(sorted_x[lo:hi], e)
        return out

    # ------------------------------------------------------------------
    # Strategy B: padded grouped GEMM
    # ------------------------------------------------------------------
    def forward_padded(
        self,
        sorted_x: torch.Tensor,
        ctx: DispatchContext,
        pad_limit_factor: float = 2.0,
    ) -> torch.Tensor:
        """Pad every expert to the same row count and run one ``bmm``.

        Fewer, larger kernels -- faster when routing is balanced, and it hides
        imbalance, which is exactly why it is *not* the default here.  The
        padding waste is ``E_local * max_n`` rows instead of ``recv_n``; the
        guard below refuses to allocate when that gets out of hand (an extreme
        skew at a large token count is how you OOM a 16 GB card).
        """
        if self.identity:
            return sorted_x

        n_experts = self.count
        slices = list(ctx.expert_slices())
        if not slices:
            return torch.empty_like(sorted_x)

        max_n = max(hi - lo for _, lo, hi in slices)
        if max_n == 0:
            return torch.empty_like(sorted_x)

        padded_rows = n_experts * max_n
        if padded_rows > pad_limit_factor * max(1, ctx.recv_n):
            # Fall back rather than allocate a buffer many times the real
            # payload.  Silent here would be worse: the timing would silently
            # stop being comparable with the loop path.
            if os.environ.get("MOE_EP_VERBOSE", "0") == "1":
                print(
                    f"[padded] fallback to loop: padded_rows={padded_rows} "
                    f"> {pad_limit_factor} x recv_n={ctx.recv_n}"
                )
            return self.forward_loop(sorted_x, ctx)

        H, Fdim = self.cfg.hidden, self.cfg.ffn
        pad_x = torch.zeros(n_experts, max_n, H, dtype=sorted_x.dtype, device=sorted_x.device)
        for e, lo, hi in slices:
            pad_x[e, : hi - lo] = sorted_x[lo:hi]

        h = F.silu(torch.bmm(pad_x, self.w_gate)) * torch.bmm(pad_x, self.w_up)
        pad_out = torch.bmm(h, self.w_down)                      # [E, max_n, H]

        out = torch.empty_like(sorted_x)
        for e, lo, hi in slices:
            out[lo:hi] = pad_out[e, : hi - lo]
        return out

    # ------------------------------------------------------------------
    def forward(
        self,
        sorted_x: torch.Tensor,
        ctx: DispatchContext,
        grouped: bool = False,
        timer: Optional[PhaseTimer] = None,
    ) -> torch.Tensor:
        if grouped:
            return self.forward_padded(sorted_x, ctx)
        return self.forward_loop(sorted_x, ctx)
