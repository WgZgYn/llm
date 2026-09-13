"""Top-k router with a controllable routing-imbalance knob.

The router is **replicated**: every rank holds identical weights (see
:mod:`ep_moe.init_utils`) and routes only its own tokens.  There is no
communication in this step -- which is exactly why a rank-dependent seed is such
a nasty bug, and why ``check_env.py`` fingerprints the parameters across ranks.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .init_utils import router_weight

BALANCE_MODES = ("natural", "uniform")


class TopKRouter(nn.Module):
    """``logits = x @ W``; softmax over the ``k`` selected logits (Mixtral-style).

    ``skew`` is the knob experiment (d) needs: it adds a constant to the logits
    of the first ``skew_hot`` experts, pushing traffic onto them.  ``skew=0``
    leaves the natural (already non-uniform) router distribution.

    ``balance="uniform"`` short-circuits to a perfectly round-robin assignment,
    which is useful for isolating the *communication* cost of EP from the cost
    of imbalance.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int = 0,
        skew: float = 0.0,
        skew_hot: int = 0,
        balance: str = "natural",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if balance not in BALANCE_MODES:
            raise ValueError(f"balance must be one of {BALANCE_MODES}")

        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.skew = float(skew)
        self.skew_hot = int(skew_hot)
        self.balance = balance
        self.layer_idx = layer_idx
        self.weight = nn.Parameter(
            router_weight(
                cfg.seed,
                layer_idx,
                (cfg.hidden, cfg.num_experts),
                cfg.torch_dtype,
                device,
            )
        )

    def extra_repr(self) -> str:
        return (
            f"E={self.num_experts}, k={self.top_k}, skew={self.skew}, "
            f"skew_hot={self.skew_hot}, balance={self.balance}"
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(topk_idx [T,k] int64, topk_w [T,k] activation-dtype)``."""
        T = x.shape[0]

        if self.balance == "uniform":
            idx = (
                torch.arange(T * self.top_k, device=x.device, dtype=torch.int64)
                % self.num_experts
            ).reshape(T, self.top_k)
            w = torch.full(
                (T, self.top_k),
                1.0 / self.top_k,
                dtype=x.dtype,
                device=x.device,
            )
            return idx, w

        # Routing logits in fp32 even when the model runs in fp16: the router is
        # tiny, and top-k over fp16 logits can flip the selection between the
        # dense reference and the EP run, which would look like a comm bug.
        logits = x.float() @ self.weight.float()          # [T, E]
        if self.skew > 0.0 and self.skew_hot > 0:
            logits = logits.clone()
            logits[:, : self.skew_hot] += self.skew

        topk_logits, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_w = torch.softmax(topk_logits, dim=-1).to(x.dtype)
        return topk_idx, topk_w


# ----------------------------------------------------------------------
# Load statistics
# ----------------------------------------------------------------------
def expert_load(topk_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Token count per global expert (device tensor, length ``num_experts``)."""
    return torch.bincount(topk_idx.reshape(-1), minlength=num_experts)


def imbalance_factor(counts: torch.Tensor) -> float:
    """``max/mean`` over experts.  1.0 is perfectly balanced.

    Returned as a python float, so it costs one host sync -- call it once per
    measured configuration, never inside the timing loop.
    """
    c = counts.float()
    mean = c.mean()
    if float(mean) <= 0.0:
        return 0.0
    return float(c.max() / mean)


def load_stats(counts: torch.Tensor) -> Dict[str, float]:
    c = counts.float()
    mean = float(c.mean())
    return {
        "expert_load_min": int(c.min().item()),
        "expert_load_max": int(c.max().item()),
        "expert_load_mean": mean,
        "imbalance_factor": (float(c.max()) / mean) if mean > 0 else 0.0,
        "empty_experts": int((c == 0).sum().item()),
    }
