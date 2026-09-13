"""All-to-all token dispatch and combine -- the heart of expert parallelism.

This is the only file in the project that calls ``all_to_all_single``, and it is
isolated for one reason: the split arithmetic is the single place where a
mistake produces *silently wrong numbers* instead of an exception, and the
traceback you eventually get points somewhere unhelpful.

The exchange
------------
Rank ``r`` holds ``T`` tokens and, for each, ``k`` chosen **global** expert ids.
Expert ``e`` lives on rank ``e // experts_per_rank``.  So each token has to make
``k`` round trips to the ranks that own its experts::

    x [T,H]  --router-->  topk_idx [T,k], topk_w [T,k]
             --dispatch (all_to_all)-->     recv [recv_n, H], grouped by local expert
             --local experts-->             out  [recv_n, H]
             --combine (all_to_all back)--> back [T*k, H]
             --scatter-add over k-->        out  [T,H]

Two things are easy to get wrong and are handled explicitly below.

**1. The output split sizes are the transpose of the input split sizes.**
``all_to_all_single(out, inp, output_split_sizes, input_split_sizes)`` sends
chunk ``j`` of ``inp`` (sized ``input_split_sizes[j]``) to rank ``j``, and
receives into ``out`` a concatenation where chunk ``i`` comes *from* rank ``i``
and has size ``output_split_sizes[i]``.  The number of rows rank ``i`` sends to
me is *its* ``input_split_sizes[my_rank]``, which I cannot know locally.  So the
split vector itself has to be exchanged first (:meth:`_exchange_splits`).
Omitting that step is the classic bug: under uniform routing the numbers happen
to be equal and everything looks fine, and the moment routing is skewed you get
either a shape mismatch or silently scrambled data.

**2. Every rank must call every collective, every time.**
This is SPMD.  A rank that returns early because it received zero tokens hangs
the other three inside NCCL.  All branches here are therefore *local* skips
(skipping a GEMM over an empty slice); none of them skip a collective.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist

from .dist_utils import EPLayout
from .timer import PhaseTimer, timed

#: Set MOE_EP_DEBUG=1 to turn on the internal consistency assertions.  They cost
#: a host sync each, so they are off by default -- but they are worth turning on
#: for the very first run on a new machine.
_DEBUG = os.environ.get("MOE_EP_DEBUG", "0") == "1"


@dataclass
class DispatchContext:
    """Metadata needed to bring the expert outputs home.

    Everything stays on-device: index tensors read back to the host for
    ``index_select``/``index_add_`` would add a sync per phase.
    """

    T: int
    k: int
    H: int
    n_pairs: int
    recv_n: int
    perm: torch.Tensor          # [T*k]     sorted slot -> original flat-pair index
    src_token: torch.Tensor     # [T*k]     flat-pair index -> token index
    recv_w: torch.Tensor        # [recv_n]  gate weights, receive order
    inv_perm: torch.Tensor      # [recv_n]  expert-sorted slot -> receive slot
    counts: torch.Tensor        # [E_local] tokens per local expert
    in_list: List[int] = field(default_factory=list)   # rows sent per destination
    out_list: List[int] = field(default_factory=list)  # rows received per source
    #: ``[(local_expert, lo, hi), ...]`` for non-empty experts only.  Computed
    #: eagerly during dispatch, where the host sync for the split lists has
    #: already been paid, so the expert loop does not pay a second one.
    slices: List[Tuple[int, int, int]] = field(default_factory=list)

    def expert_slices(self) -> Iterator[Tuple[int, int, int]]:
        """Yield ``(local_expert_idx, lo, hi)`` for non-empty local experts."""
        return iter(self.slices)

    def as_dict(self) -> dict:
        return {
            "T": self.T,
            "k": self.k,
            "H": self.H,
            "n_pairs": self.n_pairs,
            "recv_n": self.recv_n,
            "in_splits": self.in_list,
            "out_splits": self.out_list,
            "local_expert_counts": [int(c) for c in self.counts.tolist()],
        }


class AllToAllDispatcher:
    """Dispatch / combine for a single EP group."""

    def __init__(
        self,
        layout: EPLayout,
        group=None,
        fuse: bool = False,
        prefix: str = "",
    ) -> None:
        self.layout = layout
        self.group = group
        self.fuse = fuse
        #: Prefix for phase names, so a stack of layers stays attributable
        #: (``L0.dispatch.a2a`` etc.).  The report folds them back together.
        self.prefix = prefix

    # ------------------------------------------------------------------
    # Split bookkeeping
    # ------------------------------------------------------------------
    def _exchange_splits(self, in_splits: torch.Tensor) -> List[int]:
        """Exchange the split vector; returns *my* output split sizes.

        ``out_splits[i]`` = rows rank ``i`` will send me = rank ``i``'s
        ``in_splits[my_rank]``.  A plain all-to-all of a length-``world``
        vector: the splits of *that* exchange are uniform by construction, so
        no split arguments are needed.
        """
        out_splits = torch.empty_like(in_splits)
        dist.all_to_all_single(out_splits, in_splits, group=self.group)
        return [int(v) for v in out_splits.tolist()]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_w: torch.Tensor,
        timer: Optional[PhaseTimer] = None,
    ) -> Tuple[torch.Tensor, DispatchContext]:
        """Send each token to the ranks owning its ``k`` experts.

        Returns ``(sorted_x, ctx)`` where ``sorted_x`` is ``[recv_n, H]`` with
        rows grouped by *local expert id*, so the expert bank can slice it with
        contiguous ranges.
        """
        L = self.layout
        T, k = topk_idx.shape
        H = x.shape[1]
        n_pairs = T * k
        device = x.device

        # ---- 1. expand [T,H] into T*k (token, expert) pairs --------------
        with timed(timer, self.prefix + "dispatch.expand"):
            flat_expert = topk_idx.reshape(-1).contiguous()            # [T*k]
            flat_w = topk_w.reshape(-1).contiguous()                   # [T*k]
            src_token = torch.arange(T, device=device, dtype=torch.int64)
            src_token = src_token.repeat_interleave(k)                 # [T*k]
            dst_rank = L.expert_to_rank(flat_expert)                   # [T*k]

            # Stable sort clusters pairs by destination rank while preserving
            # relative order within a destination, keeping the reverse trip's
            # within-rank ordering deterministic.
            _, perm = torch.sort(dst_rank, stable=True)
            perm = perm.to(torch.int64).contiguous()
            send_src = src_token.index_select(0, perm).contiguous()
            send_x = x.index_select(0, send_src).contiguous()          # [T*k, H]
            send_exp = flat_expert.index_select(0, perm).contiguous()
            send_w = flat_w.index_select(0, perm).contiguous()

        with timed(timer, self.prefix + "dispatch.count"):
            in_splits = torch.bincount(dst_rank, minlength=L.world)    # [world] int64
            in_list = [int(v) for v in in_splits.tolist()]

        if _DEBUG:
            assert sum(in_list) == n_pairs, (
                f"split bookkeeping broken: sum(in_splits)={sum(in_list)} "
                f"!= T*k={n_pairs}"
            )

        # ---- 2. exchange the split vector (THE step people forget) ------
        with timed(timer, self.prefix + "dispatch.split_a2a"):
            out_list = self._exchange_splits(in_splits)
        recv_n = int(sum(out_list))

        # ---- 3. dispatch -------------------------------------------------
        if self.fuse:
            recv_x, recv_exp, recv_w = self._dispatch_fused(
                send_x, send_exp, send_w, in_list, out_list, recv_n, timer
            )
        else:
            with timed(timer, self.prefix + "dispatch.a2a"):
                recv_x = torch.empty(recv_n, H, dtype=x.dtype, device=device)
                recv_exp = torch.empty(recv_n, dtype=torch.int64, device=device)
                recv_w = torch.empty(recv_n, dtype=x.dtype, device=device)
                for buf, send in (
                    (recv_x, send_x),
                    (recv_exp, send_exp),
                    (recv_w, send_w),
                ):
                    dist.all_to_all_single(
                        buf, send, out_list, in_list, group=self.group
                    )

        # ---- 4. group by local expert ------------------------------------
        with timed(timer, self.prefix + "dispatch.group"):
            local_exp = (recv_exp - L.expert_start).contiguous()
            if _DEBUG and recv_n:
                assert int(local_exp.min()) >= 0 and int(local_exp.max()) < (
                    L.experts_per_rank
                ), (
                    "received an expert id this rank does not own -- the "
                    "expert->rank mapping is wrong"
                )
            _, local_perm = torch.sort(local_exp, stable=True)
            local_perm = local_perm.to(torch.int64).contiguous()
            counts = torch.bincount(
                local_exp, minlength=L.experts_per_rank
            ).to(torch.int64)
            inv_perm = torch.empty(recv_n, dtype=torch.int64, device=device)
            inv_perm.index_copy_(
                0, local_perm, torch.arange(recv_n, dtype=torch.int64, device=device)
            )
            sorted_x = recv_x.index_select(0, local_perm).contiguous()

            # One host sync here, charged to the dispatch phase.  Variable-size
            # all_to_all needs python split lists anyway, so the sync is already
            # on this path; folding the expert slices into it keeps it to one.
            counts_host = counts.tolist()
            slices: List[Tuple[int, int, int]] = []
            lo = 0
            for e, c in enumerate(counts_host):
                if c > 0:
                    slices.append((e, lo, lo + c))
                lo += c

        ctx = DispatchContext(
            T=T,
            k=k,
            H=H,
            n_pairs=n_pairs,
            recv_n=recv_n,
            perm=perm,
            src_token=src_token,
            recv_w=recv_w,
            inv_perm=inv_perm,
            counts=counts,
            in_list=in_list,
            out_list=out_list,
            slices=slices,
        )
        return sorted_x, ctx

    def _dispatch_fused(
        self, send_x, send_exp, send_w, in_list, out_list, recv_n, timer
    ):
        """Pack x / expert-id / weight into one all_to_all instead of three.

        Saves two collective launches per layer.  The expert ids are cast to the
        activation dtype, which is lossless here: ``num_experts`` is far below
        the fp16 integer range and the ids are small whole numbers.
        """
        dtype = send_x.dtype
        H = send_x.shape[1]
        with timed(timer, self.prefix + "dispatch.a2a"):
            packed = torch.cat(
                [send_x, send_exp.to(dtype).unsqueeze(1), send_w.unsqueeze(1)], dim=1
            ).contiguous()                                             # [T*k, H+2]
            recv = torch.empty(recv_n, H + 2, dtype=dtype, device=send_x.device)
            dist.all_to_all_single(recv, packed, out_list, in_list, group=self.group)
            recv_x = recv[:, :H].contiguous()
            recv_exp = recv[:, H].to(torch.int64).contiguous()
            recv_w = recv[:, H + 1].contiguous()
        return recv_x, recv_exp, recv_w

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------
    def combine(
        self,
        out_sorted: torch.Tensor,
        ctx: DispatchContext,
        timer: Optional[PhaseTimer] = None,
    ) -> torch.Tensor:
        """Send expert outputs home and scatter-add them back onto the tokens.

        The gate weights are applied *before* the reverse all-to-all: same
        arithmetic (the weight is per (token, expert) pair), strictly less data
        on the wire.
        """
        device = out_sorted.device

        with timed(timer, self.prefix + "combine.reorder"):
            out_recv = out_sorted.index_select(0, ctx.inv_perm)        # [recv_n, H]
            send_back = (out_recv * ctx.recv_w.unsqueeze(1)).contiguous()

        with timed(timer, self.prefix + "combine.a2a"):
            back = torch.empty(
                ctx.n_pairs, ctx.H, dtype=out_sorted.dtype, device=device
            )
            # Splits swapped relative to dispatch: what we received going out is
            # what we sent coming back.
            dist.all_to_all_single(
                back, send_back, ctx.in_list, ctx.out_list, group=self.group
            )

        with timed(timer, self.prefix + "combine.scatter_add"):
            out = torch.zeros(ctx.T, ctx.H, dtype=out_sorted.dtype, device=device)
            # perm[j] is the original flat-pair index of send-slot j, so the
            # token that produced row j of `back` is src_token[perm[j]].
            out.index_add_(0, ctx.src_token.index_select(0, ctx.perm), back)
        return out


# ----------------------------------------------------------------------
# Volume accounting (used by the benchmark tables)
# ----------------------------------------------------------------------
def theoretical_bytes_per_rank(
    tokens_per_rank: int, top_k: int, hidden: int, dtype_bytes: int
) -> int:
    """Dispatch payload + combine payload for one MoE layer, per rank.

    Independent of ``ep_size`` at fixed per-rank token count -- which is the
    point: the *volume* does not shrink with EP, only the number of peers it is
    spread over does.
    """
    return 2 * tokens_per_rank * top_k * hidden * dtype_bytes


def actual_bytes(
    in_list: List[int], out_list: List[int], hidden: int, dtype_bytes: int
) -> dict:
    """Measured per-rank payload, including the expert-id side channel.

    Takes the raw split lists rather than a :class:`DispatchContext` so callers
    can account for a recorded dispatch after the fact, without having to
    fabricate a context object.
    """
    dispatch_rows = sum(in_list)
    combine_rows = sum(out_list)
    dispatch_payload = dispatch_rows * hidden * dtype_bytes
    combine_payload = combine_rows * hidden * dtype_bytes
    id_overhead = dispatch_rows * 8  # int64 expert id travels alongside each row
    return {
        "dispatch_rows": dispatch_rows,
        "combine_rows": combine_rows,
        "dispatch_MB": dispatch_payload / 2**20,
        "combine_MB": combine_payload / 2**20,
        "expert_id_MB": id_overhead / 2**20,
        "total_MB": (dispatch_payload + combine_payload + id_overhead) / 2**20,
    }
