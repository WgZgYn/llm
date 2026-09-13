"""Link-level measurements: P2P access, P2P bandwidth, all_to_all bandwidth.

These are the measurements that make experiment (d) interpretable.  "The
all_to_all took 3 ms" means nothing on its own; "it took 3 ms and moved 8 MB,
against a measured 22 GB/s link" tells you whether the collective is
bandwidth-bound or latency-bound -- which is the entire difference between the
prefill-like and decode-like regimes.

Bandwidth definitions are stated explicitly wherever a number is produced,
because "GB/s" is ambiguous: all-reduce benchmarks conventionally report a
``busbw`` inflated by ``2(R-1)/R``, single-direction P2P does not, and mixing
the two makes a topology look better or worse than it is.
"""

from __future__ import annotations

import statistics
import time
from typing import Dict, List

import torch
import torch.distributed as dist


def p2p_access_matrix() -> List[List[bool]]:
    """``can_device_access_peer[i][j]`` over the local devices.

    Not guaranteed symmetric, so the full ordered matrix is built rather than a
    triangle.
    """
    n = torch.cuda.device_count()
    return [
        [bool(torch.cuda.can_device_access_peer(i, j)) for j in range(n)]
        for i in range(n)
    ]


def one_way_bandwidth_matrix(
    ctx,
    nrows: int,
    hidden: int,
    iters: int = 20,
    warmup: int = 5,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Median **one-way payload** bandwidth per directed pair, in GB/s.

    Uses a ``batch_isend_irecv`` ping-pong rather than a collective: P2P does
    not require other ranks to participate, so there is no matching problem and
    no way to hang on a stray collective.

    Timed on the **receiver**, whose ``wait()`` returns when the data has
    actually landed; a sender's returns when its buffer is free.  The figure is
    ``payload / t`` -- a genuine one direction number, so it reads lower than a
    ``busbw`` figure from an all-reduce benchmark.  That is expected, not a bug.
    """
    dev = ctx.device
    mat = torch.zeros(ctx.world, ctx.world, dtype=torch.float64, device=dev)
    payload = nrows * hidden * torch.tensor([], dtype=dtype).element_size()

    for i in range(ctx.world):
        for j in range(ctx.world):
            if i == j:
                continue
            samples = []
            for it in range(warmup + iters):
                dist.barrier()
                t0 = time.perf_counter()
                if ctx.rank == i:
                    buf = torch.ones(nrows, hidden, dtype=dtype, device=dev)
                    reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, buf, j)])
                    for r in reqs:
                        r.wait()
                elif ctx.rank == j:
                    buf = torch.empty(nrows, hidden, dtype=dtype, device=dev)
                    reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, buf, i)])
                    for r in reqs:
                        r.wait()
                dt = 0.0
                if ctx.rank == j:
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                if it >= warmup and ctx.rank == j:
                    samples.append(dt)
            if ctx.rank == j and samples:
                med = statistics.median(samples)
                if med > 0:
                    mat[i][j] = payload / med / 1e9
            dist.barrier()

    dist.all_reduce(mat, op=dist.ReduceOp.SUM)
    return mat


def all_to_all_bandwidth(
    ctx,
    rows_per_peer: int,
    hidden: int,
    iters: int = 20,
    warmup: int = 5,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """Bandwidth of a uniform-split all_to_all, at a given message size.

    Reports two views of the same measurement:

    * ``per_rank_GBps`` -- bytes this rank put on the wire divided by time.
    * ``aggregate_GBps`` -- ``world *`` the above, i.e. how much the group moved
      in total.  This is the number to compare against a link's raw bandwidth.

    Also returns the latency floor, which is what dominates the decode leg.
    """
    dev = ctx.device
    world = ctx.world
    b = torch.tensor([], dtype=dtype).element_size()
    send = torch.ones(rows_per_peer * world, hidden, dtype=dtype, device=dev)
    recv = torch.empty_like(send)
    payload = send.numel() * b

    def once() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_to_all_single(recv, send)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        once()
    ts = [once() for _ in range(iters)]

    med = statistics.median(ts)
    return {
        "world": world,
        "rows_per_peer": rows_per_peer,
        "payload_MB_per_rank": payload / 2**20,
        "ms_median": med * 1e3,
        "ms_min": min(ts) * 1e3,
        "per_rank_GBps": (payload / med / 1e9) if med > 0 else 0.0,
        "aggregate_GBps": (payload * world / med / 1e9) if med > 0 else 0.0,
    }


def classify_pairs(world: int) -> Dict[str, List[tuple]]:
    """Split the ordered GPU pairs into the spec's intra/cross groups.

    Only meaningful for the 4-GPU layout the spec describes (0,1 on one socket
    and 2,3 on the other); returns everything under ``other`` for other sizes.
    """
    if world != 4:
        return {"intra": [], "cross": [], "other": [
            (i, j) for i in range(world) for j in range(world) if i != j
        ]}
    intra = [(0, 1), (1, 0), (2, 3), (3, 2)]
    cross = [(0, 2), (2, 0), (0, 3), (3, 0), (1, 2), (2, 1), (1, 3), (3, 1)]
    return {"intra": intra, "cross": cross, "other": []}


def summarise_matrix(mat: torch.Tensor, world: int) -> Dict[str, float]:
    host = mat.cpu().tolist()
    groups = classify_pairs(world)
    out: Dict[str, float] = {}
    vals = [host[i][j] for i in range(world) for j in range(world) if i != j]
    if vals:
        out["median_GBps"] = statistics.median(vals)
        out["min_GBps"] = min(vals)
        out["max_GBps"] = max(vals)
    if groups["intra"]:
        intra = [host[i][j] for i, j in groups["intra"]]
        cross = [host[i][j] for i, j in groups["cross"]]
        out["intra_mean_GBps"] = statistics.fmean(intra)
        out["cross_mean_GBps"] = statistics.fmean(cross)
        if out["cross_mean_GBps"] > 0:
            out["intra_over_cross"] = (
                out["intra_mean_GBps"] / out["cross_mean_GBps"]
            )
    return out
