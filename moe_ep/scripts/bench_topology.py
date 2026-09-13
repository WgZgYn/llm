"""Link-level measurements: P2P bandwidth matrix and all_to_all scaling.

No MoE here on purpose.  This answers the question the MoE numbers depend on:
*what is the link capable of, and at which message size does all_to_all stop
being latency-bound?*  The decode-like regime lives far below that knee, which
is why its effective bandwidth collapses -- and you can only say that with
these measurements in hand.

Run it once normally and once with ``NCCL_P2P_DISABLE=1``; the difference is
the P2P/NVLink contribution and makes the topology experiment interpretable.

Usage::

    torchrun --standalone --nproc_per_node=4 scripts/bench_topology.py
    NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 scripts/bench_topology.py
"""

from __future__ import annotations

import argparse

import sys

import _common  # noqa: F401
import torch

from _common import make_writer
from ep_moe import init_distributed
from ep_moe.dist_utils import shutdown
from ep_moe.report import environment_facts
from ep_moe.timer import format_table
from ep_moe.topology import (
    all_to_all_bandwidth,
    one_way_bandwidth_matrix,
    p2p_access_matrix,
    summarise_matrix,
)

#: Message sizes in KB of payload per rank per peer.  Spans the decode regime
#: (a few KB) through the prefill regime (MBs), so the latency/bandwidth knee
#: is visible in one table.
DEFAULT_SIZES_KB = (1, 4, 16, 64, 256, 1024, 4096)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--sizes-kb", type=int, nargs="+", default=list(DEFAULT_SIZES_KB))
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--p2p-size-mb", type=int, default=32)
    p.add_argument("--skip-p2p", action="store_true")
    p.add_argument("--skip-a2a", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=str(_common.PROJECT_ROOT / "out"))
    p.add_argument("--no-write", action="store_true")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    ctx = init_distributed()
    writer = make_writer(
        args, default_tag=args.tag or "bench_topology", is_main=ctx.is_main
    )

    try:
        if ctx.is_main:
            print("=" * 78)
            print(f"link measurements -- ep_size={ctx.world}")
            print("=" * 78)
            import os

            print(f"  NCCL_P2P_DISABLE  {os.environ.get('NCCL_P2P_DISABLE', '(unset)')}")
            print(f"  NCCL_IB_DISABLE   {os.environ.get('NCCL_IB_DISABLE', '(unset)')}")
            print()

        # ---- P2P access + bandwidth -------------------------------------
        if not args.skip_p2p:
            access = p2p_access_matrix()
            n = torch.cuda.device_count()
            if ctx.is_main:
                rows = [
                    [f"gpu{i}"] + ["yes" if access[i][j] else "-" for j in range(n)]
                    for i in range(n)
                ]
                print(format_table([""] + [f"gpu{j}" for j in range(n)], rows,
                                   title="can_device_access_peer"))

            hidden = args.hidden
            nrows = max(1, args.p2p_size_mb * 2**20 // (hidden * 2))
            mat = one_way_bandwidth_matrix(
                ctx, nrows, hidden, iters=args.iters, warmup=args.warmup
            )
            host = mat.cpu().tolist()
            summary = summarise_matrix(mat, ctx.world)

            if ctx.is_main:
                rows = [
                    [f"gpu{i}"]
                    + [("-" if i == j else f"{host[i][j]:.2f}") for j in range(ctx.world)]
                    for i in range(ctx.world)
                ]
                print()
                print(
                    format_table(
                        [""] + [f"->gpu{j}" for j in range(ctx.world)],
                        rows,
                        title=(
                            f"one-way P2P payload bandwidth GB/s "
                            f"({nrows * hidden * 2 / 2**20:.1f} MB per transfer)"
                        ),
                    )
                )
                if "intra_mean_GBps" in summary:
                    print(f"\n  intra-group mean : {summary['intra_mean_GBps']:.2f} GB/s")
                    print(f"  cross-group mean : {summary['cross_mean_GBps']:.2f} GB/s")
                    print(f"  ratio            : {summary['intra_over_cross']:.2f}x")
                writer.add(
                    {
                        "kind": "topology_p2p",
                        "world_size": ctx.world,
                        "matrix_GBps": host,
                        **summary,
                        **environment_facts(),
                    }
                )

        # ---- all_to_all scaling ------------------------------------------
        if not args.skip_a2a:
            if ctx.is_main:
                print("\n### all_to_all bandwidth vs message size ###\n")
            rows = []
            a2a_records = []
            for kb in args.sizes_kb:
                rows_per_peer = max(1, (kb * 1024) // (args.hidden * 2))
                res = all_to_all_bandwidth(
                    ctx,
                    rows_per_peer,
                    args.hidden,
                    iters=args.iters,
                    warmup=args.warmup,
                )
                rows.append(
                    [
                        f"{res['payload_MB_per_rank']:.4f}",
                        f"{rows_per_peer}",
                        f"{res['ms_median']:.4f}",
                        f"{res['ms_min']:.4f}",
                        f"{res['per_rank_GBps']:.2f}",
                        f"{res['aggregate_GBps']:.2f}",
                    ]
                )
                a2a_records.append({**res, "kind": "topology_a2a",
                                    "world_size": ctx.world,
                                    "hidden": args.hidden})

            if ctx.is_main:
                print(
                    format_table(
                        ["payload MB/rank", "rows/peer", "ms(median)", "ms(min)",
                         "per-rank GB/s", "aggregate GB/s"],
                        rows,
                        title="uniform all_to_all",
                    )
                )
                print("\n  per-rank = bytes this rank sent / time.")
                print("  aggregate = world x that; compare it against the P2P matrix "
                      "above to see how far below the link the collective runs.")
                small = [r for r in a2a_records if r["payload_MB_per_rank"] < 0.05]
                if small:
                    print(
                        f"\n  small-message floor: ~{small[0]['ms_min']:.3f} ms at "
                        f"{small[0]['payload_MB_per_rank']:.4f} MB/rank. That fixed "
                        f"cost is what dominates the decode-like regime -- it does "
                        f"not shrink with token count."
                    )
            for r in a2a_records:
                writer.add({**r, **environment_facts()})

    finally:
        shutdown()

    if ctx.is_main:
        writer.write_csv()
        print(f"\nartifacts written to {writer.out_dir}/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
