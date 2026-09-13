"""Correctness: does expert-parallel output match a dense reference?

The reference puts **all** experts on every rank and does no communication at
all.  It shares the router and expert initialisation with the EP model (see
``ep_moe/init_utils.py``), so any difference between the two is a difference in
the *communication*, which is the only thing this script is testing.

Every rank builds its own reference and compares against its own slice of the
output, so no gathering is needed and the check runs at full parallelism.  The
cost is that each rank holds the full expert bank; ``--max-reference-MB`` guards
against picking a configuration where that does not fit.

Usage::

    for ep in 1 2 4; do
      torchrun --standalone --nproc_per_node=$ep scripts/verify_ep.py --preset tiny
    done
"""

from __future__ import annotations

import argparse
import sys

import _common  # noqa: F401
import torch
import torch.distributed as dist

from _common import (
    Checker,
    add_model_args,
    add_run_args,
    build_model_config,
    build_run_config,
    make_writer,
    setup_determinism,
    validate_against_world,
)
from ep_moe import init_distributed
from ep_moe.dist_utils import EPLayout, shutdown
from ep_moe.init_utils import make_global_tokens, slice_for_rank
from ep_moe.model import (
    DenseMoEStack,
    EPMoEStack,
    check_expert_partition,
    estimate_model_bytes,
    expert_fingerprint_map,
    memory_report,
    router_fingerprint,
)
from ep_moe.timer import format_table

TOLERANCES = {
    "fp32": {"atol": 1e-5, "rtol": 1e-4},
    "fp16": {"atol": 2e-2, "rtol": 2e-2},
    "bf16": {"atol": 5e-2, "rtol": 5e-2},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    add_run_args(p)
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--rtol", type=float, default=None)
    p.add_argument("--max-reference-MB", type=float, default=6000.0)
    p.add_argument("--skip-dense", action="store_true",
                   help="only check fingerprints and routing agreement")
    return p.parse_args()


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Report numbers, not a boolean.

    ``allclose`` alone tells you nothing useful when it fails on a machine you
    are debugging over ssh.
    """
    a32, b32 = a.float(), b.float()
    d = (a32 - b32).abs()
    denom = b32.abs().clamp_min(1e-12)
    flat_idx = int(d.reshape(-1).argmax())
    return {
        "max_abs_err": float(d.max()),
        "mean_abs_err": float(d.mean()),
        "max_rel_err": float((d / denom).max()),
        "rel_l2": float(
            d.norm() / b32.norm().clamp_min(1e-12)
        ),
        "worst_flat_index": flat_idx,
        "n_elements": a.numel(),
    }


def main() -> int:
    args = parse_args()
    ctx = init_distributed()
    chk = Checker(is_main=ctx.is_main)
    writer = make_writer(
        args, default_tag=f"verify_ep{ctx.world}", is_main=ctx.is_main
    )

    try:
        world = ctx.world
        cfg = build_model_config(args)
        run = build_run_config(args)
        validate_against_world(cfg, args, world)
        setup_determinism(args.deterministic, cfg.seed)

        tol = TOLERANCES[cfg.dtype]
        atol = args.atol if args.atol is not None else tol["atol"]
        rtol = args.rtol if args.rtol is not None else tol["rtol"]

        T_local = run.local_tokens(world)
        T_global = T_local * world

        est = estimate_model_bytes(cfg, world)
        dense_need = est["naive_ep1_expert_MB"]
        ep_need = est["expert_weights_MB"]
        total_need = dense_need + ep_need if not args.skip_dense else ep_need
        free_mb = memory_report()["device_free_MB"]

        if ctx.is_main:
            print("=" * 78)
            print("EP vs dense reference")
            print("=" * 78)
            print(f"  config          {cfg.summary()}")
            print(f"  ep_size         {world}")
            print(f"  tokens          global={T_global} per_rank={T_local}")
            print(f"  skew            {run.skew} (hot {run.skew_hot}), "
                  f"balance={args.balance}")
            print(f"  tolerance       atol={atol} rtol={rtol}")
            print(f"  memory estimate dense={dense_need:.0f} MB + ep={ep_need:.0f} MB "
                  f"= {total_need:.0f} MB, device free {free_mb:.0f} MB")
            print()

        if total_need > min(args.max_reference_MB, free_mb * 0.9):
            raise SystemExit(
                f"refusing to run: needs about {total_need:.0f} MB but the limit "
                f"is {min(args.max_reference_MB, free_mb * 0.9):.0f} MB. "
                f"Use a smaller --preset, fewer --num-layers, or --skip-dense."
            )

        # ---- build models ------------------------------------------------
        layout = EPLayout(ctx.rank, world, cfg.num_experts, cfg.top_k)
        ep_stack = EPMoEStack(
            cfg,
            layout,
            group=None,
            device=ctx.device,
            skew=run.skew,
            skew_hot=run.skew_hot,
            balance=args.balance,
            fuse=run.fuse_dispatch,
            grouped=run.grouped_gemm,
        ).to(ctx.device)

        # ---- 1. weights agree where they must ----------------------------
        maps = [None] * world
        dist.all_gather_object(maps, expert_fingerprint_map(ep_stack.layers[0].experts))
        problems = check_expert_partition(maps, cfg.num_experts)
        chk.check(not problems, "experts tile 0..E-1 exactly once",
                  "; ".join(problems) if problems else "")

        fps = [None] * world
        dist.all_gather_object(fps, router_fingerprint(ep_stack))
        chk.check(len(set(fps)) == 1, "router identical across ranks",
                  f"({fps[0]})" if len(set(fps)) == 1 else f"{fps}")

        # ---- 2. tokens ----------------------------------------------------
        x_global = make_global_tokens(cfg.seed, T_global, cfg.hidden,
                                      cfg.torch_dtype, ctx.device)
        x_local = slice_for_rank(x_global, ctx.rank, world, T_local)

        # ---- 3. routing agreement ----------------------------------------
        ep_idx, _ = ep_stack.layers[0].router(x_local)
        dense_idx = None
        dense_stack = None
        if not args.skip_dense:
            dense_stack = DenseMoEStack(
                cfg,
                device=ctx.device,
                skew=run.skew,
                skew_hot=run.skew_hot,
                balance=args.balance,
            ).to(ctx.device)
            dense_idx, _ = dense_stack.layers[0].router(x_local)
        else:
            dense_stack = None
            dense_idx, _ = ep_stack.layers[0].router(x_local)

        chk.check(
            bool(torch.equal(ep_idx, dense_idx)),
            "router picks the same experts for the same tokens",
            "if this fails the tensor comparison below is meaningless",
        )

        # ---- 4. forward ---------------------------------------------------
        with torch.no_grad():
            ep_out = ep_stack(x_local)

        if dense_stack is None:
            if ctx.is_main:
                print("\n--skip-dense: skipping the numeric comparison.")
            records = [{"tag": args.tag or "verify", "ep_size": world,
                        "routing_match": True, "skipped_dense": True}]
        else:
            with torch.no_grad():
                ref_full = dense_stack(x_global)
            ref_out = ref_full.narrow(0, ctx.rank * T_local, T_local)

            stats = diff_stats(ep_out, ref_out)
            passed = bool(
                torch.allclose(ep_out.float(), ref_out.float(), atol=atol, rtol=rtol)
            )
            chk.check(
                passed,
                f"EP output matches dense reference ({cfg.dtype})",
                f"max_abs={stats['max_abs_err']:.3e} "
                f"rel_l2={stats['rel_l2']:.3e}",
            )

            if ctx.is_main:
                print()
                print(
                    format_table(
                        ["metric", "value"],
                        [
                            ["max_abs_err", f"{stats['max_abs_err']:.6e}"],
                            ["mean_abs_err", f"{stats['mean_abs_err']:.6e}"],
                            ["max_rel_err", f"{stats['max_rel_err']:.6e}"],
                            ["rel_l2", f"{stats['rel_l2']:.6e}"],
                            ["atol / rtol", f"{atol} / {rtol}"],
                            ["worst index", str(stats["worst_flat_index"])],
                        ],
                        title="EP vs dense",
                    )
                )
                if not passed:
                    # Show the worst few elements: on a remote box this is the
                    # difference between a diagnosis and another round trip.
                    d = (ep_out.float() - ref_out.float()).abs().reshape(-1)
                    k = min(5, d.numel())
                    worst = torch.topk(d, k).indices.tolist()
                    rows = [
                        [
                            str(i),
                            f"{ref_out.reshape(-1)[i].item():.6f}",
                            f"{ep_out.reshape(-1)[i].item():.6f}",
                            f"{d[i].item():.3e}",
                        ]
                        for i in worst
                    ]
                    print()
                    print(format_table(
                        ["flat_idx", "expected", "actual", "abs_err"],
                        rows,
                        title="worst elements",
                    ))

            records = [
                {
                    "tag": args.tag or "verify",
                    "ep_size": world,
                    "tokens_per_rank": T_local,
                    "tokens_global": T_global,
                    "dtype": cfg.dtype,
                    "skew": run.skew,
                    "atol": atol,
                    "rtol": rtol,
                    "passed": passed,
                    "routing_match": True,
                    **stats,
                }
            ]

        for rec in records:
            writer.add({
                "kind": "verify_ep",
                "world_size": world,
                "config": cfg.summary(),
                **rec,
            })

    finally:
        total = torch.tensor([chk.failures], dtype=torch.int32, device=ctx.device)
        dist.all_reduce(total, op=dist.ReduceOp.MAX)
        failures = int(total.item())
        shutdown()

    if ctx.is_main:
        writer.write_csv()
        print(f"\nartifacts: {writer.jsonl_path()}")
    if ctx.is_main:
        if failures:
            print(f"\n{failures} check(s) FAILED.")
        else:
            print("\ncorrectness checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
