"""The four experiment tracks for the toy MoE EP demo.

Modes
-----
``measure``  one configuration, full phase/memory/communication report
``prefill``  ladder over large per-rank token counts (512 .. 8192)
``decode``   ladder over tiny per-rank token counts (1 .. 16) -- the leg where
             EP's fixed overhead dominates and its relative cost is worst
``mem``      memory decomposition only
``all``      prefill + decode + mem

EP size is not a knob here: it *is* ``world_size``, so the sweep is one
``torchrun --nproc_per_node=$EP`` per size (see ``scripts/run_all.sh``).

Usage::

    torchrun --standalone --nproc_per_node=4 scripts/bench_ep.py --preset tiny --mode all
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import _common  # noqa: F401
import torch
import torch.distributed as dist

from _common import (
    add_model_args,
    add_run_args,
    build_model_config,
    build_run_config,
    make_writer,
    setup_determinism,
    validate_against_world,
)
from ep_moe import init_distributed
from ep_moe.comm import actual_bytes, theoretical_bytes_per_rank
from ep_moe.config import ModelConfig, RunConfig
from ep_moe.dist_utils import EPLayout, all_gather_vector, shutdown
from ep_moe.init_utils import make_global_tokens, slice_for_rank
from ep_moe.model import (
    EPMoEStack,
    ForwardStats,
    estimate_model_bytes,
    memory_report,
    router_fingerprint,
)
from ep_moe.report import gpu_facts
from ep_moe.timer import PhaseTimer, format_table

PREFILL_LADDER = (512, 1024, 2048, 4096, 8192)
DECODE_LADDER = (1, 2, 4, 8, 16)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    add_run_args(p)
    p.add_argument(
        "--mode",
        default="measure",
        choices=["measure", "prefill", "decode", "mem", "all"],
    )
    p.add_argument("--decode-ladder", action="store_true",
                   help="alias for --mode decode with the default ladder")
    return p.parse_args()


# ======================================================================
def cross_rank_phases(ctx, timer: PhaseTimer) -> Dict[str, Dict[str, float]]:
    """Per-phase time on the slowest rank, plus the mean and the straggler ratio.

    A collective finishes when its slowest participant arrives, so ``max`` is
    the honest cost and ``max/mean`` is the imbalance signal.  Reporting only
    the local mean would make a skewed router look faster than it is.
    """
    stats = timer.stats(aggregate=True)
    names = list(stats)
    all_names = [None] * ctx.world
    dist.all_gather_object(all_names, names)
    union: List[str] = []
    for ns in all_names:
        for n in ns:
            if n not in union:
                union.append(n)
    if not union:
        return {}

    local = torch.tensor(
        [stats.get(n, {}).get("ms_mean", 0.0) for n in union],
        dtype=torch.float64,
        device=ctx.device,
    )
    gathered = all_gather_vector(local, ctx.world)          # [world, n_names]
    per_max = gathered.max(dim=0).values
    per_mean = gathered.mean(dim=0)

    out: Dict[str, Dict[str, float]] = {}
    for i, n in enumerate(union):
        mx = float(per_max[i])
        mn = float(per_mean[i])
        out[n] = {
            "ms_max_across_ranks": mx,
            "ms_mean_across_ranks": mn,
            "ms_local": stats.get(n, {}).get("ms_mean", 0.0),
            "calls": stats.get(n, {}).get("calls", 0),
            "straggler": (mx / mn) if mn > 0 else 1.0,
        }
    return out


def measure(
    ctx,
    cfg: ModelConfig,
    run: RunConfig,
    args,
    stack: EPMoEStack,
    T_local: int,
) -> dict:
    """Time ``--iters`` iterations after ``--warmup`` warmup iterations."""
    layout = stack.layout
    device = ctx.device

    x_global = make_global_tokens(cfg.seed, T_local * ctx.world, cfg.hidden,
                                  cfg.torch_dtype, device)
    x_local = slice_for_rank(x_global, ctx.rank, ctx.world, T_local)

    timer = PhaseTimer(nvtx=run.nvtx)

    # Warmup is not optional: the first all_to_all initialises the NCCL
    # communicator and the first cuBLAS call autotunes, so an unwarmed first
    # iteration can be an order of magnitude off.
    with torch.no_grad():
        for _ in range(run.warmup):
            stack(x_local, timer=None)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(run.iters):
            stack(x_local, timer=timer)
            torch.cuda.synchronize()          # one sync per iteration, never inside
            timer.resolve()

    mem = memory_report()
    phases = cross_rank_phases(ctx, timer)

    # A separate untimed pass for the side outputs.  Collecting the dispatch
    # context costs a host sync, so it must stay out of the timed region.
    stats = ForwardStats(keep_dispatches=True)
    with torch.no_grad():
        stack(x_local, timer=None, stats=stats)

    return {
        "x_local": x_local,
        "phases": phases,
        "mem": mem,
        "stats": stats,
        "timer": timer,
    }


def communication_record(ctx, cfg: ModelConfig, stats: ForwardStats, T_local: int) -> dict:
    """Measured bytes vs the closed form ``2 * T_local * k * H * dtype_bytes``."""
    b = cfg.dtype_bytes
    theory_per_layer = theoretical_bytes_per_rank(T_local, cfg.top_k, cfg.hidden, b)
    theory = theory_per_layer * cfg.num_layers

    tot_dispatch = tot_combine = tot_ids = 0.0
    rows_sent = rows_recv = 0
    for d in stats.dispatches:
        ab = actual_bytes(d["in_splits"], d["out_splits"], cfg.hidden, b)
        tot_dispatch += ab["dispatch_MB"]
        tot_combine += ab["combine_MB"]
        tot_ids += ab["expert_id_MB"]
        rows_sent += ab["dispatch_rows"]
        rows_recv += ab["combine_rows"]

    per_rank = {
        "theory_MB": theory / 2**20,
        "dispatch_MB": tot_dispatch,
        "combine_MB": tot_combine,
        "expert_id_MB": tot_ids,
        "actual_MB": tot_dispatch + tot_combine + tot_ids,
        "rows_sent": rows_sent,
        "rows_recv": rows_recv,
    }
    # Per-rank bytes vary with routing even though the mean does not -- that
    # divergence is the whole point of the imbalance experiment.
    vec = torch.tensor(
        [per_rank["actual_MB"], per_rank["dispatch_MB"], per_rank["combine_MB"]],
        dtype=torch.float64,
        device=ctx.device,
    )
    gathered = all_gather_vector(vec, ctx.world)
    per_rank["actual_MB_max"] = float(gathered[:, 0].max())
    per_rank["actual_MB_mean"] = float(gathered[:, 0].mean())
    per_rank["dispatch_MB_max"] = float(gathered[:, 1].max())
    return per_rank


def load_record(ctx, cfg: ModelConfig, stats: ForwardStats) -> dict:
    """Global expert load: gathered across ranks this is the true histogram."""
    if stats.expert_load is None:
        return {}
    local = stats.expert_load.to(torch.float64)
    gathered = all_gather_vector(local, ctx.world)          # [world, E]
    total = gathered.sum(dim=0)
    mean = float(total.mean())
    hot = int(total.argmax())
    per_rank = gathered.sum(dim=1)                          # tokens per rank

    return {
        "expert_load_total": [int(v) for v in total.tolist()],
        "expert_load_hottest_id": hot,
        "expert_load_hottest": int(total[hot]),
        "expert_load_mean": mean,
        "imbalance_factor": (float(total.max()) / mean) if mean > 0 else 0.0,
        "empty_experts_global": int((total == 0).sum()),
        "tokens_per_rank_max": int(per_rank.max()),
        "tokens_per_rank_min": int(per_rank.min()),
        "rank_load_straggler": (
            float(per_rank.max() / per_rank.mean()) if per_rank.mean() > 0 else 1.0
        ),
    }


# ======================================================================
def run_config(
    ctx,
    cfg: ModelConfig,
    run: RunConfig,
    args,
    T_local: int,
    label: str,
    writer,
) -> dict:
    layout = EPLayout(ctx.rank, ctx.world, cfg.num_experts, cfg.top_k)
    stack = EPMoEStack(
        cfg,
        layout,
        group=None,
        device=ctx.device,
        skew=run.skew,
        skew_hot=run.skew_hot,
        balance=args.balance,
        fuse=run.fuse_dispatch,
        grouped=run.grouped_gemm,
        identity=run.identity_experts,
    ).to(ctx.device)

    res = measure(ctx, cfg, run, args, stack, T_local)
    phases = res["phases"]
    # Every phase is recorded at depth 0 -- the layers are prefixed (L0.router)
    # and folded by aggregate_by_kind, but nothing nests -- so the sum over all
    # of them IS the total.  Summing a hardcoded whitelist here would silently
    # under-report the moment a phase is added or renamed.
    total_max = sum(p["ms_max_across_ranks"] for p in phases.values())
    comm = communication_record(ctx, cfg, res["stats"], T_local)
    load = load_record(ctx, cfg, res["stats"])

    # Effective bandwidth uses the hottest rank's bytes over the time the
    # communication phases actually took on the slowest rank.
    comm_ms = sum(
        phases.get(n, {}).get("ms_max_across_ranks", 0.0)
        for n in ("dispatch.a2a", "dispatch.split_a2a", "combine.a2a")
    )
    comm_bytes_max = comm["actual_MB_max"] * 2**20
    eff_gbps = (comm_bytes_max / (comm_ms / 1e3) / 1e9) if comm_ms > 0 else 0.0

    record = {
        "kind": "bench_ep",
        "tag": label,
        "ep_size": ctx.world,
        "experts_per_rank": layout.experts_per_rank,
        "tokens_per_rank": T_local,
        "tokens_global": T_local * ctx.world,
        "hidden": cfg.hidden,
        "ffn": cfg.ffn,
        "num_experts": cfg.num_experts,
        "top_k": cfg.top_k,
        "num_layers": cfg.num_layers,
        "dtype": cfg.dtype,
        "skew": run.skew,
        "skew_hot": run.skew_hot,
        "balance": args.balance,
        "fuse_dispatch": run.fuse_dispatch,
        "grouped_gemm": run.grouped_gemm,
        "identity_experts": run.identity_experts,
        "iters": run.iters,
        "warmup": run.warmup,
        "total_ms": total_max,
        "phases": phases,
        "comm_total_MB": comm["actual_MB"],
        "comm_eff_GBps": eff_gbps,
        "comm_theory_MB": comm["theory_MB"],
        "comm_actual_MB_max": comm["actual_MB_max"],
        "comm_actual_MB_mean": comm["actual_MB_mean"],
        "comm": comm,
        "peak_allocated_MB": res["mem"]["max_allocated_MB"],
        "peak_reserved_MB": res["mem"]["max_reserved_MB"],
        "expert_param_MB": stack.expert_param_bytes / 2**20,
        "replicated_param_MB": stack.replicated_param_bytes / 2**20,
        "router_fingerprint": router_fingerprint(stack),
        **load,
    }

    if ctx.is_main:
        print_phase_table(label, record, phases)
        if comm_ms > 0:
            print(
                format_table(
                    ["metric", "value"],
                    [
                        ["theory MB/rank", f"{comm['theory_MB']:.3f}"],
                        ["actual MB/rank (mean)", f"{comm['actual_MB_mean']:.3f}"],
                        ["actual MB/rank (max)", f"{comm['actual_MB_max']:.3f}"],
                        ["dispatch MB", f"{comm['dispatch_MB']:.3f}"],
                        ["combine MB", f"{comm['combine_MB']:.3f}"],
                        ["expert-id side channel MB", f"{comm['expert_id_MB']:.3f}"],
                        ["comm time on slowest rank ms", f"{comm_ms:.3f}"],
                        ["effective GB/s (max bytes)", f"{eff_gbps:.2f}"],
                        ["actual/theory", f"{comm['actual_MB'] / max(comm['theory_MB'], 1e-9):.3f}"],
                    ],
                    title="communication volume",
                )
            )
        print()

    writer.add(record)
    return record


def print_phase_table(label: str, record: dict, phases: Dict[str, Dict[str, float]]) -> None:
    if not phases:
        return
    rows = []
    for name, p in phases.items():
        rows.append(
            [
                name,
                f"{p['ms_max_across_ranks']:.3f}",
                f"{p['ms_mean_across_ranks']:.3f}",
                f"{p['ms_local']:.3f}",
                f"{p['straggler']:.3f}",
                str(int(p["calls"])),
            ]
        )
    print(
        format_table(
            ["phase", "ms(max rank)", "ms(mean rank)", "ms(local)", "straggler", "calls"],
            rows,
            title=(
                f"[{label}] ep={record['ep_size']} T_local={record['tokens_per_rank']} "
                f"T_global={record['tokens_global']} skew={record['skew']} "
                f"| total(max rank) {record['total_ms']:.3f} ms"
            ),
        )
    )


def print_memory_table(ctx, cfg: ModelConfig, records: List[dict]) -> None:
    if not ctx.is_main or not records:
        return
    est = estimate_model_bytes(cfg, ctx.world)
    rows = [
        [
            r["tag"],
            str(r["ep_size"]),
            f"{r['expert_param_MB']:.1f}",
            f"{r['replicated_param_MB']:.2f}",
            f"{r['peak_allocated_MB']:.1f}",
            f"{r['peak_reserved_MB']:.1f}",
        ]
        for r in records
    ]
    print(
        format_table(
            ["config", "ep", "expert MB/rank", "replicated MB", "peak alloc MB",
             "peak reserved MB"],
            rows,
            title="memory decomposition",
        )
    )
    print(f"  analytic expert weights at this EP: {est['expert_weights_MB']:.1f} MB/rank")
    print(f"  analytic saving vs EP=1          : {est['saving_vs_ep1_MB']:.1f} MB/rank")
    print("  note: peak_reserved includes NCCL's own buffers, which never shrink "
          "with EP;\n        that gap is the honest cost floor. Compare across EP "
          "sizes in out/ to see it.")


# ======================================================================
def main() -> int:
    args = parse_args()
    if args.decode_ladder:
        args.mode = "decode"
    ctx = init_distributed()
    writer = make_writer(
        args, default_tag=args.tag or f"bench_ep{ctx.world}", is_main=ctx.is_main
    )

    try:
        cfg = build_model_config(args)
        run = build_run_config(args)
        validate_against_world(cfg, args, ctx.world)
        setup_determinism(args.deterministic, cfg.seed)

        facts = gpu_facts(ctx.device)
        if ctx.is_main:
            print("=" * 78)
            print(f"Toy MoE EP benchmark -- ep_size={ctx.world}")
            print("=" * 78)
            print(f"  {cfg.summary()}")
            print(f"  device            {facts['name']} sm_{facts['capability']} "
                  f"({facts['sm_count']} SMs)")
            print(f"  memory            {facts['total_MB']:.0f} MB, "
                  f"{facts['free_MB']:.0f} MB free")
            print(f"  skew              {run.skew} hot={run.skew_hot} "
                  f"balance={args.balance}")
            print(f"  iters             {run.iters} (+{run.warmup} warmup)")
            print()

        est = estimate_model_bytes(cfg, ctx.world)
        if ctx.is_main and est["expert_weights_MB"] > facts["total_MB"] * 0.6:
            print(f"  [WARN] expert weights alone are "
                  f"{est['expert_weights_MB']:.0f} MB/rank of "
                  f"{facts['total_MB']:.0f} MB. Expect OOM at large token counts; "
                  f"try a smaller --preset or a larger EP.\n")

        records: List[dict] = []
        mode = args.mode

        if mode in ("measure", "mem"):
            T_local = run.local_tokens(ctx.world)
            records.append(
                run_config(ctx, cfg, run, args, T_local,
                           args.tag or f"ep{ctx.world}_T{T_local}", writer)
            )
        else:
            if mode in ("prefill", "all"):
                if ctx.is_main:
                    print("\n### prefill-like ladder (large token batches) ###\n")
                for T in PREFILL_LADDER:
                    r = RunConfig(**{**run.__dict__, "tokens_per_rank": T})
                    records.append(
                        run_config(ctx, cfg, r, args, T,
                                   f"prefill_ep{ctx.world}_T{T}", writer)
                    )
            if mode in ("decode", "all"):
                if ctx.is_main:
                    print("\n### decode-like ladder (tiny per-rank batches) ###\n")
                for T in DECODE_LADDER:
                    r = RunConfig(**{**run.__dict__, "tokens_per_rank": T})
                    records.append(
                        run_config(ctx, cfg, r, args, T,
                                   f"decode_ep{ctx.world}_T{T}", writer)
                    )

        if mode in ("mem", "all") and ctx.is_main:
            print()
            print_memory_table(ctx, cfg, records)

        if mode in ("mem", "all"):
            sm = torch.cuda.max_memory_allocated(ctx.device)
            mx = torch.tensor([float(sm)], dtype=torch.float64, device=ctx.device)
            g = all_gather_vector(mx, ctx.world)
            if ctx.is_main:
                print(f"\n  peak allocated per rank (MB): "
                      f"{[round(float(v) / 2**20, 1) for v in g[:, 0].tolist()]}")

    finally:
        shutdown()

    if ctx.is_main:
        writer.write_csv()
        notes = [
            "分阶段耗时为跨 rank 的 max（集合通信在最后一个 rank 到达时才结束），"
            "straggler = max/mean 就是路由不均衡的可观测信号。",
            "decode-like 一档（T_local 1..16）里，通信量与启动开销同量级，"
            "GB/s 会远低于 P2P 链路上限——这正是小 batch 下 EP 相对代价最差的原因。",
            "peak_reserved 比 peak_allocated 更接近真实占用：NCCL 的内部 buffer "
            "不进 torch allocator，也基本不随 EP 缩小。",
        ]
        md = writer.write_markdown(
            title=f"Toy MoE EP -- ep_size={ctx.world}",
            notes=notes,
            table_columns=[
                "tag", "ep_size", "tokens_per_rank", "top_k", "skew", "dtype",
                "total_ms", "imbalance_factor", "rank_load_straggler",
                "comm_theory_MB", "comm_actual_MB_mean", "comm_eff_GBps",
                "peak_allocated_MB", "peak_reserved_MB",
            ],
        )
        print(f"\nartifacts written to {writer.out_dir}/")
        print(f"  {writer.jsonl_path().name}")
        print(f"  {writer.csv_path().name}")
        if md:
            print(f"  {md.name}")
        print(f"\n  read back with: pandas.read_json('{writer.jsonl_path()}', lines=True)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
