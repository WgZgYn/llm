"""Preflight and self-diagnosis for a fresh 4x V100 node.

Run this **first** on the target machine.  It is sectioned and non-fatal: every
section is wrapped so one failure never masks the rest, and the exit code is set
from an all-reduced failure flag so a rank that dies does not leave the others
hanging inside a collective.

The order matters.  Sections 5 and 6 validate the ``all_to_all`` split semantics
with known data *before* any MoE code runs -- if the split/transpose arithmetic
in ``ep_moe/comm.py`` is wrong, that is where you find out, with a clear
expected-vs-actual, rather than as a mysterious numeric difference later.

Usage::

    torchrun --standalone --nproc_per_node=4 scripts/check_env.py
    torchrun --standalone --nproc_per_node=4 scripts/check_env.py --p2p-size-mb 64
"""

from __future__ import annotations

import argparse
import os
import sys

import _common  # noqa: F401  (sys.path bootstrap; must precede ep_moe)
import torch
import torch.distributed as dist

from _common import Checker, add_model_args, make_writer, section
from ep_moe import ModelConfig, init_distributed
from ep_moe.dist_utils import EPLayout, nccl_version, shutdown
from ep_moe.model import (
    EPMoEStack,
    check_expert_partition,
    expert_fingerprint_map,
    memory_report,
    router_fingerprint,
)
from ep_moe.report import (
    environment_facts,
    gpu_facts,
    nvidia_smi_topo,
    pci_topology,
)
from ep_moe.timer import format_table
from ep_moe.topology import (
    one_way_bandwidth_matrix,
    p2p_access_matrix,
    summarise_matrix,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--p2p-size-mb", type=int, default=32, help="P2P ping-pong payload")
    p.add_argument("--p2p-iters", type=int, default=20)
    p.add_argument("--p2p-warmup", type=int, default=5)
    p.add_argument("--skip-p2p", action="store_true", help="skip the bandwidth matrices")
    p.add_argument("--timeout-minutes", type=int, default=3)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=str(_common.PROJECT_ROOT / "out"))
    p.add_argument("--no-write", action="store_true")
    return p.parse_args()

# ======================================================================
def section_1_environment(chk: Checker, ctx) -> None:
    section("1. environment")
    print(f"  torch             {torch.__version__}")
    print(f"  torch.version.cuda{torch.version.cuda:>6}")
    print(f"  cudnn             {torch.backends.cudnn.version()}")
    print(f"  nccl              {nccl_version()}")
    print(f"  backend           {ctx.backend}")
    print(f"  cuda available    {torch.cuda.is_available()}")
    print(f"  device count      {torch.cuda.device_count()}")

    if not dist.is_nccl_available():
        chk.fail("NCCL backend", "not available in this torch build")
    else:
        chk.ok("NCCL backend available")

    print()
    print("  relevant environment:")
    keys = [
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES",
    ]
    for k in keys:
        print(f"    {k:<26} {os.environ.get(k, '(unset)')}")
    nccl_vars = sorted(k for k in os.environ if k.startswith(("NCCL_", "TORCH_NCCL_")))
    if nccl_vars:
        for k in nccl_vars:
            print(f"    {k:<26} {os.environ[k]}")
    else:
        print("    (no NCCL_* / TORCH_NCCL_* variables set)")

    if not os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING"):
        chk.warn(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING unset",
            "a mismatched collective will hang instead of raising. "
            "See scripts/env.example.sh",
        )

# ======================================================================
def section_2_visibility(chk: Checker, ctx) -> None:
    section("2. CUDA_VISIBLE_DEVICES consistency")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = torch.cuda.device_count()
    if cvd is None:
        chk.ok("CUDA_VISIBLE_DEVICES unset", f"{visible} device(s) visible")
    else:
        listed = len([x for x in cvd.split(",") if x.strip()])
        if listed == visible:
            chk.ok("CUDA_VISIBLE_DEVICES matches device_count", f"({listed})")
        else:
            chk.fail(
                "CUDA_VISIBLE_DEVICES mismatch",
                f"lists {listed} device(s) but torch sees {visible}",
            )
        if listed < ctx.world:
            chk.warn(
                "fewer visible devices than ranks",
                f"LOCAL_RANK can exceed {listed}; unset it for --nproc_per_node="
                f"{ctx.world}",
            )

# ======================================================================
def section_3_topology(chk: Checker, ctx, args) -> None:
    section("3. topology invariants")
    if ctx.world != torch.cuda.device_count():
        chk.warn(
            "world_size != device_count",
            f"{ctx.world} ranks, {torch.cuda.device_count()} GPUs. Fine if you "
            f"meant it; for the 4xV100 box use --nproc_per_node=4",
        )
    else:
        chk.ok("world_size == device_count", f"({ctx.world})")

    if args.ep_size is not None:
        chk.check(
            args.ep_size == ctx.world,
            "--ep-size matches world_size",
            f"({args.ep_size})",
        )

    facts = gpu_facts(ctx.device)
    print()
    print(f"  rank {ctx.rank} on {facts['name']} (cuda:{ctx.local_rank})")
    print(f"    compute capability  {facts['capability']}")
    print(f"    SM count            {facts['sm_count']}")
    print(f"    memory              {facts['total_MB']:.0f} MB total, "
          f"{facts['free_MB']:.0f} MB free")
    print(f"    bf16 native         {facts['bf16_native']}")

    major = int(facts["capability"].split(".")[0])
    if major < 8:
        chk.warn(
            f"pre-Ampere device (sm_{facts['capability'].replace('.', '')})",
            "bf16 has no hardware path here; --dtype fp16 is the correct "
            "default and bf16 timings would be meaningless",
        )
    else:
        chk.ok("Ampere or newer", "bf16 is native")

    numa = pci_topology()
    if numa:
        print()
        print("  NUMA placement (spec expects 0,1 on one socket and 2,3 on the other):")
        for k, v in sorted(numa.items()):
            print(f"    {k:<26} {v}")

    topo = nvidia_smi_topo()
    if topo:
        print()
        print("  nvidia-smi topo -m:")
        for line in topo.rstrip().splitlines():
            print(f"    {line}")
    else:
        chk.warn("nvidia-smi topo -m unavailable", "cannot confirm the link types")

# ======================================================================
def section_4_p2p_access(chk: Checker, ctx) -> None:
    section("4. P2P access matrix")
    n = torch.cuda.device_count()
    local = p2p_access_matrix()
    gathered = [None] * ctx.world
    dist.all_gather_object(gathered, local)
    mat = gathered[0]

    print("  can_device_access_peer[i][j]  (rows: from, cols: to)")
    rows = []
    for i in range(n):
        rows.append([f"gpu{i}"] + ["yes" if mat[i][j] else "-" for j in range(n)])
    print(format_table([""] + [f"gpu{j}" for j in range(n)], rows))

    if n >= 4:
        intra = [(0, 1), (1, 0), (2, 3), (3, 2)]
        cross = [(0, 2), (2, 0), (0, 3), (3, 0), (1, 2), (2, 1), (1, 3), (3, 1)]
        bad_intra = [p for p in intra if not mat[p[0]][p[1]]]
        if bad_intra:
            chk.warn("intra-group P2P unavailable", f"pairs {bad_intra} -- "
                     "these will fall back to SYS and be slow")
        else:
            chk.ok("intra-group P2P available", "pairs (0,1) and (2,3)")
        ok_cross = [p for p in cross if mat[p[0]][p[1]]]
        print(f"    cross-group pairs with P2P: {len(ok_cross)}/{len(cross)}")

# ======================================================================
def section_5_p2p_bandwidth(chk: Checker, ctx, args) -> None:
    section("5. measured P2P bandwidth (one-way payload GB/s, timed on receiver)")
    hidden = 4096
    nrows = max(1, args.p2p_size_mb * 2**20 // (hidden * 2))
    print(f"  payload {nrows * hidden * 2 / 2**20:.1f} MB per transfer, "
          f"{args.p2p_iters} iters after {args.p2p_warmup} warmup")
    print()

    mat = one_way_bandwidth_matrix(
        ctx, nrows, hidden, iters=args.p2p_iters, warmup=args.p2p_warmup
    )
    host = mat.cpu().tolist()
    rows = []
    for i in range(ctx.world):
        rows.append(
            [f"gpu{i}"]
            + [("-" if i == j else f"{host[i][j]:.2f}") for j in range(ctx.world)]
        )
    print(format_table([""] + [f"->gpu{j}" for j in range(ctx.world)], rows))

    summary = summarise_matrix(mat, ctx.world)
    med = summary.get("median_GBps", 0.0)
    print(f"\n  median one-way bandwidth: {med:.2f} GB/s")
    cold = [
        f"{i}->{j}"
        for i in range(ctx.world)
        for j in range(ctx.world)
        if i != j and 0 < host[i][j] < 0.05 * med
    ]
    if cold:
        chk.warn(
            "suspiciously slow pairs",
            f"{cold} -- possible cold start; increase --p2p-warmup",
        )

    if "intra_mean_GBps" in summary:
        mi = summary["intra_mean_GBps"]
        mc = summary["cross_mean_GBps"]
        ratio = summary.get("intra_over_cross", 0.0)
        print(f"  intra-group (0-1, 2-3) mean : {mi:.2f} GB/s")
        print(f"  cross-group (SYS)      mean : {mc:.2f} GB/s")
        print(f"  ratio intra/cross           : {ratio:.2f}x")
        if ratio < 1.1:
            chk.warn(
                "intra and cross-group bandwidth look similar",
                "NVLink/PIX may not be in use, or NCCL picked a different path; "
                "re-run with NCCL_P2P_DISABLE=1 to see the SYS-only floor",
            )
        else:
            chk.ok("intra-group is measurably faster than cross-group")
        print("  compare against `nvidia-smi topo -m` above to interpret this")

    if ctx.is_main:
        return {"matrix_GBps": host, **summary}
    return {}

# ======================================================================
def section_6_alltoall_semantics(chk: Checker, ctx) -> None:
    section("6. all_to_all_single split semantics with known data")
    world, rank, dev = ctx.world, ctx.rank, ctx.device

    # Deliberately asymmetric, different on every rank, with zeros mixed in: a
    # uniform split vector would pass even if the transpose step were missing,
    # which is exactly the bug this section exists to catch.
    if world == 1:
        in_splits = [4]
    else:
        in_splits = [((rank * 2) + j) % 3 for j in range(world)]
        if sum(in_splits) == 0:
            in_splits[0] = 1
    nrows = sum(in_splits)
    offsets = [0]
    for s in in_splits:
        offsets.append(offsets[-1] + s)

    print(f"  rank {rank}: in_splits={in_splits} (sum={nrows})")

    # Row r of the block destined for rank d carries rank*1000 + r, and column 1
    # is the destination tag, so both the ordering and the contents of every
    # received block are checkable.
    inp = torch.zeros(nrows, 2, dtype=torch.float64, device=dev)
    for d in range(world):
        lo, hi = offsets[d], offsets[d + 1]
        if hi > lo:
            inp[lo:hi, 0] = (
                torch.arange(hi - lo, dtype=torch.float64, device=dev) + rank * 1000
            )
            inp[lo:hi, 1] = float(d)

    # Exchange the split vector, then check it against the closed form computed
    # from every rank's own vector -- self-consistent even when the zero-guard
    # above fired on some rank.
    all_in = [None] * world
    dist.all_gather_object(all_in, in_splits)
    expected_out = [all_in[i][rank] for i in range(world)]

    in_t = torch.tensor(in_splits, dtype=torch.int64, device=dev)
    out_t = torch.empty_like(in_t)
    dist.all_to_all_single(out_t, in_t)
    out_splits = [int(v) for v in out_t.tolist()]

    chk.check(
        out_splits == expected_out,
        "split transpose: out_splits[i] == in_splits_of_rank_i[my_rank]",
        f"got {out_splits}, expected {expected_out}",
    )

    out = torch.zeros(sum(out_splits), 2, dtype=torch.float64, device=dev)
    dist.all_to_all_single(out, inp, out_splits, in_splits)

    ok = True
    pos = 0
    for src in range(world):
        n = out_splits[src]
        blk = out[pos : pos + n]
        pos += n
        if n == 0:
            continue
        expect_vals = torch.arange(n, dtype=torch.float64, device=dev) + src * 1000
        if not torch.equal(blk[:, 0], expect_vals):
            ok = False
            print(
                f"    [FAIL] block from rank {src}: expected "
                f"{expect_vals.tolist()} got {blk[:, 0].tolist()}"
            )
        if not bool(torch.all(blk[:, 1] == float(rank))):
            ok = False
            print(
                f"    [FAIL] block from rank {src}: destination tag "
                f"{blk[:, 1].unique().tolist()} != {rank}"
            )
    chk.check(
        ok, "received blocks come in ascending source order with intact contents"
    )

    # The reverse trip must reproduce the original exactly.  The values are whole
    # numbers, so torch.equal is the right test; allclose would be too loose.
    back = torch.zeros(nrows, 2, dtype=torch.float64, device=dev)
    dist.all_to_all_single(back, out, in_splits, out_splits)
    chk.check(
        torch.equal(back, inp),
        "inverse all_to_all with swapped splits round-trips exactly",
    )

    # Cross-check against torch's independent list-based API.  The output list
    # must be sized by the *received* counts, not the sent ones.
    if world > 1:
        in_list = list(inp.split(in_splits, dim=0))
        out_list = [
            torch.zeros(out_splits[i], 2, dtype=torch.float64, device=dev)
            for i in range(world)
        ]
        dist.all_to_all(out_list, in_list)
        ref = torch.cat([out_list[i] for i in range(world)], dim=0)
        chk.check(
            torch.equal(ref, out),
            "matches dist.all_to_all (list form) -- independent implementation",
        )

    if 0 in in_splits or 0 in out_splits:
        chk.ok("zero-size split handled", "(some peer sent/received nothing)")

# ======================================================================
def section_7_ep_roundtrip(chk: Checker, ctx, args) -> None:
    section("7. full EP dispatch/combine round trip with identity experts")
    world = ctx.world

    # Identity experts + softmax gate weights (which sum to 1 by construction)
    # means the correct output IS the input.  Every line of comm.py runs -- the
    # pair expansion, the split exchange, the local grouping, inv_perm, the
    # swapped return splits, the scatter-add -- with no expert maths to hide a
    # bug behind.
    for label, skew, top_k in (
        (f"balanced k={min(2, world)}", 0.0, min(2, world)),
        ("skewed k=1", 4.0, 1),
    ):
        cfg = ModelConfig(
            hidden=64,
            ffn=128,
            num_experts=world,
            top_k=top_k,
            num_layers=1,
            dtype="fp32",
            seed=args.seed,
        )
        layout = EPLayout(ctx.rank, world, cfg.num_experts, cfg.top_k)
        stack = EPMoEStack(
            cfg,
            layout,
            group=None,
            device=ctx.device,
            skew=skew,
            skew_hot=max(1, cfg.num_experts // 2),
            identity=True,
        ).to(ctx.device)

        T = 32
        x = torch.randn(T, cfg.hidden, dtype=torch.float32, device=ctx.device)
        with torch.no_grad():
            out = stack(x)
        err = float((out - x).abs().max())
        chk.check(
            err < 1e-5,
            f"identity round trip ({label}) reproduces the input",
            f"max_abs_err={err:.3e}",
        )

    # The skewed case above deliberately starves whole ranks: with top_k=1 and
    # skew_hot = world//2, half the ranks own no hot expert and receive zero
    # tokens.  That exercises the recv_n == 0 path, which is the one place a
    # zero-count NCCL transfer happens.  Flag it explicitly, because if this
    # build mishandles it the failure looks exotic otherwise.
    print("  note: the skewed case drives some ranks to recv_n == 0 -- a zero-count")
    print("        NCCL transfer. If it failed above, that is the likely cause.")

    # Expert placement: every expert initialised exactly once across the group.
    cfg = ModelConfig(
        hidden=32, ffn=64, num_experts=world, top_k=1, num_layers=1,
        dtype="fp32", seed=args.seed,
    )
    layout = EPLayout(ctx.rank, world, cfg.num_experts, cfg.top_k)
    stack = EPMoEStack(cfg, layout, group=None, device=ctx.device).to(ctx.device)

    maps = [None] * world
    dist.all_gather_object(maps, expert_fingerprint_map(stack.layers[0].experts))
    problems = check_expert_partition(maps, cfg.num_experts)
    chk.check(not problems, "experts tile 0..E-1 exactly once across ranks",
              "; ".join(problems) if problems else "")

    fps = [None] * world
    dist.all_gather_object(fps, router_fingerprint(stack))
    chk.check(
        len(set(fps)) == 1,
        "router weights are identical on every rank",
        f"fingerprints={fps}" if len(set(fps)) != 1 else f"({fps[0]})",
    )

# ======================================================================
def section_8_summary(chk: Checker, ctx) -> None:
    section("8. summary")
    print(f"  section failures : {chk.failures}")
    print(f"  section warnings : {chk.warnings}")
    mem = memory_report()
    print(f"  memory (rank {ctx.rank})   allocated {mem['allocated_MB']:.0f} MB, "
          f"reserved {mem['reserved_MB']:.0f} MB, "
          f"device free {mem['device_free_MB']:.0f} MB")

# ======================================================================
def main() -> int:
    args = parse_args()
    ctx = init_distributed(timeout_minutes=args.timeout_minutes)
    chk = Checker()
    writer = make_writer(args, default_tag=args.tag or f"check_env{ctx.world}")
    p2p_record: dict = {}

    try:
        if ctx.is_main:
            print("=" * 78)
            print("Toy MoE EP -- environment preflight")
            print("=" * 78)
            print(f"  world_size={ctx.world}  nproc_per_node should equal this")
            print()

        dist.barrier()
        for fn in (
            lambda: section_1_environment(chk, ctx),
            lambda: section_2_visibility(chk, ctx),
            lambda: section_3_topology(chk, ctx, args),
            lambda: section_4_p2p_access(chk, ctx),
        ):
            dist.barrier()
            try:
                fn()
            except Exception as exc:  # keep going; one bad section is not fatal
                chk.fail("section raised", f"{type(exc).__name__}: {exc}")

        if not args.skip_p2p:
            dist.barrier()
            try:
                p2p_record = section_5_p2p_bandwidth(chk, ctx, args) or {}
            except Exception as exc:
                chk.fail("P2P bandwidth section raised",
                         f"{type(exc).__name__}: {exc}")

        for fn in (
            lambda: section_6_alltoall_semantics(chk, ctx),
            lambda: section_7_ep_roundtrip(chk, ctx, args),
        ):
            dist.barrier()
            try:
                fn()
            except Exception as exc:
                chk.fail("section raised", f"{type(exc).__name__}: {exc}")

        dist.barrier()
        section_8_summary(chk, ctx)

        if ctx.is_main:
            writer.add(
                {
                    "kind": "check_env",
                    "world_size": ctx.world,
                    "failures": chk.failures,
                    "warnings": chk.warnings,
                    "router_fingerprint_ok": True,
                    **p2p_record,
                    **environment_facts(),
                }
            )

    finally:
        total = torch.tensor([chk.failures], dtype=torch.int32, device=ctx.device)
        dist.all_reduce(total, op=dist.ReduceOp.MAX)
        failures = int(total.item())
        shutdown()

    if ctx.is_main and writer.enabled:
        writer.write_csv()
        print(f"\nartifacts written to {writer.out_dir}/")

    if failures:
        print(f"\n{failures} check(s) FAILED -- do not trust benchmark numbers "
              f"until these are resolved.")
        return 1
    print("\nall checks passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
