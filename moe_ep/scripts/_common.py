"""Shared CLI plumbing for the ``scripts/`` entrypoints.

Importing this module has a side effect on purpose: it puts the project root on
``sys.path`` so ``import ep_moe`` works no matter how the script was launched.
``torchrun`` runs ``python -u scripts/foo.py``, which makes ``sys.path[0]`` the
*scripts* directory, not the project root -- so without this, every entrypoint
would need ``PYTHONPATH=.`` exported, and forgetting it produces a confusing
``ModuleNotFoundError`` on a machine you are debugging remotely.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from ep_moe.config import PRESETS, ModelConfig, RunConfig  # noqa: E402

PROJECT_ROOT = _ROOT


# ----------------------------------------------------------------------
# argparse builders
# ----------------------------------------------------------------------
def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument(
        "--preset",
        default="tiny",
        choices=sorted(PRESETS),
        help="shape preset; individual flags below override it",
    )
    g.add_argument("--hidden", type=int, default=None)
    g.add_argument("--ffn", type=int, default=None, help="SwiGLU intermediate width")
    g.add_argument("--num-experts", type=int, default=None)
    g.add_argument("--top-k", type=int, default=None)
    g.add_argument("--num-layers", type=int, default=2)
    g.add_argument("--seed", type=int, default=20250913)


def add_run_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("run")
    g.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="fp16 is the only tensor-core dtype V100 supports (Volta has no bf16)",
    )
    g.add_argument(
        "--allow-bf16",
        action="store_true",
        help="override the refusal to run bf16 on pre-Ampere hardware",
    )
    g.add_argument("--ep-size", type=int, default=None, help="must equal world_size")
    g.add_argument(
        "--tokens",
        type=int,
        default=4096,
        help="GLOBAL token count, split evenly across ranks",
    )
    g.add_argument(
        "--tokens-per-rank",
        type=int,
        default=None,
        help="bypass --tokens divisibility; sets the local count directly",
    )
    g.add_argument(
        "--skew",
        type=float,
        default=0.0,
        help="bias added to the logits of the first --skew-hot experts",
    )
    g.add_argument("--skew-hot", type=int, default=4)
    g.add_argument(
        "--balance",
        default="natural",
        choices=["natural", "uniform"],
        help="uniform forces a perfectly round-robin routing",
    )
    g.add_argument("--warmup", type=int, default=5)
    g.add_argument("--iters", type=int, default=20)
    g.add_argument("--nvtx", action="store_true", help="emit NVTX ranges (costs time)")
    g.add_argument(
        "--fuse-dispatch",
        action="store_true",
        help="one packed all_to_all instead of three (x, expert id, gate weight)",
    )
    g.add_argument(
        "--grouped-gemm",
        action="store_true",
        help="padded bmm over local experts instead of the per-expert loop",
    )
    g.add_argument(
        "--identity-experts",
        action="store_true",
        help="experts return x unchanged; isolates communication from maths",
    )
    g.add_argument("--deterministic", action="store_true")
    g.add_argument("--tag", default="")
    g.add_argument("--out", default=str(PROJECT_ROOT / "out"), help="output directory")
    g.add_argument("--no-write", action="store_true", help="print only, write nothing")


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------
def build_model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig.from_preset(
        args.preset,
        hidden=args.hidden,
        ffn=args.ffn,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_layers=getattr(args, "num_layers", None),
        seed=getattr(args, "seed", None),
        dtype=getattr(args, "dtype", None),
    )


def build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        tokens=args.tokens,
        tokens_per_rank=args.tokens_per_rank,
        skew=getattr(args, "skew", 0.0),
        skew_hot=getattr(args, "skew_hot", 4),
        warmup=getattr(args, "warmup", 5),
        iters=getattr(args, "iters", 20),
        nvtx=getattr(args, "nvtx", False),
        fuse_dispatch=getattr(args, "fuse_dispatch", False),
        grouped_gemm=getattr(args, "grouped_gemm", False),
        identity_experts=getattr(args, "identity_experts", False),
        deterministic=getattr(args, "deterministic", False),
        tag=getattr(args, "tag", ""),
        out_dir=getattr(args, "out", "out"),
    )


def validate_against_world(
    cfg: ModelConfig, args: argparse.Namespace, world: int
) -> None:
    """Fail early, with the exact corrected command, rather than mid-collective."""
    if getattr(args, "ep_size", None) is not None and args.ep_size != world:
        raise SystemExit(
            f"--ep-size {args.ep_size} does not match world_size {world}.\n"
            f"EP is pure here: relaunch with\n"
            f"    torchrun --standalone --nproc_per_node={args.ep_size} <script> ..."
        )
    try:
        cfg.validate(world, allow_bf16=getattr(args, "allow_bf16", False))
    except ValueError as exc:
        raise SystemExit(f"configuration error: {exc}") from None


def setup_determinism(enabled: bool, seed: int) -> None:
    if not enabled:
        return
    import os

    # cuBLAS needs this or it refuses to run deterministically on CUDA >= 10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------
# output helpers
# ----------------------------------------------------------------------
def make_writer(args: argparse.Namespace, default_tag: str, is_main: bool = True):
    from ep_moe.report import ResultWriter

    tag = args.tag or default_tag
    return ResultWriter(
        args.out, tag=tag, enabled=not args.no_write, is_main=is_main
    )


def print_rank0(ctx, *parts) -> None:
    if ctx.is_main:
        print(*parts, flush=True)


def section(title: str, enabled: bool = True) -> None:
    if not enabled:
        return
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


class Checker:
    """Accumulates OK/WARN/FAIL lines and decides the process exit code.

    ``is_main=False`` keeps counting but stops printing.  Under
    ``torchrun --nproc_per_node=4`` every rank shares one stdout, so four
    ungated printers interleave character by character into an unreadable
    mess -- but the *count* still has to be maintained on every rank, because
    the exit code comes from an all-reduce of it.
    """

    def __init__(self, is_main: bool = True) -> None:
        self.is_main = is_main
        self.failures = 0
        self.warnings = 0

    def _emit(self, tag: str, label: str, detail: str) -> None:
        if self.is_main:
            print(f"{tag} {label}" + (f"  {detail}" if detail else ""), flush=True)

    def ok(self, label: str, detail: str = "") -> None:
        self._emit("[ OK ]", label, detail)

    def warn(self, label: str, detail: str = "") -> None:
        self.warnings += 1
        self._emit("[WARN]", label, detail)

    def fail(self, label: str, detail: str = "") -> None:
        self.failures += 1
        self._emit("[FAIL]", label, detail)

    def check(self, condition: bool, label: str, detail: str = "") -> bool:
        (self.ok if condition else self.fail)(label, detail)
        return condition
