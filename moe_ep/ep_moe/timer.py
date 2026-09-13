"""CUDA-event phase timing, NVTX ranges, and the reporting tables.

How to time a collective correctly
----------------------------------
Three rules, all of which are easy to get wrong and all of which make the
resulting table lie:

1. **Record events inside the iteration, read them after one sync at the
   end.**  ``cudaEventElapsedTime`` needs the end event to have completed;
   reading ``elapsed_time`` mid-iteration forces a host sync, which serialises
   the pipeline and inflates every phase that follows it.  So
   ``PhaseTimer.phase`` only *records* and ``resolve()`` reads, and
   ``resolve()`` must be called after ``torch.cuda.synchronize()``.

2. **One sync per measured iteration, never inside a phase.**

3. **A collective's cost is the MAX across ranks, not the mean.**  The
   collective finishes when the slowest participant arrives.  Every timing
   this module reports is therefore accompanied by a ``straggler = max/mean``
   column; at balanced routing it is slightly above 1.0 and under a skewed
   router it is where the imbalance actually shows up.
"""

from __future__ import annotations

import contextlib
import re
import statistics
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Sequence

import torch

#: Phase names are prefixed with the layer index (``L0.router``) so a stack of
#: layers stays attributable.  ``PhaseTimer.aggregate_by_kind`` folds them back
#: into one row per phase type for the printed table.
_LAYER_PREFIX = re.compile(r"^L\d+\.")


def timed(timer: Optional["PhaseTimer"], name: str):
    """``timer.phase(name)`` when a timer is present, a no-op otherwise.

    Lets call sites stay on one line without an ``if`` and without the
    ``with A if cond else B:`` idiom, which parses correctly but is easy to
    misread.
    """
    return timer.phase(name) if timer is not None else contextlib.nullcontext()


class PhaseTimer:
    """Accumulates per-phase GPU times (in milliseconds) across iterations."""

    def __init__(self, nvtx: bool = False, enabled: bool = True) -> None:
        self.nvtx = nvtx
        self.enabled = enabled
        self._pending: List[tuple] = []          # (name, depth, start_ev, end_ev)
        self._depth = 0
        self.times: "OrderedDict[str, List[float]]" = OrderedDict()
        self.depths: Dict[str, int] = {}

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        if self.nvtx:
            torch.cuda.nvtx.range_push(name)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        depth = self._depth
        start.record()
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            end.record()
            if self.nvtx:
                torch.cuda.nvtx.range_pop()
            self._pending.append((name, depth, start, end))

    def resolve(self) -> None:
        """Read all recorded events.  Call *after* ``torch.cuda.synchronize()``."""
        if not self.enabled:
            return
        for name, depth, start, end in self._pending:
            self.times.setdefault(name, []).append(start.elapsed_time(end))
            self.depths.setdefault(name, depth)
        self._pending.clear()

    def reset(self) -> None:
        self._pending.clear()
        self.times.clear()
        self.depths.clear()

    # -- reporting --------------------------------------------------------
    def aggregate_by_kind(self) -> "OrderedDict[str, List[float]]":
        """Fold ``L0.router`` / ``L1.router`` into a single ``router`` series.

        The per-iteration values are summed index-wise, which is what makes the
        aggregate row comparable across models with different layer counts.
        """
        agg: "OrderedDict[str, List[float]]" = OrderedDict()
        for name, values in self.times.items():
            kind = _LAYER_PREFIX.sub("", name)
            if kind in agg and len(agg[kind]) == len(values):
                agg[kind] = [a + b for a, b in zip(agg[kind], values)]
            else:
                agg.setdefault(kind, list(values))
        return agg

    def top_level_names(self) -> List[str]:
        return [n for n in self.times if self.depths.get(n, 0) == 0]

    def total_ms(self) -> float:
        """Sum of the depth-0 phases.  Sub-phases overlap their parent."""
        return sum(statistics.fmean(self.times[n]) for n in self.top_level_names())

    def stats(self, aggregate: bool = False) -> Dict[str, Dict[str, float]]:
        series = self.aggregate_by_kind() if aggregate else self.times
        out: Dict[str, Dict[str, float]] = {}
        for name, values in series.items():
            if not values:
                continue
            depth = self.depths.get(name, 0)
            if aggregate:
                # After folding, a name is top-level iff none of its per-layer
                # originals were nested.
                depth = min(
                    (self.depths.get(n, 0) for n in self.times if _LAYER_PREFIX.sub("", n) == name),
                    default=0,
                )
            out[name] = {
                "ms_mean": statistics.fmean(values),
                "ms_p50": statistics.median(values),
                "ms_max": max(values),
                "calls": len(values),
                "depth": depth,
            }
        return out

    def format_table(self, aggregate: bool = True) -> str:
        stats = self.stats(aggregate=aggregate)
        if not stats:
            return "(no phases recorded)"
        total = self.total_ms() or 1e-9
        rows = []
        for name, s in stats.items():
            indent = "  " * s["depth"]
            share = (s["ms_mean"] / total * 100.0) if s["depth"] == 0 else float("nan")
            rows.append(
                [
                    f"{indent}{name}",
                    f"{s['ms_mean']:.3f}",
                    f"{s['ms_p50']:.3f}",
                    "-" if share != share else f"{share:.1f}",
                    str(int(s["calls"])),
                ]
            )
        header = ["phase", "ms(mean)", "ms(p50)", "%tot", "calls"]
        return format_table(header, rows, title="phase breakdown (per iteration)")

    def as_records(self) -> List[dict]:
        return [{"phase": k, **v} for k, v in self.stats().items()]


# --------------------------------------------------------------------------
# Timing a plain block (warmup / correctness loops)
# --------------------------------------------------------------------------
@contextlib.contextmanager
def cuda_timed() -> Iterator[torch.cuda.Event]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield end
    end.record()
    torch.cuda.synchronize()


def elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return start.elapsed_time(end)


# --------------------------------------------------------------------------
# Table formatting
# --------------------------------------------------------------------------
def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    title: Optional[str] = None,
    aligns: Optional[Sequence[str]] = None,
) -> str:
    """Minimal dependency-free ASCII table."""
    cols = len(headers)
    cells = [[str(c) for c in r] for r in rows]
    widths = [len(str(h)) for h in headers]
    for r in cells:
        for i in range(cols):
            widths[i] = max(widths[i], len(r[i]) if i < len(r) else 0)

    if aligns is None:
        aligns = ["<"] + [">"] * (cols - 1)

    def line(char: str = "-") -> str:
        return "+".join(char * (w + 2) for w in widths)

    def fmt(vals: Sequence[str]) -> str:
        return "|".join(
            f" {vals[i]:{aligns[i]}{widths[i]}} " for i in range(cols)
        )

    out = []
    if title:
        out.append(title)
    out.append(line("-"))
    out.append(fmt([str(h) for h in headers]))
    out.append(line("="))
    for r in cells:
        padded = list(r) + [""] * (cols - len(r))
        out.append(fmt(padded))
    out.append(line("-"))
    return "\n".join(out)


def format_matrix(
    mat: Sequence[Sequence[object]],
    labels: Sequence[str],
    title: str,
    fmt: str = "{:.2f}",
) -> str:
    headers = [""] + [f"->{l}" for l in labels]
    rows = []
    for i, row in enumerate(mat):
        rows.append(
            [labels[i]] + [fmt.format(v) if v is not None else "-" for v in row]
        )
    return format_table(headers, rows, title=title)
