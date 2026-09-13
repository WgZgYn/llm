"""Output plumbing: JSON/CSV artifacts, terminal tables, markdown summary.

Every measured configuration produces one JSON record.  Records append to a
single JSONL file per run so the whole sweep reads back with one line::

    pandas.read_json("out/bench.jsonl", lines=True)

The markdown summary is generated from those records rather than written by
hand, so it cannot drift from the numbers it describes.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: Phase dicts reach the writer in two shapes and both key on "mean ms":
#: ``PhaseTimer.stats()`` produces ``ms_mean`` (one rank's local view), while
#: ``bench_ep.cross_rank_phases()`` produces ``ms_mean_across_ranks`` (the
#: gathered view).  Reading the wrong one used to raise KeyError in the
#: markdown path and -- worse -- silently write empty columns in the CSV path.
#: Both are accepted here so a phase table is never quietly blank.
_PHASE_MS_KEYS = ("ms_mean_across_ranks", "ms_mean")


def phase_mean_ms(phase: Any) -> Optional[float]:
    """Mean milliseconds for a phase dict of either shape, or None."""
    if not isinstance(phase, dict):
        return None
    for key in _PHASE_MS_KEYS:
        if key in phase:
            value = phase[key]
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion so a stray tensor can never break the dump."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

class ResultWriter:
    """Writes the per-configuration artifacts under ``out_dir``."""

    def __init__(
        self,
        out_dir: str | os.PathLike,
        tag: str = "run",
        enabled: bool = True,
        is_main: bool = True,
    ) -> None:
        self.enabled = enabled
        #: Under torchrun every rank constructs a writer, but only rank 0 may
        #: touch the files.  Four processes appending to one JSONL interleave
        #: at arbitrary byte offsets and corrupt it, so the guard lives here
        #: rather than at each call site where it would eventually be forgotten.
        self.is_main = is_main
        self.tag = tag
        self.out_dir = Path(out_dir)
        self.records: List[Dict[str, Any]] = []
        if self.enabled and self.is_main:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------
    def jsonl_path(self) -> Path:
        return self.out_dir / f"{self.tag}.jsonl"

    def csv_path(self) -> Path:
        return self.out_dir / f"{self.tag}.csv"

    def markdown_path(self) -> Path:
        return self.out_dir / f"{self.tag}_report.md"

    # -- writing ----------------------------------------------------------
    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        if not self.enabled or not self.is_main:
            return
        with open(self.jsonl_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

    def write_csv(self, rows: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        """Flatten the records into a CSV for quick plotting.

        Keys are unioned in first-seen order; the phase dict is exploded into
        ``phase.<name>.ms`` columns so a sweep plots without extra parsing.
        """
        if not self.enabled or not self.is_main:
            return
        rows = list(self.records if rows is None else rows)
        if not rows:
            return

        flat: List[Dict[str, Any]] = []
        for r in rows:
            row: Dict[str, Any] = {}
            for k, v in r.items():
                if k == "phases" and isinstance(v, dict):
                    for pname, pv in v.items():
                        row[f"phase.{pname}.ms"] = phase_mean_ms(pv)
                elif isinstance(v, (dict, list)):
                    row[k] = json.dumps(_jsonable(v))
                else:
                    row[k] = v
            flat.append(row)

        keys: List[str] = []
        for row in flat:
            for k in row:
                if k not in keys:
                    keys.append(k)

        import csv

        with open(self.csv_path(), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for row in flat:
                w.writerow(row)

    # -- markdown ---------------------------------------------------------
    def write_markdown(
        self,
        title: str,
        env: Optional[Dict[str, Any]] = None,
        notes: Optional[Sequence[str]] = None,
        table_columns: Optional[Sequence[str]] = None,
    ) -> Optional[Path]:
        if not self.enabled or not self.is_main or not self.records:
            return None

        env = env or environment_facts()
        parts: List[str] = [f"# {title}", ""]

        parts.append("## 运行环境")
        parts.append("")
        parts.append("| key | value |")
        parts.append("| --- | --- |")
        for k, v in env.items():
            parts.append(f"| `{k}` | `{v}` |")
        parts.append("")

        parts.append("## 结果")
        parts.append("")
        cols = list(table_columns or _default_columns(self.records))
        parts.append("| " + " | ".join(cols) + " |")
        parts.append("| " + " | ".join("---" for _ in cols) + " |")
        for r in self.records:
            parts.append(
                "| " + " | ".join(_cell(r.get(c)) for c in cols) + " |"
            )
        parts.append("")

        parts.append("## 分阶段耗时（每层求和，单位 ms）")
        parts.append("")
        phase_names = _phase_union(self.records)
        if phase_names:
            header = ["config"] + phase_names
            parts.append("| " + " | ".join(header) + " |")
            parts.append("| " + " | ".join("---" for _ in header) + " |")
            for r in self.records:
                phases = r.get("phases") or {}
                cells = [str(r.get("tag") or r.get("label") or "?")]
                for p in phase_names:
                    ms = phase_mean_ms(phases.get(p))
                    cells.append(f"{ms:.3f}" if ms is not None else "-")
                parts.append("| " + " | ".join(cells) + " |")
            parts.append("")
        else:
            parts.append("(no phase timings recorded)")
            parts.append("")

        if notes:
            parts.append("## 结论要点")
            parts.append("")
            for n in notes:
                parts.append(f"- {n}")
            parts.append("")

        path = self.markdown_path()
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

# ----------------------------------------------------------------------
def _cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (dict, list)):
        return json.dumps(_jsonable(v), ensure_ascii=False)
    return str(v)

def _phase_union(records: Iterable[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for r in records:
        for n in (r.get("phases") or {}):
            if n not in names:
                names.append(n)
    return names

def _default_columns(records: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = [
        "tag",
        "ep_size",
        "tokens_per_rank",
        "tokens_global",
        "top_k",
        "skew",
        "dtype",
        "total_ms",
        "imbalance_factor",
        "straggler_factor",
        "peak_allocated_MB",
        "peak_reserved_MB",
        "comm_total_MB",
        "comm_eff_GBps",
    ]
    present = {k for r in records for k in r}
    cols = [c for c in preferred if c in present]
    cols += sorted(present - set(cols) - {"phases", "dispatches"})
    return cols

def environment_facts() -> Dict[str, Any]:
    import torch

    def _safe(fn, default="n/a"):
        try:
            return fn()
        except Exception:
            return default

    facts = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": _safe(lambda: torch.version.cuda),
        "gpu_count": _safe(lambda: torch.cuda.device_count(), 0),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        from .dist_utils import nccl_version

        facts["nccl"] = nccl_version()
    except Exception:
        facts["nccl"] = "n/a"
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ):
        if os.environ.get(var):
            facts[f"env.{var}"] = os.environ[var]
    return facts

def gpu_facts(device: torch.device) -> Dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_MB": round(total / 2**20, 1),
        "free_MB": round(free / 2**20, 1),
        "bf16_native": _bf16_native(props.major),
    }

def _bf16_native(major: int) -> bool:
    """bf16 tensor cores arrived with Ampere (sm_80).  Volta is sm_70."""
    return major >= 8

def pci_topology() -> Dict[str, str]:
    """NUMA node per GPU, when the platform exposes it.

    Directly tests the spec's assumption that GPUs 0/1 share a socket and 2/3
    share the other.  Best-effort: returns an empty dict off Linux or without
    sysfs, and never raises.
    """
    out: Dict[str, str] = {}
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            bus = getattr(props, "pci_bus_id", None)
            if bus is None:
                continue
            numa = Path(f"/sys/bus/pci/devices/{bus}/numa_node")
            if numa.exists():
                out[f"gpu{i}_numa_node"] = numa.read_text().strip()
    except Exception:
        pass
    return out

def nvidia_smi_topo() -> str:
    """``nvidia-smi topo -m`` -- the authoritative NVLink/PCIe/SYS picture."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            return res.stdout
    except Exception:
        pass
    return ""

def print_header(title: str, facts: Dict[str, Any]) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    for k, v in facts.items():
        print(f"  {k:<24} {v}")
    print()
