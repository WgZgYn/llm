#!/usr/bin/env bash
#
# First pass: ~2 minutes, EP in {1,4}, balanced routing only.
# Use this to confirm the box is sane before spending time on the full sweep.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

PRESET="${PRESET:-tiny}"
OUT="${OUT:-$ROOT/out/quick_$(date +%Y%m%d_%H%M%S)}"
PORT="${MASTER_PORT:-29578}"
mkdir -p "$OUT"

echo "=== preflight ==="
torchrun --standalone --nproc_per_node=4 --master_port="$PORT" \
  "$HERE/check_env.py" --skip-p2p --out "$OUT"

for EP in 1 4; do
  echo
  echo "=== verify_ep ep=$EP ==="
  torchrun --standalone --nproc_per_node="$EP" --master_port="$PORT" \
    "$HERE/verify_ep.py" --ep-size "$EP" --preset "$PRESET" --out "$OUT" \
    --tag "quick_verify_ep${EP}"

  echo
  echo "=== bench_ep ep=$EP (prefill + decode) ==="
  torchrun --standalone --nproc_per_node="$EP" --master_port="$PORT" \
    "$HERE/bench_ep.py" --ep-size "$EP" --preset "$PRESET" --mode all \
    --warmup 2 --iters 5 --out "$OUT" --tag "quick_ep${EP}"
done

echo
echo "artifacts in $OUT"
