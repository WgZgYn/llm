#!/usr/bin/env bash
#
# Full sweep: one torchrun per EP size, because ep_size IS world_size here.
#
# Launching a fresh process group per size (rather than slicing sub-groups out
# of a fixed 4-rank job) means a partial-group bug is impossible by
# construction.  It costs three startups, which is nothing next to the value of
# not debugging a sub-group all_to_all over ssh.
#
#   bash scripts/run_all.sh                       # tiny, the default
#   PRESET=wide OUT=out/wide bash scripts/run_all.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

PRESET="${PRESET:-tiny}"
EP_SIZES="${EP_SIZES:-1 2 4}"
OUT="${OUT:-$ROOT/out/$(date +%Y%m%d_%H%M%S)}"
PORT="${MASTER_PORT:-29577}"
TOKENS="${TOKENS:-4096}"
SKEWS="${SKEWS:-0.0 3.0}"

mkdir -p "$OUT"

echo "==================================================================="
echo " preset=$PRESET  ep_sizes=[$EP_SIZES]  output=$OUT"
echo " tokens(global)=$TOKENS  skews=[$SKEWS]"
echo "==================================================================="

run() {
  local ep="$1"; shift
  echo
  echo "--- torchrun --nproc_per_node=$ep  $* ---"
  torchrun --standalone --nproc_per_node="$ep" --master_port="$PORT" \
    "$HERE/bench_ep.py" --ep-size "$ep" --preset "$PRESET" --out "$OUT" "$@"
}

# ---- 1. correctness first: a fast table of EP vs dense -------------------
echo
echo "### correctness (EP vs dense) ###"
for EP in $EP_SIZES; do
  torchrun --standalone --nproc_per_node="$EP" --master_port="$PORT" \
    "$HERE/verify_ep.py" --ep-size "$EP" --preset "$PRESET" --out "$OUT" \
    --tag "verify_ep${EP}" || echo "!! verify_ep at ep=$EP returned non-zero"
done

# ---- 2. prefill / decode ladders, balanced and skewed --------------------
for EP in $EP_SIZES; do
  for SKEW in $SKEWS; do
    run "$EP" --mode all --skew "$SKEW" --tag "ep${EP}_skew${SKEW}"
  done
done

# ---- 3. link measurements, with and without P2P --------------------------
echo
echo "### topology (P2P enabled) ###"
torchrun --standalone --nproc_per_node=4 --master_port="$PORT" \
  "$HERE/bench_topology.py" --out "$OUT" --tag "topology_p2p_on" || true

echo
echo "### topology (P2P disabled -> SYS floor) ###"
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 --master_port="$PORT" \
  "$HERE/bench_topology.py" --out "$OUT" --tag "topology_p2p_off" || true

echo
echo "==================================================================="
echo " done. artifacts in $OUT"
echo "   python -c \"import pandas as pd; print(pd.read_json('$OUT/bench_ep1.jsonl', lines=True))\""
echo "==================================================================="
