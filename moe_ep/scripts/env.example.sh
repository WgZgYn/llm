#!/usr/bin/env bash
#
# Environment knobs worth setting on the 4xV100 box.
#   source scripts/env.example.sh
#
# The first two are the highest-value pair in this file.

# --- hang -> traceback ---------------------------------------------------
# Without these, a mismatched all_to_all split hangs until the process group
# timeout.  With them you get a Python traceback naming the rank that failed,
# so a bug costs 3 minutes instead of 30.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# --- NCCL logging --------------------------------------------------------
# INFO for the very first run only.  With 4 ranks it is slow and noisy, so
# drop back to WARN once the job runs.  NCCL_DEBUG_FILE keeps the four ranks'
# logs from interleaving into soup.
export NCCL_DEBUG=WARN
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
# export NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log

# --- topology levers -----------------------------------------------------
# Set NCCL_P2P_DISABLE=1 to force the SYS path.  Running bench_topology.py
# with and without it is what makes the intra- vs cross-group numbers
# interpretable -- the delta IS the NVLink/PIX contribution.
# export NCCL_P2P_DISABLE=1

# Set this if the box has no InfiniBand HCA but NCCL still probes for one
# (a classic multi-minute stall at init).  Leave unset if there IS a
# ConnectX card.
# export NCCL_IB_DISABLE=1

# Only needed with IB disabled on some builds, and a wrong value is a top
# cause of init failure.  Single-node usually needs nothing here.
# export NCCL_SOCKET_IFNAME=lo

# --- determinism ---------------------------------------------------------
# Required for --deterministic on CUDA >= 10.2; cuBLAS refuses otherwise.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- CPU oversubscription -------------------------------------------------
# Four ranks each spawning a full BLAS thread pool on one box adds jitter to
# the straggler column.  Worth setting for measurement runs.
export OMP_NUM_THREADS=1

# --- NOT set on purpose --------------------------------------------------
# CUDA_VISIBLE_DEVICES stays UNSET for --nproc_per_node=4.  Setting it to
# fewer entries than ranks makes LOCAL_RANK exceed the visible count, which
# either errors or silently binds the wrong GPU.
# unset CUDA_VISIBLE_DEVICES
