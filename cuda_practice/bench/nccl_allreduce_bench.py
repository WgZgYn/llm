# nccl_collective_bench.py
#
# PyTorch + NCCL collective communication microbenchmark
#
# 目标：
#   测量不同 message size 下：
#     - total latency
#     - estimated fixed latency
#     - estimated transfer time
#     - payload bandwidth
#     - NCCL-style bus bandwidth
#
# 运行：
#
#   torchrun --nproc-per-node 4 \
#       nccl_collective_bench.py \
#       --op allreduce
#
#   torchrun --nproc-per-node 2 \
#       nccl_collective_bench.py \
#       --op allreduce
#
# 输出：
#   size
#   total latency
#   transfer time
#   fixed latency
#   payload bandwidth
#   bus bandwidth
#   efficiency

import argparse
import math
import os
import statistics

import torch
import torch.distributed as dist


OPS = [
    "allreduce",
    "broadcast",
    "reduce",
    "allgather",
    "reduce_scatter",
    "all_to_all",
]


def human_bytes(n):
    n = float(n)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024


def parse_size(s):
    s = s.strip().upper()

    units = {
        "K": 1024,
        "M": 1024 ** 2,
        "G": 1024 ** 3,
    }

    if s[-1] in units:
        return int(float(s[:-1]) * units[s[-1]])

    return int(s)


def busbw_factor(op, world_size):
    """
    NCCL-style bus bandwidth factor.

    S = input message size per rank.

    AllReduce:
        2 * (N-1) / N

    ReduceScatter:
        2 * (N-1) / N

    AllToAll:
        2 * (N-1) / N

    AllGather:
        (N-1) / N

    Broadcast / Reduce:
        1
    """

    n = world_size

    if op in ("allreduce", "reduce_scatter", "all_to_all"):
        return 2 * (n - 1) / n

    if op == "allgather":
        return (n - 1) / n

    return 1.0


def run_collective(op, tensor, world_size):

    if op == "allreduce":

        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

    elif op == "broadcast":

        dist.broadcast(
            tensor,
            src=0,
        )

    elif op == "reduce":

        dist.reduce(
            tensor,
            dst=0,
            op=dist.ReduceOp.SUM,
        )

    elif op == "allgather":

        outputs = [
            torch.empty_like(tensor)
            for _ in range(world_size)
        ]

        dist.all_gather(
            outputs,
            tensor,
        )

    elif op == "reduce_scatter":

        chunks = list(
            tensor.chunk(world_size)
        )

        output = torch.empty_like(chunks[0])

        dist.reduce_scatter(
            output,
            chunks,
            op=dist.ReduceOp.SUM,
        )

    elif op == "all_to_all":

        chunks_in = list(
            tensor.chunk(world_size)
        )

        chunks_out = [
            torch.empty_like(c)
            for c in chunks_in
        ]

        dist.all_to_all(
            chunks_out,
            chunks_in,
        )

    else:
        raise ValueError(op)


def make_sizes(min_b, max_b, steps):

    lo = math.log2(min_b)
    hi = math.log2(max_b)

    sizes = []

    for i in range(steps):

        x = lo + (hi - lo) * i / (steps - 1)

        sizes.append(
            int(2 ** x)
        )

    return sorted(set(sizes))


def barrier():

    dist.barrier()

    # Make sure previous CUDA work is finished.
    torch.cuda.synchronize()


def measure(op, tensor, world_size, warmup, iters):

    # --------------------------------------------------
    # Warmup
    # --------------------------------------------------

    for _ in range(warmup):
        run_collective(
            op,
            tensor,
            world_size,
        )

    barrier()

    # --------------------------------------------------
    # Measure
    # --------------------------------------------------

    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()

    for _ in range(iters):

        run_collective(
            op,
            tensor,
            world_size,
        )

    end.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)

    latency_us = (
        elapsed_ms
        * 1000
        / iters
    )

    # Collect latency from all ranks.
    latency_tensor = torch.tensor(
        [latency_us],
        device=tensor.device,
        dtype=torch.float64,
    )

    gathered = [
        torch.empty_like(latency_tensor)
        for _ in range(world_size)
    ]

    dist.all_gather(
        gathered,
        latency_tensor,
    )

    rank_times = [
        x.item()
        for x in gathered
    ]

    return {
        "local": latency_us,
        "min": min(rank_times),
        "mean": statistics.mean(rank_times),
        "median": statistics.median(rank_times),
        "max": max(rank_times),
    }


def linear_fit(sizes, times_us):
    """
    Fit:

        T = alpha + beta * S

    where:

        alpha = fixed latency
        beta  = seconds / byte

    Returns:

        alpha_us
        bandwidth_GBps
    """

    xs = [
        float(x)
        for x in sizes
    ]

    ys = [
        float(y)
        for y in times_us
    ]

    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    )

    denominator = sum(
        (x - x_mean) ** 2
        for x in xs
    )

    if denominator == 0:
        return 0.0, 0.0

    beta = numerator / denominator

    alpha = y_mean - beta * x_mean

    # beta = us / byte
    #
    # bandwidth:
    #
    # byte / us
    # = byte / second * 1e-6
    #
    # GB/s = byte/s / 1e9
    #
    # therefore:
    # GB/s = 1 / beta / 1e3

    bandwidth_gbps = (
        1.0 / beta / 1000
        if beta > 0
        else 0.0
    )

    return alpha, bandwidth_gbps


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--op",
        choices=OPS,
        default="allreduce",
    )

    parser.add_argument(
        "--min-size",
        default="1K",
    )

    parser.add_argument(
        "--max-size",
        default="1G",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--output",
        default="",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Init distributed
    # --------------------------------------------------

    dist.init_process_group(
        backend="nccl"
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            0,
        )
    )

    torch.cuda.set_device(
        local_rank
    )

    device = torch.device(
        "cuda",
        local_rank,
    )

    # --------------------------------------------------
    # Sizes
    # --------------------------------------------------

    min_b = parse_size(
        args.min_size
    )

    max_b = parse_size(
        args.max_size
    )

    sizes = make_sizes(
        min_b,
        max_b,
        args.steps,
    )

    factor = busbw_factor(
        args.op,
        world_size,
    )

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    if rank == 0:

        print()
        print("=" * 105)

        print(
            f"NCCL Collective Benchmark"
        )

        print(
            f"op={args.op} "
            f"world_size={world_size}"
        )

        print(
            f"message size = per-rank input size"
        )

        print(
            f"bus bandwidth factor = {factor:.4f}"
        )

        print("=" * 105)

        print(
            f"{'Size':>12} "
            f"{'Total(us)':>12} "
            f"{'Transfer(us)':>14} "
            f"{'Fixed(us)':>12} "
            f"{'Payload(GB/s)':>15} "
            f"{'BusBW(GB/s)':>13} "
            f"{'Eff.(%)':>10}"
        )

        print("-" * 105)

    results = []

    # --------------------------------------------------
    # Benchmark
    # --------------------------------------------------

    for nbytes in sizes:

        # float32 => 4 bytes
        n_elements = max(
            1,
            nbytes // 4,
        )

        tensor = torch.rand(
            n_elements,
            device=device,
            dtype=torch.float32,
        )


        result = measure(
            args.op,
            tensor,
            world_size,
            args.warmup,
            args.iters,
        )

        # Use MAX rank time as collective time.
        #
        # Collective completes only when the slowest
        # rank has completed.
        total_us = result["max"]

        # Payload bandwidth:
        #
        # S / T
        payload_gbps = (
            nbytes
            / (total_us * 1e-6)
            / 1e9
        )

        # NCCL-style bus bandwidth.
        busbw_gbps = (
            factor
            * nbytes
            / (total_us * 1e-6)
            / 1e9
        )

        results.append(
            {
                "size": nbytes,
                "total_us": total_us,
                "payload_gbps": payload_gbps,
                "busbw_gbps": busbw_gbps,
            }
        )

        del tensor

    # --------------------------------------------------
    # Estimate fixed latency + bandwidth
    #
    # Only use larger messages for fitting.
    # Small messages are heavily affected by launch/
    # synchronization overhead.
    # --------------------------------------------------

    fit_start = max(
        0,
        len(results) // 2,
    )

    fit_sizes = [
        r["size"]
        for r in results[fit_start:]
    ]

    fit_times = [
        r["total_us"]
        for r in results[fit_start:]
    ]

    fixed_us, fitted_bw = linear_fit(
        fit_sizes,
        fit_times,
    )

    # --------------------------------------------------
    # Print final table
    # --------------------------------------------------

    for r in results:

        transfer_us = max(
            0.0,
            r["total_us"] - fixed_us,
        )

        payload_bw = r[
            "payload_gbps"
        ]

        efficiency = (
            payload_bw
            / fitted_bw
            * 100
            if fitted_bw > 0
            else 0
        )

        if rank == 0:

            print(
                f"{human_bytes(r['size']):>12} "
                f"{r['total_us']:>12.2f} "
                f"{transfer_us:>14.2f} "
                f"{fixed_us:>12.2f} "
                f"{payload_bw:>15.2f} "
                f"{r['busbw_gbps']:>13.2f} "
                f"{efficiency:>9.1f}%"
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    if rank == 0:

        print()
        print("=" * 70)

        print(
            f"Estimated fixed latency : "
            f"{fixed_us:.2f} us"
        )

        print(
            f"Estimated payload BW    : "
            f"{fitted_bw:.2f} GB/s"
        )

        print(
            f"NCCL bus BW factor      : "
            f"{factor:.4f}"
        )

        print("=" * 70)

    # --------------------------------------------------
    # CSV
    # --------------------------------------------------

    if args.output and rank == 0:

        with open(
            args.output,
            "w",
        ) as f:

            f.write(
                "size_bytes,"
                "total_us,"
                "transfer_us,"
                "fixed_us,"
                "payload_gbps,"
                "busbw_gbps,"
                "efficiency\n"
            )

            for r in results:

                transfer_us = max(
                    0.0,
                    r["total_us"]
                    - fixed_us,
                )

                efficiency = (
                    r["payload_gbps"]
                    / fitted_bw
                    * 100
                    if fitted_bw > 0
                    else 0
                )

                f.write(
                    f"{r['size']},"
                    f"{r['total_us']:.4f},"
                    f"{transfer_us:.4f},"
                    f"{fixed_us:.4f},"
                    f"{r['payload_gbps']:.4f},"
                    f"{r['busbw_gbps']:.4f},"
                    f"{efficiency:.2f}\n"
                )

        print(
            f"CSV saved to {args.output}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()