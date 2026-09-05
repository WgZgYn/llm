# nccl_allreduce_bench.py —— NCCL 集合通信原语基准（多机/多卡）
#
# 用 PyTorch 的 NCCL 后端测各种集合通信原语的"消息大小 vs 耗时/带宽"关系。
# 只支持 Linux + 多 GPU（NCCL 不支持 Windows；单进程单卡也跑不了集合通信）。
#
# 运行（单机多卡 / 多机）：
#   torchrun --nproc_per_node=8 --nnodes=2 --node_rank=<0/1> \
#            --master_addr=<ip> --master_port=29500 \
#            nccl_allreduce_bench.py --op allreduce --max-size 1G
#
# 输出：终端表格 + 可选 CSV（--output） + 可选带宽-大小曲线图（--plot）。
import argparse
import os
import time

import torch
import torch.distributed as dist

# 支持的集合通信原语
OPS = ["allreduce", "broadcast", "reduce", "allgather", "reduce_scatter", "all_to_all"]


def human_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024


def busbw_factor(op, n):
    """NCCL 的 bus bandwidth 系数：数据实际穿过总线的是 S 的多少倍。

    对应"总传输量"与单卡消息量 S 的关系：
      - allreduce / reduce_scatter / all_to_all: 2*(n-1)/n * S
      - allgather: (n-1)/n * S
      - broadcast / reduce: S（单卡视角，无 n 相关缩放）
    """
    if op in ("allreduce", "reduce_scatter", "all_to_all"):
        return 2 * (n - 1) / n
    if op == "allgather":
        return (n - 1) / n
    return 1.0


def run_collective(op, tensor, out_list, world_size, rank):
    """执行一次集合通信。tensor 是每个 rank 的输入，out_list 是 gather 类输出。"""
    if op == "allreduce":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "broadcast":
        dist.broadcast(tensor, src=0)
    elif op == "reduce":
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    elif op == "allgather":
        dist.all_gather(out_list, tensor)
    elif op == "reduce_scatter":
        # 输入切成 world_size 块，各块求和后 scatter 回各 rank
        chunks = list(tensor.chunk(world_size))
        out = torch.empty_like(chunks[0])
        dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM)
    elif op == "all_to_all":
        chunks_in = list(tensor.chunk(world_size))
        chunks_out = [torch.empty_like(c) for c in chunks_in]
        dist.all_to_all(chunks_out, chunks_in)
    else:
        raise ValueError(op)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=OPS, default="allreduce")
    p.add_argument("--min-size", default="1K", help="起始消息大小（每卡），如 1K")
    p.add_argument("--max-size", default="1G", help="最大消息大小（每卡）")
    p.add_argument("--steps", type=int, default=10, help="对数扫点的步数")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--output", default="", help="输出 CSV 路径")
    p.add_argument("--plot", action="store_true", help="画带宽-大小曲线")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"NCCL 集合通信基准  op={args.op}  world_size={world_size}")
        print(f"消息大小按每卡计（S bytes），带宽用 NCCL bus bandwidth 口径")
        print(f"{'size':>12}  {'latency(us)':>12}  {'busbw(GB/s)':>12}")

    def parse_size(s):
        s = s.strip().upper()
        mult = {"K": 1024, "M": 1024**2, "G": 1024**3}
        if s[-1] in mult:
            return int(float(s[:-1]) * mult[s[-1]])
        return int(s)

    min_b, max_b = parse_size(args.min_size), parse_size(args.max_size)
    # 对数扫点：从 min 到 max 取 steps 个点
    sizes = []
    import math
    lo, hi = math.log2(min_b), math.log2(max_b)
    for i in range(args.steps):
        sizes.append(int(2 ** (lo + (hi - lo) * i / (args.steps - 1))))
    sizes = sorted(set(sizes))

    factor = busbw_factor(args.op, world_size)
    rows = []

    for nbytes in sizes:
        n_float = max(1, nbytes // 4)
        tensor = torch.rand(n_float, device=dev)  # S bytes
        out_list = [torch.empty_like(tensor) for _ in range(world_size)]

        # warmup
        for _ in range(args.warmup):
            run_collective(args.op, tensor, out_list, world_size, rank)
        torch.cuda.synchronize()

        # 计时：事件包住一个循环，取平均
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            run_collective(args.op, tensor, out_list, world_size, rank)
        end.record()
        torch.cuda.synchronize()
        t_us = start.elapsed_time(end) / args.iters * 1e3  # 单次，微秒

        busbw = factor * nbytes / (t_us * 1e-6) / 1e9  # GB/s
        rows.append((nbytes, t_us, busbw))
        if rank == 0:
            print(f"{human_bytes(nbytes):>12}  {t_us:>12.2f}  {busbw:>12.2f}")

    # 汇总（可选：各 rank 的最小/平均，这里简单起见 rank 0 直接输出）
    if args.output and rank == 0:
        with open(args.output, "w") as f:
            f.write("size_bytes,latency_us,busbw_gbps\n")
            for nbytes, t_us, bw in rows:
                f.write(f"{nbytes},{t_us:.2f},{bw:.2f}\n")
        print(f"已保存 CSV -> {args.output}")

    if args.plot and rank == 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [r[0] for r in rows]
            ys = [r[2] for r in rows]
            plt.figure()
            plt.plot(xs, ys, "-o")
            plt.xscale("log")
            plt.xlabel("message size per rank (bytes)")
            plt.ylabel("bus bandwidth (GB/s)")
            plt.title(f"NCCL {args.op}, world_size={world_size}")
            plt.grid(True, which="both", ls="--", alpha=0.4)
            out = args.output.rsplit(".", 1)[0] + ".png" if args.output else f"nccl_{args.op}.png"
            plt.savefig(out, dpi=110)
            print(f"已保存图 -> {out}")
        except ImportError:
            print("未安装 matplotlib，跳过画图（pip install matplotlib 后可用 --plot）")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
