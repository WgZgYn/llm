# 04 · NCCL 多机集合通信

> 对应脚本：`bench/nccl_allreduce_bench.py`

## 重要前提

**NCCL 只支持 Linux + 多 GPU。** 本机（Windows + 单卡 RTX 4060）跑不了：`torch.distributed.is_nccl_available()` 为 `False`。这套内容要放到**远程 Linux 多机/多卡**环境跑。

## 什么是 NCCL

NCCL（NVIDIA Collective Communications Library）是 GPU 之间做**集合通信**的标准库。深度学习里的数据并行（DDP）、张量并行、流水线并行，底层都是这些原语：

| 原语 | 语义 | 典型用途 |
|---|---|---|
| **allreduce** | 所有卡各自有 `S`，求和/平均后每卡都拿到全量结果 | DDP 梯度同步 |
| **broadcast** | 根卡把 `S` 广播给所有卡 | 分发模型参数 |
| **reduce** | 所有卡把 `S` 归约到一张卡（dst） | 汇总损失/指标 |
| **allgather** | 每卡各有 `S`，聚合后每卡拿到 `n·S` | 张量并行前的切分合并 |
| **reduce_scatter** | 每卡有 `n·S`，归约后按块 scatter，每卡得 `S` | ZeRO / FSDP |
| **all_to_all** | 每卡向每张卡发不同数据块 | MoE 专家路由 |

> `S` 表示"每张卡的消息大小"。n 是卡数（world_size）。

## 传输量与带宽的关系（要做的实验）

NCCL 的"传输量"要看两个口径：

- **算法带宽 algbw**：`S / t`，只看单卡消息量。
- **总线带宽 busbw**：考虑数据实际穿过互连的量。对 allreduce 是 `2·(n-1)/n · S / t`。

`busbw` 是更公平的口径：它刻画的是"互连总线"上的真实流量。allreduce 里每张卡既要把自己的数据发出去、又要收别人的，总线流量约为 `2·(n-1)/n·S`。

**关键预期：带宽不是常数，随消息大小呈"两段式"曲线**：

1. **小消息（latency-bound）**：消息越小，通信延迟固定开销（launch、握手、内存访问）占比越大，带宽很低。这段主要受**延迟**限制。
2. **大消息（bandwidth-bound）**：消息足够大后，带宽趋于平台值（受 PCIe/NVLink/网卡带宽限制），进入带宽瓶颈区。

实验目的就是扫一遍消息大小（如 1 KB → 1 GB），画出这条曲线，找到：
- 从 latency-bound 拐到 bandwidth-bound 的"拐点"大小；
- 平台带宽（≈ 网卡/互连的理论带宽）；
- 不同原语、不同卡数的曲线差异。

## 多机运行方式

脚本用 `torch.distributed`，通过 `torchrun` 启动（自动设好 `RANK`/`WORLD_SIZE`/`MASTER_ADDR` 等环境变量）：

```bash
# 单机 8 卡
torchrun --nproc_per_node=8 bench/nccl_allreduce_bench.py --op allreduce --max-size 1G

# 两台机器各 8 卡（在每台机器上分别执行，node_rank 不同）
# 机器 0：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
         --master_addr=<机器0的IP> --master_port=29500 \
         bench/nccl_allreduce_bench.py --op allreduce --max-size 1G --output result.csv --plot
# 机器 1：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
         --master_addr=<机器0的IP> --master_port=29500 \
         bench/nccl_allreduce_bench.py --op allreduce --max-size 1G
```

常用参数：`--op`（原语）、`--min-size/--max-size`（扫点范围）、`--steps`（扫点数）、`--iters`（计时迭代次数，小消息可调大）、`--output`（存 CSV）、`--plot`（画带宽-大小曲线，需 matplotlib）。

## 结果怎么看

单机内（NVLink/PCIe）和跨机（InfiniBand/RoCE）曲线形态类似，但平台带宽和拐点不同：
- **跨机**通常带宽更低、拐点更大（网络延迟更高）。
- **allreduce** 带宽随卡数增长而下降更明显（总流量更大）。

做多次实验（`--iters` 调大、或重复多次取中位数）是为了压掉网络抖动，尤其是跨机场景。

## 实验建议清单

1. 固定 8 卡，扫 `allreduce` 的 1K→1G，记录曲线，找拐点与平台带宽。
2. 换 `allgather` / `reduce_scatter` / `all_to_all`，对比同一大小下的带宽。
3. 对比"单机 8 卡" vs "2 机 × 4 卡"，观察跨机对带宽/拐点的影响。
4. 固定中等大小（如 128 MB），多次重复，看跨机时延的抖动（分布）。
