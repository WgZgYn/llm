# CUDA / NCCL 实践笔记

以**学习研究**为目标，覆盖两类内容：

1. **CUDA Runtime API**（流 / 事件 / 同步 / 异步）—— 在本机（单卡）直接跑，每个示例都带真实计时指标。
2. **NCCL 集合通信**（allreduce 等）—— 需放到远程 Linux 多机/多卡环境跑。

## 目录结构

```
cuda_practice/
├── python/                 # 可运行：torch.cuda 版（本机 Windows 单卡即可跑）
│   ├── 01_streams_basic.py     流的基本概念
│   ├── 02_async_memcpy.py      同步/异步拷贝 & pinned 内存
│   ├── 03_overlap.py           拷贝与计算的重叠
│   ├── 04_events_timing.py     事件与精确计时
│   ├── 05_sync_granularity.py  同步粒度与代价
│   ├── 06_stream_semantics.py  跨流依赖 (wait_event)
│   ├── 07_cuda_graph.py        CUDA Graph 消启动开销
│   └── run_all.py              一次跑完 01~07
├── cuda_native/            # 参考：原生 CUDA C++（真正 cudaStreamCreate 等 API 签名）
│   ├── 01~06_*.cu + common.h
│   └── Makefile                （Linux 编译；Windows 需 MSVC）
├── bench/
│   └── nccl_allreduce_bench.py NCCL 多机集合通信基准
├── docs/                   # 学习笔记（重点）
│   ├── 01_streams.md
│   ├── 02_events.md
│   ├── 03_sync_async.md
│   └── 04_nccl.md
└── build.ps1               # 编译 cuda_native（需 MSVC，否则去 Linux）
```

## 环境

- **本机（Windows 单卡 RTX 4060）**：`python/` 下所有示例用 `torch.cuda`，直接跑。
- **远程（Linux 多机多卡）**：`bench/nccl_allreduce_bench.py`，用 `torchrun` 启动。
- **原生 C++ 示例**：`cuda_native/` 需 nvcc +（Windows 用 MSVC，或 Linux 用 Makefile）。

## 快速开始（本机）

```bash
# 进入项目 uv 虚拟环境后：
python cuda_practice/python/run_all.py        # 一次跑完 01~06
# 或单独跑某个：
python cuda_practice/python/03_overlap.py
```

每个脚本开头注释里都写了它对应的 CUDA API 映射，跑完对照 `docs/` 看解释。

## 跑 NCCL（远程 Linux 多机）

```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
         --master_addr=<ip> --master_port=29500 \
         cuda_practice/bench/nccl_allreduce_bench.py --op allreduce --max-size 1G --plot
```

详见 `docs/04_nccl.md`。

## 核心结论速览

- **启动是异步的**，内核真实耗时只能用 **event** 测（主机时钟测的是"排队"）。
- **流内 FIFO 串行**，**流间独立**，但真正稳定的并行来自 **copy engine + SM 的重叠**。
- **异步拷贝必须用 pinned 内存**，否则退化为同步。
- **同步越细越便宜**：`event.sync < stream.sync < device.sync`（约 0.2 / 0.5 / 3.7 µs）。
- **跨流依赖**用 `wait_event` 精确到"点"，别动不动就 `device.synchronize()`。
- **NCCL 带宽 vs 消息大小**呈两段：小消息 latency-bound、大消息 bandwidth-bound，扫点找拐点与平台带宽。
