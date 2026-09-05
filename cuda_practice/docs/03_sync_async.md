# 03 · 同步（Sync）与异步（Async）

> 对应示例：`python/02_async_memcpy.py`（异步拷贝）、`python/03_overlap.py`（重叠）、`python/05_sync_granularity.py`（同步粒度）
> 原生参考：`cuda_native/02/03/05_*.cu`

## 1. 异步拷贝与 pinned 内存

拷贝分两种：**同步** `cudaMemcpy`（阻塞到拷完才返回）和**异步** `cudaMemcpyAsync`（排进流就返回）。

但**异步拷贝要真正异步，主机内存必须是 pinned（page-locked）**。普通 `malloc` 出来的内存是 pageable 的，驱动会在内部"同步 + 中转（staging buffer）"，既慢又失去异步性。示例实测：

| 方式 | 耗时 | 带宽 | 主机是否被阻塞 |
|---|---|---|---|
| 同步拷贝 (pageable) | 20.6 ms | 13.0 GB/s | 是 |
| 异步拷贝 (pageable) | 20.5 ms | 13.1 GB/s | **是**（退化为同步） |
| 异步拷贝 (pinned)  | 20.0 ms | 13.4 GB/s | **否**（主机 0.04 ms 就返回） |

关键观察：`pageable + non_blocking` 的"主机返回"用了 20.5 ms，而 `pinned + non_blocking` 只用了 0.04 ms——前者根本没异步。所以做异步流水线时，主机侧缓冲必须 `pin_memory()`（原生用 `cudaMallocHost`）。

## 2. 拷贝与计算的重叠（overlap）

拷贝走 **copy engine（DMA）**，计算走 **SM**，是不同硬件单元，所以能"一边拷贝一边算"。前提：拷贝/计算放**不同流**，且用 pinned + 异步。示例实测：

```
单独执行：拷贝 20.1 ms，计算 15.6 ms，串行应约 35.7 ms
同流串行    : 36.0 ms
双流重叠    : 20.3 ms   （≈ max(拷贝, 计算) = 20.1 ms）
节省比例    : 43.7%
```

同一条流里是 FIFO 队列，拷贝没结束就轮不到计算；拆到两条流后，计算躲进拷贝的"影子"里并行执行。这就是流水线的本质——训练里常见的数据加载（拷贝）与前后向（计算）重叠。

## 3. 同步的粒度与代价

CUDA 有三档同步，**越细代价越低、对并发破坏越小**。示例实测（空闲同步，1000 次平均）：

| API | 粒度 | 单次开销 |
|---|---|---|
| `torch.cuda.synchronize()` | 整个设备所有流 | 3.7 µs |
| `stream.synchronize()` | 某一条流 | 0.47 µs |
| `event.synchronize()` | 某一个"点" | 0.24 µs |

**粒度语义**也很重要：示例里 `s_comp.synchronize()`（只等短计算）花了 `1.3 ms`，而 `torch.cuda.synchronize()`（等所有流，含长拷贝）花了 `38.8 ms`。前者只等自己关心的那条流，后者把无关的长拷贝也一起等了。

## 4. 阻塞等待 vs 轮询查询

- **阻塞**：`synchronize()` —— 主机干等，不占 CPU，但做不了别的事。
- **轮询**：`query()` —— 非阻塞，主机可以"边等边干别的活"（示例里用计数器模拟，查了 5400 多次）。

两者耗时相近，区别在于轮询期间主机是"活"的。代价是轮询占 CPU、有额外延迟，通常用 `cudaStreamQuery`/`cudaEventQuery` 做"忙等 + 间隔 yield"。

## 5. 什么时候用什么同步

- 只想确认**某个点**完成 → `event.synchronize()`（最便宜）。
- 只想确认**某条流**完成 → `stream.synchronize()`。
- 要**全部收敛**（如测总耗时、退出前清理）→ `device.synchronize()`。
- 想在等待期间**做别的事** → `query()` 轮询。
- **少用 `device.synchronize()`**：它会打断所有并发，等于把多流流水线"拍平"。
