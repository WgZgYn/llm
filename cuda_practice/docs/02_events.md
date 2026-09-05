# 02 · 事件（Event）与精确计时

> 对应示例：`python/04_events_timing.py`　·　原生参考：`cuda_native/04_events_timing.cu`

## 一句话理解

**事件是"流上的一个标记/时间戳"。** 把它记录（record）到某条流里，当这条流执行到那个位置时，事件就"完成"了。事件有两个用途：**计时** 和 **做同步点**。

## 核心 API

| CUDA Runtime (C++) | torch.cuda (Python) | 作用 |
|---|---|---|
| `cudaEventCreate(&e)` | `e = torch.cuda.Event(enable_timing=True)` | 创建事件 |
| `cudaEventCreateWithFlags(&e, cudaEventDisableTiming)` | `Event(enable_timing=False)` | 创建"只做同步、不计时"的事件（更省） |
| `cudaEventRecord(e, s)` | `e.record(stream)` | 把事件记录到流 `s`（省略则当前流） |
| `cudaEventSynchronize(e)` | `e.synchronize()` | 阻塞等事件完成 |
| `cudaEventQuery(e)` | `e.query()` | 非阻塞查：完成了吗（bool） |
| `cudaEventElapsedTime(&ms, a, b)` | `a.elapsed_time(b)` | 两个事件之间的毫秒数 |

## 为什么主机时钟测不准内核耗时

内核启动是异步的。用主机时钟 `time.time()` 包住一次 kernel 调用，测到的是"**排队**"的时间（`~0.05 ms`），不是执行时间。示例输出：

```
主机时钟(错误): 0.05 ms   <- 只测到排队
事件计时(正确): 2.24 ms   <- 真实执行时间
```

事件记录的是 **GPU 时间线（device timeline）** 上的时刻，所以 `elapsed_time` 读出来的是内核真实运行时长。

## 读耗时前必须先同步

`elapsed_time` 要求两个事件都已"完成"。如果内核还没算完就去读，会得到 `cudaErrorNotReady`（原生）或读不准。所以要么先 `synchronize()`，要么用 `query()` 轮询到完成。

示例里故意让一个长 kernel 跑着就去 `elapsed_time`，展示"未同步就查询"的失败；`synchronize` 之后才能读到正确值。

## enable_timing=False

事件在内部要维护 GPU 时间戳，这有少量开销。如果你只是拿事件做"同步点"（例如跨流依赖，见 `06_stream_semantics`），而不需要计时，就用 `Event(enable_timing=False)`，创建/记录开销更低。这类事件**不能**调用 `elapsed_time`。

## 事件的两类用途小结

1. **计时**：`record(start)` → 干活 → `record(end)` → `synchronize` → `elapsed_time`。
2. **跨流依赖**：`stream.wait_event(event)` 让一条流等另一条流上的某个点（见 `06`）。
