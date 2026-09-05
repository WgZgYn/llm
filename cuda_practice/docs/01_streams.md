# 01 · 流（Stream）

> 对应示例：`python/01_streams_basic.py`　·　原生参考：`cuda_native/01_streams_basic.cu`

## 一句话理解

**流是 GPU 上的一条任务队列（FIFO）。** 你提交到同一条流上的操作，会严格按提交顺序执行；不同流上的操作之间没有先后顺序，可以并发。

内核（kernel）启动是**异步**的：主机把任务丢进流里就立刻返回，不会等它算完。想要"确定算完了"，必须显式同步。

## 核心 API

| CUDA Runtime (C++) | torch.cuda (Python) | 作用 |
|---|---|---|
| `cudaStreamCreate(&s)` | `s = torch.cuda.Stream()` | 创建一条流 |
| `cudaStreamDestroy(s)` | （GC 自动回收） | 销毁流 |
| `kernel<<<g,b,0,s>>>` | `with torch.cuda.stream(s): ...` | 把操作排到流 `s` |
| `cudaStreamSynchronize(s)` | `s.synchronize()` | 等这条流上的活干完 |
| `cudaStreamQuery(s)` | `s.query()` | 非阻塞查：这条流干完了吗（返回 bool） |
| `cudaDeviceSynchronize()` | `torch.cuda.synchronize()` | 等**整个设备**所有流 |

## 三个关键性质（对应示例的三段输出）

1. **启动异步**：主机 `matmul` 启动只需 `~0.05 ms`，而内核真正跑了 `~2 ms`。中间这段"没等"就是异步的意义——主机可以继续发下一个任务，或去准备下一批数据。

2. **流内 FIFO**：同一条流上连续两个 matmul 总耗时 `~5 ms`，≈ 单次 `~2 ms` 的两倍。说明第二个必须等第一个结束——队列是串行的。

3. **流间独立**：把"长拷贝"放 `s_copy`、把"短计算"放 `s_comp`，`s_comp.synchronize()` 只花 `~1 ms` 就返回，而 `s_copy.query()` 显示拷贝还在跑。说明 `s_comp` 没有等 `s_copy`——两条流互不等待。

## 一个容易被忽略的事实

流之间"没有先后顺序"**不等于**"一定能并行"。两条流上的**计算类**任务最终都要抢占同一批 SM（计算核心）：

- 一个 matmul 往往已经占满所有 SM，此时另一条流上的 matmul 只能排队等 SM 空出来——**表现为串行**。
- 真正稳定的"并行"来自**不同硬件单元**：拷贝走 **copy engine（DMA）**，计算走 **SM**，两者才能同时进行（详见 `03_overlap`）。

所以演示"流间独立"时，示例刻意用"长拷贝 + 短计算"的组合，而不是"两个计算"。

## 默认流 vs 非默认流

- **默认流（stream 0）**：什么都不传时的流。在原生 CUDA 里它是"遗留默认流"，会与其它**阻塞流**隐式同步，容易悄悄串行化程序。
- **非默认流**：`torch.cuda.Stream()` 创建的是**非阻塞流**，与默认流之间不隐式同步，需要依赖时必须用 `wait_event` 显式声明（见 `06_stream_semantics`）。

## 建议实验

改 `01_streams_basic.py` 里的 `n = 2048`（matmul 尺寸），观察：
- 尺寸越小，"启动异步"与"实际执行"的差距越小（启动开销占比变大）；
- 把第 3 段的"短计算"换成 `matmul`，再看 `s_comp.synchronize()`——它会变成 ~20 ms，因为和拷贝的 SM 争用（其实是排队等待）。

---

## 补充 1：事件记在哪个流上？同步的作用范围

以 `cuda_native/01_streams_basic.cu` 第 3 段为例：

```cuda
CHECK(cudaEventRecord(start));                 // 没传流 → 记在"默认流(stream 0)"
spin_kernel<<<..., s1>>>(...);                 // s1（阻塞流）
spin_kernel<<<..., s2>>>(...);                 // s2（阻塞流）
CHECK(cudaEventRecord(stop));                  // 也没传流 → 也在默认流
CHECK(cudaEventSynchronize(stop));             // 等 stop
```

**`start` 和 `stop` 都记在默认流（stream 0）上**，不在 s1/s2 上。那它俩怎么"同时管住" s1、s2 呢？靠的是**遗留默认流的隐式同步**：

- `cudaStreamCreate` 造出来的是**阻塞流（blocking stream）**，会与默认流**隐式同步**；
- 默认流在记录 `stop` 之前，会先等所有阻塞流（s1、s2）里已有的活干完；
- 所以 `stop` 这个事件，天然排在"两个 kernel 都结束"之后，`elapsed_time(start, stop)` 就是两个 kernel 的总耗时。

**同步的作用范围**，由粗到细：

| 调用 | 作用范围 | 备注 |
|---|---|---|
| `cudaDeviceSynchronize()` | 整个设备所有流 | 最粗，别随便用 |
| `cudaStreamSynchronize(s)` | 只等流 `s` | 等一条 |
| `cudaEventSynchronize(e)` | 等事件 `e` 所在流的"那个点" | 等一个点 |
| `cudaStreamWaitEvent(s, e)` | 让流 `s` 等事件 `e`（跨流依赖） | 不阻塞主机，GPU 内部同步 |
| 默认流隐式同步 | 默认流 ↔ 所有阻塞流 | 上面的 `stop` 靠的就是它 |

**注意 .cu 与 .py 的一个差别**：原生 `.cu` 用的是**阻塞流 + 默认流隐式同步**，所以不用显式同步 s1/s2 就能测总耗时；而 `torch.cuda.Stream()` 创建的是**非阻塞流**，与默认流不隐式同步，所以 Python 版必须显式 `s_comp.synchronize()` / `torch.cuda.synchronize()`。这不是谁对谁错，而是**流是否阻塞**这一标志（`cudaStreamNonBlocking`）决定的——想精确控制依赖时，非阻塞流 + `wait_event` 更可控。

## 补充 2：串行 → 并发，以及 CUDA Graph / 框架编排

### 2.1 流内串行 ≠ GPU 一直满载

流内算子是**串行（有先后顺序）**的，但两个算子之间并不是"无缝衔接"，中间有：

1. **启动开销（launch overhead）**：主机把 kernel 排进流 + 驱动调度上 GPU，单次约几 µs；
2. **依赖停顿**：下一个 kernel 等上一个的输入就绪。

当算子本身很小（几 µs 算完）时，启动开销占比极高，GPU 大量时间在空转。

### 2.2 多流能提升利用率吗？——能，但有条件

你的直觉是对的：**当前一个 kernel 没有占满 SM（占用率/occupancy 低）时，多流可以把"并发 kernel"调度到空闲的 SM 上，填满空档，提升利用率**。前提：

- 各 kernel 的网格/资源不足以占满所有 SM（有空闲资源）；
- 用非阻塞流（或不同阻塞流之间）；
- 两个 kernel 的寄存器/共享内存之和不超过 SM 上限。

反过来：像 `spin_kernel<<<256 块>>>` 这种一个 kernel 就占满 ~24 个 SM 的，两个流也叠不起来——这正是第 3 段注释写"**若**≈单 kernel 耗时，说明并发"的原因：**并发是条件成立的**。

最稳定、最常被利用的两种"并行"是：

1. **不同硬件单元**：拷贝（copy engine）+ 计算（SM），见 `03_overlap`；
2. **多个小算子打包**：见下面的 CUDA Graph。

### 2.3 CUDA Graph：消掉启动开销

把一整段 kernel 序列"录制"成一个图，之后 `replay` 一次就把整段跑完，只花 1 次启动开销。实测（`python/07_cuda_graph.py`）：

```
普通逐个启动 200 个小 sin：1.44 ms   （200 次启动开销 + 空转）
CUDA Graph 单次 replay：    0.30 ms   （1 次启动）
加速比：4.86x
```

API 对应：`cudaStreamBeginCapture/EndCapture` → `torch.cuda.graph(g)`；`cudaGraphInstantiate/Launch` → `g.replay()`。

**限制**：图里 kernel 的输入输出**形状/地址固定**（复用同一批缓冲）；不能有动态控制流或 host 回调；形状变了要重录。

### 2.4 框架是怎么编排的

- **PyTorch**：默认每个线程一个"当前流"，所有 op 排到当前流。数据并行（DDP）里，梯度 `allreduce` 用**独立非阻塞流**，与反向传播的**计算重叠**；推理侧用 `torch.cuda.CUDAGraph`；`torch.compile`（Inductor）把多个小 op **融合成更少的 kernel**（也是一种"减少启动次数"）。
- **NCCL**：内部用独立流 + 事件，把通信和计算重叠（见 `docs/04_nccl.md`）。
- **TensorRT / TensorRT-LLM / vLLM**：把整个模型编译成一张**执行图**（≈ CUDA Graph），预调优 kernel，运行时几乎零启动开销——这是推理延迟做到极致的核心手段。

一句话：**"多流"解决的是"并发/重叠"，"CUDA Graph + 算子融合"解决的是"启动开销"。** 两者正交，现代推理栈通常是"图 + 流"一起用。

