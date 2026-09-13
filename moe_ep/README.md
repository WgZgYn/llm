# Toy MoE Expert Parallelism on 4×V100

一个最小但完整、可测量、可理解的 PyTorch Distributed **Expert Parallelism (EP)** Demo。
单机 4×Tesla V100 16GB，NCCL，**只做 forward inference**。

不是生产级 MoE，不追求极致性能。目标是把下面 8 件事用可以跑起来的代码和可以读的数字讲清楚：

| # | 问题 | 对应实验 |
|---|---|---|
| 1 | Router 如何把 token 分配给 expert | 实验 A/B（router 相位耗时 + expert 直方图） |
| 2 | EP 如何把同一个 MoE Layer 的不同 expert 分到不同 GPU | 实验 C（参数字节数 / rank） |
| 3 | token 如何经 all-to-all dispatch 到目标 expert 所在 GPU | 实验 B（`dispatch.*` 相位）+ `check_env.py` §6 |
| 4 | expert 算完如何经 inverse all-to-all 返回 | 实验 B（`combine.*` 相位）+ `--identity-experts` 往返测试 |
| 5 | EP 的显存收益来自哪里 | 实验 C |
| 6 | EP 的通信代价来自哪里 | 实验 D（实测字节 vs `2·T·k·H·b`） |
| 7 | token 数 / EP size / routing imbalance / GPU 拓扑如何影响性能 | 实验 B/D + `bench_topology.py` |
| 8 | Prefill-like 大 batch 与 Decode-like 小 batch 为何行为迥异 | 实验 B 的 prefill/decode 双阶梯 |

---

## 环境要求

- Linux 单机，**4 × Tesla V100 16GB**（sm_70 / Volta）
- PyTorch ≥ 2.x + CUDA 12.x，`torch.distributed` + NCCL
- Python ≥ 3.10（开发时用的是 3.12.9）
- **不依赖** DeepSpeed / Megatron / DeepEP / TP / PP；只用 `torch` + 标准库

> ⚠️ **V100 是 Volta，硬件不支持 bf16。** 它只有 fp16 tensor core。`--dtype bf16` 会被
> 直接拒绝（除非显式 `--allow-bf16`），因为 cuBLAS 在 sm_70 上没有 bf16 路径，测出来的
> 时间没有意义。默认 `--dtype fp16`，需要干净数值时用 `--dtype fp32`。
>
> 本项目**不使用 `torch.autocast`**：隐式类型转换会让"这个相位到底跑的 fp16 还是 fp32"
> 变得说不清，计时和显存归因都会失真。

## 部署

整个目录自包含，直接拷到目标机即可：

```bash
scp -r moe_ep user@v100-host:~/
ssh user@v100-host
cd ~/moe_ep
source scripts/env.example.sh     # 强烈建议：见下面"为什么"
```

所有入口脚本自己会把项目根加进 `sys.path`，**不需要**额外设 `PYTHONPATH`，也不依赖 cwd。

---

## 快速开始（请按顺序，第 2 步不要跳过）

```bash
# 1) 先看环境事实（单进程，不需要 torchrun）
python scripts/check_env.py --help

# 2) 门禁：环境 + P2P 带宽矩阵 + all_to_all 语义冒烟测试
source scripts/env.example.sh
torchrun --standalone --nproc_per_node=4 scripts/check_env.py

# 3) 正确性：EP 输出对齐 dense 参考
for ep in 1 2 4; do
  torchrun --standalone --nproc_per_node=$ep scripts/verify_ep.py --preset tiny
done

# 4) 两分钟快速一遍（EP ∈ {1,4}，只看均衡路由）
bash scripts/run_quick.sh

# 5) 完整扫描
bash scripts/run_all.sh
PRESET=wide OUT=out/wide bash scripts/run_all.sh
```

**为什么第 2 步不能跳过：** 它在任何 MoE 代码跑起来之前，用**已知数据 + 不等长 split**
验证 `all_to_all_single` 的 split 语义。`ep_moe/comm.py` 里最容易出错、且错了之后症状
最难看的一步是"输出 split 是输入 split 的转置"——如果漏了，均匀路由下数字恰好相等、一切
正常，一旦路由倾斜就变成静默的数据错位。`check_env.py` §6 专门用非均匀 split 把这个问题
在几秒钟内暴露成一行 expected-vs-actual。

**为什么建议先 `source scripts/env.example.sh`：** 其中
`TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 把"split 不匹配导致挂死 30 分钟"变成"3 分钟后给出
指名到 rank 的 Python traceback"。在一台你只能 ssh 上去的机器上，这个差别很大。

### 关于 sweep 的编排

EP size **不是**一个运行时开关——`world_size` 就是 `ep_size`（纯 EP，不做 sub-group 切分）。
所以每个 EP size 起一次独立的 `torchrun --nproc_per_node=$EP`。代价是启动三次，换来的是
**partial-group 类 bug 在构造上不可能发生**。

---

## 实验

### A. 正确性（EP vs dense 参考）

```bash
for ep in 1 2 4; do
  torchrun --standalone --nproc_per_node=$ep scripts/verify_ep.py --preset tiny
done
torchrun --standalone --nproc_per_node=4 scripts/verify_ep.py --preset tiny --skew 4.0   # 制造极端不均衡
```

每个 rank 额外构建一份 **dense 参考模型**（放全部 E 个 expert），对本 rank 的 token 切片
做前向，然后本地比对——不需要 gather。参考模型与 EP 模型共用同一套权重初始化
（`(seed, global_expert_id)` 决定权重），所以两者的差异只可能来自通信。

打印的是**数字而不是布尔**：`max_abs_err` / `mean_abs_err` / `max_rel_err` / `rel_l2` /
最差元素下标，失败时还会列出最差的 5 个元素。在远程机器上，`allclose` 返回 False 本身
说明不了任何问题。

同时会检查：

- **routing 一致性**：dense 与 EP 的 `topk_idx` 必须逐元素相等。不相等的话，下面的张量
  比对毫无意义。
- **expert 划分**：跨 rank all-gather 后必须恰好覆盖 `0..E-1` 各一次。
- **router 指纹**：每个 rank 的 router 权重哈希必须相同。这是全项目最阴的失败模式——各
  rank seed 不一致会导致 token 被送到错误的 expert，**不崩溃**，只是数值悄悄错掉。

### B. 分阶段计时 + prefill/decode 对照

```bash
torchrun --standalone --nproc_per_node=4 scripts/bench_ep.py --preset tiny --mode all
torchrun --standalone --nproc_per_node=4 scripts/bench_ep.py --preset tiny --mode all --skew 3.0
```

输出两张表：相位耗时表与通信量表。

相位：`router` / `dispatch.expand` / `dispatch.count` / `dispatch.split_a2a` /
`dispatch.a2a` / `dispatch.group` / `expert` / `combine.reorder` / `combine.a2a` /
`combine.scatter_add`。层号会被折叠（`L0.router` + `L1.router` → `router`）。

三条计时约定，写在 `ep_moe/timer.py` 里也写在表头上：

1. **事件在一次迭代内记录，迭代末尾统一 sync 一次后读取。** `elapsed_time` 要求 end
   event 已完成，如果在相位中间读就会强制一次 host sync，把后面所有相位都撑大。
2. **一次迭代只 sync 一次，绝不在相位内部 sync。**
3. **集合通信的耗时按跨 rank 的 max 报告。** 集合通信在最后一个 rank 到达时才结束，所以
   max 才是真实代价；`straggler = max/mean` 就是路由不均衡的可观测信号。表头写了
   "ms(max rank)"，不要看到 `max > mean` 就以为出错。

### C. 显存对比

`--mode all` / `--mode mem` 会打印显存分解表。跨 EP size 比较请把三次运行的 `out/`
放在一起看。

同时报告 `max_memory_allocated` 和 `max_memory_reserved`：**NCCL 自己的通信 buffer 不进
torch allocator**，只体现在 reserved 里，而且基本不随 EP 缩小。只看 allocated 会高估 EP
的显存收益——而"EP 的显存收益从哪来"正是这个 demo 要回答的问题之一，所以这里刻意把不好
看的那一半也打出来。

`verify_ep.py` 会预估显存并在超限时**拒绝运行**而不是 OOM。

### D. 拓扑 / 通信量

```bash
torchrun --standalone --nproc_per_node=4 scripts/bench_topology.py
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 scripts/bench_topology.py
```

- 有向 P2P 带宽矩阵（组内 0-1/2-3 vs 跨组 SYS），**单向 payload GB/s，在接收端计时**。
  这个定义与 all-reduce benchmark 常用的 `busbw` 不同，会偏低，是预期的。
- `all_to_all` 带宽随消息大小变化，从 1KB 到 4MB——**延迟/带宽拐点**就在这里。
- 两次运行（P2P 开/关）的差值就是 P2P 链路的贡献。

通信量表把实测字节与闭式解 `2·T_local·k·H·dtype_bytes`（dispatch + combine）对比，并单独
列出 expert id 侧信道（每行 8 字节 int64）。倾斜路由那一行是重点：**每 rank 字节数分化，
均值几乎不变，但集合通信时间由 max 决定，所以时间反而上升**——这一行就是学习目标 7。

---

## 代码结构

```
moe_ep/
├── ep_moe/
│   ├── config.py        ModelConfig / RunConfig / PRESETS，含显存估算与校验
│   ├── dist_utils.py    进程组初始化（3 分钟超时 + LOCAL_RANK 硬失败）、EPLayout、跨 rank 归约
│   ├── init_utils.py    确定性初始化：专家 e 的权重只取决于 (seed, 全局 expert id)
│   ├── router.py        TopKRouter（skew 注入）+ 负载统计
│   ├── comm.py     ★    all_to_all dispatch/combine —— 全项目唯一调用 all_to_all_single 的文件
│   ├── experts.py       SwiGLU expert bank，per-expert 循环 / padded grouped GEMM 两种实现
│   ├── moe_layer.py     EPMoELayer 与 DenseMoELayer（参考实现）
│   ├── model.py         EPMoEStack / DenseMoEStack、参数指纹、显存与显存模型
│   ├── timer.py         PhaseTimer（相位计时 + NVTX）、表格格式化
│   ├── topology.py      P2P 访问矩阵、单向带宽矩阵、all_to_all 带宽扫描
│   └── report.py        JSON/CSV/markdown 产物
└── scripts/
    ├── check_env.py  ★  门禁：环境 + 拓扑 + all_to_all 语义 + identity 往返
    ├── verify_ep.py     正确性：EP vs dense
    ├── bench_ep.py      四项实验：相位计时 / prefill-decode / 显存 / 通信量
    ├── bench_topology.py 链路测量
    ├── run_all.sh / run_quick.sh
    └── env.example.sh
```

`comm.py` 被单独隔离成一个文件，理由只有一个：split 算术是整个项目里唯一**错了不报错、
只给出错误数字**的地方。隔离出来，`check_env.py` 才能用已知数据对它做独立验证。

---

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--preset` | `tiny` | tiny(H=1024,F=4096,E=8,k=2) / medium(H=2048,F=8192,E=8) / wide(H=2048,F=8192,E=32) |
| `--hidden --ffn --num-experts --top-k --num-layers` | 跟随 preset | 逐项覆盖 preset |
| `--dtype` | `fp16` | `fp32` / `fp16`；`bf16` 在 Volta 上被拒绝 |
| `--ep-size` | `world_size` | 必须等于 world_size，否则给出修正后的 torchrun 命令 |
| `--tokens` | 4096 | **全局** token 数，按 rank 均分（保证 EP=1/2/4 可比） |
| `--tokens-per-rank` | — | 直接指定本地 token 数，绕过整除约束 |
| `--skew` / `--skew-hot` | 0 / 4 | 给前 N 个 expert 的 logits 加常数，制造可控不均衡 |
| `--balance` | `natural` | `uniform` 强制完美轮转路由（隔离通信代价与不均衡代价） |
| `--warmup` / `--iters` | 5 / 20 | 共享机器上建议 `--iters 50` 取最终数字 |
| `--identity-experts` | 关 | expert 直接返回 x，用于把通信 bug 与数学 bug 分开 |
| `--fuse-dispatch` | 关 | 把 x / expert id / gate weight 打包成一次 all_to_all（省两次 launch） |
| `--grouped-gemm` | 关 | padded bmm 代替 per-expert 循环（更快但会抹平不均衡信号） |
| `--nvtx` | 关 | 输出 NVTX range，配合 `nsys profile --trace=cuda,nvtx` |
| `--deterministic` | 关 | 会设置 `CUBLAS_WORKSPACE_CONFIG` |
| `--mode` | `measure` | `measure` / `prefill` / `decode` / `mem` / `all` |
| `--out` / `--tag` / `--no-write` | `out/` / — | 产物目录与命名 |

---

## 预期定性结果

- **G1 Routing**：把 router logits 取 top-k 得到**全局** expert id。`skew=0` 时负载**也
  不是均匀的**——expert 直方图本身就值得看。
- **G2 EP 切分**：每 rank 持有 `E/world_size` 个 expert，参数字节数按 `1/EP` 下降。
- **G3/G4 dispatch/combine**：`--identity-experts` 下输出**精确等于**输入（softmax 权重
  和为 1，identity expert 原样返回）。这条能过，说明 comm.py 的每一行都对。
- **G5 显存**：expert 权重按 `1/EP` 缩小；`peak_reserved` 缩小得**更少**；all_to_all 的
  中间 buffer **完全不随 EP 缩小**；router 是完全复制的。**净收益 < 朴素的 1/EP 收益。**
- **G6 通信代价**：每 rank 每层约 `2·T_local·k·H·b` 字节。tiny 配置下（H=1024, k=2, fp16）
  是每 token 每 rank 8KB——字节数很小，代价主要由固定启动开销与 host sync 主导，
  GB/s 一栏会远低于 P2P 链路上限。
- **G7 token 数 / EP / 不均衡 / 拓扑**：`T_local` 变大 → 通信转向带宽受限，GB/s 向 P2P
  平台爬升；EP 变大 → 消息更多更小 → 更差；加 skew → 字节均值不变但 **max 的耗时上升**
  （看 straggler 一栏），且跨组（0-2, 1-3）流量重的那个 rank 就是 straggler。
- **G8 prefill vs decode**：`T_local = 1..16` 时相位表几乎全是固定开销，`dispatch.split_a2a`
  这种小集合通信的占比反而上升，总耗时**几乎与 T_local 无关**，EP 的相对代价最差；
  `T_local = 4096` 时 expert GEMM 占主导，通信 GB/s 接近链路平台。

---

## 排错

| 症状 | 原因 / 处理 |
|---|---|
| 挂住不动 | `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,COLL`，并设 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`。进程组超时是 3 分钟，别等 30 分钟。看是哪个 rank 没到 barrier。 |
| `size mismatch` / 数据错位 | 十有八九是 `output_split_sizes` 没转置。跑 `check_env.py` §6。 |
| 数值不对但形状对 | 先看 `verify_ep.py` 的 routing 一致性与 router 指纹。seed 用了 rank 相关的值是经典原因。 |
| `ModuleNotFoundError: ep_moe` | 入口脚本自己会加 sys.path；如果你在别处 import，需要 `PYTHONPATH=.`。 |
| 第一次迭代特别慢 | NCCL communicator 初始化 + cuBLAS autotune。warmup 不是可选项。 |
| `CUDA_VISIBLE_DEVICES` 相关怪象 | `--nproc_per_node=4` 时**不要设**它。设成少于 4 个条目会让 `LOCAL_RANK` 越界或绑错卡。 |
| OOM | 看是哪个 buffer：`wide` + EP=1 的 expert 权重约 2.15GB，dense 参考再来一份。用 `--preset medium`、`--num-layers 1` 或 `--skip-dense`。 |
| all_to_all 卡在某个 rank | 集合通信是 SPMD 的：任何 rank 提前 return 都会让其余 rank 挂死。检查是不是有跨过集合通信的分支。 |
| 同一配置两次结果不一致 | `index_add_` 在 CUDA 上是浮点非确定的。想要确定就得改用 sort-based 分段求和，或者接受容差。 |
| 极端 skew 下某个 rank 收到 0 个 token 后出错 | 这是**零长度 NCCL 传输**路径（`recv_n == 0`）。`check_env.py` §7 的 skewed 用例会故意触发它，就是想让这个问题在门禁阶段暴露而不是在 benchmark 里。如果只有它失败，说明这个 NCCL 构建对零长度 all_to_all 的处理有问题——先把 `--skew` 调小。 |

---

## 核心结论速览

1. EP 的显存收益 ≈ `(E - E/EP) × 3·H·F × dtype_bytes`，但会被**不随 EP 缩小的 all_to_all
   buffer 与 NCCL 内部 buffer** 部分抵消，token 越多抵消越多。
2. EP 的通信量**不随 EP 缩小**——它随每 rank 的 token 数线性增长，EP 只改变"摊到几个对端
   上"。EP 越大，消息越多越小，越吃亏。
3. 集合通信的耗时由**最慢的 rank** 决定，路由不均衡直接转化为拖尾，而不是被平均掉。
4. 小消息（decode-like）下 `all_to_all` 是延迟受限的：固定开销不随 token 数下降，所以
   decode 阶段的 EP 相对代价最差。
5. 组内链路（0-1, 2-3）与跨组（SYS）的带宽差，决定了谁来当 straggler。
6. `--identity-experts` 的往返不变式（输出 == 输入）是验证 dispatch/combine 正确性最快、
   最强的单条测试。

---

## 产物

`out/` 下每个配置一条 JSON 记录，追加写入 `*.jsonl`：

```python
import pandas as pd
df = pd.read_json("out/20260913_120000/bench_ep4.jsonl", lines=True)
df[["tag", "total_ms", "imbalance_factor", "comm_eff_GBps", "peak_reserved_MB"]]
```

同时生成 `*.csv`（把 `phases` 摊平成 `phase.<name>.ms` 列）和自动汇总的
`*_report.md`（含表格与结论）。markdown 是从记录生成的，不会和数字脱节。
