"""GPT 语言模型 —— 完整的生成式预训练 Transformer。

这是项目的核心模型文件，整合了 nanoGPT 的全部设计改进。

用法:
    from llm import GPT, GPTConfig

    # 从零初始化
    config = GPTConfig.from_preset("gpt2")
    model = GPT(config)

    # 从 HuggingFace 加载预训练权重
    model = GPT.from_pretrained("gpt2")

    # 生成文本
    ids = model.generate(tokenizer.encode("Hello"), max_new_tokens=50)
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..config.model_config import GPTConfig
from ..model.layer import LayerNorm, Block


class GPT(nn.Module):
    """完整的 GPT 语言模型。

    整合了 nanoGPT 的全部设计改进:
    1.  融合 QKV 投影（CausalSelfAttention 中）
    2.  Pre-LN 风格（Block 中）
    3.  可配置 bias（Linears + LayerNorms）
    4.  Flash Attention（CausalSelfAttention 中）
    5.  Weight Tying（wte ↔ lm_head 权重共享）
    6.  可学习位置嵌入（nn.Embedding）
    7.  缩放残差初始化（c_proj 权重特殊处理）
    8.  分组 Weight Decay（configure_optimizers）
    9.  分层 Dropout（attn_dropout + resid_dropout）
    10. from_pretrained() 加载 HF GPT-2
    11. crop_block_size() 模型手术
    12. estimate_mfu() 硬件利用率监控
    13. generate() temperature + top-k 采样
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # ── Transformer 主体（使用 ModuleDict 便于状态管理）──
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),       # token 嵌入
            wpe=nn.Embedding(config.block_size, config.n_embd),       # 位置嵌入（可学习）
            drop=nn.Dropout(config.dropout),                           # 嵌入 dropout
            h=nn.ModuleList([self._make_block(config, i) for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),          # 最终 LayerNorm
        ))

        # ── 输出头 ──
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ── Weight Tying: 共享输入嵌入和输出投影的权重 ──
        # 这是 GPT-2 的标准做法，节省 d_model × vocab_size 个参数
        # 注: torch.compile() 下可能产生无害的 UserWarning
        self.transformer.wte.weight = self.lm_head.weight

        # ── 权重初始化 ──
        self.apply(self._init_weights)
        # 残差投影特殊缩放: 控制深层网络的激活方差增长
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # 打印参数量
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    # ═══════════════════════════════════════════════════════════════════
    # 参数与权重
    # ═══════════════════════════════════════════════════════════════════

    def get_num_params(self, non_embedding: bool = True) -> int:
        """返回模型参数量。

        non_embedding=True: 减去位置嵌入的参数量
        （token 嵌入因 weight tying 实际用作输出权重，所以保留）
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()  # 位置嵌入不算
        return n_params

    def _make_block(self, config: GPTConfig, layer_idx: int) -> nn.Module:
        """构造一个 Transformer Block。子类重写此方法以实现变种（如 MoE）。"""
        return Block(config.n_embd, config.n_head, config.block_size,
                     config.bias, config.dropout)

    def _init_weights(self, module):
        """统一的权重初始化。

        - Linear:   Normal(0, 0.02), bias → zeros
        - Embedding: Normal(0, 0.02)
        - LayerNorm: weight → ones, bias → zeros（默认不变）
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ═══════════════════════════════════════════════════════════════════
    # 前向传播
    # ═══════════════════════════════════════════════════════════════════

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """前向传播。

        参数:
            idx:    输入 token ids [batch_size, seq_len]
            targets: 目标 token ids [batch_size, seq_len]，为 None 时仅预测最后位置

        返回:
            (logits, loss): loss 仅在 targets 不为 None 时有值
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"序列长度 {t} 超过 block_size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # ── 嵌入层 ──
        tok_emb = self.transformer.wte(idx)          # token 嵌入 [B, T, n_embd]
        pos_emb = self.transformer.wpe(pos)           # 位置嵌入 [T, n_embd]
        x = self.transformer.drop(tok_emb + pos_emb)  # 相加 + dropout

        # ── Transformer blocks ──
        for block in self.transformer.h:
            x = block(x)

        # ── 最终 LayerNorm ──
        x = self.transformer.ln_f(x)

        # ── 输出 ──
        if targets is not None:
            # 训练模式: 对所有位置计算 loss
            logits = self.lm_head(x)  # [B, T, vocab_size]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

            # ── 收集 MoE 辅助 loss ──
            if self.training:
                for block in self.transformer.h:
                    moe = getattr(block, 'mlp', None)
                    if moe is not None and hasattr(moe, 'pop_aux_loss'):
                        loss = loss + moe.pop_aux_loss()
        else:
            # 推理优化: 只计算最后一个位置的 logits
            # 用 [-1] 而不是 [-1:] 保留时间维度（torch.compile 友好）
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # ═══════════════════════════════════════════════════════════════════
    # 文本生成
    # ═══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """自回归文本生成。

        参数:
            idx:             初始 token ids [batch_size, prefix_len]
            max_new_tokens:  最多生成多少个新 token
            temperature:     温度（>1 更随机，<1 更确定，=1 原始分布）
            top_k:           只从概率最高的 k 个 token 中采样（None = 不限制）

        返回:
            完整序列（输入 + 生成的）[batch_size, prefix_len + new_tokens]
        """
        for _ in range(max_new_tokens):
            # 如果序列过长，裁剪到 block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )

            # 前向传播（只取最后位置的 logits）
            logits, _ = self(idx_cond)

            # 温度缩放
            logits = logits[:, -1, :] / temperature

            # Top-K 筛选
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    # ═══════════════════════════════════════════════════════════════════
    # 模型手术
    # ═══════════════════════════════════════════════════════════════════

    def crop_block_size(self, block_size: int):
        """缩小模型的 block_size。

        场景: 加载 GPT-2 预训练权重（block_size=1024）后，
        想用于更小上下文的简单任务。减少位置嵌入和支持的序列长度。

        实现: 截断 wpe 权重、更新 config、裁剪 causal mask（如果有）。
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        # 裁剪每个 block 的 causal mask buffer（仅在不使用 Flash Attention 时有）
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # ═══════════════════════════════════════════════════════════════════
    # 硬件利用率估算
    # ═══════════════════════════════════════════════════════════════════

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """估算 Model FLOPs Utilization (MFU)。

        以 A100 bfloat16 峰值 312 TFLOPS 为基准，logger 会自动修正为实际 GPU。

        参数:
            fwdbwd_per_iter: 每次 optimizer step 处理的序列数
                             (gradient_accumulation_steps × batch_size × world_size)
            dt:              单次迭代耗时（秒）

        返回:
            MFU 比例 (0.0 ~ 1.0+)
        """
        # FLOPs 估算: 参考 PaLM 论文 Appendix B
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size

        # flops_per_token: 每个 token 的前向+反向 FLOPs
        flops_per_token = 6 * N + 12 * L * H * Q * T
        # flops_per_seq: 一个完整序列 (长度 T) 的前向+反向 FLOPs
        flops_per_seq = flops_per_token * T
        # flops_per_iter: 一次 optimizer step 的总 FLOPs
        flops_per_iter = flops_per_seq * fwdbwd_per_iter

        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 bfloat16 peak (logger 中按实际 GPU 修正)
        return flops_achieved / flops_promised

    # ═══════════════════════════════════════════════════════════════════
    # 预训练权重加载
    # ═══════════════════════════════════════════════════════════════════

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: dict | None = None):
        """从 HuggingFace GPT-2 预训练权重加载模型。

        支持: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

        HF 的 GPT-2 使用 Conv1D 而非 nn.Linear（权重是转置的），
        此方法自动处理转置。

        用法:
            model = GPT.from_pretrained("gpt2")
            model = GPT.from_pretrained("gpt2-medium", override_args={'dropout': 0.1})
        """
        from transformers import GPT2LMHeadModel

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args), \
            "目前只支持覆盖 'dropout' 参数"

        print(f"loading weights from pretrained gpt: {model_type}")

        # ── 根据 model_type 确定架构 ──
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']

        # ── 创建并加载 ──
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias')
                      and not k.endswith('.attn.bias')]

        # ── Conv1D → Linear 权重转置 ──
        transposed = [
            'attn.c_attn.weight', 'attn.c_proj.weight',
            'mlp.c_fc.weight', 'mlp.c_proj.weight',
        ]
        assert len(sd_hf_keys) == len(sd_keys), \
            f"keys 数量不匹配: {len(sd_hf_keys)} != {len(sd_keys)}"

        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                # Conv1D 权重需要转置
                assert sd_hf[k].shape[::-1] == sd[k].shape, \
                    f"shape 不匹配: {k} {sd_hf[k].shape[::-1]} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
