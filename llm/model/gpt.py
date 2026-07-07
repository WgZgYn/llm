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
                     config.bias, config.dropout,
                     checkpoint=config.gradient_checkpointing)

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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                kv_cache: list[dict] | None = None):
        """前向传播，支持 KV-Cache 推理加速。

        参数:
            idx:      [B, T] 或 [B, 1] (cache 模式下通常只有 1 个新 token)
            targets:  [B, T]，None 时仅推理
            kv_cache: 每个 Block 一个 dict {'k': ..., 'v': ...}，None 时不使用
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"序列长度 {t} 超过 block_size {self.config.block_size}"

        # 位置编码: cache 模式下只取最后一个位置
        if kv_cache is not None and len(kv_cache) > 0 and 'k' in kv_cache[0]:
            cache_len = kv_cache[0]['k'].size(2)  # 已经缓存的长度
            pos = torch.arange(cache_len, cache_len + t, dtype=torch.long, device=device)
        else:
            cache_len = 0
            pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # 穿行 blocks，逐个传递 kv_cache
        for i, block in enumerate(self.transformer.h):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x = block(x, layer_cache)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.ignore_index)
            if self.training:
                for block in self.transformer.h:
                    moe = getattr(block, 'mlp', None)
                    if moe is not None and hasattr(moe, 'pop_aux_loss'):
                        loss = loss + moe.pop_aux_loss()
        else:
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
        use_cache: bool = True,
    ) -> torch.Tensor:
        """自回归文本生成，可选 KV-Cache 加速。

        参数:
            idx:             初始 token ids [batch_size, prefix_len]
            max_new_tokens:  最多生成多少个新 token
            temperature:     温度
            top_k:           top-k 采样
            use_cache:       是否使用 KV-Cache（默认 True）
        """
        # ── 没有 cache 的慢路径（原逻辑，兼容）──
        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size \
                    else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

        # ── KV-Cache 路径（快）──
        n_layers = self.config.n_layer
        kv_cache = [{} for _ in range(n_layers)]

        next_input = idx  # [B, T] or [B, 1]
        for step in range(max_new_tokens):
            logits, _ = self(next_input, kv_cache=kv_cache)

            # 采样下一个 token
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_input = torch.multinomial(probs, num_samples=1)  # [B, 1]

            idx = torch.cat((idx, next_input), dim=1)

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
