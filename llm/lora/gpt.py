"""LoRAGPT —— GPT + LoRA 低秩适配。

完全模仿 MoEGPT 的架构（llm/model/moe_gpt.py）：
    继承 GPT，重写 _make_block() 钩子，在指定层注入 LoRA。

用法:
    from llm.lora import LoRAConfig, LoRAGPT
    from llm.config import GPTConfig

    # 方式 1: 从预训练权重加载（推荐）
    lora_config = LoRAConfig(r=8, alpha=16.0)
    model = LoRAGPT.from_pretrained("gpt2", lora_config)

    # 方式 2: 从零训练
    config = GPTConfig(vocab_size=50304, block_size=512, n_layer=6, ...)
    lora_config = LoRAConfig(r=4, alpha=8.0)
    model = LoRAGPT(config, lora_config)

    # 方式 3: 从本地 checkpoint 恢复
    model = LoRAGPT(config, lora_config)
    model.load_state_dict(torch.load("ckpt.pt")['model'], strict=False)
"""

import torch
import torch.nn as nn

from ..config.model_config import GPTConfig
from ..model.gpt import GPT
from ..model.layer import Block
from .config import LoRAConfig
from .block import LoRABlock
from .linear import LoRALinear


class LoRAGPT(GPT):
    """GPT with LoRA (Low-Rank Adaptation)。

    不重写 __init__，不重写 forward。只重写 _make_block() 钩子。

    参数:
        config:       GPTConfig（与普通 GPT 完全一样）
        lora_config:  LoRAConfig（LoRA 适配器配置）
    """

    def __init__(self, config: GPTConfig, lora_config: LoRAConfig):
        self._lora_config = lora_config
        super().__init__(config)
        # super().__init__ 中的 self.apply(self._init_weights) 会
        # 把 lora_A/lora_B 也重新初始化为 normal(0,0.02)，
        # 需要在此处恢复 LoRA 的正确初始化
        self._reset_lora_weights()

    # ═══════════════════════════════════════════════════════════════
    # _make_block 钩子（与 MoEGPT 模式一致）
    # ═══════════════════════════════════════════════════════════════

    def _make_block(self, config: GPTConfig, layer_idx: int) -> nn.Module:
        """GPT 构造每个 Block 时调用此钩子。

        LoRAGPT 在 target_layers 指定的层返回 LoRABlock，
        其余层返回普通 Block。
        """
        lc = self._lora_config

        if self._layer_needs_lora(layer_idx, lc):
            return LoRABlock(
                config.n_embd, config.n_head, config.block_size,
                config.bias, config.dropout,
                lora_config=lc,
                checkpoint=config.gradient_checkpointing,
            )
        else:
            return Block(
                config.n_embd, config.n_head, config.block_size,
                config.bias, config.dropout,
                checkpoint=config.gradient_checkpointing,
            )

    @staticmethod
    def _layer_needs_lora(layer_idx: int, lc: LoRAConfig) -> bool:
        """判断第 layer_idx 层是否需要 LoRA。"""
        tl = lc.target_layers
        if tl is None:
            return True  # 全部层
        if isinstance(tl, tuple):
            return tl[0] <= layer_idx < tl[1]  # 范围
        if isinstance(tl, list):
            return layer_idx in tl  # 指定索引列表
        return True

    # ═══════════════════════════════════════════════════════════════
    # from_pretrained —— 从 HuggingFace 预训练权重加载
    # ═══════════════════════════════════════════════════════════════

    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        lora_config: LoRAConfig | None = None,
        override_args: dict | None = None,
    ) -> "LoRAGPT":
        """从 HuggingFace GPT-2 预训练权重加载 LoRAGPT。

        直接加载 HF 权重并映射到 LoRAGPT，处理以下差异:
        - HF Conv1D → nn.Linear 转置
        - HF mlp.c_fc/c_proj → 本地 mlp.net.0/net.2 命名
        - Attention Linear → LoRALinear.weight buffer

        用法:
            lora_config = LoRAConfig(r=8, alpha=16.0)
            model = LoRAGPT.from_pretrained("gpt2", lora_config)
        """
        from transformers import GPT2LMHeadModel

        if lora_config is None:
            lora_config = LoRAConfig()

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}

        print(f"Loading pretrained weights from HuggingFace: {model_type}")

        # ── Step 1: 确定架构参数 ──
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']

        model_config = GPTConfig(**config_args)

        # ── Step 2: 创建 LoRAGPT ──
        lora_model = cls(model_config, lora_config)

        # ── Step 3: 加载 HF 权重 ──
        print(f"  Downloading {model_type} from HuggingFace...")
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # ── Step 4: 构建 HF → LoRAGPT key 映射并拷贝权重 ──
        sd_lora = lora_model.state_dict()
        transposed = [
            'attn.c_attn.weight', 'attn.c_proj.weight',
            'mlp.c_fc.weight', 'mlp.c_proj.weight',
        ]
        # HF MLP 命名 → 本地 MLP Sequential 命名
        mlp_name_map = {
            'mlp.c_fc': 'mlp.net.0',
            'mlp.c_proj': 'mlp.net.2',
        }

        loaded = 0
        skipped_lora = 0

        for hf_key, hf_tensor in sd_hf.items():
            # 跳过 HF 特有的非权重 key
            if hf_key.endswith('.attn.masked_bias') or hf_key.endswith('.attn.bias'):
                continue

            # 转换 HF key → 本地 key
            local_key = hf_key

            # MLP 命名转换: c_fc → net.0, c_proj → net.2
            for hf_name, local_name in mlp_name_map.items():
                if hf_name in local_key:
                    local_key = local_key.replace(hf_name, local_name)
                    break

            if local_key not in sd_lora:
                # LoRA 特有的 key（lora_A/lora_B），跳过
                skipped_lora += 1
                continue

            # ── Conv1D → Linear 转置 ──
            # HF GPT-2 的权重存储为 (in_features, out_features)，
            # 而 nn.Linear 存储为 (out_features, in_features)。
            # 对于非方阵 (如 c_attn: [768,2304]) shape 不同可检测；
            # 对于方阵 (如 c_proj: [768,768]) 必须强制转置。
            if any(hf_key.endswith(w) for w in transposed):
                target = sd_lora[local_key]
                if hf_tensor.shape == target.shape and hf_tensor.shape[0] != hf_tensor.shape[1]:
                    # 非方阵且 shape 相同 → 直接复制
                    target.copy_(hf_tensor)
                else:
                    # 需要转置（HF [in,out] → 我们 [out,in]）
                    target.copy_(hf_tensor.t().contiguous())
            else:
                # bias / LayerNorm / embedding → 直接复制
                if hf_tensor.shape == sd_lora[local_key].shape:
                    sd_lora[local_key].copy_(hf_tensor)
                else:
                    print(f"  [SKIP] shape mismatch: {hf_key} "
                          f"HF{list(hf_tensor.shape)} vs local{list(sd_lora[local_key].shape)}")
                    continue

            loaded += 1

        print(f"  Loaded {loaded} weights from HF (skipped {skipped_lora} LoRA-only keys)")

        # ── Step 5: 冻结 base weights ──
        lora_model.freeze_base_weights(
            train_wpe=lora_config.train_wpe,
            train_ln=lora_config.train_ln,
        )

        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total = lora_model.get_num_params()
        print(f"  LoRA trainable: {trainable/1e6:.2f}M / {total/1e6:.1f}M "
              f"({trainable/total*100:.2f}%)")

        return lora_model

    # ═══════════════════════════════════════════════════════════════
    # 冻结 / 解冻 base weights
    # ═══════════════════════════════════════════════════════════════

    def _reset_lora_weights(self):
        """恢复 LoRA 的正确初始化。

        GPT.__init__ 中的 self.apply(self._init_weights) 会将
        lora_A 初始化为 normal(0,0.02)，lora_B 初始化为 normal(0,0.02)。
        正确的初始化是: lora_A = kaiming_uniform, lora_B = zeros。
        """
        import math

        for module in self.modules():
            if isinstance(module, LoRALinear):
                # lora_A: kaiming uniform
                nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
                # lora_B: zeros（关键！确保初始 LoRA 增量为 0）
                nn.init.zeros_(module.lora_B.weight)

    def freeze_base_weights(self, train_wpe: bool = False, train_ln: bool = False):
        """冻结所有非 LoRA 参数，只保留 LoRA 矩阵可训练。

        LoRALinear 的 base weight 以 buffer 存储，天然不参与训练。
        此方法额外冻结: wte, wpe, ln_f, lm_head, LayerNorm 等。

        参数:
            train_wpe: True = 位置嵌入也参与训练（算术任务需要位置感知）
            train_ln:  True = LayerNorm 也参与训练（适合分布变化大的任务）
        """
        for name, param in self.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            elif train_wpe and 'wpe' in name:
                param.requires_grad = True
            elif train_ln and ('ln_' in name or 'ln_f' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        """解冻所有参数（用于全量微调对比实验）。"""
        for param in self.parameters():
            param.requires_grad = True
