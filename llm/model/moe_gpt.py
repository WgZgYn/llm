"""MoEGPT —— 使用 MoE FFN 的 GPT 变种。

通过重写 GPT._make_block() 钩子，将指定层的 MLP 替换为 MoE_FFN。
其余全部功能（forward / generate / from_pretrained / estimate_mfu）直接继承。

用法:
    from llm.model.moe_gpt import MoEGPT

    config = GPTConfig(vocab_size=50304, block_size=1024, n_layer=12, ...)
    model = MoEGPT(config, num_experts=8, top_k=2, moe_layers=(3, 9))
"""

from ..config.model_config import GPTConfig
from ..model.gpt import GPT
from ..model.layer import Block
from ..model.moe import MoE_FFN


class MoEGPT(GPT):
    """GPT with Mixture of Experts FFN。

    不重写 __init__，不重写 forward。只重写 _make_block() 钩子。

    参数:
        config:        GPTConfig（与普通 GPT 完全一样）
        num_experts:   expert 总数（默认 8）
        top_k:         每个 token 激活的 expert 数（默认 2）
        moe_layers:    (start, end) 用 MoE 的层范围，None=全部层
                       e.g. (3, 9) → 第 3~8 层 MoE，0~2 和 9~11 保持 MLP
        router_jitter: 训练时 router 输入噪声（0=不加）
    """

    def __init__(
        self,
        config: GPTConfig,
        num_experts: int = 8,
        top_k: int = 2,
        moe_layers: tuple[int, int] | None = None,
        balance_coef: float = 0.01,
        z_loss_coef: float = 0.001,
    ):
        self._num_experts = num_experts
        self._top_k = top_k
        self._moe_start, self._moe_end = moe_layers or (0, config.n_layer)
        self._balance_coef = balance_coef
        self._z_loss_coef = z_loss_coef
        super().__init__(config)

    def _make_block(self, config: GPTConfig, layer_idx: int):
        """GPT 构造每个 Block 时调用此钩子。MoEGPT 在指定层注入 MoE。"""
        if self._moe_start <= layer_idx < self._moe_end:
            ffn = MoE_FFN(
                n_embd=config.n_embd,
                num_experts=self._num_experts,
                top_k=self._top_k,
                bias=config.bias,
                dropout=config.dropout,
                balance_coef=self._balance_coef,
                z_loss_coef=self._z_loss_coef,
            )
        else:
            ffn = None
        return Block(config.n_embd, config.n_head, config.block_size,
                     config.bias, config.dropout, ffn=ffn)
