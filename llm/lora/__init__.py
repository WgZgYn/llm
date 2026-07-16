"""llm.lora —— LoRA (Low-Rank Adaptation) 模块。

快速开始:
    from llm.lora import LoRAConfig, LoRAGPT, merge_all_lora
    from llm.config import GPTConfig

    config = GPTConfig.from_preset("gpt2")
    lora_config = LoRAConfig(r=8, alpha=16.0)
    model = LoRAGPT.from_pretrained("gpt2", lora_config)

    # 训练...
    # ...

    # 推理时融合
    merge_all_lora(model)
    output = model.generate(...)
"""

from .config import LoRAConfig
from .linear import LoRALinear
from .attention import LoRAAttention
from .mlp import LoRAMLP
from .block import LoRABlock
from .gpt import LoRAGPT
from .utils import (
    lora_state_dict,
    save_lora_adapters,
    load_lora_adapters,
    merge_all_lora,
    unmerge_all_lora,
    count_lora_params,
    print_lora_info,
)

__all__ = [
    # 核心
    "LoRAConfig",
    "LoRALinear",
    # 子类层级
    "LoRAAttention",
    "LoRAMLP",
    "LoRABlock",
    "LoRAGPT",
    # 工具
    "lora_state_dict",
    "save_lora_adapters",
    "load_lora_adapters",
    "merge_all_lora",
    "unmerge_all_lora",
    "count_lora_params",
    "print_lora_info",
]
