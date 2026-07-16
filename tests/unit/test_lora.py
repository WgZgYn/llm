"""LoRA 单元测试。

测试内容:
1. LoRALinear 前向传播正确性
2. merge / unmerge 无损往返
3. LoRAGPT 创建 + 前向传播
4. from_pretrained 权重迁移
5. freeze_base_weights 正确性
6. save/load LoRA adapters
"""

import os
import sys
import tempfile
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from llm.config import GPTConfig
from llm.model.gpt import GPT
from llm.lora.config import LoRAConfig
from llm.lora.linear import LoRALinear
from llm.lora.attention import LoRAAttention
from llm.lora.mlp import LoRAMLP
from llm.lora.block import LoRABlock
from llm.lora.gpt import LoRAGPT
from llm.lora.utils import (
    lora_state_dict,
    save_lora_adapters,
    load_lora_adapters,
    merge_all_lora,
    unmerge_all_lora,
    count_lora_params,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def small_config():
    """创建轻量 GPT 配置用于快速测试。"""
    return GPTConfig(
        vocab_size=65,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=True,
    )


@pytest.fixture
def lora_config():
    """默认 LoRA 配置。"""
    return LoRAConfig(r=4, alpha=8.0, dropout=0.0, target_modules='attn')


# ═══════════════════════════════════════════════════════════════════
# LoRALinear 测试
# ═══════════════════════════════════════════════════════════════════

class TestLoRALinear:
    """测试 LoRALinear 核心功能。"""

    def test_forward_shape(self):
        """forward 输出 shape 应与普通 Linear 一致。"""
        lora = LoRALinear(64, 128, r=4, alpha=8.0)
        x = torch.randn(2, 10, 64)
        y = lora(x)
        assert y.shape == (2, 10, 128)

    def test_forward_batch(self):
        """支持 2D 输入 [batch, features] 和 3D 输入 [batch, seq, features]。"""
        lora = LoRALinear(64, 128, r=4, alpha=8.0)

        # 2D
        x2 = torch.randn(8, 64)
        y2 = lora(x2)
        assert y2.shape == (8, 128)

        # 3D
        x3 = torch.randn(4, 16, 64)
        y3 = lora(x3)
        assert y3.shape == (4, 16, 128)

    def test_initial_zero_delta(self):
        """训练开始时（B=0），LoRA 增量为 0，不改变输出。"""
        lora = LoRALinear(64, 64, r=4, alpha=8.0)
        # 设置已知 base weight
        with torch.no_grad():
            lora.weight.copy_(torch.eye(64))

        x = torch.randn(4, 64)
        y = lora(x)

        # Wx + 0 = Wx
        expected = x @ torch.eye(64)
        assert torch.allclose(y, expected, atol=1e-6)

    def test_merge_unmerge_roundtrip(self):
        """merge → unmerge 后，weight 完全恢复。"""
        lora = LoRALinear(64, 64, r=4, alpha=8.0)
        w_orig = lora.weight.data.clone()

        # 手动设置 non-zero LoRA 权重
        nn.init.normal_(lora.lora_A.weight, std=0.1)
        nn.init.normal_(lora.lora_B.weight, std=0.1)

        lora.merge()
        assert lora.merged
        assert not torch.allclose(lora.weight, w_orig, atol=1e-6)

        lora.unmerge()
        assert not lora.merged
        assert torch.allclose(lora.weight, w_orig, atol=1e-6)

    def test_merged_forward_equals_unmerged(self):
        """merge 后的 forward 应该等于 merge 前的 forward。"""
        lora = LoRALinear(64, 64, r=4, alpha=8.0)

        # 设置固定权重
        with torch.no_grad():
            lora.weight.normal_(std=0.1)
            lora.lora_A.weight.normal_(std=0.01)
            lora.lora_B.weight.normal_(std=0.01)

        x = torch.randn(16, 64)

        # 未融合 forward
        y_unmerged = lora(x)

        # 融合后 forward
        lora.merge()
        y_merged = lora(x)

        assert torch.allclose(y_unmerged, y_merged, atol=1e-5)

    def test_merge_idempotent(self):
        """多次 merge 应该是幂等的。"""
        lora = LoRALinear(64, 64, r=4, alpha=8.0)
        nn.init.normal_(lora.lora_A.weight, std=0.1)
        nn.init.normal_(lora.lora_B.weight, std=0.1)

        lora.merge()
        w1 = lora.weight.data.clone()
        lora.merge()  # 第二次 merge 应该是 no-op
        w2 = lora.weight.data.clone()

        assert torch.allclose(w1, w2, atol=1e-6)

    def test_trainable_params(self):
        """只有 lora_A 和 lora_B 的 weight 是 Parameter（requires_grad=True）。
        base weight 是 buffer（不在 parameters() 中）。
        """
        lora = LoRALinear(64, 128, r=4, alpha=8.0)

        param_names = {n for n, _ in lora.named_parameters()}
        assert 'lora_A.weight' in param_names
        assert 'lora_B.weight' in param_names
        assert 'weight' not in param_names  # weight 是 buffer！

    def test_init_from_pretrained(self):
        """init_from_pretrained 正确复制权重。"""
        original = nn.Linear(64, 128, bias=True)
        nn.init.normal_(original.weight, std=0.1)

        lora = LoRALinear(64, 128, r=4, alpha=8.0, bias=True)
        lora.init_from_pretrained(original.weight.data)
        # 同时复制 bias
        with torch.no_grad():
            lora.bias.data.copy_(original.bias.data)

        assert torch.allclose(lora.weight, original.weight.data, atol=1e-6)
        assert torch.allclose(lora.bias, original.bias.data, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
# LoRAAttention 测试
# ═══════════════════════════════════════════════════════════════════

class TestLoRAAttention:
    """测试 LoRAAttention 功能。"""

    def test_creation(self, lora_config):
        """创建 LoRAAttention 并验证模块类型。"""
        attn = LoRAAttention(
            n_embd=64, n_head=4, block_size=64,
            bias=True, dropout=0.0,
            lora_config=lora_config,
        )
        assert isinstance(attn.c_attn, LoRALinear)
        assert isinstance(attn.c_proj, LoRALinear)

    def test_forward_shape(self, lora_config):
        """前向传播 shape 正确。"""
        attn = LoRAAttention(
            n_embd=64, n_head=4, block_size=64,
            bias=True, dropout=0.0,
            lora_config=lora_config,
        )
        x = torch.randn(2, 32, 64)
        y = attn(x)
        assert y.shape == (2, 32, 64)


# ═══════════════════════════════════════════════════════════════════
# LoRAMLP 测试
# ═══════════════════════════════════════════════════════════════════

class TestLoRAMLP:
    """测试 LoRAMLP 功能。"""

    def test_creation_target_mlp(self):
        """target_modules='mlp' 时 MLP 使用 LoRA。"""
        lora_cfg = LoRAConfig(r=4, alpha=8.0, target_modules='mlp')
        mlp = LoRAMLP(64, bias=True, dropout=0.0, lora_config=lora_cfg)
        assert isinstance(mlp.net[0], LoRALinear)
        assert isinstance(mlp.net[2], LoRALinear)

    def test_creation_target_attn(self):
        """target_modules='attn' 时 MLP 不应用 LoRA。"""
        lora_cfg = LoRAConfig(r=4, alpha=8.0, target_modules='attn')
        mlp = LoRAMLP(64, bias=True, dropout=0.0, lora_config=lora_cfg)
        assert isinstance(mlp.net[0], nn.Linear)  # 保持普通 Linear
        assert isinstance(mlp.net[2], nn.Linear)

    def test_forward_shape(self):
        """前向传播 shape 正确。"""
        lora_cfg = LoRAConfig(r=4, alpha=8.0, target_modules='mlp')
        mlp = LoRAMLP(64, bias=True, dropout=0.0, lora_config=lora_cfg)
        x = torch.randn(2, 32, 64)
        y = mlp(x)
        assert y.shape == (2, 32, 64)


# ═══════════════════════════════════════════════════════════════════
# LoRABlock 测试
# ═══════════════════════════════════════════════════════════════════

class TestLoRABlock:
    """测试 LoRABlock 功能。"""

    def test_creation(self, lora_config):
        """创建 LoRABlock 验证子模块类型。"""
        block = LoRABlock(
            n_embd=64, n_head=4, block_size=64,
            bias=True, dropout=0.0,
            lora_config=lora_config,
        )
        assert isinstance(block.attn, LoRAAttention)
        assert isinstance(block.mlp, nn.Module)

    def test_forward_shape(self, lora_config):
        """前向传播 shape 正确。"""
        block = LoRABlock(
            n_embd=64, n_head=4, block_size=64,
            bias=True, dropout=0.0,
            lora_config=lora_config,
        )
        x = torch.randn(2, 32, 64)
        y = block(x)
        assert y.shape == (2, 32, 64)

    def test_kv_cache_support(self, lora_config):
        """KV-Cache 功能正常。"""
        block = LoRABlock(
            n_embd=64, n_head=4, block_size=64,
            bias=True, dropout=0.0,
            lora_config=lora_config,
        )

        x1 = torch.randn(1, 1, 64)
        cache = {}
        y1 = block(x1, cache)
        assert 'k' in cache
        assert y1.shape == (1, 1, 64)

        # 第二个 token（使用 cache）
        x2 = torch.randn(1, 1, 64)
        y2 = block(x2, cache)
        assert y2.shape == (1, 1, 64)
        assert cache['k'].shape[2] == 2  # 已缓存 2 个 token


# ═══════════════════════════════════════════════════════════════════
# LoRAGPT 测试
# ═══════════════════════════════════════════════════════════════════

class TestLoRAGPT:
    """测试 LoRAGPT 完整功能。"""

    def test_creation(self, small_config, lora_config):
        """创建 LoRAGPT 并验证 Block 类型。"""
        model = LoRAGPT(small_config, lora_config)
        assert len(model.transformer.h) == 2
        for block in model.transformer.h:
            assert isinstance(block, LoRABlock)

    def test_target_layers_range(self, small_config):
        """测试 target_layers 范围过滤。"""
        lora_cfg = LoRAConfig(r=4, alpha=8.0, target_layers=(1, 2))
        model = LoRAGPT(small_config, lora_cfg)
        assert isinstance(model.transformer.h[0], nn.Module)  # Block (plain)
        assert isinstance(model.transformer.h[1], LoRABlock)

    def test_target_layers_list(self, small_config):
        """测试 target_layers 列表过滤。"""
        lora_cfg = LoRAConfig(r=4, alpha=8.0, target_layers=[0])
        model = LoRAGPT(small_config, lora_cfg)
        assert isinstance(model.transformer.h[0], LoRABlock)
        assert not isinstance(model.transformer.h[1], LoRABlock)

    def test_forward(self, small_config, lora_config):
        """前向传播返回正确的 shape 和 loss。"""
        model = LoRAGPT(small_config, lora_config)
        x = torch.randint(0, 65, (2, 32))
        y = torch.randint(0, 65, (2, 32))
        logits, loss = model(x, y)
        assert logits.shape == (2, 32, 65)
        assert loss is not None
        assert loss.item() > 0

    def test_generate(self, small_config, lora_config):
        """generate 方法正常工作。"""
        model = LoRAGPT(small_config, lora_config)
        x = torch.randint(0, 65, (1, 4))
        out = model.generate(x, max_new_tokens=8, temperature=1.0)
        assert out.shape[1] == 12  # 4 prompt + 8 new

    def test_freeze_base_weights(self, small_config, lora_config):
        """freeze_base_weights 后只有 LoRA 参数可训练。"""
        model = LoRAGPT(small_config, lora_config)
        model.freeze_base_weights()

        for name, param in model.named_parameters():
            if 'lora_' in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_count_params_after_freeze(self, small_config, lora_config):
        """冻结后 trainable_params ≈ lora_params。"""
        model = LoRAGPT(small_config, lora_config)
        model.freeze_base_weights()
        info = count_lora_params(model)
        assert info['trainable_params'] == info['lora_params']


# ═══════════════════════════════════════════════════════════════════
# State Dict & 持久化测试
# ═══════════════════════════════════════════════════════════════════

class TestLoRAPersistence:
    """测试 LoRA adapter 的保存/加载。"""

    def test_lora_state_dict(self, small_config, lora_config):
        """lora_state_dict 只包含 LoRA 参数。"""
        model = LoRAGPT(small_config, lora_config)
        sd = lora_state_dict(model)
        assert all('lora_' in k for k in sd.keys())
        assert len(sd) > 0

    def test_save_load_roundtrip(self, small_config, lora_config):
        """保存后加载，LoRA 参数应完全一致。"""
        model = LoRAGPT(small_config, lora_config)
        model.freeze_base_weights()

        # 手动设置一些非零 LoRA 权重
        for module in model.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.lora_A.weight, std=0.1)
                nn.init.normal_(module.lora_B.weight, std=0.1)

        lora_before = lora_state_dict(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'lora_adapters.pt')

            # 保存
            save_lora_adapters(model, path, lora_config)

            # 创建新模型并加载
            model2 = LoRAGPT(small_config, lora_config)
            loaded_cfg = load_lora_adapters(model2, path)

            assert loaded_cfg is not None
            lora_after = lora_state_dict(model2)

            assert lora_before.keys() == lora_after.keys()
            for k in lora_before:
                assert torch.allclose(lora_before[k], lora_after[k], atol=1e-6), \
                    f"Mismatch in {k}"

    def test_merge_all(self, small_config, lora_config):
        """merge_all_lora 后所有 LoRALinear 应该已融合。"""
        model = LoRAGPT(small_config, lora_config)
        count = merge_all_lora(model)
        assert count > 0

        for module in model.modules():
            if isinstance(module, LoRALinear):
                assert module.merged

    def test_full_merge_unmerge_roundtrip(self, small_config, lora_config):
        """merge → forward → unmerge → forward 输出一致。"""
        model = LoRAGPT(small_config, lora_config)

        # 设置一些随机 LoRA 权重
        for module in model.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.lora_A.weight, std=0.01)
                nn.init.normal_(module.lora_B.weight, std=0.01)

        x = torch.randint(0, 65, (2, 16))

        with torch.no_grad():
            logits_before, _ = model(x)

        merge_all_lora(model)
        with torch.no_grad():
            logits_merged, _ = model(x)

        unmerge_all_lora(model)
        with torch.no_grad():
            logits_after, _ = model(x)

        # 融合前后输出一致
        assert torch.allclose(logits_before, logits_merged, atol=1e-4), \
            "Merged forward differs from unmerged!"
        # Unmerge 后恢复一致
        assert torch.allclose(logits_before, logits_after, atol=1e-6), \
            "Unmerge didn't fully restore!"


# ═══════════════════════════════════════════════════════════════════
# LoRAConfig 测试
# ═══════════════════════════════════════════════════════════════════

class TestLoRAConfig:
    """测试 LoRAConfig 基本功能。"""

    def test_default(self):
        cfg = LoRAConfig()
        assert cfg.r == 8
        assert cfg.alpha == 16.0
        assert cfg.scale == 2.0

    def test_scale(self):
        cfg = LoRAConfig(r=4, alpha=8.0)
        assert cfg.scale == 2.0

    def test_to_from_dict(self):
        cfg = LoRAConfig(r=16, alpha=32.0, target_modules='all')
        d = cfg.to_dict()
        cfg2 = LoRAConfig.from_dict(d)
        assert cfg2.r == 16
        assert cfg2.alpha == 32.0
        assert cfg2.target_modules == 'all'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
