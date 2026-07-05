"""Mixture of Experts FFN —— 稀疏激活的前馈网络。

参考: Switch Transformer (Fedus et al. 2021), ST-MoE (Zoph et al. 2022), Mixtral (Jiang et al. 2024)

核心思想:
    标准 FFN: 每个 token 都走同一个 MLP（参数量 C，计算量 C）
    MoE FFN:  N 个 expert（各是一个小 MLP），每个 token 只走 top-k 个
            → 参数量 N×C，计算量 ≈ k×C（k << N）

内置 Load Balancing Loss + Router Z-Loss，防止 expert 坍塌。

用法:
    from llm.model.moe import MoE_FFN

    ffn = MoE_FFN(n_embd=768, num_experts=8, top_k=2)
    out = ffn(x)                          # 前向传播
    aux = ffn.pop_aux_loss()              # 取辅助 loss（每次 forward 后调用）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE_FFN(nn.Module):
    """Mixture of Experts FFN —— 替换标准 MLP 的 drop-in 组件。

    每个 expert: Linear(n_embd → 4*n_embd) → GELU → Linear(4*n_embd → n_embd)
    Router:      Linear(n_embd → num_experts) → softmax → top-k

    参数:
        n_embd:         隐藏维度
        num_experts:    expert 总数（建议 8）
        top_k:          每个 token 激活的 expert 数（建议 2）
        bias:           Linear 中是否使用 bias
        dropout:        expert 内部 dropout
        balance_coef:   Load Balancing Loss 系数（0=禁用，建议 0.01）
        z_loss_coef:    Router Z-Loss 系数（0=禁用，建议 0.001）
    """

    def __init__(
            self,
            n_embd: int,
            num_experts: int = 8,
            top_k: int = 2,
            bias: bool = True,
            dropout: float = 0.0,
            balance_coef: float = 0.01,
            z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_coef = balance_coef
        self.z_loss_coef = z_loss_coef

        # ── N 个 expert，每个就是标准 MLP ──
        self.experts = nn.ModuleList([
            MLP(n_embd, bias=bias, dropout=dropout, expand=4)
            for _ in range(num_experts)
        ])

        # ── Router ──
        self.router = nn.Linear(n_embd, num_experts, bias=False)

        # ── 辅助 loss（forward 后累积在此，pop 取走并清零）──
        self._aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] 或 [N, C]

        返回: 与输入同 shape 的输出
        辅助 loss 存于 self._aux_loss，调用方用 pop_aux_loss() 取走
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, C = x.shape
            x_flat = x.view(B * T, C)
        else:
            x_flat = x

        N_tokens, _ = x_flat.shape
        E = self.num_experts
        k = self.top_k

        # ── Router logits + probs ──
        router_logits = self.router(x_flat)  # [N, E]
        router_probs = F.softmax(router_logits, dim=-1)  # [N, E]

        # ── top-k ──
        top_weights, top_indices = torch.topk(router_probs, k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # ── 辅助 loss ──
        if self.training:
            self._aux_loss = self._compute_aux_loss(
                router_logits, router_probs, top_indices)

        # ── 稀疏计算 ──
        out = torch.zeros_like(x_flat)
        for expert_idx in range(E):
            mask = (top_indices == expert_idx).any(dim=-1)  # [N]
            token_ids = mask.nonzero(as_tuple=True)[0]
            if token_ids.numel() == 0:
                continue

            tids, kpos = (top_indices == expert_idx).nonzero(as_tuple=True)
            weights = top_weights[tids, kpos]

            expert_out = self.experts[expert_idx](x_flat[token_ids])
            out[token_ids] += weights.unsqueeze(-1) * expert_out

        if x.dim() == 3:
            out = out.view(B, T, C)
        return out

    def pop_aux_loss(self) -> torch.Tensor:
        """取走累积的辅助 loss 并清零。每次 forward 后调用。"""
        v = self._aux_loss
        self._aux_loss = 0.0
        if isinstance(v, torch.Tensor):
            return v
        return torch.tensor(0.0, device=self.router.weight.device)

    # ── 辅助 loss 实现 ──

    def _compute_aux_loss(self, logits, probs, top_indices):
        """Load Balancing Loss + Router Z-Loss。"""
        loss = torch.tensor(0.0, device=logits.device)

        # ① Load Balancing Loss (Switch Transformer)
        #    f_i = fraction of tokens where expert i is in top-k
        #    P_i = mean router probability for expert i
        #    loss = E × Σ(f_i × P_i)
        if self.balance_coef > 0:
            # Count how many tokens go to each expert
            expert_counts = torch.zeros(self.num_experts, device=logits.device)
            expert_counts.scatter_add_(0, top_indices.view(-1),
                                       torch.ones_like(top_indices.view(-1), dtype=torch.float))
            f = expert_counts / (top_indices.numel() / self.top_k)  # [E], normalized
            P = probs.mean(dim=0)  # [E]
            balance_loss = self.num_experts * (f * P).sum()
            loss = loss + self.balance_coef * balance_loss

        # ② Router Z-Loss (ST-MoE)
        #    penalizes large router logits to stabilize training
        if self.z_loss_coef > 0:
            z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
            loss = loss + self.z_loss_coef * z_loss

        return loss


from .layer import MLP  # 复用标准 FFN
