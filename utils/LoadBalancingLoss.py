# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Expert Load Balancing Loss
"""

import torch
import torch.nn.functional as F



class LoadBalancingLoss:
    """
    The sparse gating can result in a load balancing issue known as routing collapse, where the
    model predominantly selects only a few experts, limiting training opportunities for others.
    The LoadBalancingLoss is an auxiliary loss designed to mitigate this issue by encouraging an
    even distribution of tokens across experts.
    See https://arxiv.org/abs/2101.03961
    """

    def __init__(self, n_experts, top_k, alpha=0.02) -> None:
        assert int(n_experts) >= 0, "n_experts must be non-negative"
        top_k= int(top_k) if int(n_experts) > 0 else 0
        assert top_k <= int(n_experts), "top_k must be less than or equal to n_experts"
        self.n_experts= int(n_experts)
        self.top_k= top_k
        self.alpha= alpha


    def extra_repr(self):
        return f"alpha={self.alpha}"


    def __call__(self, gate_logits, padding_mask=None):
        """
        - padding_mask: a tensor of shape [batch_size, seq_len] normally passed into the model to
        indicate which token positions are "real" versus which are just padding (data-validity mask).
        """
        if gate_logits is None or not isinstance(gate_logits, (tuple, list)) or gate_logits[0] is None:
            return torch.zeros([])

        device= gate_logits[0].device
        concatenated_gate_logits= torch.cat(
            [layer_gate.to(device) for layer_gate in gate_logits], dim=0
        )  # shape [n_moe_layers * B * T, n_experts]

        routing_weights= F.softmax(concatenated_gate_logits, dim=-1)
        _, selected_experts= torch.topk(routing_weights, self.top_k, dim=-1)
        expert_mask= F.one_hot(selected_experts, num_classes=self.n_experts)

        if padding_mask is None:
            # Compute the percentage of tokens routed to each expert
            tokens_per_expert= torch.mean(expert_mask.float(), dim=0)
            # Compute the average probability of routing to these experts
            router_prob_per_expert= torch.mean(routing_weights, dim=0)
        else:
            # checking elementwise that each entry is 0 or 1
            is_binary= torch.logical_or(padding_mask == 0, padding_mask == 1)
            # ensure every position is True
            assert is_binary.all(), "padding_mask must contain only 0s and 1s"

            batch_size, seq_len= padding_mask.shape
            # recover how many MoE layers contributed gating logits -> (n_moe_layers * B * T) // (B * T)
            n_moe_layers= concatenated_gate_logits.shape[0] // (batch_size * seq_len)
            # Align the padding mask with the expert_mask, which has shape [n_experts, K, B * T]
            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_padding_mask= (
                padding_mask[None, :, :, None, None]  # [B, T] -> [1, B, T, 1, 1]
                .expand((n_moe_layers, batch_size, seq_len, 2, self.n_experts))
                .reshape(-1, 2, self.n_experts)
                .to(device)
            )
            # expert_attention_mask ends up with shape [n_moe_layers * B * T, K, n_experts]

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert= torch.sum(expert_mask.float() * expert_padding_mask, dim=0) / torch.sum(
                expert_padding_mask, dim=0
            )
            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_padding_mask= (
                padding_mask[None, :, :, None]  # [B, T] -> [1, B, T, 1]
                .expand((n_moe_layers, batch_size, seq_len, self.n_experts))
                .reshape(-1, self.n_experts)
                .to(device)
            )
            # router_per_expert_padding_mask shape [n_moe_layers * B * T, n_experts]

            # Compute the average probability of routing to these experts
            router_prob_per_expert= torch.sum(routing_weights * router_per_expert_padding_mask, dim=0) / torch.sum(
                router_per_expert_padding_mask, dim=0
            )

        overall_loss= torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))

        return self.alpha * self.n_experts * overall_loss
