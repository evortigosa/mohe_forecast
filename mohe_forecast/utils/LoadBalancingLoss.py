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
        return f"n_experts={self.n_experts}, top_k={self.top_k}, alpha={self.alpha}"


    @staticmethod
    def flatten_router_probs(router_probs):
        """
        Concatenate all MoE layer router probabilities into shape [n_moe_layers * B * T, E].
        """
        if router_probs is None or not isinstance(router_probs, (tuple, list)):
            return None

        valid= [x for x in router_probs if x is not None]
        if len(valid) == 0:
            return None

        flattened= []
        for x in valid:
            if x.ndim == 2:         # [N, E]
                flattened.append(x)
            elif x.ndim == 3:       # [B, T, E]
                flattened.append(x.reshape(-1, x.size(-1)))
            else:
                raise ValueError(f"Expected probs with shape [N,E] or [B,T,E], got {tuple(x.shape)}")

        return torch.cat(flattened, dim=0)  # [n_moe_layers * B * T, E]


    @staticmethod
    def compute_metrics(expert_mask, routing_probs, device):
        """
        Compute detached monitoring metrics from router probabilities.
        TODO: get metrics when padding_mask is not None
        """
        num_tokens= routing_probs.size(0)
        # hard load: fraction of selected slots assigned to each expert
        token_counts= expert_mask.sum(dim=(0, 1))  # [E]
        hard_fraction= token_counts / token_counts.sum().clamp_min(1.0)  # [E]
        # soft importance: average routing probability mass
        prob_mass= routing_probs.sum(dim=0)  # [E]
        soft_fraction= prob_mass / prob_mass.sum().clamp_min(1e-12)  # [E]
        # mean token entropy of router distribution
        entropy= -(routing_probs * torch.log(routing_probs.clamp_min(1e-12))).sum(dim=-1).mean()
        # collapse indicators
        dead_experts= (token_counts == 0).sum()
        # coefficient of variation: std / mean
        cv_hard= hard_fraction.std(unbiased=False) / hard_fraction.mean().clamp_min(1e-12)
        cv_soft= soft_fraction.std(unbiased=False) / soft_fraction.mean().clamp_min(1e-12)

        return {
            "hard_fraction": hard_fraction.detach(),
            "soft_fraction": soft_fraction.detach(),
            "token_counts": token_counts.detach(),
            "prob_mass": prob_mass.detach(),
            "entropy": entropy.detach(),
            "dead_experts": dead_experts.detach(),
            "cv_hard": cv_hard.detach(),
            "cv_soft": cv_soft.detach(),
            "num_tokens": torch.tensor(num_tokens, device=device),
        }


    def compute_layerwise_metrics(self, router_probs, device):
        layer_metrics= {}
        for layer_id, probs in enumerate(router_probs):
            if probs is None:
                continue

            if probs.ndim == 3:
                probs= probs.reshape(-1, probs.size(-1))  # [N, E]
            elif probs.ndim != 2:
                raise ValueError(f"Expected probs with shape [B,T,E] or [N,E], got {tuple(probs.shape)}")

            _, selected_experts= torch.topk(probs, self.top_k, dim=-1)
            expert_mask= F.one_hot(selected_experts, num_classes=self.n_experts).float()
            layer_metrics[layer_id]= self.compute_metrics(expert_mask, probs, device)

        return layer_metrics


    def __call__(self, router_probs, padding_mask=None, return_metrics=False):
        """
        router_probs: a list of MoE layer router probabilities (softmax router logits) over all experts.
        - padding_mask: a tensor of shape [batch_size, seq_len] normally passed into the model to
        indicate which token positions are "real" versus which are just padding (data-validity mask).
        """
        concatenated_router_probs= self.flatten_router_probs(router_probs)  # [N, E]
        if concatenated_router_probs is None:
            return torch.zeros([]), None, None

        device= concatenated_router_probs.device
        _, selected_experts= torch.topk(concatenated_router_probs, self.top_k, dim=-1)  # [N, K]
        expert_mask= F.one_hot(selected_experts, num_classes=self.n_experts).float()  # [N, K, E]

        if padding_mask is None:
            # compute the percentage of tokens routed to each expert (hard load)
            tokens_per_expert= expert_mask.sum(dim=(0, 1))
            tokens_per_expert= tokens_per_expert / tokens_per_expert.sum().clamp_min(1.0)
            # compute the average probability of routing to these experts (soft importance)
            router_prob_per_expert= concatenated_router_probs.mean(dim=0)
        else:
            # checking elementwise that each entry is 0 or 1
            is_binary= torch.logical_or(padding_mask == 0, padding_mask == 1)
            # ensure every position is True
            assert is_binary.all(), "padding_mask must contain only 0s and 1s"
            batch_size, seq_len= padding_mask.shape
            assert concatenated_router_probs.shape[0] % (batch_size * seq_len) == 0, \
                "Cannot infer n_moe_layers: concatenated router probabilities size is not divisible by B*T"

            # recover how many MoE layers contributed router probs -> (n_moe_layers * B * T) // (B * T)
            n_moe_layers= concatenated_router_probs.shape[0] // (batch_size * seq_len)
            # align the padding mask with the expert_mask, which has shape [N, K, E]
            # compute the percentage of tokens routed to each expert
            token_mask= (
                padding_mask[None, :, :]
                .expand(n_moe_layers, batch_size, seq_len)
                .reshape(-1).to(device).float()
            )  # [N]
            slot_mask= token_mask[:, None, None]  # [N,1,1]
            token_counts= (expert_mask * slot_mask).sum(dim=(0, 1))   # [E]
            tokens_per_expert= token_counts / token_counts.sum().clamp_min(1.0)

            # compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_padding_mask= (
                padding_mask[None, :, :, None]  # [B, T] -> [1, B, T, 1]
                .expand((n_moe_layers, batch_size, seq_len, self.n_experts))
                .reshape(-1, self.n_experts).to(device)
            )
            # router_per_expert_padding_mask shape [n_moe_layers * B * T, n_experts]

            # compute the average probability of routing to these experts
            router_prob_per_expert= torch.sum(
                concatenated_router_probs * router_per_expert_padding_mask, dim=0
            ) / torch.sum(router_per_expert_padding_mask, dim=0)

        overall_loss= torch.sum(tokens_per_expert * router_prob_per_expert)
        loss= self.alpha * self.n_experts * overall_loss

        if not return_metrics:
            return loss, None, None

        global_metrics= self.compute_metrics(expert_mask, concatenated_router_probs, device)
        layer_metrics = self.compute_layerwise_metrics(router_probs, device)

        return loss, global_metrics, layer_metrics
