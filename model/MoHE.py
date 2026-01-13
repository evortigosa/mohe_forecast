# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
MoHE Modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



"""
Feed Forward Networks
"""


class FeedForward(nn.Module):
    """
    The Feed Forward Network (FFN) with (optional) Gated Linear Unit (GLU) architecture.
    The use of a gated mechanism enhances the expressivity of the FFN by introducing a gating.
    This is more flexible than traditional MLP layers and is proven effective in many Transformer
    variants like Llama 3, GPT-NeoX, or PaLM.
    - This module can switch between a GLU-based FFN and a standard FFN based on the glu flag.
    """

    def __init__(self, d_model, d_ff, n_outputs=None, dropout=0.2, glu=False, bias=False) -> None:
        super(FeedForward, self).__init__()
        # First linear projection (always used)
        self.up_proj= nn.Linear(d_model, d_ff, bias=bias)
        # Gated Linear Unit (GLU) activation when glu=True
        if glu:
            self.gate_proj= nn.Linear(d_model, d_ff, bias=bias)
            self.actv_fn= nn.SiLU()
        else:
            # Alternative: no gating
            self.gate_proj= None
            self.actv_fn= nn.GELU()
        # Dropout layer (applied after gating or activation)
        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None
        # Final down projection
        ffn_out= d_model if n_outputs is None else n_outputs
        self.down_proj= nn.Linear(d_ff, ffn_out, bias=bias)

        # initialize Linear modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


    def forward(self, x):
        # apply GLU activation when glu=True
        if self.gate_proj is not None:
            # elementwise multiply the gate and the features
            x= self.actv_fn(self.up_proj(x)) * self.gate_proj(x)
        else:
            x= self.actv_fn(self.up_proj(x))
        if self.dropout is not None:
            x= self.dropout(x)
        x= self.down_proj(x)

        return x



class ConvFeedForward(nn.Module):
    """
    Feed Forward Network (FFN) based on convolutions, with optional SwiGLU‐style gating.
    This module first applies a regular (up-)convolution. If GLU is enabled, an additional conv
    computes a gating mechanism, and the activation is applied only on the up‐conv branch before
    performing an element-wise multiplication with the expanded gating features.
    See https://arxiv.org/abs/1612.08083 and https://arxiv.org/abs/2104.00298
    - This module can switch between a SwiGLU ConvFFN and a ConvFFN based on the glu flag.
    """

    def __init__(self, d_model, d_ff, n_outputs=None, dropout=0.2, glu=False, bias=False) -> None:
        super(ConvFeedForward, self).__init__()
        # First projection (always used)
        self.up_conv= nn.Conv1d(d_model, d_ff, kernel_size=1, stride=1, padding=0, bias=bias)
        # Gated Linear Unit (GLU) activation when glu=True
        if glu:
            self.gate_conv= nn.Conv1d(d_model, d_ff, kernel_size=1, stride=1, padding=0, bias=bias)
            self.actv_fn= nn.SiLU()
        else:
            # Alternative: no gating
            self.gate_conv= None
            self.actv_fn= nn.GELU()
        # Dropout layer (applied after gating or activation)
        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None
        # Final down projection
        ffn_out= d_model if n_outputs is None else n_outputs
        self.down_conv= nn.Conv1d(d_ff, ffn_out, kernel_size=1, stride=1, padding=0, bias=bias)

        # initialize Conv modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


    def forward(self, x):
        x= x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        # apply GLU activation when glu=True
        if self.gate_conv is not None:
            # elementwise multiply the gate and the features
            x= self.actv_fn(self.up_conv(x)) * self.gate_conv(x)
        else:
            x= self.actv_fn(self.up_conv(x))
        if self.dropout is not None:
            x= self.dropout(x)
        x= self.down_conv(x)

        return x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)



class DwConvFeedForward(nn.Module):
    """
    Feed Forward Network (FFN) based on depthwise separable convolutions, with optional SwiGLU gating.
    This module first applies a depthwise convolution followed by a pointwise convolution to expand
    the feature dimension (up‐conv). If GLU is enabled, an additional pointwise conv computes a
    gating mechanism, and the SiLU activation is applied only on the pw_conv branch before performing
    an element-wise multiplication with the gating features.
    - This module can switch between a GLU-based DwConvFFN and a DwConvFFN based on the glu flag.
    """

    def __init__(self, d_model, d_ff, n_outputs=None, dropout=0.2, glu=False, bias=False) -> None:
        super(DwConvFeedForward, self).__init__()
        # Up-Conv -- Shared depthwise separable convolution (applied along the time dimension)
        self.dw_conv= nn.Conv1d(
            d_model, d_model, kernel_size=3, stride=1, padding=1, groups=d_model, bias=bias
        )
        # Up-Conv -- Pointwise convolution (expansion)
        self.pw_conv= nn.Conv1d(d_model, d_ff, kernel_size=1, stride=1, padding=0, bias=bias)
        # Gated Linear Unit (GLU) activation when glu=True
        if glu:
            # Additional pointwise convolution to compute the gate
            self.gate_conv= nn.Conv1d(d_model, d_ff, kernel_size=1, stride=1, padding=0, bias=bias)
            self.actv_fn= nn.SiLU()
        else:
            # Alternative: no gating
            self.gate_conv= None
            self.actv_fn= nn.GELU()
        # Dropout layer (applied after gating or activation)
        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None
        # Final down projection
        ffn_out= d_model if n_outputs is None else n_outputs
        self.down_conv= nn.Conv1d(d_ff, ffn_out, kernel_size=1, stride=1, padding=0, bias=bias)

        # initialize Conv modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


    def forward(self, x):
        x= x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        # gate only the expansion step, otherwise the gate is forced to learn a direct mapping
        x= self.dw_conv(x)
        # apply GLU activation when glu=True
        if self.gate_conv is not None:
            # elementwise multiply the gate and the features
            x= self.actv_fn(self.pw_conv(x)) * self.gate_conv(x)
        else:
            x= self.actv_fn(self.pw_conv(x))
        if self.dropout is not None:
            x= self.dropout(x)
        x= self.down_conv(x)

        return x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)



class FANLayer(nn.Module):
    """
    The Fourier Analysis Network (FAN) layer.
    - gate (bool) defines a learnable "scalar‐gating" parameter that controls how much to use the
    periodic vs. non-periodic components. When gate=True, the FANLayer learns a single parameter
    g in (0, 1) that interpolates between periodic (cos/sin) and non‐periodic components; when
    gate=False, the FANLayer omits that scalar gate and simply concatenates cos(p), sin(p), and a
    non‐periodic transform.
    - freq_scale (float) multiplies the std of Wp to set the bandwidth (higher -> higher freq).
    - Adapted from https://arxiv.org/abs/2410.02675
    """

    def __init__(self, input_dim, output_dim, gate=False, bias=False, freq_scale=1.0,
                 is_last=False) -> None:
        super(FANLayer, self).__init__()
        assert output_dim % 4 == 0, "output_dim must be divisible by 4"
        # p_output_dim is set to a quarter of output_dim as in the original paper
        p_output_dim    = output_dim // 4
        p_bar_output_dim= output_dim - p_output_dim * 2
        # For cosine and sine (periodic) components
        self.Wp= nn.Linear(input_dim, p_output_dim, bias=bias)
        if gate:
            self.gate= nn.Parameter(torch.zeros(1))
            self.actv_fn= nn.SiLU() if not is_last else nn.Identity()
        else:
            self.gate= None
            self.actv_fn= nn.GELU() if not is_last else nn.Identity()
        # For the non-periodic component
        self.Wp_bar= nn.Linear(input_dim, p_bar_output_dim, bias=bias)

        # normal_ presented the faster convergence for FANs
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 * float(freq_scale))
                if m.bias is not None: nn.init.zeros_(m.bias)


    def forward(self, x):
        p    = self.Wp(x)                    # periodic cos/sin‐part
        p_bar= self.actv_fn(self.Wp_bar(x))  # non‐periodic part

        if self.gate is not None:
            g= torch.sigmoid(self.gate)      # scalar in (0, 1)
            return torch.cat((g * torch.cos(p), g * torch.sin(p), (1 - g) * p_bar), dim=-1)

        return torch.cat((torch.cos(p), torch.sin(p), p_bar), dim=-1)



class FANFeedForward(nn.Module):
    """
    The Feed Forward Network (FFN) based on the FAN architecture.
    - The activation function is implemented inside the FANLayer.
    - fan_gate (bool) defines the learnable "scalar‐gating" parameter inside each FANLayer.
    - This module can switch between a "GLU‐style" FAN-FFN and a FAN-FFN on the glu flag.
    - freq_scale (float) multiplies the std of Wp to set the bandwidth (higher -> higher freq).
    - nn.init.normal_ on this FAN network improves convergence.
    """

    def __init__(self, d_model, d_ff, n_outputs=None, dropout=0.2, fan_gate=False, glu=False,
                 bias=False, freq_scale=1.0) -> None:
        super(FANFeedForward, self).__init__()
        # FAN up layer
        self.up_fan= FANLayer(d_model, d_ff, fan_gate, bias=bias, freq_scale=freq_scale)
        # Gated Linear Unit (GLU) when glu=True
        if glu:
            self.gate_proj= FANLayer(d_model, d_ff, fan_gate, bias=bias, freq_scale=freq_scale)
        else:
            self.gate_proj= None
        # Dropout layer (applied after gating or activation)
        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None
        # FAN down layer
        self.down_fan= FANLayer(d_ff, d_model, fan_gate, bias=bias, freq_scale=freq_scale)
        # Final projection when n_outputs is not None
        if (n_outputs is not None) and (n_outputs != d_model):
            self.WL= nn.Linear(d_model, n_outputs, bias=bias)
            # initialize the Linear module with Glorot / fan_avg
            nn.init.xavier_uniform_(self.WL.weight)
            if self.WL.bias is not None: nn.init.zeros_(self.WL.bias)
        else:
            self.WL= None


    def forward(self, x):
        # apply GLU activation when glu=True
        if self.gate_proj is not None:
            # elementwise multiply the gate and the features
            x= self.up_fan(x) * torch.sigmoid(self.gate_proj(x))
        else:
            x= self.up_fan(x)
        if self.dropout is not None:
            x= self.dropout(x)
        x= self.down_fan(x)
        if self.WL is not None:
            x= self.WL(x)

        return x



"""
Mixture-of-Experts (MoE)
"""


class MoEFeedForward(nn.Module):
    """
    The Sparse Mixture-of-Experts (MoE) module for heterogeneous experts (MoHE). Delegate the
    modeling of diverse time series patterns to sparse specialized experts in a data-driven manner
    through a sparce gating function (only K of N experts per token) for expert assignments.
    - When n_experts=0, forward the input into a single FFN module; MoE otherwise.
    - ffn_type (str): defines the shared_expert type from 'mlp' for MLP-FFN, 'conv' for Conv-FFN,
    'dwconv' for DwConv-FFN, or 'fan' for FAN-FFN.
    - experts_type (str): defines the routed experts from 'mlp' for MLP-FFN or 'fan' for FAN-FFN.
    See https://arxiv.org/abs/2410.10469 and https://arxiv.org/abs/2409.16040
    """

    def __init__(self, d_model, d_ff, dropout=0.2, ffn_type='dwconv', fan_gate=False, glu=False,
                 n_experts=8, top_k=2, experts_type='fan', bias=False) -> None:
        super(MoEFeedForward, self).__init__()
        assert n_experts >= 0, "n_experts must be non-negative"
        # store gating logits for auxiliary load-balancing regularizers (losses)
        self.router_logits= None

        # shared fallback expert -- ensures no token is unprocessed if its top-k experts happen
        # to be poorly trained or overflowed
        self.shared_expert= self.get_ffn(ffn_type, d_model, d_ff, dropout, fan_gate, glu, bias)

        if n_experts == 0:
            self.experts= None
            self.top_k= 0
        else:
            assert top_k > 0, "top_k must be > 0"
            self.top_k= min(top_k, n_experts)

            if isinstance(experts_type, str):
                experts_type= [experts_type for _ in range(n_experts)]
            else:
                assert all(isinstance(item, str) for item in experts_type), \
                    "experts_type must be a list of strings"
                assert len(experts_type) >= n_experts, \
                    "experts_type must be a string or a list of length n_experts"

            # controls contribution from fallback expert
            self.shared_gating= nn.Linear(d_model, 1, bias=False)

            # n_experts routed expert modules
            self.experts= nn.ModuleList([
                self.get_expert_ffn(experts_type[i], d_model, d_ff, dropout, fan_gate, glu, bias)
                for i in range(n_experts)
            ])
            # experts gating to generate token-to-expert affinity scores
            self.gating= nn.Linear(d_model, n_experts, bias=False)

            # initialize gating modules with Glorot / fan_avg
            nn.init.xavier_uniform_(self.shared_gating.weight)
            nn.init.xavier_uniform_(self.gating.weight)


    def get_ffn(self, ffn_type, d_model, d_ff, dropout, fan_gate, glu, bias):
        if ffn_type == 'conv':
            return ConvFeedForward(d_model, d_ff, None, dropout, glu, bias)
        elif ffn_type == 'dwconv':
            return DwConvFeedForward(d_model, d_ff, None, dropout, glu, bias)
        elif ffn_type == 'mlp':
            return FeedForward(d_model, d_ff, None, dropout, glu, bias)
        else:
            return FANFeedForward(d_model, d_ff, None, dropout, fan_gate, glu, bias)


    def get_expert_ffn(self, expert_type, d_model, d_ff, dropout, fan_gate, glu, bias):
        ffn_type= 'mlp' if expert_type == 'mlp' else 'fan'

        return self.get_ffn(ffn_type, d_model, d_ff, dropout, fan_gate, glu, bias)


    def forward(self, x):
        B, T, C= x.size()

        # with no sparse routed experts
        if self.experts is None:
            return self.shared_expert(x)

        # with sparse routed experts
        x_squashed= x.view(-1, C)  # (B * T, C)

        # compute gating logits and probabilities via softmax
        self.router_logits= self.gating(x_squashed)  # (B * T, n_experts)
        router= F.softmax(self.router_logits.float(), dim=-1)
        # select top-k experts for each token (softmax scores and indices) -> (B * T, K)
        router, selected_experts= torch.topk(router, self.top_k, dim=-1)
        # renormalize over top-k so they sum to 1 -- keeps MoE as a convex mixture
        router= router / router.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # cast back to x dtype
        router= router.to(x.dtype)

        # one hot the selected experts -- (B * T, K, n_experts) -> (n_experts, K, B * T)
        expert_mask= F.one_hot(selected_experts, num_classes=len(self.experts)).permute(2, 1, 0)
        # output buffer
        results= torch.zeros_like(x_squashed)

        for expert_idx, expert in enumerate(self.experts):
            # expert_mask[i] tells us which (rank, token) pairs route to expert i
            # retrieve pairs where this expert is selected
            rank_idx, token_idx= torch.where(expert_mask[expert_idx])  # (K, B * T)
            # index the correct inputs and compute the expert output for the current expert
            # we route individual token embeddings, not whole sequences
            expert_inputs = x_squashed[None, token_idx].reshape(-1, C)
            routing_weight= router[token_idx, rank_idx, None]

            # apply expert and routing weight by gate
            current_expert= expert(expert_inputs) * routing_weight
            results.index_add_(0, token_idx, current_expert.to(x_squashed.dtype))

        # shared fallback expert always applied
        shared_out= self.shared_expert(x) * F.sigmoid(self.shared_gating(x))
        results= results.view(B, T, C) + shared_out

        return results.contiguous()
