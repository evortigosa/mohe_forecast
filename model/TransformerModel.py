# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
The Transformer Architecture
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from einops import repeat
from .Normalization import RMSNorm, DynamicTanh



"""
Rotary Positional Encoding (RoPE)
"""


class QKRoPE(nn.Module):
    """
    The Rotary Positional Encoding (RoPE) functions for Query and Key embeddings.
    It precomputes complex rotation factors (in polar form) and applies them to the Query and Key
    embeddings after reshaping the last dimension into pairs.
    - Based on https://arxiv.org/abs/2407.21783
    """

    def __init__(self, d_head, block_size, theta=10000.0) -> None:
        super(QKRoPE, self).__init__()
        assert d_head % 2 == 0, "d_head must be even for RoPE"
        self.block_size= block_size
        self.theta= theta
        # computing inverse frequencies for each pair in the head dimension in a register_buffer
        self.register_buffer('inv_freq', 1.0 / (self.theta ** (
            torch.arange(0, d_head, 2, dtype=torch.int64)[: (d_head // 2)].float() / d_head
        )))


    def extra_repr(self):
        return f"block_size={self.block_size}, theta={self.theta}"


    def precompute_freqs_cis(self, max_len, device):
        """
        Precompute the complex rotation factors for the given sequence length.
        """
        # computing positions vector
        t= torch.arange(max_len, dtype=torch.int64, device=device).type_as(self.inv_freq)
        # freqs gives all the angles for all the position of tokens in the sequence
        freqs= torch.outer(t, self.inv_freq)
        # the rotation matrix needs to be converted to complex numbers in polar form
        freqs_cis= torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        """
        Reshape freqs_cis for broadcast to match the dimensions of x.
        """
        ndim= x.ndim
        assert 0 <= 1 < ndim, "x should have at least 2 dimensions"
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), \
            "The last two dimension of freqs_cis and x must match"
        # create a shape that has dimension 1 for all dims except dim1 (T) and last dim
        shape= [d if i == 1 or i == ndim -1 else 1 for i, d in enumerate(x.shape)]

        return freqs_cis.view(*shape)


    def forward(self, q, k, start_pos, inference):
        if self.theta <= 0.:
            return q, k

        B, T, _, _= q.shape  # shape (B, T, nh, dh)

        if inference:
            # compute rotation matrix for each position in the sequence
            freqs_cis= self.precompute_freqs_cis(self.block_size * 2, q.device)
            # during inference, we should only take the rotation matrix range from the current
            # position of the tokens
            freqs_cis= freqs_cis[start_pos : start_pos + T]
        else:
            # compute rotation matrix to Query and Key for training
            freqs_cis= self.precompute_freqs_cis(self.block_size, q.device)

        # applying rotary positional encoding to both Query and Key embedding together
        # q/k_ci[B, T, n_(kv_)heads, dh/2] -- reshape last dimension into pairs
        q_ci= torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ci= torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

        # reshape freqs_cis for broadcast to match the dimensions of x
        freqs_cis= self.reshape_for_broadcast(freqs_cis, q_ci)

        # q/k_out[B, T, n_(kv_)heads, dh]
        q_out= torch.view_as_real(q_ci * freqs_cis).flatten(3)
        k_out= torch.view_as_real(k_ci * freqs_cis).flatten(3)

        return q_out.type_as(q), k_out.type_as(k)



""" 
KV Cache
"""


class KVCache:
    """
    --- THIS IS A WIP ---
    Implements a Key-Value Cache module for sliding-window time-series forecasting.
    Based on in-place left-shifts, avoiding repeated concatenations or allocations.
    """

    def __init__(self, block_size, n_kv_heads, d_head) -> None:
        self.block_size= block_size
        self.n_kv_heads= n_kv_heads
        self.d_head= d_head
        self.is_empty= True
        # None until init_cache is called
        self.k_cache= None
        self.v_cache= None
        self.batch_size= None
        self.device= None
        self.dtype = None


    def is_none(self) -> bool:
        """
        Check if caches are unallocated and update is_empty flag.
        """
        if self.k_cache is None or self.v_cache is None:
            self.is_empty= True
            return True
        return False


    def cache_validation(self, batch_size, n_heads, d_head, device, dtype) -> bool:
        """
        Verify and validate cache metadata.
        """
        if self.batch_size is None or self.device is None or self.dtype is None:
            return False
        if (batch_size != self.batch_size) or (n_heads != self.n_kv_heads) or (d_head != self.d_head):
            return False
        if (device != self.device) or (dtype != self.dtype):
            return False
        return True


    def cache_invalidation(self) -> None:
        """
        Logically free the caches.
        """
        if not self.is_none():
            self.k_cache.zero_()
            self.v_cache.zero_()
        self.is_empty= True


    def alloc_buffers(self, batch_size, device, dtype) -> None:
        """
        Allocate internal buffers and set metadata.
        """
        self.batch_size= batch_size
        self.device= device
        self.dtype = dtype
        shape= (self.batch_size, self.block_size, self.n_kv_heads, self.d_head)
        # use zeros for deterministic memory
        self.k_cache= torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.v_cache= torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.is_empty= True


    def init_cache(self, k, v) -> None:
        """
        Initialize cache from a full context window.
        """
        assert k.ndim == 4 and v.ndim == 4, "k and v must be 4D: (B, T, n_heads, d_head)"
        B, T, nh, dh= k.shape
        assert T  == self.block_size, f"k length {T} must equal block_size {self.block_size}"
        assert nh == self.n_kv_heads, f"n_kv_heads mismatch: {nh} vs {self.n_kv_heads}"

        if not self.cache_validation(B, nh, dh, k.device, k.dtype):
            # allocate new buffers when needed or metadata mismatch
            self.alloc_buffers(B, k.device, k.dtype)
        self.is_empty= False
        # copy (detach incoming to avoid keeping graph)
        self.k_cache.copy_(k.detach())
        self.v_cache.copy_(v.detach())


    def get_kv(self):
        """
        Return current caches (raises if empty).
        """
        if self.is_empty or self.is_none():
            raise RuntimeError("KV Cache is empty.")
        return self.k_cache, self.v_cache


    def update(self, k, v):
        """
        Append T new KV vectors and drop the oldest T new (sliding window update).
        """
        assert k.ndim == 4 and v.ndim == 4, "k and v must be 4D: (B, T, n_heads, d_head)"
        B, T_new, nh, dh= k.shape

        if T_new > self.block_size:
            raise ValueError(f"Sequence ({T_new}) cannot exceed block_size ({self.block_size})")

        # nothing to do -> return current cache (or raise if empty)
        if T_new == 0:
            if self.is_empty or self.is_none():
                raise RuntimeError("KV Cache is empty and no tokens provided.")
            return self.k_cache, self.v_cache

        # cache is empty -> initialize from a full context window.
        if self.is_empty or self.is_none():
            self.init_cache(k, v)
            return self.k_cache, self.v_cache

        # cache is not empty -> verify shapes / device
        if not self.cache_validation(B, nh, dh, k.device, k.dtype):
            raise RuntimeError("k shape/device/dtype must match existing cache metadata")

        # shift left by T_new -> [:, :block_size - T_new, :, :]= [:, T_new:, :, :]
        if T_new < self.block_size:
            T_keep= self.block_size - T_new
            self.k_cache[:, :T_keep, :, :].copy_(self.k_cache[:, T_new:, :, :].clone())
            self.v_cache[:, :T_keep, :, :].copy_(self.v_cache[:, T_new:, :, :].clone())
            # write new tokens at end
            self.k_cache[:, T_keep:, :, :].copy_(k.detach())
            self.v_cache[:, T_keep:, :, :].copy_(v.detach())
        else:
            # T_new == block_size -> full replace
            self.k_cache.copy_(k.detach())
            self.v_cache.copy_(v.detach())

        return self.k_cache, self.v_cache



"""
Differential Attention
"""


class DifferentialAttention(nn.Module):
    """
    The Differential Attention Module.
    Note that FlashAttention can be enabled on the fly through the setting of flash_attn in the
    forward method.
    - Based on https://arxiv.org/abs/2410.05258
    """

    def __init__(self, n_heads, d_head, depth, dropout_module) -> None:
        super(DifferentialAttention, self).__init__()
        self.d_head= d_head
        # depth represents the current layer index
        self.depth= depth
        self.lambda_init= self.lambda_init_fn(depth)
        # learnable vectors to compose the learnable lambda term
        self.lambda_q1= nn.Parameter(torch.zeros((n_heads, d_head), dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_k1= nn.Parameter(torch.zeros((n_heads, d_head), dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_q2= nn.Parameter(torch.zeros((n_heads, d_head), dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.lambda_k2= nn.Parameter(torch.zeros((n_heads, d_head), dtype=torch.float32).normal_(mean=0.0, std=0.1))

        self.dropout= dropout_module
        self.scaling= 1.0 / math.sqrt(d_head)
        self.diff_norm= RMSNorm(2 * d_head)


    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)


    def extra_repr(self):
        return f"depth={self.depth}, lambda_init={self.lambda_init}"


    def forward(self, q, k, v, mask, flash_attn):
        B, _, T, _= v.size()  # shape (B, nh, T, dh)

        # lambda is derived from a composition of four learnable vectors
        lambda_1= torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2= torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_final= lambda_1 - lambda_2 + self.lambda_init
        # we extended lambda as a per head modulator
        lambda_final= lambda_final.view(1, lambda_final.size(-1), 1, 1)
        lambda_final= repeat(lambda_final, '1 d 1 1 -> b d 1 1', b=B)

        if not flash_attn:
            # Differential Attention
            attn= (q @ k.transpose(-2, -1)) * self.scaling
            # apply causal mask (when the mask is not None)
            if mask is not None:
                attn= attn.masked_fill(mask[:,:,:T,:T]== 0, float('-inf'))
            # normalize Attention scores
            attn= F.softmax(attn, dim=-1, dtype=torch.float32).type_as(attn)
            # differential mechanism
            attn= attn.view(B, -1, 2, T, T)
            attn= attn[:, :, 0] - lambda_final * attn[:, :, 1]

            attn= self.dropout(attn)
            # compute Attention output
            y= attn @ v  # (B, nh, T, dh)
        else:
            # Differential FlashAttention
            q= q.reshape(B, -1, T, 2, self.d_head)
            k= k.reshape(B, -1, T, 2, self.d_head)
            # query and key matrices are split into two groups
            q1, q2= q[:, :, :, 0], q[:, :, :, 1]
            k1, k2= k[:, :, :, 0], k[:, :, :, 1]
            # compute Attention using FlashAttention kernels
            y1= F.scaled_dot_product_attention(
                q1, k1, v, dropout_p=self.dropout.p, is_causal=mask
            )
            y2= F.scaled_dot_product_attention(
                q2, k2, v, dropout_p=self.dropout.p, is_causal=mask
            )
            y= y1 - lambda_final * y2

        # headwise norm to maintain training stability and scale
        y= self.diff_norm(y)
        y= y * (1 - self.lambda_init)

        return y



"""
Group Query Multi-Headed Attention
"""


class MultiHeadedAttention(nn.Module):
    """
    The Group Query Multi-Headed Attention Module [RoPE and Group Query Attention].
    Note that FlashAttention can be enabled on the fly through the setting of flash_attn in the
    forward method.
    """

    def __init__(self, depth, d_model, block_size, n_heads, n_kv_heads, dropout=0.2, diff_attn=False,
                 bias=False, rope_theta=10000.0) -> None:
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model= d_model
        self.n_heads= n_heads
        # if n_kv_heads is None, use n_heads for key/value; else use provided value
        self.n_kv_heads= n_heads if n_kv_heads is None else n_kv_heads
        assert self.n_kv_heads <= n_heads, "n_kv_heads must be <= n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep= n_heads // self.n_kv_heads
        # when diff_attn is active, halve the effective head dimension for q and k (but not for v)
        self.diff_factor= 1 if not diff_attn else 2
        self.d_head = d_model // n_heads // self.diff_factor
        self.scaling= 1.0 / math.sqrt(self.d_head)

        # query, key, value projections
        self.q_proj= nn.Linear(d_model, d_model, bias=True)
        self.k_proj= nn.Linear(d_model, d_model // self.n_rep, bias=True)
        self.v_proj= nn.Linear(d_model, d_model // self.n_rep, bias=True)
        # Rotary Positional Encoding module
        self.ropenc= QKRoPE(self.d_head, block_size, rope_theta)
        # regularization
        dropout_module= nn.Dropout(p=dropout)
        # differential Attention module
        self.diff_attn= None if not diff_attn else DifferentialAttention(
            n_heads, self.d_head, depth, dropout_module
        )
        self.dropout= dropout_module if not diff_attn else None
        # output projection
        self.o_proj= nn.Linear(d_model, d_model, bias=bias)

        # initialize Linear modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


    @staticmethod
    def norm(x):
        """
        Purely functional RMSNorm with no learnable params
        """
        return F.rms_norm(x, (x.size(-1),))


    @staticmethod
    def repeat_kv(x, n_rep):
        """
        Repeat Key or Value tensor along the head dimension to match the Query head count, i.e.,
        if the number of Key/Value heads is less than Query heads, this function expands the
        Key/Value embeddings with the required number of repetition
        """
        B, T, n_kv_heads, dh= x.shape

        if n_rep== 1:
            return x

        return (
            x[:, :, :, None, :].expand(
                B, T,  n_kv_heads,  n_rep,  dh
            ).reshape(
                B, T, (n_kv_heads * n_rep), dh
            )
        )


    def forward(self, xq, xk, xv, start_pos, inference, causal_mask=None, flash_attn=True):
        B, T, C= xq.size()  # x(batch_size, sequence length, d_model)
        assert C == self.d_model, "Input embedding dimension must match model embedding dimension"

        # calculate query, key, values for all heads
        q= self.q_proj(xq)  # q -> (B, T, C)
        k= self.k_proj(xk)  # k -> (B, T, C // n_rep)
        v= self.v_proj(xv)  # v -> (B, T, C // n_rep)
        # reshape for Group Query Multi-Headed Attention (double n_heads for q and k when diff_attn)
        q= q.view(B, -1, self.n_heads    * self.diff_factor,  self.d_head)  # q view -> (B, T, nh,   dh)
        k= k.view(B, -1, self.n_kv_heads * self.diff_factor,  self.d_head)  # k view -> (B, T, nkvh, dh)
        v= v.view(B, -1, self.n_kv_heads,  self.diff_factor * self.d_head)  # v view -> (B, T, nkvh, dh)
        # apply RoPE to query and key embeddings
        q, k= self.ropenc(q, k, start_pos, inference)
        # q, k= self.norm(q), self.norm(k)  # QK norm
        # here, key and value shapes are not the same with query, which has to be to compute
        # Attention scores
        k= self.repeat_kv(k, self.n_rep)  # k -> (B, T, nh, dh)
        v= self.repeat_kv(v, self.n_rep)  # v -> (B, T, nh, dh)
        # to compute Attention, we need to bring heads at dim 1 and T at dim 2
        q= q.transpose(1, 2).contiguous()
        k= k.transpose(1, 2).contiguous()  # q,k,v transp -> (B, nh, T, dh)
        v= v.transpose(1, 2).contiguous()

        if flash_attn and (not inference):
            # applies FlashAttention
            is_causal= False if causal_mask is None else True

            if self.diff_attn is None:
                y= F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout.p, is_causal=is_causal
                )
            else:
                y= self.diff_attn(q, k, v, is_causal, flash_attn)
        else:
            # implements Attention
            if self.diff_attn is None:
                # the original 'scaled dot product'
                attn= (q @ k.transpose(-2, -1)) * self.scaling  # attn -> (B, nh, T, T)
                # apply causal mask (when the mask is not None)
                if causal_mask is not None:
                    attn= attn.masked_fill(causal_mask[:,:,:T,:T]== 0, float('-inf'))
                # normalize Attention scores
                attn= F.softmax(attn, dim=-1, dtype=torch.float32).type_as(attn)
                attn= self.dropout(attn)
                # compute Attention output
                y= attn @ v  # (B, nh, T, dh)
            else:
                y= self.diff_attn(q, k, v, causal_mask, flash_attn)

        # concatenate multi-head outputs -- re-assembly all head outputs side by side
        y= y.transpose(1, 2).contiguous().view(B, -1, C)  # (B, T, nh, dh) -> (B, T, C)
        # output projection
        return self.o_proj(y)



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

    def __init__(self, d_model, d_ff, dropout=0.2, ffn_type='conv', fan_gate=False, glu=False,
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
            # controls contribution from fallback expert
            self.shared_gating= nn.Linear(d_model, 1, bias=False)

            assert top_k > 0, "top_k must be > 0"
            self.top_k= min(top_k, n_experts)

            if isinstance(experts_type, str):
                experts_type= [experts_type for _ in range(n_experts)]
            else:
                assert all(isinstance(item, str) for item in experts_type), \
                    "experts_type must be a list of strings"
                assert len(experts_type) >= n_experts, \
                    "experts_type must be a string or a list of length n_experts"

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



"""
Transformer Block
"""


class TransformerBlock(nn.Module):
    """
    The Transformer Block (Encoder/Decoder-only, pre-normalization version).
    - If multi_modal=True, we have an extra cross-attention module to incorporate exogenous
    covariates and allow for multi-modal learning.
    - If is_causal=True, we have a Decoder Transformer; otherwise, an Encoder Transformer.
    - norm_type (str): 'layer' for LayerNorm, 'rms' for RMSNorm, or 'dyt' for DynamicTanh.
    - If diff_attn=True, we use Differential Attention.
    - MoHE. ffn_type (str): the shared expert that can be 'mlp' for MLP-FFN, 'conv' for Conv-FFN,
    'dwconv' for DwConv-FFN, or 'fan' for FAN-FFN. experts_type (str): multiple routed experts that
    can be 'mlp' for MLP-FFN or 'fan' for FAN-FFN.
    Note that FlashAttention can be enabled on the fly in the MultiHeadedAttention module
    through the setting of flash_attn in the forward method.
    """

    def __init__(self, multi_modal, depth, d_model=384, block_size=672, n_heads=12, n_kv_heads=6,
                 d_ff=768, dropout=0.2, drop_path=0.3, norm_type='rms', diff_attn=False,
                 ffn_type='dwconv', glu=False, n_experts=8, top_k_experts=2, experts_type='fan',
                 bias=False, rope_theta=10000.0) -> None:
        super(TransformerBlock, self).__init__()

        # Self-Attention module to endogenous series
        self.norm1= self.get_norm(norm_type, d_model, init_alpha=0.6)
        self.s_att= MultiHeadedAttention(
            depth, d_model, block_size, n_heads, n_kv_heads, dropout, diff_attn, bias, rope_theta
        )
        self.drop_path1= DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Cross-Attention module to incorporate exogenous covariates
        if multi_modal:
            self.norm2= self.get_norm(norm_type, d_model, init_alpha=0.6)
            self.c_att= MultiHeadedAttention(
                depth, d_model, block_size, n_heads, n_kv_heads, dropout, False, bias, rope_theta
            )
            self.drop_path2= DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        else:
            self.c_att= None

        # Mixture-of-Experts (MoE) module allows heterogeneous experts (MoHE)
        self.norm3= self.get_norm(norm_type, d_model, init_alpha=0.2)
        self.ffn  = MoEFeedForward(
            d_model, d_ff, dropout, ffn_type, False, glu, n_experts, top_k_experts, experts_type,
            bias
        )
        self.drop_path3= DropPath(drop_path) if drop_path > 0.0 else nn.Identity()


    def get_norm(self, norm_type, d_model, init_alpha=0.5):
        if norm_type == 'rms':
            return RMSNorm(d_model)
        elif norm_type == 'dyt':
            return DynamicTanh(d_model, init_alpha)
        else:
            return nn.LayerNorm(d_model)


    def forward(self, x, x_cross, start_pos, inference, causal_mask=None, flash_attn=True):
        x_norm= self.norm1(x)
        x= x + self.drop_path1(self.s_att(
            x_norm, x_norm, x_norm, start_pos, inference, causal_mask, flash_attn
        ))
        if (self.c_att is not None) and (x_cross is not None):
            x_norm= self.norm2(x)
            x= x + self.drop_path2(self.c_att(  # no causal_mask in cross-attention
                x_norm, x_cross, x_cross, start_pos, inference, None, flash_attn
            ))
        x_norm= self.norm3(x)
        x= x + self.drop_path3(self.ffn(x_norm))

        return x, self.ffn.router_logits



"""
Transformer Model
"""


class TransformerModel(nn.Module):
    """
    A Transformer model is essentially a stack of N Encoder/Decoder Blocks.
    - If multi_modal=True, we have an extra cross-attention module to incorporate exogenous
    covariates and allow for multi-modal learning.
    - If is_causal=True, we have a Decoder Transformer; otherwise, an Encoder Transformer.
    - norm_type (str): 'layer' for LayerNorm, 'rms' for RMSNorm, or 'dyt' for DynamicTanh.
    - If diff_attn=True, we use differential attention.
    - MoHE. ffn_type (str): the shared expert that can be 'mlp' for MLP-FFN, 'conv' for Conv-FFN,
    'dwconv' for DwConv-FFN, or 'fan' for FAN-FFN. experts_type (str): multiple routed experts that
    can be 'mlp' for MLP-FFN or 'fan' for FAN-FFN.
    """

    def __init__(self, multi_modal, is_causal, n_layer=8, d_model=384, block_size=672, n_heads=12,
                 n_kv_heads=6, d_ff=768, dropout=0.2, drop_path=0.3, norm_type='rms', flash_attn=True,
                 diff_attn=False, ffn_type='dwconv', glu=False, n_experts=8, top_k_experts=2,
                 experts_type='fan', bias=False, rope_theta=10000.0) -> None:
        super(TransformerModel, self).__init__()
        # block_size represents the max sequence length
        self.block_size= block_size
        # create a lower triangular matrix (2D tensor)
        self.register_buffer(
            'causal_mask_buffer',
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False
        )
        # set the causal mask and FlashAttention use
        self.flash_attn = flash_attn
        self.causal_mask= None
        self.def_causal_mask(is_causal, self.flash_attn)
        # stochastic decay according to each TransformerBlock depth
        sdp_rates= [x.item() for x in torch.linspace(0, drop_path, n_layer)]

        # define a stack of TransformerBlocks
        self.transformer= nn.ModuleList([
            TransformerBlock(
                multi_modal, depth, d_model, block_size, n_heads, n_kv_heads, d_ff, dropout,
                sdp_rates[depth], norm_type, diff_attn, ffn_type, glu, n_experts, top_k_experts,
                experts_type, bias, rope_theta
            ) for depth in range(n_layer)
        ])
        # final normalization layer after the last TransformerBlock
        self.final_norm= self.transformer[-1].get_norm(norm_type, d_model, init_alpha=0.2)


    def def_causal_mask(self, is_causal, flash_attn=True):
        """
        If is_causal=True, we have a Decoder Transformer; otherwise, an Encoder Transformer.
        """
        self.flash_attn= flash_attn

        if is_causal and (not self.flash_attn):
            # causal mask tensor on the Attention outputs when TransformerModel is a Decoder, i.e.,
            # current steps depend on the past only
            self.causal_mask= self.causal_mask_buffer
        elif is_causal and self.flash_attn:
            # causal mask when TransformerModel is a Decoder using FlashAttention
            self.causal_mask= True
        else:
            # no causal mask when TransformerModel is an Encoder
            self.causal_mask= None


    def forward(self, x, x_cross, start_pos, inference):
        B, T, C= x.size()  # x(batch_size, sequence length, d_model)
        assert T <= self.block_size, \
            f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        if self.causal_mask is not None:
            if inference:
                self.def_causal_mask(is_causal=True, flash_attn=False)
            else:
                self.def_causal_mask(is_causal=True, flash_attn=self.flash_attn)

        if isinstance(self.causal_mask, torch.Tensor):
            if self.causal_mask.device != x.device:
                self.causal_mask= self.causal_mask.to(x.device)

        all_router_logits= ()
        # forward the embedding through the transformer
        for block in self.transformer:
            x, router_logits= block(
                x, x_cross, start_pos, inference, self.causal_mask, self.flash_attn
            )
            all_router_logits += (router_logits,)

        return self.final_norm(x), all_router_logits
