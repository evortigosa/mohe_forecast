# -*- coding: utf-8 -*-
"""
# The Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE) Model
Input Embedding Modules
"""

import math
import torch
import torch.nn as nn
from .Normalization import RMSNorm



"""
# Input Modules
"""


class PositionalEmbedding(nn.Module):
    """
    Implements the standard PE function as in https://arxiv.org/abs/1706.03762.
    """

    def __init__(self, block_size, d_model, base_val=10000.0) -> None:
        super(PositionalEmbedding, self).__init__()
        self.block_size= block_size
        self.base_val= base_val

        # create a long tensor of block_size positions
        position= torch.arange(0, block_size).unsqueeze(1)
        frequencies= torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(base_val) / d_model)
        )
        # create an empty placeholder
        wpe= torch.zeros(block_size, d_model).float()
        wpe.require_grad= False
        # iterating over each element in the sequence using sin and cos
        wpe[:, 0::2]= torch.sin(position * frequencies)
        wpe[:, 1::2]= torch.cos(position * frequencies)
        # register_buffer -- it is not saved in the state_dict nor optimized
        self.register_buffer('wpe', wpe.unsqueeze(0), persistent=False)


    def extra_repr(self):
        return f"block_size={self.block_size}, base={self.base_val}"


    def forward(self, x):
        x= x + self.wpe[:, : x.size(1)].to(x.device)

        return x



class PatchMasking(nn.Module):
    """
    Applies random masking to the patch embeddings for self-supervised training tasks.
    A specified fraction (mask_ratio) of patches is set to zero.
    - If has_cls_tk=True, the class token (first token) is not masked.
    """

    def __init__(self, mask_ratio=0.2, has_cls_tk=False) -> None:
        super(PatchMasking, self).__init__()
        assert 0.0 <= mask_ratio < 1.0, "mask_ratio must be in [0, 1)"
        self.mask_ratio= mask_ratio
        self.has_cls_tk= has_cls_tk


    def extra_repr(self):
        return f"mask_ratio={self.mask_ratio}, cls_token={self.has_cls_tk}"


    def forward(self, x, x_cross=None):
        if (not self.training) or self.mask_ratio== 0.0:
            return x, x_cross

        if self.has_cls_tk:
            # ensure the class token will not be masked
            cls= x[:, 0, :]
            x  = x[:, 1:, :]
            if x_cross is not None:
                cls_cross= x_cross[:, 0, :]
                x_cross  = x_cross[:, 1:, :]

        B, P, C= x.size()  # (batch_size, num_patches, d_model)
        # create a binary mask of shape (B, P): True means the patch is masked
        mask= torch.rand(B, P, dtype=x.dtype, device=x.device) < self.mask_ratio
        # expand mask to match x dimensions (B, P, 1)
        mask= mask.unsqueeze(-1)

        # set masked positions to zero
        x= x.masked_fill(mask, value=0.0)
        if self.has_cls_tk:
            x= torch.cat((cls.unsqueeze(1), x), dim=1)

        if x_cross is not None:
            x_cross= x_cross.masked_fill(mask, value=0.0)
            if self.has_cls_tk:
                x_cross= torch.cat((cls_cross.unsqueeze(1), x_cross), dim=1)

        return x, x_cross



class PatchMaskingMAE(nn.Module):
    """
    Performs random masking to the patch embeddings for self-supervised training tasks.
    Masked Autoencoder (MAE) style: Only the visible patches are keep to feed the model.
    - If has_cls_tk=True, the class token (first token) is not masked.
    Adapted from https://arxiv.org/abs/2111.06377
    """

    def __init__(self, mask_ratio=0.2, has_cls_tk=False) -> None:
        super(PatchMaskingMAE, self).__init__()
        assert 0.0 <= mask_ratio < 1.0, "mask_ratio must be in [0, 1)"
        self.mask_ratio= mask_ratio
        self.has_cls_tk= has_cls_tk


    def extra_repr(self):
        return f"mask_ratio={self.mask_ratio}, cls_token={self.has_cls_tk}"


    def forward(self, x, x_cross=None):
        if (not self.training) or self.mask_ratio== 0.0:
            return x, x_cross, None, None

        if self.has_cls_tk:
            # ensure the class token will not be masked
            cls= x[:, 0, :]
            x  = x[:, 1:, :]
            if x_cross is not None:
                cls_cross= x_cross[:, 0, :]
                x_cross  = x_cross[:, 1:, :]

        B, P, C= x.size()  # (batch_size, num_patches, d_model)
        # determine the number of patches to keep
        pto_keep= int(P * (1 - self.mask_ratio))
        # generate random indices for masking -- noise in [0, 1]
        # ascend: small is keep, large is remove
        ids_shuffle= torch.rand(B, P, dtype=x.dtype, device=x.device).argsort(dim=1)
        ids_restore= torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep= ids_shuffle[:, :pto_keep]
        x_masked= torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
        if self.has_cls_tk:
            x_masked= torch.cat((cls.unsqueeze(1), x_masked), dim=1)

        if x_cross is not None:
            x_cross_masked= torch.gather(x_cross, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
            if self.has_cls_tk:
                x_cross_masked= torch.cat((cls_cross.unsqueeze(1), x_cross_masked), dim=1)
        else:
            x_cross_masked= None

        # generate the binary mask: 0 is keep, 1 is remove
        mask= torch.ones([B, P], device=x.device)
        mask[:, :pto_keep]= 0
        # unshuffle to get the binary mask
        mask= torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, x_cross_masked, mask, ids_restore



class IntTemporalEmbedding(nn.Module):
    """
    Initializes the Temporal Embedding module for handling calendar covariates (hour‑of‑day,
    day‑of‑week, month) as extra channels.
    - This IntTemporalEmbedding was designed to take integer bucket indices (so use timeenc=0)
    - Assume feature ordering: (B, C, T) where C = [month, day, weekday, hour, (minute)].
    - C must be at least 4 (month, day, weekday, hour) and optional 5th feature -> minute.
    """

    def __init__(self, out_channels, d_model, minute_size=4):
        super(IntTemporalEmbedding, self).__init__()
        # cardinalities of each calendar feature [month, day, weekday, hour, (minute)]
        month_size  = 13
        day_size    = 32
        weekday_size= 7
        hour_size   = 24

        # per‑feature embeddings
        self.month_embed  = nn.Embedding(month_size, d_model)
        self.day_embed    = nn.Embedding(day_size, d_model)
        self.weekday_embed= nn.Embedding(weekday_size, d_model)
        self.hour_embed   = nn.Embedding(hour_size, d_model)
        self.minute_embed = nn.Embedding(minute_size, d_model)
        # # project concatenated time features from 5*d_model to out_channels
        self.proj_time= nn.Linear(5*d_model, out_channels, bias=False)
        # fuse the concatenated 2*out_channels time/data embedding to out_channels
        self.fuse= nn.Linear(2*out_channels, out_channels, bias=False)

        # initialize Linear modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


    def forward(self, x_mark, x):
        x_mark = x_mark.long()
        _, C, _= x_mark.size()  # (B, C, T) integer‐indexed time features
        assert 4 <= C <= 5, f"Expected time channels 4 or 5; got C={C}"
        # 'T' or 'min' is the consistent naming for minutes
        freq= 'min' if C== 5 else 'h'  # freq= 'h' if C== 4

        month  = self.month_embed(x_mark[:, 0, :])
        day    = self.day_embed(x_mark[:, 1, :])
        weekday= self.weekday_embed(x_mark[:, 2, :])
        hour   = self.hour_embed(x_mark[:, 3, :])
        if freq== 'min':
            min_data= x_mark[:, 4, :]
        else:
            min_data= torch.zeros_like(hour)
        minute= self.minute_embed(min_data)

        # concatenate along the embedding dimension
        x_mark= torch.cat([month, day, weekday, hour, minute], dim=-1)
        x_mark= self.proj_time(x_mark)  # (B, T, C)
        x= x.permute(0, 2, 1)           # (B, T, C)

        te= torch.cat([x, x_mark], dim=-1)  # (B, T, 2*C)
        te= self.fuse(te)                   # (B, T, C)
        # fuse time embeddings -- now we can then patchify
        return te.permute(0, 2, 1)  # (B, C, T)



class ContTemporalEmbeddingV3(nn.Module):
    """
    Initializes the Temporal Embedding module for handling calendar covariates (hour‑of‑day,
    day‑of‑week, month) as extra channels.
    - This ContTemporalEmbedding was designed to take continuous‐valued time data (timeenc=1)
    - Assume feature ordering: (B, C, T) where C = [(minute), hour, weekday, day, month].
    - C must be at least 4 (hour, weekday, day, month) and optional 5th feature -> minute.
    """

    def __init__(self, out_channels, d_model):
        super(ContTemporalEmbeddingV3, self).__init__()
        # expected C = 5 for [minute, hour, weekday, day, month]
        time_channels= 5

        # project raw data channels to d_model
        self.proj_data= nn.Linear(out_channels, d_model, bias=False)
        # project time features to d_model
        self.proj_time= nn.Linear(time_channels, d_model, bias=False)
        # fuse the concatenated 2*d_model time/data embedding to out_channels
        self.fuse= nn.Linear(2*d_model, out_channels, bias=False)

        # initialize Linear modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


    def forward(self, x_mark, x):
        B, C, T= x_mark.size()  # (B, C, T) integer‐indexed time features
        assert 4 <= C <= 5, f"Expected time channels 4 or 5; got C={C}"
        # 'T' or 'min' is the consistent naming for minutes
        freq= 'min' if C== 5 else 'h'  # freq= 'h' if C== 4

        if freq== 'h':
            minute= torch.zeros(B, 1, T, device=x_mark.device, dtype=x_mark.dtype) -0.50
            x_mark= torch.cat([minute, x_mark], dim=1)

        x     = self.proj_data(x.permute(0, 2, 1))       # (B, T, d_model)
        x_mark= self.proj_time(x_mark.permute(0, 2, 1))  # (B, T, d_model)

        te= torch.cat([x, x_mark], dim=-1)  # (B, T, 2*d_model)
        te= self.fuse(te)                   # (B, T, C)
        # fuse time embeddings -- now we can then patchify
        return te.permute(0, 2, 1)  # (B, C, T)



class ContTemporalEmbedding(nn.Module):
    """
    Initializes the Temporal Embedding module for handling calendar covariates (hour‑of‑day,
    day‑of‑week, month) as extra channels.
    - This ContTemporalEmbedding was designed to take continuous‐valued time data (timeenc=1)
    - Assume feature ordering: (B, C, T) where C = [(minute), hour, weekday, day, month].
    - C must be at least 4 (hour, weekday, day, month) and optional 5th feature -> minute.
    """

    def __init__(self, out_channels, d_model):
        super(ContTemporalEmbedding, self).__init__()
        # expected C = 5 for [minute, hour, weekday, day, month]
        time_channels= 5
        # project to the highest dim to avoid bottlenecks
        d_hidden= max(time_channels, out_channels, d_model)

        # project raw data channels to d_hidden
        self.proj_data= nn.Linear(out_channels, d_hidden, bias=False)
        # project time features to d_hidden
        self.proj_time= nn.Linear(time_channels, d_hidden, bias=False)
        # fuse the concatenated 2*d_hidden time/data embedding to out_channels
        self.fuse= nn.Linear(2*d_hidden, out_channels, bias=False)

        # initialize Linear modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


    def forward(self, x_mark, x):
        B, C, T= x_mark.size()  # (B, C, T) integer‐indexed time features
        assert 4 <= C <= 5, f"Expected time channels 4 or 5; got C={C}"
        # 'T' or 'min' is the consistent naming for minutes
        freq= 'min' if C== 5 else 'h'  # freq= 'h' if C== 4

        if freq== 'h':
            minute= torch.zeros(B, 1, T, device=x_mark.device, dtype=x_mark.dtype) -0.50
            x_mark= torch.cat([minute, x_mark], dim=1)

        x     = self.proj_data(x.permute(0, 2, 1))       # (B, T, d_hidden)
        x_mark= self.proj_time(x_mark.permute(0, 2, 1))  # (B, T, d_hidden)

        te= torch.cat([x, x_mark], dim=-1)  # (B, T, 2*d_hidden)
        te= self.fuse(te)                   # (B, T, C)
        # fuse time embeddings -- now we can then patchify
        return te.permute(0, 2, 1)  # (B, C, T)



class PatchEmbeddingV3(nn.Module):
    """
    Initializes the Embedding module. Applies either depthwise separable or regular convolutions
    to patch the input sequence and applies normalization + dropout.
    - v3: uses RMSNorm as the normalization layer.
    """

    def __init__(self, patch_width, channels, d_model, dropout=0.2) -> None:
        super(PatchEmbeddingV3, self).__init__()
        self.patch_width= patch_width
        self.d_model= d_model
        channels= 1

        # define convolutional patch embedding
        self.embed= nn.Conv1d(  # (batch_size, d_model, num_patches)
            channels, d_model, kernel_size=patch_width, stride=patch_width, bias=False
        )
        # define normalization and dropout modules for regularization
        self.norm= RMSNorm(d_model)
        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None

        # initialize Conv modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


    def extra_repr(self):
        return f"patch_width={self.patch_width}, d_model={self.d_model}"


    def forward(self, ts):
        # ts -> (batch_size, channels/features, seq_length)
        B, C, T= ts.size()
        ts= ts.reshape(-1, T).unsqueeze(1)  # (batch_size * channels/features, 1, seq_length)
        # ensure channel independence, batch_size assume batch_size * channels/features

        x= self.embed(ts)
        # x -> (B * C, d_model, num_patches)
        x= x.permute(0, 2, 1)
        # x -> (B * C, num_patches, d_model)
        if self.dropout is not None:
            x= self.dropout(self.norm(x))
        else:
            x= self.norm(x)

        return x.contiguous()  # (B * C, num_patches, d_model)



class PatchEmbedding(nn.Module):
    """
    Initializes the Embedding module. Applies either depthwise separable or regular convolutions
    to patch the input sequence and applies normalization + dropout.
    - v4: uses GroupNorm as the normalization layer.
    """

    def __init__(self, patch_width, channels, d_model, dropout=0.2) -> None:
        super(PatchEmbedding, self).__init__()
        self.patch_width= patch_width
        self.d_model= d_model
        channels= 1

        # define convolutional patch embedding
        self.embed= nn.Conv1d(  # (batch_size, d_model, num_patches)
            channels, d_model, kernel_size=patch_width, stride=patch_width, bias=False
        )
        # define normalization and dropout modules for regularization
        self.norm= nn.GroupNorm(num_groups=1, num_channels=d_model)
        # single group, equivalent with a LayerNorm
        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None

        # initialize Conv modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


    def extra_repr(self):
        return f"patch_width={self.patch_width}, d_model={self.d_model}"


    def forward(self, ts):
        # ts -> (batch_size, channels/features, seq_length)
        B, C, T= ts.size()
        ts= ts.reshape(-1, T).unsqueeze(1)  # (batch_size * channels/features, 1, seq_length)
        # ensure channel independence, batch_size assume batch_size * channels/features

        x= self.embed(ts)
        # x -> (B * C, d_model, num_patches)
        if self.dropout is not None:
            x= self.dropout(self.norm(x))
        else:
            x= self.norm(x)
        x= x.permute(0, 2, 1)
        # x -> (B * C, num_patches, d_model)

        return x.contiguous()  # (B * C, num_patches, d_model)



class MultiModalEmbedding(nn.Module):
    """
    This module handles the embedding of exogenous covariates in order to allow for multi-modal
    learning.
    """

    def __init__(self, patch_width, out_channels, d_model, dropout=0.2, norm_type='group') -> None:
        super(MultiModalEmbedding, self).__init__()

        self.covariates= ContTemporalEmbedding(out_channels, d_model)
        if norm_type == 'rms':
            self.patchfy= PatchEmbeddingV3(patch_width, out_channels, d_model, dropout)
        else:
            self.patchfy= PatchEmbedding(patch_width, out_channels, d_model, dropout)


    def forward(self, x_mark, x):
        x= self.covariates(x_mark, x)

        return self.patchfy(x)



class EmbeddingDecoderMAE(nn.Module):
    """
    Expand the input for a Masked Autoencoder (MAE) style decoder. Only the visible patches are
    keep by a MAE style encoder model. This module places zero tokens at the masked positions
    to feed the MAE style decoder.
    Adapted from https://arxiv.org/abs/2111.06377
    """

    def __init__(self, enc_d_model, dec_d_model, has_cls_tk=False, bias=False) -> None:
        super(EmbeddingDecoderMAE, self).__init__()
        self.has_cls_tk= has_cls_tk
        self.decoder_embed= nn.Linear(enc_d_model, dec_d_model, bias=bias)
        self.mask_token= nn.Parameter(torch.zeros(1, 1, dec_d_model))

        # initialize nn.Linear modules with Glorot / fan_avg
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        if self.decoder_embed.bias is not None: nn.init.zeros_(self.decoder_embed.bias)
        torch.nn.init.normal_(self.mask_token, std=.02)


    def forward(self, x, ids_restore, cls_token=None):
        if cls_token is not None:
            self.has_cls_tk= True
            x= torch.cat([cls_token.unsqueeze(1), x], dim=1)

        # embed tokens to decoder d_model
        x= self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens= self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        if self.has_cls_tk:
            # temporarily remove the cls token from the input
            x_= torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        else:
            x_= torch.cat([x, mask_tokens], dim=1)

        # unshuffle
        x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        if self.has_cls_tk:
            # append cls token from the input
            x= torch.cat([x[:, :1, :], x_], dim=1)
            return x

        return x_
