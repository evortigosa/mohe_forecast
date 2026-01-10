# -*- coding: utf-8 -*-
"""
# The Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE) Model
"""

import math
import inspect
import torch
import torch.nn as nn
from einops import repeat
from dataclasses import asdict
from .Normalization import RMSNorm, InstanceNorm, RevIN
from .TransformerModel import FeedForward, ConvFeedForward, DwConvFeedForward, FANLayer, FANFeedForward
from .TransformerModel import TransformerModel
from .Config import BaseConfig



"""
# Input Modules
"""


def round_channels(channels, width_mult=1, divisor=8, min_value=None):
    """
    Round number of channels based on width multiplier.
    Ensure that all layers have a channel number that is divisible by 'divisor'.
    - This helps with efficient hardware utilization.
    """
    if min_value is None:
        min_value= divisor

    new_channels= channels * width_mult
    new_channels= max(min_value, int(new_channels + divisor / 2) // divisor * divisor)
    # Prevent rounding down by more than 10%
    if new_channels < 0.9 * channels:
        new_channels += divisor

    return int(new_channels)



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
        # itterating over each element in the sequence using sin and cos
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
    """

    def __init__(self, mask_ratio=0.2) -> None:
        super(PatchMasking, self).__init__()
        assert 0.0 <= mask_ratio < 1.0, "mask_ratio must be in [0, 1)"
        self.mask_ratio= mask_ratio


    def extra_repr(self):
        return f"mask_ratio={self.mask_ratio}"


    def forward(self, x, x_cross=None, has_cls_tk=False):
        if (not self.training) or self.mask_ratio== 0.0:
            return x, x_cross

        if has_cls_tk:
            # ensure the class token will not be masked
            cls= x[:, 0, :]
            x  = x[:, 1:, :]
            if (x_cross is not None):
                cls_cross= x_cross[:, 0, :]
                x_cross  = x_cross[:, 1:, :]

        B, P, C= x.size()  # (batch_size, num_patches, d_model)
        # create a binary mask of shape (B, P): True means the patch is masked
        mask= torch.rand(B, P, dtype=x.dtype, device=x.device) < self.mask_ratio
        # expand mask to match x dimensions (B, P, 1)
        mask= mask.unsqueeze(-1)

        # set masked positions to zero
        x= x.masked_fill(mask, value=0.0)
        if has_cls_tk:
            x= torch.cat((cls.unsqueeze(1), x), dim=1)

        if (x_cross is not None):
            x_cross= x_cross.masked_fill(mask, value=0.0)
            if has_cls_tk:
                x_cross= torch.cat((cls_cross.unsqueeze(1), x_cross), dim=1)

        return x, x_cross



class PatchMaskingMAE(nn.Module):
    """
    Performs random masking to the patch embeddings for self-supervised training tasks.
    Masked Autoencoder (MAE) style: Only the visible patches are keep to feed the model.
    Adapted from https://arxiv.org/abs/2111.06377
    """

    def __init__(self, mask_ratio=0.2) -> None:
        super(PatchMaskingMAE, self).__init__()
        assert 0.0 <= mask_ratio < 1.0, "mask_ratio must be in [0, 1)"
        self.mask_ratio= mask_ratio


    def extra_repr(self):
        return f"mask_ratio={self.mask_ratio}"


    def forward(self, x, x_cross=None, has_cls_tk=False):
        if (not self.training) or self.mask_ratio== 0.0:
            return x, x_cross, None, None

        if has_cls_tk:
            # ensure the class token will not be masked
            cls= x[:, 0, :]
            x  = x[:, 1:, :]
            if (x_cross is not None):
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
        if has_cls_tk:
            x_masked= torch.cat((cls.unsqueeze(1), x_masked), dim=1)

        if (x_cross is not None):
            x_cross_masked= torch.gather(x_cross, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
            if has_cls_tk:
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
    keep by a MAE style encoder model. This module places random tokens in the masked positions
    to feed the MAE style decoder.
    Adapted from https://arxiv.org/abs/2111.06377
    """

    def __init__(self, enc_d_model, dec_d_model, bias=False) -> None:
        super(EmbeddingDecoderMAE, self).__init__()
        self.decoder_embed= nn.Linear(enc_d_model, dec_d_model, bias=bias)
        self.mask_token= nn.Parameter(torch.zeros(1, 1, dec_d_model))

        # initialize nn.Linear modules with Glorot / fan_avg
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        if self.decoder_embed.bias is not None: nn.init.zeros_(self.decoder_embed.bias)
        torch.nn.init.normal_(self.mask_token, std=.02)


    def forward(self, x, ids_restore, has_cls_tk=False):
        # embed tokens to decoder d_model
        x= self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens= self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        if has_cls_tk:
            # temporarily remove the cls token
            x_= torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        else:
            x_= torch.cat([x, mask_tokens], dim=1)

        # unshuffle
        x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        if has_cls_tk:
            # append cls token
            x= torch.cat([x[:, :1, :], x_], dim=1)
            return x

        return x_



"""
# Output Modules
"""


class OutputBlock(nn.Module):
    """
    The output projection head.
    - If fine_tune=True: out_proj is a single projection layer; otherwise, it assumes an FFN module
    according to ffn_type (str) -- 'mlp' for MLP-FFN, 'conv' for Conv-FFN, 'dwconv' for DwConv-FFN,
    or 'fan' for FAN-FFN.
    """

    def __init__(self, forecasting, d_model, d_ff, n_outputs, dropout=0.2, ffn_type='mlp', bias=False,
                 fine_tune=False) -> None:
        super(OutputBlock, self).__init__()
        # in fine_tune mode we have only a simplified projection head -- see ViT
        if fine_tune:
            if ffn_type== 'fan':
                self.out_proj= FANLayer(d_model, n_outputs, bias=bias, is_last=True)
            else:
                self.out_proj= nn.Linear(d_model, n_outputs, bias=bias)

                # initialize non-FAN projection modules with Glorot / fan_avg
                nn.init.xavier_uniform_(self.out_proj.weight)
                if self.out_proj.bias is not None: nn.init.zeros_(self.out_proj.bias)
        else:
            if ffn_type == 'conv' and forecasting:
                self.out_proj= ConvFeedForward(d_model, d_ff, n_outputs, dropout, glu=False, bias=bias)
            elif ffn_type == 'dwconv' and forecasting:
                self.out_proj= DwConvFeedForward(d_model, d_ff, n_outputs, dropout, glu=False, bias=bias)
            elif ffn_type == 'fan':
                self.out_proj= FANFeedForward(
                    d_model, d_ff, n_outputs, dropout, fan_gate=False, glu=False, bias=bias
                )
            else:
                self.out_proj= FeedForward(d_model, d_ff, n_outputs, dropout, glu=False, bias=bias)


    def forward(self, x):
        x= self.out_proj(x)

        return x



class UnPatchV3(nn.Module):
    """
    Initializes the Reverse Patch Embedding module. Applies convolutions to reverse (decode) the
    patch embedding back to input sequence shape allowing for SSL-Encoding.
    See https://arxiv.org/abs/2201.03545
    """

    def __init__(self, patch_width, channels, d_model, dropout=0.2, bias=False) -> None:
        super(UnPatchV3, self).__init__()
        assert d_model % patch_width == 0, "d_model must be divisible by patch_width"
        self.channels = channels
        pw_d_model= round_channels(d_model // patch_width)
        hidden_dim= round_channels(pw_d_model * 4)
        out_channels= 1
        # calculate kernel_size and padding of the depthwise conv based on patch_width
        dks= min(max(((patch_width // 2) - 1), 1), 7)  # [1, 7]
        dks= dks - 1 if dks % 2 == 0 else dks
        dpd= dks // 2

        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None
        self.unpatch= nn.Sequential(
            nn.ConvTranspose1d(  # (batch_size, d_model, num_patches)
                d_model, pw_d_model, kernel_size=patch_width, stride=patch_width, bias=False
            ),
            nn.GELU(),
            nn.Conv1d(           # depthwise conv
                pw_d_model, pw_d_model, kernel_size=dks, stride=1, padding=dpd, groups=pw_d_model,
                bias=False
            ),
            nn.GroupNorm(num_groups=1, num_channels=pw_d_model),
            nn.Conv1d(pw_d_model, hidden_dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU(),           # projection phase
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )                        # (batch_size, channels/features, seq_length)

        # initialize Conv modules and norm
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


    def forward(self, x):
        # x -> (batch_size * channels/features, num_patches, d_model)
        if self.dropout is not None:
            x= self.dropout(x)
        # upsample and decode the patch embeddings
        x = x.permute(0, 2, 1)  # (B, P, C) -> (B, C, P)
        ts= self.unpatch(x)
        # ts -> (batch_size * channels/features, 1, seq_length)
        ts= ts.reshape(-1, self.channels, ts.size(-1))
        # ts -> (batch_size, channels/features, seq_length)

        return ts.contiguous()



class UnPatch(nn.Module):
    """
    Initializes the Reverse Patch Embedding module. Applies convolutions to reverse (decode) the
    patch embedding back to input sequence shape allowing for SSL-Encoding.
    See https://arxiv.org/abs/2201.03545
    """

    def __init__(self, patch_width, channels, d_model, dropout=0.2, bias=False) -> None:
        super(UnPatch, self).__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        self.channels = channels
        hidden_dim= round_channels(d_model // 4)
        out_channels= 1
        # calculate kernel_size and padding of the depthwise conv based on patch_width
        dks= min(max(((patch_width // 2) - 1), 1), 7)  # [1, 7]
        dks= dks - 1 if dks % 2 == 0 else dks
        dpd= dks // 2

        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None
        self.unpatch= nn.Sequential(
            nn.ConvTranspose1d(  # (batch_size, d_model, num_patches)
                d_model, d_model, kernel_size=patch_width, stride=patch_width, bias=False
            ),
            nn.GELU(),
            nn.Conv1d(           # depthwise conv
                d_model, d_model, kernel_size=dks, stride=1, padding=dpd, groups=d_model, bias=False
            ),
            nn.GroupNorm(num_groups=1, num_channels=d_model),
            nn.Conv1d(d_model, hidden_dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU(),           # projection phase
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )                        # (batch_size, channels/features, seq_length)

        # initialize Conv modules and norm
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


    def forward(self, x):
        # x -> (batch_size * channels/features, num_patches, d_model)
        if self.dropout is not None:
            x= self.dropout(x)
        # upsample and decode the patch embeddings
        x = x.permute(0, 2, 1)  # (B, P, C) -> (B, C, P)
        ts= self.unpatch(x)
        # ts -> (batch_size * channels/features, 1, seq_length)
        ts= ts.reshape(-1, self.channels, ts.size(-1))
        # ts -> (batch_size, channels/features, seq_length)

        return ts.contiguous()



class LinearUnPatch(nn.Module):
    """
    Initializes the Linear Reverse Patch Embedding module. Applies a linear projection to reverse
    (decode) the patch embedding back to input sequence shape allowing for SSL-Encoding.
    - From a (B, P, C) tensor into a (B, D, H) forecast.
    """

    def __init__(self, n_patches, channels, d_model, n_outputs, dropout=0.2,
                 bias=False, individual=False) -> None:
        super(LinearUnPatch, self).__init__()
        self.channels  = channels
        self.individual= individual
        input_dim= n_patches * d_model

        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None
        if self.individual:
            # individual linear mapping from P * C to T
            self.proj= nn.ModuleList([
                nn.Linear(input_dim, n_outputs, bias=bias) for _ in range(channels)
            ])
        else:
            self.proj= nn.Linear(input_dim, n_outputs, bias=bias)

        # initialize Linear modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


    def forward(self, x):
        BC, _, _= x.shape  # (batch_size * channels/features, num_patches, d_model)

        if self.individual:
            B= BC // self.channels
            # flatten patches+embed into one vector per batch
            x_flat= x.reshape(B, self.channels, -1)
            # (batch_size, channels/features, num_patches * d_model)
            c_ts= []
            for i, c_proj in enumerate(self.proj):
                c_x_flat= x_flat[:, i, :]   # for each channel -> (batch_size, num_patches * d_model)
                if self.dropout is not None:
                    c_x_flat= self.dropout(c_x_flat)
                # project to T
                c_x_flat= c_proj(c_x_flat)  # (batch_size, seq_length)
                c_ts.append(c_x_flat)

            ts= torch.stack(c_ts, dim=1)    # (batch_size, channels/features, seq_length)
        else:
            # flatten patches+embed into one vector per batch
            x_flat= x.reshape(BC, -1)
            # (batch_size * channels/features, num_patches * d_model)
            if self.dropout is not None:
                x_flat= self.dropout(x_flat)
            # project to T
            ts= self.proj(x_flat)  # (batch_size * channels/features, seq_length)
            ts= ts.reshape(-1, self.channels, ts.size(-1))  # (batch_size, channels, seq_length)

        return ts.contiguous()  # (batch_size, channels, seq_length)



class DecoderHead(nn.Module):
    """
    Define the final projection head for Decoder-only (generative) models. (receives feature_maps
    to UnPatch, output shape -> [batch_size, channels/features, seq_length]).
    """

    def __init__(self, patch_width, n_patches, channels, d_model, d_ff, n_outputs, dropout=0.2,
                 head_type='mlp', bias=False, fine_tune=False, unpatch='conv') -> None:
        super(DecoderHead, self).__init__()
        # decoder projection head
        self.d_head= OutputBlock(True, d_model, d_ff, d_model, dropout, head_type, bias, fine_tune)
        if unpatch == 'linear':
            self.unpatch= LinearUnPatch(n_patches, channels, d_model, n_outputs, dropout, bias)
        else:
            self.unpatch= UnPatch(patch_width, channels, d_model, dropout, bias)


    def forward(self, x):
        x= self.d_head(x)

        return self.unpatch(x)



class EncoderSSLHead(nn.Module):
    """
    Define the final head for Encoder-only models under SSL pre-training mode (receives
    feature_maps to UnPatch, output shape -> [batch_size, channels/features, seq_length]
    when mask_type is not 'mae'; otherwise, outputs feature_maps).
    """

    def __init__(self, patch_width, n_patches, channels, d_model, d_ff, n_outputs, dropout=0.2,
                 mask_type='mae', head_type='mlp', bias=False, fine_tune=False, unpatch='conv') -> None:
        super(EncoderSSLHead, self).__init__()
        # encoder under SSL pre-training mode
        if mask_type == 'mae':
            self.e_head= nn.Identity()
        else:
            if unpatch == 'linear':
                unpatch= LinearUnPatch(n_patches, channels, d_model, n_outputs, dropout, bias)
            else:
                unpatch= UnPatch(patch_width, channels, d_model, dropout, bias)

            self.e_head= nn.Sequential(
                OutputBlock(True, d_model, d_ff, d_model, dropout, head_type, bias, fine_tune),
                unpatch,
            )


    def forward(self, x):
        x= self.e_head(x)

        return x



class EncoderHead(nn.Module):
    """
    Define the final head for Encoder-only models.
    - If forecasting=True: forecasting head to produce an entire sequence of future real values
    (receives feature_maps, output shape -> [batch_size, channels/features, n_outputs]);
    classification head otherwise (receives cls_tokens, output shape -> [batch_size, n_outputs]).
    """

    def __init__(self, forecasting, patch_width, n_patches, channels, d_model, d_ff, n_outputs,
                 dropout=0.2, head_type='mlp', bias=False, fine_tune=False, unpatch='conv') -> None:
        super(EncoderHead, self).__init__()
        self.forecasting= forecasting
        self.channels= channels

        if forecasting:
            # encoder forecasting head
            self.e_head= OutputBlock(True, d_model, d_ff, d_model, dropout, head_type, bias, fine_tune)
            if unpatch == 'linear':
                self.unpatch= LinearUnPatch(n_patches, channels, d_model, n_outputs, dropout, bias)
            else:
                self.unpatch= UnPatch(patch_width, channels, d_model, dropout, bias)
        else:
            # encoder classification head
            self.e_head= OutputBlock(
                False, channels*d_model, d_ff, n_outputs, dropout, head_type, bias, fine_tune
            ) if n_outputs > 0 else nn.Identity()


    def forward(self, x):
        if self.forecasting:
            x= self.e_head(x)
            return self.unpatch(x)

        x= x.reshape(x.shape[0], -1)
        return self.e_head(x)



""" 
# The Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE) Model
"""


class TSFTransformer(nn.Module):
    """
    Initializes a Time-Series Forecasting Transformer (TSFT) model.
    - width_factor controls the number of output forecast patches at each forward pass (when the
    model (non-SSL Encoders) is trained to predict more or less than the next single patch).
    - If multi_modal=True, we have an extra cross-attention module to incorporate exogenous
    covariates and allow for multi-modal learning.
    - If is_causal=True, we have a Decoder Transformer forecaster; otherwise, an Encoder Transformer.
    - If is_causal=False and mask_ratio > 0.0, applies patch masking for self-supervised (SSL)
    training objective (Decoders are naturally trained in SSL mode by using causal masks).
    - If is_causal=False, not SSL, and forecasting=True, we forecast future values; if is_causal=False,
    not SSL, and forecasting=False, we perform time-series classification.
    - norm_type (str): 'layer' for LayerNorm, 'rms' for RMSNorm, or 'dyt' for DynamicTanh.
    - If diff_attn=True, we use differential attention.
    - MoHE. ffn_type (str): the shared expert that can be 'mlp' for MLP-FFN, 'conv' for Conv-FFN,
    'dwconv' for DwConv-FFN, or 'fan' for FAN-FFN. experts_type (str): multiple routed experts that
    can be 'mlp' for MLP-FFN or 'fan' for FAN-FFN.
    - If rope_theta<=0, RoPE is disabled and the sinusoidal positional embedding is used.
    """

    def __init__(
            self, patch_width:int, channels:int, n_outputs:int, width_factor:float, multi_modal:bool,
            is_causal=False, forecasting=True, mask_ratio=0., mask_type='mae', n_layer=6, d_model=256, block_size=672,
            n_heads=8, n_kv_heads=4, d_ff=512, dropout=0.2, drop_path=0.3, norm_type='rms', flash_attn=True,
            diff_attn=False, ffn_type='dwconv', glu=False, n_experts=8, top_k_experts=2, experts_type='fan',
            output_head_type='mlp', fine_tune=True, unpatch='conv', bias=False, rope_theta=10000.0,
            use_input_norm=True, emb_norm_type='layer', output_head_dropout=0., cls_token=False
    ) -> None:
        super(TSFTransformer, self).__init__()
        assert patch_width > 0, "patch_width must be greater than zero"
        self.patch_width= int(patch_width)
        self.block_size = int(block_size)
        # ensure that the input time window is divisible by patch_width
        assert self.block_size % self.patch_width == 0, \
            f"block_size ({self.block_size}) must be divisible by patch_width ({self.patch_width})"
        assert self.block_size >= self.patch_width, \
            f"block_size ({self.block_size}) must be greater than or equal to patch_width ({self.patch_width})"
        # standardize text-based hyperparameters
        norm_type= norm_type.lower()
        emb_norm_type= emb_norm_type.lower()
        mask_type= mask_type.lower()
        ffn_type= ffn_type.lower()
        experts_type= experts_type.lower()
        output_head_type= output_head_type.lower()
        unpatch= unpatch.lower()

        self.n_outputs= int(n_outputs)
        self.is_causal= is_causal
        # ensure mask_ratio is only available for Encoders
        mask_ratio= mask_ratio if (not is_causal) else 0.0
        # ensure forecasting mode for Encoders under no SSL objective
        self.forecasting= forecasting if mask_ratio == 0.0 else False
        # ensure forecasting mode for Decoders
        self.forecasting= True if self.is_causal else self.forecasting
        # calculate the dimension of the patch space
        patch_dim= self.block_size // self.patch_width
        # control the width of the output patch (step horizon) during forecasting
        self.width_factor= width_factor
        # control the first and last positions of the time points generated during forecasting
        self.forecast_fst= 0
        self.forecast_lst= 0
        # "online" normalization to help the model focus on residual dynamics
        self.input_norm= InstanceNorm(dim2reduce=-1, eps=1e-5) if use_input_norm else None
        #self.input_norm= RevIN(channels, eps=1e-5) if use_input_norm else None

        # define the patch embedding for converting TS tokens
        if emb_norm_type == 'rms':
            self.t_embedding= PatchEmbeddingV3(self.patch_width, channels, d_model, dropout)
        else:
            self.t_embedding= PatchEmbedding(self.patch_width, channels, d_model, dropout)
        # define the patch embedding for exogenous calendar covariates
        if multi_modal:
            self.c_embedding= MultiModalEmbedding(self.patch_width, channels, d_model, dropout, emb_norm_type)
        else:
            self.c_embedding= None

        # define CLS token as a learnable parameter and initialize it with normal distributions
        if cls_token:
            self.cls_token= nn.Parameter(torch.randn(1, 1, d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
            patch_dim= patch_dim + 1
        else:
            self.cls_token= None

        # define SSL patch masking with a mask_ratio (Encoder-only)
        if mask_ratio > 0.0 and mask_type == 'mae':
            rope_theta= 0.0
            self.mask_layer= PatchMaskingMAE(mask_ratio)
        elif mask_ratio > 0.0:
            self.mask_layer= PatchMasking(mask_ratio)
        else:
            self.mask_layer= None

        if rope_theta <= 0.0:
            self.pos_emb= PositionalEmbedding(patch_dim, d_model)
        else:
            self.pos_emb= None

        # define the backbone transformer model
        self.backbone= TransformerModel(
            multi_modal, is_causal, n_layer, d_model, patch_dim, n_heads, n_kv_heads, d_ff, dropout,
            drop_path, norm_type, flash_attn, diff_attn, ffn_type, glu, n_experts, top_k_experts,
            experts_type, bias, rope_theta
        )

        patch_dim= patch_dim - 1 if cls_token else patch_dim
        # identity transformation (no change to the tensor)
        self.latent_space= nn.Identity()

        # define the final head according to the model and task objective
        if is_causal:
            self.head= DecoderHead(
                self.patch_width, patch_dim, channels, d_model, d_ff, self.block_size, output_head_dropout,
                output_head_type, bias, fine_tune, unpatch
            )
        else:
            if self.mask_layer is not None:
                self.head= EncoderSSLHead(
                    self.patch_width, patch_dim, channels, d_model, d_ff, self.block_size, output_head_dropout,
                    mask_type, output_head_type, bias, fine_tune, unpatch
                )
            else:
                out_dim= self.block_size if self.forecasting else self.n_outputs
                self.head= EncoderHead(
                    self.forecasting, self.patch_width, patch_dim, channels, d_model, d_ff, out_dim,
                    output_head_dropout, output_head_type, bias, fine_tune, unpatch
                )
        self.set_horizon(self.forecast_lst)

        self.config= BaseConfig(
            self.patch_width, channels, self.n_outputs, self.width_factor, multi_modal,
            self.is_causal, self.forecasting, mask_ratio, mask_type, n_layer, d_model, self.block_size,
            n_heads, n_kv_heads, d_ff, dropout, drop_path, norm_type, flash_attn, diff_attn,
            ffn_type, glu, n_experts, top_k_experts, experts_type, output_head_type, fine_tune,
            unpatch, bias, rope_theta, use_input_norm, emb_norm_type, output_head_dropout, cls_token
        )


    @classmethod
    def from_config(cls, cfg):
        cfg_map= asdict(cfg)
        # filter to only accept parameters that __init__ accepts
        sig= inspect.signature(cls.__init__)
        valid= set(sig.parameters) - {"self"}
        filtered= {k: v for k, v in cfg_map.items() if k in valid}

        return cls(**filtered)


    def disable_ssl_mode(self, head):
        assert not self.is_causal, "SSL mode is only available for Encoder-only models"
        assert isinstance(head, EncoderHead), "Head must be an EncoderHead for disabling SSL mode"
        self.mask_layer= None
        self.head= head
        self.forecasting= self.head.forecasting

        return "SSL mode disabled"


    def enable_ssl_mode(self, head, mask_ratio=0.2):
        assert not self.is_causal, "SSL mode is only available for Encoder-only models"
        assert isinstance(head, EncoderSSLHead), "Head must be an EncoderSSLHead for enabling SSL mode"
        self.mask_layer= PatchMasking(mask_ratio)
        self.head= head
        self.forecasting= False

        return f"SSL mode enabled with mask_ratio={mask_ratio}"


    def switch_model_type(self, head, flash_attn=None):
        # get the inverse of self.is_causal
        is_causal = not self.is_causal
        flash_attn= self.backbone.flash_attn if flash_attn is None else flash_attn

        if is_causal:
            assert isinstance(head, DecoderHead), "Head must be a DecoderHead for switching model type"
            self.head= head
            self.is_causal  = True
            self.forecasting= True
            self.mask_layer = None
        else:
            assert isinstance(head, EncoderHead), "Head must be an EncoderHead for switching model type"
            self.head= head
            self.is_causal  = False
            self.forecasting= self.head.forecasting
            self.mask_layer = None

        self.backbone.def_causal_mask(self.is_causal, flash_attn)

        return f"Model type switched to is_causal={self.is_causal}"


    def set_horizon(self, forecast_cut=0) -> None:
        """
        - forecast_cut controls the amount of last time points generated during each iteration
        when width_factor >= 1. If positive, cuts from the end to patch. If negative define a
        forecast step from the end, so it is possible to isolate up to the last predicted time
        point in a sequence.
        """
        width_factor= self.width_factor
        forecast_cut= int(forecast_cut)

        # Preliminaries to ensure alignment with training
        if self.is_causal:
            # prevent having more than next patch generation for Decoders
            width_factor= 1
            if forecast_cut > 0:
                forecast_cut= 0
            else:
                forecast_cut= forecast_cut if abs(forecast_cut) < self.patch_width else -self.patch_width
        elif self.mask_layer is not None:
            width_factor= self.block_size
            forecast_cut= 0

        if width_factor < 1:
            # decrease the generated forecast step (f_patch_width) from patch_width
            f_patch_width= self.patch_width
            end_f_patch_width= f_patch_width - int(width_factor * self.patch_width)

            if f_patch_width == end_f_patch_width:
                raise ValueError("width_factor too small; no effective reduction")
        else:
            # increase the generated forecast step (f_patch_width) from the prediction end
            f_patch_width= int(width_factor * self.patch_width)  # patch_width for Decoders
            end_f_patch_width= 0

            if forecast_cut > 0 and forecast_cut < f_patch_width:
                # decrease the size of the generated tail (from prediction end)
                end_f_patch_width= forecast_cut  # Encoders-only
            else:
                if forecast_cut < 0:
                    # define a forecast step from the last predicted time point
                    f_patch_width= abs(forecast_cut)  # up to patch_width for Decoders

        f_patch_width= min(f_patch_width, self.block_size)

        self.forecast_fst= f_patch_width
        self.forecast_lst= end_f_patch_width


    @torch.inference_mode()
    def forecast(self, ts, ts_mark=None, ts_mark_future=None):
        """
        Perform autoregressive forecasting patch-by-patch until get the forecast horizon.
        - when n_outputs==1, we perform time-series forecasting/regression.
        """
        assert not isinstance(self.head, EncoderSSLHead), "Forecasting is not enabled for EncoderSSLHead"
        assert self.forecasting, "Forecasting is not enabled"
        assert self.width_factor > 0.0, "width_factor must be greater than zero"

        f_patch_width    = int(self.forecast_fst)
        end_f_patch_width= int(self.forecast_lst)

        f_step= f_patch_width - end_f_patch_width
        assert f_step > 0, "f_step must be positive"
        n_patches= math.ceil(self.n_outputs / f_step)

        B, C, T= ts.size()
        assert T >= f_step, f"Initial sequence length {T} must be >= f_step {f_step}"
        round_t= int(n_patches * f_step)
        out= torch.zeros([B, C, round_t], device=ts.device, dtype=ts.dtype)

        self.eval()
        try:
            for i in range(n_patches):
                logits, *_= self.forward(ts, 0, ts_mark=ts_mark)
                if end_f_patch_width > 0:
                    future= logits[:, :, -f_patch_width:-end_f_patch_width]
                else:
                    future= logits[:, :, -f_patch_width:]  # get the last (newest) prediction
                ts= ts[:, :, f_step:]                      # drop the oldest forecasting step
                ts= torch.cat((ts, future), dim=-1)        # append the new prediction

                if ts_mark is not None and ts_mark_future is not None:
                    # we also have to slide the context, otherwise our time stamps will drift
                    # out of alignment
                    next_mark_future= ts_mark_future[:, :, i*f_step:(i+1)*f_step]
                    ts_mark= ts_mark[:, :, f_step:]
                    ts_mark= torch.cat([ts_mark, next_mark_future], dim=-1)

                out[:, :, i*f_step:(i+1)*f_step]= future  # store the forecasting

            last_token= round_t - self.n_outputs
            if last_token > 0:
                out= out[:, :, :-last_token]     # extract exactly the prediction horizon
        finally:
            self.train()

        return out


    def extra_repr(self):
        if self.is_causal:
            return "--- Decoder-only model with causal Attention ---"
        return "--- Encoder-only model ---"


    def forward(self, ts, start_pos=0, ts_mark=None):
        B, C, T= ts.size()  # ts (batch_size, channels/features, seq_length)
        assert T <= self.block_size, \
            f'Cannot forward sequence of length {T}, time window is only {self.block_size}'
        # when not in training mode inference is activated and set to 'True'
        inference= False if self.training else True

        if self.input_norm is not None:
            ts= self.input_norm(ts, 'norm')

        x= self.t_embedding(ts)  # (B * C, P, d_model)

        has_cls_tk= False
        if self.cls_token is not None:
            has_cls_tk= True
            # repeat a class token (CLS) for each sequence in the batch
            cls_tk= repeat(self.cls_token, '1 1 d -> b 1 d', b=B*C)
            # append CLS tokens with patch embeddings
            x= torch.cat((cls_tk, x), dim=1)  # (B * C, 1+P, d_model)

        if self.pos_emb is not None:
            x= self.pos_emb(x)

        # embed covariates (if any) to forward it into the cross-attention modules
        if (self.c_embedding is not None) and (ts_mark is not None):
            x_cross= self.c_embedding(ts_mark, ts)  # (B * C, P, d_model)

            if self.cls_token is not None:
                # repeat a CLS token also for each sequence in cross-attention modules
                cross_cls_tk= repeat(self.cls_token, '1 1 d -> b 1 d', b=B*C)
                # append CLS tokens with covariate patch embeddings
                x_cross= torch.cat((cross_cls_tk, x_cross), dim=1)  # (B * C, 1+P, d_model)

            if self.pos_emb is not None:
                x_cross= self.pos_emb(x_cross)
        else:
            x_cross= None

        # patch masking when in SSL mode
        mask, ids_restore= None, None
        if self.mask_layer is not None:
            if isinstance(self.mask_layer, PatchMaskingMAE):
                x, x_cross, mask, ids_restore= self.mask_layer(x, x_cross, has_cls_tk)
            else:
                x, x_cross= self.mask_layer(x, x_cross, has_cls_tk)

        # forward the embeddings through the transformer
        x, router_logits= self.backbone(x, x_cross, start_pos, inference)

        if self.is_causal or self.forecasting or (self.mask_layer is not None):
            # full feature map (representing individual patch embeddings)
            ft_map= x[:, 1:] if (self.cls_token is not None) else x
            out= self.latent_space(ft_map)  # (B * C, P, d_model)
        else:
            # out receives the class token for classification tasks only (encoder and no SSL head)
            cls_tk= x[:, 0] if (self.cls_token is not None) else x.mean(dim=1)
            out= self.latent_space(cls_tk.reshape(B, C, -1))  # (B, C, d_model)
        # the output head generates logits according to the task
        logits= self.head(out)

        cls_tk= x[:, 0] if (self.cls_token is not None) else None

        if (mask is not None) and (ids_restore is not None):
            return logits, router_logits, cls_tk, mask, ids_restore

        if (self.input_norm is not None) and logits.ndim == ts.ndim:
            logits= self.input_norm(logits, 'denorm')

        if cls_tk is not None:
            return logits, router_logits, cls_tk

        return logits, router_logits


    def setup_optimizer(self, learning_rate, weight_decay, betas=(0.9, 0.95), verbose=False):
        """
        Splitting up the parameters that should be weight decayed and those that should not.
        Thanks to @karpathy
        """
        # get the device of the model by checking one of its parameters
        device= next(self.parameters()).device
        # start with candidate parameters (that require grad)
        param_dict= {pn: p for pn, p in self.named_parameters()}
        param_dict= {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups: any 2D parameters will be weight dacayed, otherwise no
        # i.e., all weight tensors in matmuls + embeddings decay; all biases and norms do not
        # most of the parameters will be decayed
        decay_params  = [p for n, p in param_dict.items() if p.dim()>= 2]
        nodecay_params= [p for n, p in param_dict.items() if p.dim() < 2]  # one-dim tensors
        optim_groups= [
            {'params':   decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params  = sum(p.numel() for p in decay_params)
        num_nodecay_params= sum(p.numel() for p in nodecay_params)
        # create AdamW optimizer and use the fused version of it if available
        fused_available= 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # fused is faster when it is available and when running on cuda
        use_fused= fused_available and device.type == 'cuda'
        # create a AdamW PyTorch optimizer -- bug fix of Adam
        optimizer= torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, eps=1e-8, fused=use_fused
        )
        if verbose:
            print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
            print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")
            print(f"Using fused AdamW: {use_fused}")

        return optimizer
