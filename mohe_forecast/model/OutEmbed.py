# -*- coding: utf-8 -*-
"""
# The Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE) Model
Output Embedding Modules
"""

import torch
import torch.nn as nn
from .MoHE import FeedForward, ConvFeedForward, DwConvFeedForward, FANLayer, FANFeedForward



"""
# Output Modules
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
