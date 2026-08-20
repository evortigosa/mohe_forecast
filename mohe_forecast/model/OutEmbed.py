# -*- coding: utf-8 -*-
"""
# The Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE) Model
Output Embedding Modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
                 fine_tune=False, is_causal=False) -> None:
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
                self.out_proj= DwConvFeedForward(d_model, d_ff, n_outputs, dropout, glu=False, bias=bias, is_causal=is_causal)
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

    def __init__(self, patch_width, channels, d_model, dropout=0.2, bias=False, ch_independence=True,
                 is_causal=False) -> None:
        super(UnPatchV3, self).__init__()
        assert d_model % patch_width == 0, "d_model must be divisible by patch_width"
        self.channels= channels
        pw_d_model= round_channels(d_model // patch_width)
        hidden_dim= round_channels(pw_d_model * 4)
        out_channels= 1 if ch_independence else channels
        # calculate kernel_size and padding of the depthwise conv based on patch_width
        dks= min(max(((patch_width // 2) - 1), 1), 7)  # [1, 7]
        dks= dks - 1 if dks % 2 == 0 else dks
        # The ConvTranspose1d is non-overlapping (kernel=stride=patch_width), so each patch decodes
        # into its own patch_width samples -- safe. The depthwise conv that follows runs over the
        # upsampled sample axis, and with symmetric padding (dpd= dks//2) output sample t reads
        # sample t+dpd, which at a patch boundary belongs to the next patch. Under is_causal that
        # is a future leak on exactly the samples the forecast is read from. causal=True left-pads
        # instead; parameter shapes are unchanged
        self.is_causal= bool(is_causal)
        self.left_pad= (dks - 1) if self.is_causal else 0
        dpd= 0 if self.is_causal else (dks // 2)

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
        x= x.permute(0, 2, 1)  # (B, P, C) -> (B, C, P)
        if self.is_causal:
            # left-pad at the sample level: the ConvTranspose1d expands patches by patch_width, so
            # the pad is applied after upsampling, inside the sequential, via a functional pad here
            ts= self.unpatch[0](x)             # ConvTranspose1d -> samples
            ts= self.unpatch[1](ts)            # GELU
            ts= F.pad(ts, (self.left_pad, 0))  # causal left pad before the depthwise conv
            for layer in self.unpatch[2:]:
                ts= layer(ts)
        else:
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

    def __init__(self, patch_width, channels, d_model, dropout=0.2, bias=False, ch_independence=True,
                 is_causal=False) -> None:
        super(UnPatch, self).__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        self.channels= channels
        hidden_dim= round_channels(d_model // 4)
        out_channels= 1 if ch_independence else channels
        # calculate kernel_size and padding of the depthwise conv based on patch_width
        dks= min(max(((patch_width // 2) - 1), 1), 7)  # [1, 7]
        dks= dks - 1 if dks % 2 == 0 else dks
        # The ConvTranspose1d is non-overlapping (kernel=stride=patch_width), so each patch decodes
        # into its own patch_width samples -- safe. The depthwise conv that follows runs over the
        # upsampled sample axis, and with symmetric padding (dpd= dks//2) output sample t reads
        # sample t+dpd, which at a patch boundary belongs to the next patch. Under is_causal that
        # is a future leak on exactly the samples the forecast is read from. causal=True left-pads
        # instead; parameter shapes are unchanged
        self.is_causal= bool(is_causal)
        self.left_pad= (dks - 1) if self.is_causal else 0
        dpd= 0 if self.is_causal else (dks // 2)

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
        x= x.permute(0, 2, 1)  # (B, P, C) -> (B, C, P)
        if self.is_causal:
            # left-pad at the sample level: the ConvTranspose1d expands patches by patch_width, so
            # the pad is applied after upsampling, inside the sequential, via a functional pad here
            ts= self.unpatch[0](x)             # ConvTranspose1d -> samples
            ts= self.unpatch[1](ts)            # GELU
            ts= F.pad(ts, (self.left_pad, 0))  # causal left pad before the depthwise conv
            for layer in self.unpatch[2:]:
                ts= layer(ts)
        else:
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

    def __init__(self, n_patches, channels, d_model, n_outputs, dropout=0.2, bias=False, ch_independence=True,
                 individual=False) -> None:
        super(LinearUnPatch, self).__init__()
        self.channels  = channels
        self.individual= individual
        self.out_channels= 1 if ch_independence else channels
        self.n_patches= n_patches
        input_dim= n_patches * d_model

        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None
        if self.individual:
            # individual linear mapping from P * C to T
            self.proj= nn.ModuleList([
                nn.Linear(input_dim, n_outputs, bias=bias) for _ in range(channels)
            ])
        else:
            n_outputs= n_outputs * self.out_channels
            self.proj= nn.Linear(input_dim, n_outputs, bias=bias)

        # initialize Linear modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


    def _fit_patches(self, x):
        """
        During autoregressive generation the context length (the patch count P) varies, while the
        flatten-linear projection expects exactly n_patches. We keep the most recent n_patches when P
        is larger, and left-pad with zeros (on the oldest side) when P is smaller, so the real patches
        always occupy the most-recent projection slots. When P == n_patches (the standard fixed-window
        training case) this is a no-op, so training is unchanged.
        """
        P= x.size(1)
        if P == self.n_patches:
            return x
        if P > self.n_patches:
            # keep the most recent patches
            return x[:, -self.n_patches:, :].contiguous()
        # P < n_patches -> left-pad the patch dimension with zeros.
        # F.pad order for a 3D tensor is (d_model_left, d_model_right, patch_left, patch_right)
        return F.pad(x, (0, 0, self.n_patches - P, 0))


    def forward(self, x):
        # fit the (possibly variable) patch count to the fixed size the projection expects
        x= self._fit_patches(x)
        B, _, _= x.shape  # (batch_size * channels/features, num_patches, d_model)

        if self.individual:
            if self.out_channels == 1:
                B= B // self.channels
                # flatten patches+embed into one vector per batch
                x_flat= x.reshape(B, self.channels, -1)
            else:
                x_flat= x.reshape(B, -1).unsqueeze(1).repeat(1, self.channels, 1)
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
            x_flat= x.reshape(B, -1)
            # (batch_size * channels/features, num_patches * d_model)
            if self.dropout is not None:
                x_flat= self.dropout(x_flat)
            # project to T
            ts= self.proj(x_flat)  # (batch_size * channels/features, seq_length)
            ts= ts.reshape(-1, self.channels, ts.size(-1) // self.out_channels)

        return ts.contiguous()  # (batch_size, channels/features, seq_length)



class DecoderHead(nn.Module):
    """
    Define the final projection head for Decoder-only (generative) models. (receives feature_maps
    to UnPatch, output shape -> [batch_size, channels/features, seq_length]).
    """

    def __init__(self, patch_width, n_patches, channels, d_model, d_ff, n_outputs, dropout=0.2,
                 head_type='mlp', bias=False, fine_tune=False, ch_independence=True) -> None:
        super(DecoderHead, self).__init__()
        # decoder projection head
        self.d_head= OutputBlock(True, d_model, d_ff, d_model, dropout, head_type, bias, fine_tune, is_causal=True)
        self.unpatch= UnPatch(patch_width, channels, d_model, dropout, bias, ch_independence, is_causal=True)


    def forward(self, x):
        x= self.d_head(x)

        return self.unpatch(x)



class EncoderHead(nn.Module):
    """
    Define the final head for Encoder-only models.
    - If forecasting=True: forecasting head to produce an entire sequence of future real values
    (receives feature_maps, output shape -> [batch_size, channels/features, n_outputs]);
    classification head otherwise (receives cls_tokens, output shape -> [batch_size, n_outputs]).
    """

    def __init__(self, forecasting, patch_width, n_patches, channels, d_model, d_ff, n_outputs, dropout=0.2,
                 mask_type='mae', head_type='mlp', bias=False, fine_tune=False, unpatch='conv',
                 ch_independence=True) -> None:
        super(EncoderHead, self).__init__()
        self.forecasting= forecasting
        self.mask_type= mask_type

        # MAE head (decoder side) under SSL pre-training mode
        if mask_type == 'mae':
            self.e_head= nn.Linear(d_model, patch_width, bias=bias)  # embedding to patch
        elif forecasting:
            # encoder forecasting head
            self.e_head= OutputBlock(True, d_model, d_ff, d_model, dropout, head_type, bias, fine_tune)
            if unpatch == 'linear':
                self.unpatch= LinearUnPatch(n_patches, channels, d_model, n_outputs, dropout, bias, ch_independence)
            else:
                self.unpatch= UnPatch(patch_width, channels, d_model, dropout, bias, ch_independence)
        else:
            # encoder classification head
            channels= channels if ch_independence else 1
            self.e_head= OutputBlock(
                False, channels*d_model, d_ff, n_outputs, dropout, head_type, bias, fine_tune
            ) if n_outputs > 0 else nn.Identity()


    def forward(self, x):
        if self.mask_type == 'mae':
            return self.e_head(x)

        if self.forecasting:
            x= self.e_head(x)
            return self.unpatch(x)

        x= x.reshape(x.shape[0], -1)
        return self.e_head(x)



class EncoderSSLHead(nn.Module):
    """
    Define the final head for Encoder-only models under SSL pre-training mode (receives
    feature_maps to UnPatch, output shape -> [batch_size, channels/features, seq_length]
    when mask_type is not 'mae'; otherwise, outputs feature_maps).
    """

    def __init__(self, patch_width, n_patches, channels, d_model, d_ff, n_outputs, dropout=0.2,
                 mask_type='mae', head_type='mlp', bias=False, fine_tune=False, unpatch='conv',
                 ch_independence=True) -> None:
        super(EncoderSSLHead, self).__init__()
        # MAE head (encoder side) under SSL pre-training mode
        if mask_type == 'mae':
            self.e_head= nn.Identity()
        else:
            if unpatch == 'linear':
                unpatch= LinearUnPatch(n_patches, channels, d_model, n_outputs, dropout, bias, ch_independence)
            else:
                unpatch= UnPatch(patch_width, channels, d_model, dropout, bias, ch_independence)

            self.e_head= nn.Sequential(
                OutputBlock(True, d_model, d_ff, d_model, dropout, head_type, bias, fine_tune),
                unpatch,
            )


    def forward(self, x):
        x= self.e_head(x)

        return x
