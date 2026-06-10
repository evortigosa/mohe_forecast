# -*- coding: utf-8 -*-
"""
# The Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
The MoHETS-MAE Model
"""

import torch
import torch.nn as nn
from dataclasses import replace
from .Normalization import InstanceNorm
from .InEmbed import EmbeddingDecoderMAE
from .TSFT import TSFTransformer
from .Config import ModelConfig



class MohetsMAE(nn.Module):
    """
    Initializes a Masked Autoencoder (MAE) with MoHETS backbone.
    - encoder_config and decoder_config are dataclass objects to initialize the backbone.
    --- THIS IS A WIP ---
    """

    def __init__(self, patch_width:int, channels:int, mask_ratio:float, use_input_norm:bool,
                 cls_token:bool, encoder_config:ModelConfig, decoder_config:ModelConfig) -> None:
        super(MohetsMAE, self).__init__()
        self.channels= int(channels)
        assert 0.0 < mask_ratio < 1.0, "mask_ratio must be in (0, 1)"
        self.input_norm= InstanceNorm(dim2reduce=-1, eps=1e-5) if use_input_norm else None

        # MAE encoder side
        enc_overrides= {
            "patch_width": patch_width, "channels": channels, "is_causal": False,
            "mask_ratio": mask_ratio, "mask_type": 'mae', "use_input_norm": False,
            "cls_token": cls_token,
        }
        self.encoder= TSFTransformer.from_config(replace(encoder_config, **enc_overrides))

        # MAE decoder side
        self.decoder_embedding= EmbeddingDecoderMAE(
            encoder_config.d_model, decoder_config.d_model, cls_token, encoder_config.bias
        )
        dec_block_size= self.encoder.block_size
        if cls_token:
            dec_block_size += self.encoder.cls_token.size(1) * patch_width

        dec_overrides= {
            "patch_width": patch_width, "channels": channels, "is_causal": False,
            "mask_ratio": 0., "mask_type": 'mae', "use_input_norm": False,
            "cls_token": False, "emb_norm_type": None, "block_size": dec_block_size,
        }
        self.decoder= TSFTransformer.from_config(replace(decoder_config, **dec_overrides))


    def set_mask_ratio(self, mask_ratio=0.75) -> None:
        assert 0.0 <= mask_ratio < 1.0, "mask_ratio must be in [0, 1)"
        self.encoder.mask_layer.mask_ratio= mask_ratio


    def patchify(self, ts):
        """
        Non-learnable patch method. The input image is divided into patches and flattened with no learnable
        parameters like in Embedding modules.
        ts: (B, C, T)
        x:  (B*C, P, patch_width)
        """
        B, C, T= ts.size()
        p= self.encoder.t_embedding.patch_width
        assert T % p == 0, \
            f"Sequence length ({T}) must be divisible by patch_width ({p})"

        n= T // p
        x= ts.reshape(B * C, 1, n, p)  # (batch_size * channels/features, 1, n_patches, patch_width)
        x= torch.einsum('bcnp->bnpc', x)
        x= x.reshape(B * C, n, p * 1)

        return x


    def unpatchify(self, x):
        """
        Non-learnable unpatch method. The input image is restored from patches with no learnable parameters.
        x:  (B*C, P, patch_width)
        ts: (B, C, T)
        """
        BC, n, pw= x.size()
        p= self.encoder.t_embedding.patch_width
        assert pw == p and BC % self.channels == 0
        B= BC // self.channels

        x= x.reshape(BC, n, p, 1)  # (batch_size * channels/features, n_patches, patch_width, 1)
        x= torch.einsum('bnpc->bcnp', x)
        ts= x.reshape(B, self.channels * 1, n * p)

        if (self.input_norm is not None) and x.ndim == ts.ndim:
            ts= self.input_norm(ts, 'denorm')

        return ts


    def forward_encoder(self, ts, mask_ratio, ts_mark=None):
        B, C, T= ts.size()  # ts (batch_size, channels/features, seq_length)
        assert T <= self.encoder.block_size, \
            f'Cannot forward sequence of length {T}, time window is only {self.encoder.block_size}'

        if self.input_norm is not None:
            ts= self.input_norm(ts, 'norm')

        self.set_mask_ratio(mask_ratio)
        latent, enc_router, cls_token, mask, ids_restore= self.encoder(ts, ts_mark=ts_mark)

        return latent, enc_router, cls_token, mask, ids_restore


    def forward_decoder(self, latent, ids_restore, cls_token, ts_mark=None):
        # cls token will be handled by the decoder_embedding (when it is present)
        latent= self.decoder_embedding(latent, ids_restore, cls_token)
        # discard cls token when it is present
        logits, dec_router, *_= self.decoder(latent, ts_mark=ts_mark, ext_cls_token=cls_token)

        return logits, dec_router


    def forward_loss(self, ts, ts_pred, mask, criterion=nn.MSELoss(reduction='none')):
        """
        Take the MSE loss on removed patches.
        ts: (B, C, T)
        ts_pred: (BC, P, patch_width)
        mask: (BC, P) -> 0 is keep, 1 is removing
        """
        target= self.patchify(ts)
        # "online" normalization
        if self.input_norm is not None:
            target= self.input_norm(target, 'norm')

        loss= (ts_pred - target)**2 if criterion is None else criterion(ts_pred, target)
        loss= loss.mean(dim=-1)  # (BC, P) mean loss per patch
        mask_sum= mask.sum()
        if mask_sum.item() > 0.:
            loss= (loss * mask).sum() / mask_sum  # mean loss on removed patches
        else:
            loss= loss.mean()  # mean loss on all patches (no removed patches)

        return loss


    def forward(self, ts, mask_ratio=0.75, ts_mark=None, criterion=None):
        """
        - ts_mark is an optional input for exogenous covariates.
        """
        latent, enc_router, cls_token, mask, ids_restore= self.forward_encoder(ts, mask_ratio, ts_mark)
        logits, dec_router= self.forward_decoder(latent, ids_restore, cls_token, ts_mark)
        loss= self.forward_loss(ts, logits, mask, criterion)  # logits -> (B*C, P, patch_width)

        return loss, logits, enc_router, dec_router
