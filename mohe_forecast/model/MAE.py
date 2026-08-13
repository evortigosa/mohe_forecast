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
from .Config import ModelConfig, TinyConfig, SmallConfig, BaseConfig, LargeConfig, UltraConfig



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
        # NOTE: the encoder strips the CLS from latent and returns it separately, so the tensor
        # fed to the decoder embedding never carries a CLS. We pass has_cls_tk=False and let
        # EmbeddingDecoderMAE.forward prepend the externally-supplied CLS token when present.
        self.decoder_embedding= EmbeddingDecoderMAE(
            encoder_config.d_model, decoder_config.d_model, False, encoder_config.bias
        )
        dec_block_size= self.encoder.block_size
        if cls_token:
            dec_block_size += self.encoder.cls_token.size(1) * patch_width

        dec_overrides= {
            "patch_width": patch_width, "channels": channels, "is_causal": False,
            "mask_ratio": 0., "mask_type": 'mae', "use_input_norm": False, "rope_theta": 0.,
            "cls_token": False, "emb_norm_type": None, "block_size": dec_block_size,
            # the MAE decoder operates in latent space and does not cross-attend to raw covariates
            "multi_modal": None,
        }
        self.decoder= TSFTransformer.from_config(replace(decoder_config, **dec_overrides))


    @property
    def config(self):
        """
        The encoder config. MohetsMAE wraps two backbones, so it has no single config: the encoder is
        the part that transfers to the downstream forecaster, so it is the one exposed here. The decoder
        config remains available as model.decoder.config.
        """
        return self.encoder.config


    @property
    def mask_layer(self):
        """ The mask_layer held by the encoder. """
        return self.encoder.mask_layer


    @property
    def mask_ratio(self) -> float:
        """
        The mask ratio currently held by the encoder's masking layer.
        NOTE: forward(ts, mask_ratio=0.75, ...) calls set_mask_ratio() on every pass, so this value
        reflects the last forward. Always pass mask_ratio explicitly rather than relying on the default.
        """
        return float(self.mask_layer.mask_ratio)


    def set_mask_ratio(self, mask_ratio=0.75) -> None:
        assert 0.0 <= mask_ratio < 1.0, "mask_ratio must be in [0, 1)"
        self.mask_layer.mask_ratio= float(mask_ratio)


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


    def patch_stats(self, ts):
        """
        Per-patch mean / stdev of the RAW series: exactly the statistics forward_loss uses to build the
        norm_pix target (same eps, same unbiased=False), so a reconstruction can be mapped back to the
        original units without duplicating (and drifting from) that math.
        ts:     (B, C, T)
        p_mean: (B*C, P, 1), p_stdev: (B*C, P, 1)
        """
        assert self.input_norm is not None, \
            "patch_stats is only meaningful when input_norm is enabled (norm_pix targets)"

        target= self.patchify(ts)
        p_mean = target.mean(dim=-1, keepdim=True)
        p_stdev= torch.sqrt(target.var(dim=-1, keepdim=True, unbiased=False) + self.input_norm.eps)

        return p_mean, p_stdev


    def unpatchify(self, x, p_mean=None, p_stdev=None):
        """
        Non-learnable unpatch method. The input image is restored from patches with no learnable parameters.
        x:  (B*C, P, patch_width)
        ts: (B, C, T)
        Two normalizations are in play and they must not be mixed up:
        - the encoder input is instance-normalized (input_norm, stats (B, C, 1));
        - the decoder target is per-patch normalized (norm_pix, stats (B*C, P, 1)), because forward_loss
          normalizes each target patch by its own mean/stdev.
        The decoder's logits therefore live in per-patch space whenever input_norm is enabled, so pass the
        ground-truth p_mean/p_stdev (see patch_stats) to invert those stats: the result is already in raw
        units and the instance denorm is correctly skipped. Denormalizing logits with the instance stats
        instead is a normalization mismatch and does not reconstruct the series.
        With input_norm disabled there is no per-patch norm, logits are raw patches, and this is a reshape.
        """
        BC, n, pw= x.size()
        p= self.encoder.t_embedding.patch_width
        assert pw == p and BC % self.channels == 0
        B= BC // self.channels

        denorm_pix= (p_mean is not None) and (p_stdev is not None)
        if denorm_pix:
            # invert the per-patch (norm_pix) normalization (done in patch space, before the reshape)
            x= x * p_stdev + p_mean

        x= x.reshape(BC, n, p, 1)  # (batch_size * channels/features, n_patches, patch_width, 1)
        x= torch.einsum('bnpc->bcnp', x)
        ts= x.reshape(B, self.channels * 1, n * p)

        # instance denorm only applies when x was in instance-normalized space: if the per-patch stats were
        # inverted above, ts is already in raw units and denormalizing again would corrupt it.
        if (not denorm_pix) and (self.input_norm is not None):
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


    def forward_decoder(self, latent, ids_restore, cls_token):
        """
        The MAE decoder operates in latent space and does not consume exogenous covariates: those
        are incorporated on the encoder side.
        """
        # cls token is handled by the decoder_embedding and stripped inside the decoder head.
        latent= self.decoder_embedding(latent, ids_restore, cls_token)
        logits, dec_router, *_= self.decoder(latent, ext_cls_token=cls_token)

        return logits, dec_router


    def forward_loss(self, ts, ts_pred, mask, criterion=nn.MSELoss(reduction='none')):
        """
        Take the MSE loss on removed patches.
        ts: (B, C, T)
        ts_pred: (BC, P, patch_width)
        mask: (BC, P) -> 0 is keep, 1 is removing
        """
        target= self.patchify(ts)
        # per-patch target normalization (norm_pix_loss, He et al. 2021). Computed inline with local stats
        # so it does not overwrite self.input_norm's encoder-input statistics (the [B, C, 1] stats stored
        # during forward_encoder, which unpatchify/denorm rely on). This is numerically identical to the
        # previous self.input_norm(target, 'norm') call (per-patch mean, biased variance, eps inside the sqrt)
        if self.input_norm is not None:
            p_mean = target.mean(dim=-1, keepdim=True)
            p_stdev= torch.sqrt(target.var(dim=-1, keepdim=True, unbiased=False) + self.input_norm.eps)
            target = (target - p_mean) / p_stdev

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
        logits, dec_router= self.forward_decoder(latent, ids_restore, cls_token)
        loss= self.forward_loss(ts, logits, mask, criterion)  # logits -> (B*C, P, patch_width)

        return loss, logits, mask, enc_router, dec_router



# =================================================================================================
# MoHETS-MAE architectures
# =================================================================================================
# Naming follows the MAE reference implementation (He et al., facebookresearch/mae):
#   mohets_mae_<encoder>_dec<width>d<depth>b -> encoder preset + decoder width / depth
# e.g. mohets_mae_base_dec128d2b == Base encoder (MoHE) + a 128-dim, 2-block dense decoder.
#
# Design rationale
# ----------------
# 1. The decoder is dense (n_experts=0 -> shared expert only, no router). Two reasons:
#    - it is discarded after pre-training, so MoE capacity there never transfers to the forecaster;
#    - at mask_ratio=0.75 about 76% of its input tokens are copies of one learned mask vector, which
#      makes token-level expert routing degenerate and would pollute the load-balancing statistics.
#    MoE capacity is therefore kept where it transfers: the encoder.
# 2. The decoder is shallow (2 blocks). He et al. found a 1-block decoder costs only ~0.1 pt of
#    fine-tuning accuracy vs 8 blocks (decoder depth matters mainly for linear probing). This pays off
#    more here than in ViT-MAE: the encoder sees only the visible patches (10 of 42 at mask_ratio=0.75)
#    while the decoder runs on all 42, so a decoder block costs ~4x an encoder block of equal width.
#    The *_dec*d4b / *_dec256d6b variants keep the preset's full depth.


def mohets_mae_decoder_config(preset:ModelConfig, n_layer:int=2, **overrides) -> ModelConfig:
    """
    Build a dense (n_experts=0) MAE-decoder config from an existing preset. Only these are changed:
    - n_layer                  : trimmed (see note 2 above);
    - n_experts / top_k_experts: 0 -> dense shared-expert FFN, no router (MoEFeedForward skips the
                                 routing path entirely and forwards through 'ffn_type' alone);
    - dropout / drop_path      : 0 -> a reconstruction module needs no stochastic depth.
    """
    assert preset.ffn_type is not None, "ffn_type must be set when n_experts=0: it is the dense FFN"

    config= replace(
        preset, n_layer=n_layer,
        n_experts=0, top_k_experts=0,  # dense: shared expert only, no routed experts, no router
        dropout=0.0, drop_path=0.0,    # reconstruction module: no stochastic depth / dropout
    )
    return replace(config, **overrides) if overrides else config


def _mohets_mae(encoder_config:ModelConfig, decoder_config:ModelConfig, patch_width:int=16,
                channels:int=1, mask_ratio:float=0.75, use_input_norm:bool=True,
                cls_token:bool=False, **encoder_overrides) -> MohetsMAE:
    """
    Shared builder. Extra keyword arguments are applied to the encoder config, so the backbone can
    be tuned without rebuilding it, e.g. mohets_mae_base(channels=7, n_experts=4, drop_path=0.0).
    """
    if encoder_overrides:
        encoder_config= replace(encoder_config, **encoder_overrides)

    return MohetsMAE(
        patch_width=patch_width, channels=channels, mask_ratio=mask_ratio, use_input_norm=use_input_norm,
        cls_token=cls_token, encoder_config=encoder_config, decoder_config=decoder_config,
    )


# --- recommended architectures: shallow (2-block) dense decoder ----------------------------------

def mohets_mae_small_dec64d2b(decoder_overrides:dict|None=None, **kwargs) -> MohetsMAE:
    """ Small encoder (MoHE 4L/128d) + Tiny-width dense decoder (64d, 4h/2kv, 2 blocks). """
    decoder= mohets_mae_decoder_config(TinyConfig(), 2, **(decoder_overrides or {}))
    return _mohets_mae(SmallConfig(), decoder, **kwargs)


def mohets_mae_base_dec128d2b(decoder_overrides:dict|None=None, **kwargs) -> MohetsMAE:
    """ Base encoder (MoHE 6L/256d) + Small-width dense decoder (128d, 4h/2kv, 2 blocks). """
    decoder= mohets_mae_decoder_config(SmallConfig(), 2, **(decoder_overrides or {}))
    return _mohets_mae(BaseConfig(), decoder, **kwargs)


def mohets_mae_large_dec128d2b(decoder_overrides:dict|None=None, **kwargs) -> MohetsMAE:
    """ Large encoder (MoHE 8L/384d) + Small-width dense decoder (128d, 4h/2kv, 2 blocks). """
    decoder= mohets_mae_decoder_config(SmallConfig(), 2, **(decoder_overrides or {}))
    return _mohets_mae(LargeConfig(), decoder, **kwargs)


def mohets_mae_ultra_dec128d2b(decoder_overrides:dict|None=None, **kwargs) -> MohetsMAE:
    """ Ultra encoder (MoHE 12L/512d) + Small-width dense decoder (128d, 4h/2kv, 2 blocks). """
    decoder= mohets_mae_decoder_config(SmallConfig(), 2, **(decoder_overrides or {}))
    return _mohets_mae(UltraConfig(), decoder, **kwargs)


def mohets_mae_ultra_dec256d2b(decoder_overrides:dict|None=None, **kwargs) -> MohetsMAE:
    """ Ultra encoder (MoHE 12L/512d) + Base-width dense decoder (256d, 8h/4kv, 2 blocks). """
    decoder= mohets_mae_decoder_config(BaseConfig(), 2, **(decoder_overrides or {}))
    return _mohets_mae(UltraConfig(), decoder, **kwargs)


# --- full-preset-depth variants: same pairings, decoder depth left at the preset's own n_layer ----
# (untrimmed decoders; useful for linear probing / frozen-representation quality, ~2-3x decoder cost)

def mohets_mae_small_dec64d4b(decoder_overrides:dict|None=None, **kwargs) -> MohetsMAE:
    """ Small encoder + full-depth Tiny decoder (64d, 4 blocks). """
    decoder= mohets_mae_decoder_config(TinyConfig(), 4, **(decoder_overrides or {}))
    return _mohets_mae(SmallConfig(), decoder, **kwargs)


def mohets_mae_base_dec128d4b(decoder_overrides:dict|None=None, **kwargs) -> MohetsMAE:
    """ Base encoder + full-depth Small decoder (128d, 4 blocks). """
    decoder= mohets_mae_decoder_config(SmallConfig(), 4, **(decoder_overrides or {}))
    return _mohets_mae(BaseConfig(), decoder, **kwargs)


def mohets_mae_large_dec128d4b(decoder_overrides:dict|None=None, **kwargs) -> MohetsMAE:
    """ Large encoder + full-depth Small decoder (128d, 4 blocks). """
    decoder= mohets_mae_decoder_config(SmallConfig(), 4, **(decoder_overrides or {}))
    return _mohets_mae(LargeConfig(), decoder, **kwargs)


def mohets_mae_ultra_dec128d4b(decoder_overrides:dict|None=None, **kwargs) -> MohetsMAE:
    """ Ultra encoder + full-depth Small decoder (128d, 4 blocks). """
    decoder= mohets_mae_decoder_config(SmallConfig(), 4, **(decoder_overrides or {}))
    return _mohets_mae(UltraConfig(), decoder, **kwargs)


def mohets_mae_ultra_dec256d6b(decoder_overrides:dict|None=None, **kwargs) -> MohetsMAE:
    """ Ultra encoder + full-depth Base decoder (256d, 6 blocks). """
    decoder= mohets_mae_decoder_config(BaseConfig(), 6, **(decoder_overrides or {}))
    return _mohets_mae(UltraConfig(), decoder, **kwargs)


# --- set recommended archs ------------------------------------------------------------------------
mohets_mae_small= mohets_mae_small_dec64d2b   # decoder:  64 dim, 2 blocks (Tiny width)
mohets_mae_base = mohets_mae_base_dec128d2b   # decoder: 128 dim, 2 blocks (Small width)
mohets_mae_large= mohets_mae_large_dec128d2b  # decoder: 128 dim, 2 blocks (Small width)
mohets_mae_ultra= mohets_mae_ultra_dec128d2b  # decoder: 128 dim, 2 blocks (Small width)
