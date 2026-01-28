# -*- coding: utf-8 -*-
"""
# The Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE) Model
The MoHETS Model
"""

import math
import inspect
import torch
import torch.nn as nn
from einops import repeat
from dataclasses import asdict
from .Normalization import InstanceNorm
from .InEmbed import PatchEmbedding, PatchEmbeddingV3, MultiModalEmbedding
from .InEmbed import PatchMasking, PatchMaskingMAE, PositionalEmbedding
from .OutEmbed import DecoderHead, EncoderHead, EncoderSSLHead
from .TransformerModel import TransformerModel
from .Config import BaseConfig



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
        use_input_norm=True, emb_norm_type='layer', output_head_dropout=0., use_qk_norm=False, headwise_attn_gate=False,
        cls_token=False
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
        mask_type= mask_type.lower()
        norm_type= norm_type.lower()
        ffn_type= ffn_type.lower()
        experts_type= experts_type.lower()
        output_head_type= output_head_type.lower()
        unpatch= unpatch.lower()
        emb_norm_type= emb_norm_type.lower() if isinstance(emb_norm_type, str) else emb_norm_type

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
        use_input_norm = use_input_norm and (emb_norm_type is not None) and not (mask_ratio > 0.0 and mask_type == 'mae')
        self.input_norm= InstanceNorm(dim2reduce=-1, eps=1e-5) if use_input_norm else None
        #self.input_norm= RevIN(channels, eps=1e-5) if use_input_norm else None

        # define the patch embedding for converting TS tokens
        if emb_norm_type is None:
            self.t_embedding= nn.Identity()  # enable custom, external input embeddings
        elif emb_norm_type == 'rms':
            self.t_embedding= PatchEmbeddingV3(self.patch_width, channels, d_model, dropout)
        else:
            self.t_embedding= PatchEmbedding(self.patch_width, channels, d_model, dropout)
        # define the patch embedding for exogenous calendar covariates
        if multi_modal is None:
            self.c_embedding= nn.Identity()  # enable custom, external covariate embeddings
        elif multi_modal:
            emb_norm_type= 'rms' if emb_norm_type is None else emb_norm_type
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
            self.mask_layer= PatchMaskingMAE(mask_ratio, cls_token)
        elif mask_ratio > 0.0:
            self.mask_layer= PatchMasking(mask_ratio, cls_token)
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
            experts_type, bias, rope_theta, use_qk_norm, headwise_attn_gate
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
            unpatch, bias, rope_theta, use_input_norm, emb_norm_type, output_head_dropout, use_qk_norm,
            headwise_attn_gate, cls_token
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


    def forward(self, ts, start_pos=0, ts_mark=None, ext_cls_token=None):
        """
        - ts_mark is an optional input for exogenous covariates.
        - ext_cls_token is an optional input for an external CLS token.
        """
        B, C, T= ts.size()  # ts (batch_size, channels/features, seq_length)
        assert T <= self.block_size, \
            f'Cannot forward sequence of length {T}, time window is only {self.block_size}'
        # when not in training mode inference is activated and set to 'True'
        inference= False if self.training else True

        if self.input_norm is not None:
            ts= self.input_norm(ts, 'norm')

        x= self.t_embedding(ts)  # (B * C, P, d_model)

        if self.cls_token is not None:
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
                x, x_cross, mask, ids_restore= self.mask_layer(x, x_cross)
            else:
                x, x_cross= self.mask_layer(x, x_cross)

        # forward the embeddings through the transformer
        x, router_logits= self.backbone(x, x_cross, start_pos, inference)

        has_cls_tk= (self.cls_token is not None) or (ext_cls_token is not None)

        if self.is_causal or self.forecasting or (self.mask_layer is not None):
            # full feature map (representing individual patch embeddings)
            ft_map= x[:, 1:] if has_cls_tk else x
            out= self.latent_space(ft_map)  # (B * C, P, d_model)
        else:
            # out receives the class token for classification tasks only (encoder and no SSL head)
            cls_tk= x[:, 0] if has_cls_tk else x.mean(dim=1)
            out= self.latent_space(cls_tk.reshape(B, C, -1))  # (B, C, d_model)
        # the output head generates logits according to the task
        logits= self.head(out)

        cls_tk= x[:, 0] if has_cls_tk else None

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
            optim_groups, lr=learning_rate, betas=betas, eps=1e-10, fused=use_fused
        )
        if verbose:
            print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
            print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")
            print(f"Using fused AdamW: {use_fused}")

        return optimizer
