# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Model Configuration
"""

from dataclasses import dataclass



@dataclass
class BaseConfig:
    """
    - patch_width (int): Default is 16. Defines the patch length.
    - channels (int): Default is 1. Defines the number of channels (features/time series).
    - n_outputs (int): Default is 96. Defines the forecasting horizon.
    - width_factor (float): Default is 1.5. Defines the output horizon as multiple of patch_width.
    - multi_modal (bool|None): Default is True. Enables the cross-attention module for covariate injection.
    - is_causal (bool): Default is False. Encoder Transformer (False) or Decoder Transformer (True).
    - forecasting (bool): Default is True. Defines the model as a forecaster.
    - mask_ratio (float): Default is 0. Enables representation learning.
    - mask_type (str): Default is 'mae' for a MAE-style masking. This can be 'mae' or 'random'.
    - n_layer (int): Default is 6. Defines the number of transformer layers/blocks.
    - d_model (int): Default is 256. Defines the model dimension.
    - block_size (int): Default is 672. Defines the max size of the look-back window.
    - n_heads (int): Default is 8. Defines the number of query heads.
    - n_kv_heads (int): Default is 4. Defines the number of key/value heads.
    - d_ff (int): Default is 512. Defines the hidden dimensionality for experts/FFN.
    - dropout (float): Default is 0.2. Defines the dropout rate.
    - drop_path (float): Default is 0.3. Defines the DropPath rate.
    - norm_type (str): Default is 'rms'. This can be 'layer', 'rms', 'dyt' (experimental).
    - flash_attn (bool): Default is True. Enables FlashAttention.
    - diff_attn (bool): Default is False. Enables Differential Attention (experimental).
    - ffn_type (str): Default is 'dwconv'. This can be 'mlp', 'conv', 'dwconv', 'fan'.
    - glu (bool): Default is False. Enables the Gated Linear Unit (GLU) architecture for experts.
    - n_experts (int): Default is 8. Defines the number of MoHE experts.
    - top_k_experts (int): Default is 2. Defines the number of MoHE activated experts.
    - experts_type (str): Default is 'fan'. This can be 'mlp', 'fan'.
    - output_head_type (str): Default is 'mlp'. This can be 'mlp', 'conv', 'dwconv', 'fan'.
    - fine_tune (bool): Default is True. When False, enables an extra layer before unpatching.
    - unpatch (str): Default is 'conv'. This can be 'mlp', 'conv'.
    - bias (bool): Default is False. Enables bias for all learning modules.
    - rope_theta (float): Default is 10000.0. RoPE base value
    - use_input_norm (bool): Default is True. Enables an "online" normalization from Non-stationary Transformer.
    - emb_norm_type (str|None): Default is 'layer'. This can be 'layer', 'rms', or None.
    - output_head_dropout (float): Default is 0. Dropout before unpatching head.
    - use_qk_norm (bool): Default is False. Enables the QK functional RMSNorm after RoPE.
    - headwise_attn_gate (bool): Default is False. Enables headwise attention gate.
    - cls_token (bool): Default is False. Enables an extra classification token.
    """
    patch_width:int= 16
    channels:int= 1
    n_outputs:int= 96
    width_factor:float= 1.5
    multi_modal:bool|None= True    # enables the cross-attention module
    is_causal:bool= False          # Encoder Transformer (False) or Decoder Transformer (True)
    forecasting:bool= True
    mask_ratio:float= 0.           # enables representation learning
    mask_type:str= 'mae'           # mae, random
    n_layer:int= 6
    d_model:int= 256
    block_size:int= 672
    n_heads:int= 8
    n_kv_heads:int= 4
    d_ff:int= 512
    dropout:float= 0.2
    drop_path:float= 0.3
    norm_type:str= 'rms'           # layer, rms, dyt
    flash_attn:bool= True          # enables FlashAttention
    diff_attn:bool= False          # enables Differential Attention (WIP)
    ffn_type:str= 'dwconv'         # mlp, conv, dwconv, fan
    glu:bool= False                # enables the Gated Linear Unit (GLU) architecture for experts
    n_experts:int= 8
    top_k_experts:int= 2
    experts_type:str= 'fan'        # mlp, fan
    output_head_type:str= 'mlp'    # mlp, conv, dwconv, fan
    fine_tune:bool= True           # enables an extra layer before unpatching
    unpatch:str= 'conv'            # mlp, conv
    bias:bool= False               # enables bias for all learning modules
    rope_theta:float= 10000.0      # RoPE base value
    use_input_norm:bool= True      # "online" normalization from Non-stationary Transformer
    emb_norm_type:str|None= 'layer'  # layer, rms
    output_head_dropout:float= 0.  # dropout before unpatching
    use_qk_norm:bool= False
    headwise_attn_gate:bool= False
    cls_token:bool= False


@dataclass
class TinyConfig(BaseConfig):
    patch_width:int= 8
    width_factor:float= 3
    n_layer:int= 4
    d_model:int= 64
    n_heads:int= 4
    n_kv_heads:int= 2
    d_ff:int= 128


@dataclass
class SmallConfig(BaseConfig):
    n_layer:int= 4
    d_model:int= 128
    n_heads:int= 4
    n_kv_heads:int= 2
    d_ff:int= 256


@dataclass
class LargeConfig(BaseConfig):
    n_layer:int= 8
    d_model:int= 384
    n_heads:int= 12
    n_kv_heads:int= 6
    d_ff:int= 768
