# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Normalization Modules
"""


import torch
import torch.nn as nn



class RMSNorm(nn.Module):
    """
    Root Mean Square normalization (RMSNorm).
    """

    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super(RMSNorm, self).__init__()
        self.normalized_shape= normalized_shape
        # scaling parameter weight initialized with ones and size equal to normalized_shape
        self.weight= nn.Parameter(torch.ones(normalized_shape))
        self.eps= eps


    def extra_repr(self):
        return f"({self.normalized_shape},), eps={self.eps}, elementwise_affine=True"


    def _norm(self, x):
        # compute the RMS norm along the last dimension
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


    def forward(self, x):
        x_norm= self._norm(x.float()).type_as(x)

        return x_norm * self.weight



class DynamicTanh(nn.Module):
    """
    Dynamic Tanh is an element-wise operation that replaces normalization layers in Transformers.
    - Defined in https://arxiv.org/abs/2503.10622
    """

    def __init__(self, normalized_shape, init_alpha_value=0.5):
        super().__init__()
        self.normalized_shape= normalized_shape
        self.init_alpha_value= init_alpha_value
        # scaling parameter alpha initialized with a single init_alpha_value
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha_value)
        self.weight= nn.Parameter(torch.ones(normalized_shape))
        self.bias  = nn.Parameter(torch.zeros(normalized_shape))


    def extra_repr(self):
        return f"({self.normalized_shape},), init_alpha_value={self.init_alpha_value}"


    def forward(self, x):
        # x has shape [batch_size, seq_length, d_model]
        x= torch.tanh(self.alpha * x)

        return x * self.weight + self.bias



"""
Instance Normalization
"""


class InstanceNorm(nn.Module):
    """
    Initializes an "online" normalization from Non-stationary Transformer.
    See https://arxiv.org/abs/2205.14415
    """

    def __init__(self, dim2reduce=-1, eps=1e-5) -> None:
        super(InstanceNorm, self).__init__()
        self.dim2reduce= dim2reduce
        self.eps= eps


    def extra_repr(self):
        return f"dim={self.dim2reduce}, eps={self.eps}"


    def forward(self, x, mode:str):
        if mode == 'norm':
            x= self._normalize(x)
        elif mode == 'denorm':
            x= self._denormalize(x)
        else: raise NotImplementedError

        return x


    def _normalize(self, x):
        self.mean= x.mean(dim=self.dim2reduce, keepdim=True).detach()
        x= x - self.mean
        self.stdev= torch.sqrt(
            torch.var(x, dim=self.dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x /= self.stdev

        return x


    def _denormalize(self, x):
        x= x * self.stdev + self.mean

        return x



class RevIN(nn.Module):
    """
    Cloned from https://openreview.net/forum?id=cGDAkQo1C0p
    Extensions:
    - Permute channels dimension of input tensor to match with TSFTransformer inputs.
    - Included extra_repr method.
    """

    def __init__(self, num_features:int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features= num_features
        self.eps= eps
        self.affine= affine
        if self.affine:
            self._init_params()

    def extra_repr(self):
        return f"num_features={self.num_features}, eps={self.eps}, elementwise_affine={self.affine}"

    def forward(self, x, mode:str):
        x= x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

        if mode == 'norm':
            self._get_statistics(x)
            x= self._normalize(x)
        elif mode == 'denorm':
            x= self._denormalize(x)
        else:
            raise NotImplementedError
        return x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight= nn.Parameter(torch.ones(self.num_features))
        self.affine_bias  = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce= tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev= torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x= x - self.mean
        x= x / self.stdev
        if self.affine:
            x= x * self.affine_weight
            x= x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x= x - self.affine_bias
            x= x / (self.affine_weight + self.eps*self.eps)
        x= x * self.stdev
        x= x + self.mean
        return x
