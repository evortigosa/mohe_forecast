# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Time Series Data Augmentation Module
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class TSAugmentation(nn.Module):
    """
    --- WIP ---
    Time series augmentation module for multivariate data.
    """

    def __init__(self, channels, max_epochs, jitter_sigma=0.02, jitter_prob=0.5, warp_strength=0.1,
                 warp_prob=0.3, mag_sigma=0.02, mag_prob=0.3) -> None:
        super(TSAugmentation, self).__init__()
        self.channels= int(channels)
        self.max_epochs= int(max_epochs)
        # augmentation magnitudes and base probabilities
        self.jitter_sigma= jitter_sigma
        self.jitter_prob = jitter_prob
        self.warp_strength= warp_strength
        self.warp_prob= warp_prob
        self.mag_sigma= mag_sigma
        self.mag_prob = mag_prob
        # epoch scheduling
        self.epoch= 0
        # store initial values so scheduling is deterministic immediately
        self.jitter_prob_init= float(jitter_prob)
        self.warp_prob_init= float(warp_prob)
        self.mag_prob_init= float(mag_prob)


    def _ensure_btc(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B,T,C) or (B,C,T); got shape {tuple(x.shape)}")
        # if last dim equals channels: (B, T, C) -> (B, T, C)
        if x.shape[2]== self.channels:
            return x, False
        # if middle dim equals channels: (B, C, T) -> (B, T, C)
        if x.shape[1]== self.channels:
            return x.transpose(1, 2), True

        raise ValueError(
            f"Cannot infer channel dimension for channels={self.channels}."
            f"x.shape = {tuple(x.shape)}"
        )


    def step_epoch(self) -> None:
        """
        Schedule augmentation probabilities to decrease over epochs.
        """
        self.epoch += 1
        t= min(max(self.epoch / max(1, self.max_epochs), 0.0), 1.0)  # fraction in [0,1]
        floor= 0.1
        self.jitter_prob= float(self.jitter_prob_init * (1.0 - t) + floor)
        self.warp_prob= float(self.warp_prob_init * (1.0 - t) + floor)
        self.mag_prob= float(self.mag_prob_init * (1.0 - t) + floor)
        # safety clamp
        self.jitter_prob= max(0.0, min(1.0, self.jitter_prob))
        self.warp_prob= max(0.0, min(1.0, self.warp_prob))
        self.mag_prob= max(0.0, min(1.0, self.mag_prob))


    def _jitter(self, x):
        """
        Add Gaussian noise to each timestep and channel.
        https://arxiv.org/abs/1706.00527
        """
        if torch.rand((), device=x.device).item() >= self.jitter_prob or self.jitter_sigma <= 0:
            return x
        noise= torch.randn_like(x) * self.jitter_sigma
        return x + noise


    def _time_warp(self, x):
        """
        Apply smooth temporal warping by interpolating stretched/compressed time axis.
        """
        if torch.rand((), device=x.device).item() >= self.warp_prob or self.warp_strength <= 0:
            return x
        B, T, C= x.shape
        device, dtype= x.device, x.dtype
        # base time grid (for generating the warp)
        t= torch.linspace(-1.0, 1.0, steps=T, device=device, dtype=dtype)  # (T,)
        # build a smooth positive velocity field g(t) per sample
        num_waves= 3  # 3 sine waves
        # per-sample random frequencies and phases
        freqs = torch.rand(B, num_waves, device=device, dtype=dtype) * 0.5 + 0.5      # (B, num_waves)
        phases= torch.rand(B, num_waves, device=device, dtype=dtype) * (2 * math.pi)  # (B, num_waves)
        # evaluate sum of sines -> shape (B, T)
        t_expand= t.view(1, 1, T)                   # (1,1,T)
        freqs = freqs.view(B, num_waves, 1)         # (B,num_waves,1)
        phases= phases.view(B, num_waves, 1)        # (B,num_waves,1)
        waves = torch.sin(2.0 * math.pi * freqs * t_expand + phases)  # (B, num_waves, T)
        waves_sum= waves.sum(dim=1)                 # (B, T)
        # a positive velocity field -> ensure > 0 (softplus is smooth and > 0)
        g= 1.0 + self.warp_strength * waves_sum     # (B, T)
        g= F.softplus(g) + 1e-6                     # (B, T) strictly > 0
        # integrate (cumulative sum) to get strictly increasing mapping use simple cumulative sum
        cum= torch.cumsum(g, dim=1)                 # (B, T)
        # normalize cumulative to [0, 1], then to [-1, 1]
        start= cum[:, 0:1]                          # (B,1)
        end= cum[:, -1:]                            # (B,1)
        norm= (cum - start) / (end - start + 1e-6)  # (B, T) in [0,1]
        t_warped= norm * 2.0 - 1.0                  # (B, T) in [-1,1]
        # expand to per-(batch,channel) mapping
        t_warped_bc= t_warped.unsqueeze(1).expand(B, C, T)      # (B, C, T)
        # construct grid for grid_sample: (B*C, T, 2)
        zeros= torch.zeros_like(t_warped_bc)                    # (B, C, T)
        # stack as (x, y) where x=0 (width), y=t_warped (height)
        grid = torch.stack([zeros, t_warped_bc], dim=-1)        # (B, C, T, 2)
        grid = grid.view(B * C, T, 1, 2)                        # (B*C, H_out=T, W_out=1, 2)
        # reshape input to (B*C, 1, H_in=T, W_in=1)
        x_reshaped= x.permute(0, 2, 1).reshape(B * C, 1, T, 1)  # (B*C, 1, T, 1)
        # sample
        x_warped= F.grid_sample(
            x_reshaped, grid, mode='bilinear', padding_mode='border', align_corners=True
        )  # (B*C, 1, T, 1)
        # reshape back to (B, T, C)
        return x_warped.view(B, C, T, 1).squeeze(-1).permute(0, 2, 1).contiguous()


    def _magnitude_warp(self, x, knot=4):
        """
        Apply magnitude warping by multiplying with a smooth random curve.
        """
        if torch.rand((), device=x.device).item() >= self.mag_prob or self.mag_sigma <= 0.0:
            return x
        B, T, C= x.shape
        device, dtype= x.device, x.dtype
        K= knot + 2
        # uniformly spaced time samples in [0,1]
        t    = torch.linspace(0.0, 1.0, steps=T, device=device, dtype=dtype)    # (T,)
        knots= 1.0 + self.mag_sigma * torch.randn((B, C, K), device=device, dtype=dtype)  # (B, C, K)
        # compute segment indices for each t (index into knots)
        j= torch.floor(t * (K - 1)).to(torch.long).clamp(0, K - 2)  # (T,)
        # expand to (B, C, T) for gather
        idx= j.view(1, 1, T).expand(B, C, T)                        # (B, C, T)
        y_j = torch.gather(knots, dim=2, index=idx)                 # (B, C, T)
        y_j1= torch.gather(knots, dim=2, index=idx + 1)             # (B, C, T)
        # alpha in [0, 1] for interpolation within each segment
        jf= j.to(dtype)
        alpha= (t - jf / (K - 1)) / (1.0 / (K - 1))                 # (T,)
        alpha= alpha.clamp(0.0, 1.0).view(1, 1, T).expand(B, C, T)  # (B,C,T)
        curve= (1.0 - alpha) * y_j + alpha * y_j1                   # (B,C,T)
        # apply curve: curve -> (B, T, C)
        curve= curve.permute(0, 2, 1).contiguous()                  # (B,T,C)

        return x * curve


    def forward(self, x):
        """
        Apply augmentations to input time series.
        """
        x, transposed= self._ensure_btc(x)
        x= self._jitter(x)
        x= self._time_warp(x)
        x= self._magnitude_warp(x)
        if transposed:
            x= x.transpose(1, 2)  # back to (B, C, T)

        return x.contiguous()
