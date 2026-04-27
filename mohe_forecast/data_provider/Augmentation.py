# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Time Series Data Augmentation Module
"""

import math
import torch
import torch.nn.functional as F



class TSAugmentation:
    """
    --- WIP ---
    Time series augmentation module for multivariate data.
    - augment_channels is a list of channel indices to augment, i.e., keep the remaining channels
    unchanged. Do not apply augmentation in (blindly) channels with:
    - continuous target variables
    - observed covariates
    - calendar/time covariates
    - binary indicators
    - missingness masks
    - static identifiers
    For multivariate forecasting:
    - jitter: yes, mild, after normalization
    - magnitude warp: yes, mild, only for continuous channels
    - time warp: optional, use carefully
    - calendar/time covariates: never augment
    - target consistency: augment full context+future before splitting, or use weak input-side noise
    to avoid creating artificial pairs where the input dynamics no longer match the future target dynamics.
    """

    def __init__(self, channels, max_epochs, jitter_sigma=0.02, jitter_prob=0.5, warp_strength=0.1,
                 warp_prob=0.3, mag_sigma=0.02, mag_prob=0.3, augment_channels=None, layout="auto") -> None:
        self.channels= int(channels)
        self.max_epochs= int(max_epochs)
        self.augment_channels= augment_channels
        self.layout= layout
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
        self.warp_prob_init  = float(warp_prob)
        self.mag_prob_init   = float(mag_prob)


    def _ensure_btc(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor; got shape {tuple(x.shape)}")

        if self.layout == "BTC":
            if x.shape[2] != self.channels:
                raise ValueError(f"Expected (B,T,C) with C={self.channels}; got {tuple(x.shape)}")
            return x, False  # last dim equals channels: (B, T, C) -> (B, T, C)

        if self.layout == "BCT":
            if x.shape[1] != self.channels:
                raise ValueError(f"Expected (B,C,T) with C={self.channels}; got {tuple(x.shape)}")
            return x.transpose(1, 2), True  # middle dim equals channels: (B, C, T) -> (B, T, C)

        if self.layout == "auto":
            if x.shape[2] == self.channels and x.shape[1] == self.channels:
                raise ValueError(
                    f"Ambiguous shape {tuple(x.shape)}: both dim 1 and dim 2 match channels={self.channels}. "
                    "Use layout='BTC' or layout='BCT'."
                )
            if x.shape[2] == self.channels:  # if last dim equals channels: (B, T, C) -> (B, T, C)
                return x, False
            if x.shape[1] == self.channels:  # if middle dim equals channels: (B, C, T) -> (B, T, C)
                return x.transpose(1, 2), True

        raise ValueError(f"Cannot infer layout for shape {tuple(x.shape)} and channels={self.channels}")


    def set_epoch(self, epoch:int) -> None:
        self.epoch= int(epoch)


    def step_epoch(self, floor:float=0.1) -> None:
        """
        Schedule augmentation probabilities to decrease over epochs.
        """
        assert floor >= 0., "floor must be a non-negative value"
        self.epoch += 1
        t= min(max(self.epoch / max(1, self.max_epochs), 0.0), 1.0)

        def decay(p0):
            p0= float(p0)
            # avoid increasing probabilities if p0 < floor
            local_floor= min(floor, p0)
            p= local_floor + (p0 - local_floor) * (1.0 - t)
            return max(0.0, min(1.0, p))

        self.jitter_prob= decay(self.jitter_prob_init)
        self.warp_prob  = decay(self.warp_prob_init)
        self.mag_prob   = decay(self.mag_prob_init)


    def _jitter(self, x):
        """
        Add Gaussian noise to each timestep and channel (per-sample augmentation).
        Best practice: standardize/normalize training data first, then apply jitter in normalized
        space.
        https://arxiv.org/abs/1706.00527
        """
        if self.jitter_sigma <= 0 or self.jitter_prob <= 0:
            return x
        B= x.shape[0]
        mask= (torch.rand(B, 1, 1, device=x.device) < self.jitter_prob).to(x.dtype)
        noise= torch.randn_like(x) * self.jitter_sigma

        return x + mask * noise


    def _time_warp(self, x):
        """
        Apply smooth temporal warping by interpolating stretched/compressed time axis.
        """
        if torch.rand((), device=x.device).item() >= self.warp_prob or self.warp_strength <= 0:
            return x
        B, T, C= x.shape
        # compute the augmentation in FP32 and cast back at the end
        orig_dtype= x.dtype
        x= x.float()
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
        # a positive velocity field -> ensure > 0
        waves_sum= waves_sum / math.sqrt(num_waves)
        g= torch.exp(self.warp_strength * waves_sum)  # (B, T) strictly > 0
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
        return x_warped.view(B, C, T, 1).squeeze(-1).permute(0, 2, 1).contiguous().to(orig_dtype)


    def _magnitude_warp(self, x, knot=4):
        """
        Apply magnitude warping by multiplying with a piecewise-linear random curve.
        """
        if torch.rand((), device=x.device).item() >= self.mag_prob or self.mag_sigma <= 0.0:
            return x
        B, T, C= x.shape
        # compute the augmentation in FP32 and cast back at the end
        orig_dtype= x.dtype
        x= x.float()
        device, dtype= x.device, x.dtype
        K= knot + 2
        # uniformly spaced time samples in [0,1]
        t    = torch.linspace(0.0, 1.0, steps=T, device=device, dtype=dtype)    # (T,)
        knots= torch.exp(self.mag_sigma * torch.randn((B, C, K), device=device, dtype=dtype))  # (B, C, K)
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

        return (x * curve).to(orig_dtype)


    def __call__(self, x, jitter=True, time_warp=True, magnitude_warp=True):
        """
        Apply augmentations to input time series.
        """
        x, transposed= self._ensure_btc(x)

        if self.augment_channels is not None:
            out= x.clone()
            xa = x[:, :, self.augment_channels]

            if jitter:
                xa= self._jitter(xa)
            if time_warp:
                xa= self._time_warp(xa)
            if magnitude_warp:
                xa= self._magnitude_warp(xa)

            out[:, :, self.augment_channels]= xa
            x= out
        else:
            if jitter:
                x= self._jitter(x)
            if time_warp:
                x= self._time_warp(x)
            if magnitude_warp:
                x= self._magnitude_warp(x)

        if transposed:
            x= x.transpose(1, 2)

        return x.contiguous()
