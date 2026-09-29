# -*- coding: utf-8 -*-
"""
MoHETS --- velocity (rate-of-change) reconstruction loss.
"""

import torch
import torch.nn as nn


class VelocityLoss(nn.Module):
    """
    Squared error between the within-patch first differences of prediction and target.

        L= lambda * (dx_pred - dx_target)^2,   dx[i]= x[i + stride] - x[i]

    Differences are taken along the last axis only. For MohetsMAE the tensors reaching the criterion
    are (B*C, P, patch_width) in original temporal order (forward_decoder applies ids_restore before
    forward_loss), so the last axis is contiguous sampling steps inside one patch. Differencing
    ACROSS a patch boundary is deliberately not done: under per-patch masking, patch k may be a
    prediction while k+1 is ground truth, and which is which changes with every mask draw.
    Notes:
    - Level-invariant. Adding a constant to the prediction leaves this loss unchanged, so it scores
      dynamics only. That is the point: MSE alone cannot distinguish "right level, wrong shape" from
      "wrong level, right shape".
    - norm_pix costs nothing. With target= (x - mu_p) / sigma_p the patch mean cancels under
      differencing, leaving dx / sigma_p, i.e. a scale-normalised rate of change.
    - stride. At 15-min sampling with CGM MARD ~9%, consecutive readings carry independent
      measurement noise and differencing amplifies it (Var[dx]= 2*sigma^2). stride=2 gives 30-min
      differences and is less noise-sensitive.
    - Computed in float32: differencing two nearby O(1) values in bf16 loses several bits to
      cancellation.
    Args:
    - stride (int): difference lag in samples along the last axis.
    - reduction (str): 'none' -> (..., K - stride) elementwise; 'mean' / 'sum' -> scalar.
    Composing with a base loss (shapes differ, so reduce the last axis first). lam is applied
    inside this module, so do NOT scale the result again:
        mse, vel= nn.MSELoss(reduction='none'), VelocityLoss(lam=0.1)
        e= mse(pred, target)                             # (B*C, P, patch_width)
        v= vel(pred, target).mean(dim=-1, keepdim=True)  # (B*C, P, 1), already weighted by lam
        loss= e + v                                      # (B*C, P, patch_width)
    so that loss.mean(dim=-1) == mse_per_patch + lam * vel_per_patch, leaving
    MohetsMAE.forward_loss's masking and reduction untouched.
    """

    def __init__(self, stride:int=1, lam:float=0.1, reduction:str="none") -> None:
        super().__init__()
        if not isinstance(stride, int) or stride < 1:
            raise ValueError(f"stride must be a positive int, got {stride!r}.")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean' or 'sum', got {reduction!r}.")
        if not isinstance(lam, (int, float)) or lam < 0.0:
            raise ValueError(f"lam must be a non-negative number, got {lam!r}; a negative weight "
                             "would reward mismatched dynamics and leave the loss unbounded below.")
        
        self.stride= stride
        self.lam= lam
        self.reduction= reduction


    def extra_repr(self) -> str:
        return f"stride={self.stride}, lambda={self.lam}, reduction='{self.reduction}'"


    def forward(self, pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}.")
        if pred.shape[-1] <= self.stride:
            raise ValueError(
                f"last axis is {pred.shape[-1]} but stride is {self.stride}: no differences exist. "
                "Reduce stride or increase patch_width."
            )

        s= self.stride
        d_pred=   pred[..., s:].float() -   pred[..., :-s].float()
        d_true= target[..., s:].float() - target[..., :-s].float()
        err= self.lam * ((d_pred - d_true) ** 2)

        if self.reduction == "mean":
            return err.mean()
        if self.reduction == "sum":
            return err.sum()
        
        return err.to(pred.dtype)



"""
MoHETS --- composed reconstruction criterion (pointwise error + velocity penalty).
"""


class ReconstructionLoss(nn.Module):
    """
    Element-wise reconstruction criterion:

        L_ij= base(pred, target)_ij + mean_k( velocity(pred, target)_ik )

    so that forward_loss's own reduction gives

        L.mean(dim=-1) == base_per_patch + lam * velocity_per_patch

    and its masking, mask_sum==0 fallback and (loss*mask).sum()/mask_sum stay untouched. lam lives
    inside VelocityLoss, so the result is already weighted -- do not scale it again here.
    Args:
    - base (nn.Module | None): pointwise term, reduction='none'. Defaults to nn.MSELoss.
    - velocity (nn.Module | None): VelocityLoss, reduction='none'. None -> plain base loss.
    """

    def __init__(self, base:nn.Module=None, velocity:nn.Module=None) -> None:
        super().__init__()
        self.base= nn.MSELoss(reduction="none") if base is None else base
        self.velocity= velocity
        # MaeTrainer rejects any criterion whose reduction is not 'none'
        self.reduction= "none"

        for name, mod in (("base", self.base), ("velocity", self.velocity)):
            if mod is not None and getattr(mod, "reduction", None) != "none":
                raise ValueError(
                    f"{name} criterion must use reduction='none' (got {getattr(mod, 'reduction', None)!r}):"
                    f" forward_loss applies .mean(dim=-1) per patch before masking."
                )


    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


    def forward(self, pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        loss= self.base(pred, target)                                            # (B*C, P, patch_width)
        if self.velocity is not None:
            loss= loss + self.velocity(pred, target).mean(dim=-1, keepdim=True)  # (B*C, P, 1)

        return loss
