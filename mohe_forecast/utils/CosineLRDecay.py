# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Cosine Learning Rate (LR) Decay
"""

import math



class CosineLRDecay:
    """
    Modulates learning rate (LR) based on the iteration number which LR there should be.
    - Call .step() after each batch (i.e. every optimizer step).
    """

    def __init__(self, optimizer, min_lr, max_lr, warmup_steps=10, max_steps=50) -> None:
        assert warmup_steps < max_steps, "warmup_steps must be less than max_steps"
        self.optimizer= optimizer
        self.min_lr= min_lr
        self.max_lr= max_lr
        self.warmup_steps= int(warmup_steps)
        self.max_steps= int(max_steps)
        self.last_step= 0
        self.last_lr= None


    def extra_repr(self):
        return f"min_lr={self.min_lr}, max_lr={self.max_lr}, warmup_steps={self.warmup_steps}"


    def get_last_lr(self):
        """ Returns the last computed learning rate. """
        return self.last_lr


    def get_lr(self, it):
        """ Computes the learning rate at a given iteration 'step'. """
        # 1) linear warmup for warmup_iters steps iterations
        if it< self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        # 2) beyond max_steps, use the minimum learning rate
        if it>= self.max_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio= (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        # coeff starts at 1 and goes to 0
        coeff= 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return self.min_lr + coeff * (self.max_lr - self.min_lr)


    def step(self):
        """ Updates the learning rate for all parameter groups in the optimizer. """
        self.last_lr= self.get_lr(self.last_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr']= self.last_lr

        self.last_step += 1
