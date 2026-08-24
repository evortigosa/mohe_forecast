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

    # the schedule shape: changing any of these changes the LR trajectory
    _CONFIG_KEYS= ("min_lr", "max_lr", "warmup_steps", "max_steps")
    # the mutable progress of the schedule
    _STATE_KEYS= ("last_step", "last_lr")

    def __init__(self, optimizer, min_lr, max_lr, warmup_steps=10, max_steps=50) -> None:
        assert warmup_steps < max_steps, "warmup_steps must be less than max_steps"
        self.optimizer= optimizer
        self.min_lr= min_lr
        self.max_lr= max_lr
        self.warmup_steps= int(warmup_steps)
        self.max_steps= int(max_steps)
        self.last_step= 0
        self.last_lr= None

        assert all(hasattr(self, k) for k in (*self._STATE_KEYS, *self._CONFIG_KEYS)), \
            "state/config key names must match the attributes set in __init__"


    def extra_repr(self):
        return f"min_lr={self.min_lr:.2e}, max_lr={self.max_lr:.2e}, warmup_steps={self.warmup_steps}"


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


    def state_dict(self) -> dict:
        """
        Serializable scheduler state. The optimizer is deliberately excluded: it is saved separately
        and re-bound on load, exactly as torch.optim.lr_scheduler does.
        - the progress counters are what must be restored to continue a run;
        - the configuration is stored for validation only, so that resuming against a different
        schedule shape is reported rather than silently changing the LR trajectory.
        """
        return {k: getattr(self, k) for k in (*self._STATE_KEYS, *self._CONFIG_KEYS)}


    def load_state_dict(self, state_dict:dict, restore_config=False) -> list:
        """
        Restore the schedule progress.
        - restore_config=False (default): keep this instance's configuration and restore only the
        counters, so a deliberate change (e.g. resuming with more epochs -> larger max_steps) is
        honoured. Differing keys are returned so the caller can log them.
        - restore_config=True: also adopt the checkpointed configuration, for an exact continuation.
        Returns the list of configuration keys that differ from the checkpoint.
        """
        missing= [k for k in self._STATE_KEYS if k not in state_dict]
        if missing:
            raise KeyError(f"CosineLRDecay.load_state_dict | missing keys: {missing}")

        mismatched= [
            k for k in self._CONFIG_KEYS if k in state_dict and state_dict[k] != getattr(self, k)
        ]
        if restore_config:
            for k in self._CONFIG_KEYS:
                if k in state_dict:
                    setattr(self, k, state_dict[k])
            assert self.warmup_steps < self.max_steps, "warmup_steps must be less than max_steps"

        self.last_step= int(state_dict["last_step"])
        self.last_lr= state_dict["last_lr"]
        # re-apply the checkpointed LR so the optimizer is consistent before the next step(),
        # which matters when the optimizer state itself was not restored
        if self.last_lr is not None and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr']= self.last_lr

        return mismatched
