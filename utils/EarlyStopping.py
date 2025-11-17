# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Early Stopping
"""

import math



class EarlyStopping:
    """
    Early stopping utility to terminate training when the loss does not improve sufficiently.
    """

    def __init__(self, patience=7, min_delta=1e-4, mode="min", verbose=True):
        assert patience >= 0, "patience must be non-negative"
        assert min_delta >= 0, "min_delta must be non-negative"
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.patience= int(patience)
        self.min_delta= float(min_delta)
        self.mode= mode
        self.verbose= verbose
        self.counter= 0
        self.best_loss= math.inf if mode == "min" else -math.inf
        self.early_stop= False


    def extra_repr(self):
        return f"patience={self.patience}, eps={self.min_delta}"


    def is_improvement(self, current) -> bool:
        """
        Return True if current is an improvement over best depending on mode.
        """
        if self.mode == "min":
            return current < (self.best_loss - self.min_delta)
        else:
            return current > (self.best_loss + self.min_delta)


    def __call__(self, current_loss:float, epoch:int):
        """
        Check if the training should be stopped early.
        """
        # validation of input
        if current_loss is None or (
            isinstance(current_loss, float) and (math.isnan(current_loss) or math.isinf(current_loss))
        ):
            # treat NaN/Inf as non-improving; increment
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop= True
                if self.verbose:
                    print(f"[EarlyStopping] Stopping: non-finite metric at epoch {epoch}.")

            return self.early_stop

        if self.is_improvement(current_loss):
            # improvement -> record best and reset counter
            self.best_loss= current_loss
            self.best_epoch= int(epoch)
            self.counter= 0
        else:
            # no sufficient improvement -> increment counter
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop= True
                if self.verbose:
                    print(f"[EarlyStopping] Early stopping triggered at epoch {epoch}.")

        return self.early_stop
