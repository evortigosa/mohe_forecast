# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Early Stopping
"""

import numpy as np



class EarlyStopping:
    """
    Early stopping utility to terminate training when the loss does not improve sufficiently.
    """

    def __init__(self, patience=7, eps=1e-4, verbose=True):
        self.patience= int(patience)
        self.eps= eps
        self.verbose= verbose
        self.counter= 0
        self.last_loss= None
        self.early_stop= False


    def extra_repr(self):
        return f"patience={self.patience}, eps={self.eps}"


    def __call__(self, current_loss, epoch):
        """
        Check if the training should be stopped early.
        """
        if self.last_loss is None:
            self.last_loss= current_loss
        else:
            loss_change= np.abs(self.last_loss - current_loss)

            # if the improvement is smaller than eps, increment the counter
            if loss_change < self.eps:
                self.counter += 1

                if self.counter >= self.patience:
                    self.early_stop= True
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
            else:
                self.counter= 0
            # update last_loss to the current loss
            self.last_loss= current_loss

        return self.early_stop
