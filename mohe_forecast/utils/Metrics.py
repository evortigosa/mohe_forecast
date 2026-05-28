# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Evaluation Metrics
"""

import numpy as np
import torch
import torch.nn.functional as F



class SumEvaluationMetric:
    def __init__(self, name, init_val= 0.0) -> None:
        """
        Base metric class to accumulate a sum of evaluation metric values.
        """
        self.name = name
        self.value= init_val

    def reset(self) -> None:
        """
        Reset the metric to its initial value.
        """
        self.value= 0.0

    def push(self, labels, preds, **kwargs) -> None:
        """
        Update the metric value by accumulating the value from _calculate.
        """
        self.value += self._calculate(labels, preds, **kwargs)

    def _calculate(self, labels, preds, **kwargs):
        """
        Calculate the metric for the current mini-batch.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __str__(self) -> str:
        return f'{self.name}: {self.value}'



class MSEMetric(SumEvaluationMetric):
    
    def __init__(self, init_val= 0.0) -> None:
        super().__init__('MSE', init_val)

    def _calculate(self, labels, preds, **kwargs):
        """
        Compute the Mean Squared Errors (MSE) for the current batch.
        """
        return F.mse_loss(labels, preds, reduction='mean')



class RMSEMetric(SumEvaluationMetric):

    def __init__(self, init_val= 0.0) -> None:
        super().__init__('RMSE', init_val)

    def _calculate(self, labels, preds, **kwargs):
        """
        Compute the Root Mean Squared Errors (RMSE) for the current batch.
        """
        return torch.sqrt(F.mse_loss(labels, preds, reduction='mean'))



class MAEMetric(SumEvaluationMetric):

    def __init__(self, init_val= 0.0) -> None:
        super().__init__('MAE', init_val)

    def _calculate(self, labels, preds, **kwargs):
        """
        Compute the Sum of Absolute Errors (MAE) for the current batch.
        """
        return F.l1_loss(labels, preds, reduction='mean')



def get_metrics(trainer_obj, test_loader):
    """
    Build metric objects from full predictions and ground-truth tensors.
    In distributed mode:
    - all ranks call trainer_obj.test(...);
    - only the main rank receives/uses full preds and trues;
    - non-main ranks return (None, None).
    """
    preds, trues= trainer_obj.test(test_loader)

    if not trainer_obj._is_main_process():
        return None, None
    if preds is None or trues is None:
        raise RuntimeError("Main process received None preds or labels from trainer.test().")

    mse_metric= MSEMetric(init_val=0.0)
    mae_metric= MAEMetric(init_val=0.0)
    mse_metric.push(trues, preds)
    mae_metric.push(trues, preds)

    return mse_metric, mae_metric



def eval_forecast_horizons(trainer_obj, data_name, test_loader_96=None, test_loader_192=None,
                           test_loader_336=None, test_loader_720=None):
    """
    Evaluate multiple forecast horizons and return (avg_mse, avg_mae), i.e., {96, 192, 336, 720}
    In distributed mode:
    - all ranks execute the same horizon sequence;
    - all ranks call test(), because test() uses distributed collectives;
    - only the main rank prints and accumulates final metric values.
    """
    avg_mse= []
    avg_mae= []

    def eval_one_horizon(horizon, loader):
        if loader is None:
            return
        # must happen on all ranks
        trainer_obj._print(f"\nForecast horizon: {horizon}")
        trainer_obj.set_forecast_horizon(horizon)
        mse_metric, mae_metric= get_metrics(trainer_obj, loader)

        if trainer_obj._is_main_process():
            avg_mse.append(mse_metric.value)
            avg_mae.append(mae_metric.value)
            print(mse_metric)
            print(mae_metric)

        trainer_obj._barrier()

    trainer_obj._print(f"\n{data_name}")
    eval_one_horizon( 96, test_loader_96)
    eval_one_horizon(192, test_loader_192)
    eval_one_horizon(336, test_loader_336)
    eval_one_horizon(720, test_loader_720)

    if not trainer_obj._is_main_process():
        return None, None
    if len(avg_mse) == 0:
        return float("nan"), float("nan")

    return np.mean(avg_mse).item(), np.mean(avg_mae).item()
