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



def get_metrics(trainer_class, test_loader):
    """
    Build metric objects from full predictions and ground-truth tensors.
    """
    mean_loss, preds, trues= trainer_class.test(test_loader)

    mse_metric= MSEMetric(init_val=0.0)
    mae_metric= MAEMetric(init_val=0.0)
    mse_metric.push(trues, preds)
    mae_metric.push(trues, preds)

    return mse_metric, mae_metric



def eval_forecast_horizons(model, trainer, data_name, test_loader_96=None, test_loader_192=None,
                           test_loader_336=None, test_loader_720=None):
    """
    Evaluate multiple forecast horizons and return (avg_mse, avg_mae) across provided horizons.
    Forecast horizons: {96, 192, 336, 720}
    """
    avg_mse= []
    avg_mae= []

    print(data_name)
    if test_loader_96 is not None:
        print("Forecast horizon: 96")
        model.n_outputs= 96
        mse_metric, mae_metric= get_metrics(trainer, test_loader_96)
        avg_mse.append(mse_metric.value)
        avg_mae.append(mae_metric.value)
        print(mse_metric)
        print(mae_metric)

    if test_loader_192 is not None:
        print("\nForecast horizon: 192")
        model.n_outputs= 192
        mse_metric, mae_metric= get_metrics(trainer, test_loader_192)
        avg_mse.append(mse_metric.value)
        avg_mae.append(mae_metric.value)
        print(mse_metric)
        print(mae_metric)

    if test_loader_336 is not None:
        print("\nForecast horizon: 336")
        model.n_outputs= 336
        mse_metric, mae_metric= get_metrics(trainer, test_loader_336)
        avg_mse.append(mse_metric.value)
        avg_mae.append(mae_metric.value)
        print(mse_metric)
        print(mae_metric)

    if test_loader_720 is not None:
        print("\nForecast horizon: 720")
        model.n_outputs= 720
        mse_metric, mae_metric= get_metrics(trainer, test_loader_720)
        avg_mse.append(mse_metric.value)
        avg_mae.append(mae_metric.value)
        print(mse_metric)
        print(mae_metric)

    if len(avg_mse) == 0:
        return float("nan"), float("nan")
    return np.mean(avg_mse).item(), np.mean(avg_mae).item()
