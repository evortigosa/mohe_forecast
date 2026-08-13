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



def chunked_mse_mae(labels:torch.Tensor, preds:torch.Tensor, chunk_size:int=32):
    """
    Memory-safe MSE/MAE computation for very large labels/preds tensors (e.g., 720+ horizons).
    - labels, preds: CPU or GPU tensors with identical shape.
    - chunk_size: number of samples along dim=0 per chunk.
    """
    if labels.shape != preds.shape:
        raise ValueError(
            f"labels and preds must have the same shape, got {labels.shape} and {preds.shape}"
        )

    total_sq_error= 0.0
    total_abs_error= 0.0
    total_count= 0
    n= labels.size(0)

    for start in range(0, n, chunk_size):
        end= min(start + chunk_size, n)
        y= labels[start:end].float()
        p= preds[start:end].float()

        diff= p - y

        total_sq_error += diff.square().sum(dtype=torch.float64).item()
        total_abs_error += diff.abs().sum(dtype=torch.float64).item()
        total_count += diff.numel()

        del y, p, diff

    mse= total_sq_error / max(total_count, 1)
    mae= total_abs_error / max(total_count, 1)

    return mse, mae



def get_metrics(trainer_obj, test_loader, dynamic_window=True, chunk_size=None):
    """
    Build metric objects from full predictions and ground-truth tensors.
    In distributed mode:
    - all ranks call trainer_obj.test(...);
    - only the main rank receives/uses full preds and trues;
    - non-main ranks return (None, None).
    """
    preds, trues= trainer_obj.test(test_loader, dynamic_window=dynamic_window)

    if not trainer_obj._is_main_process():
        return None, None
    if preds is None or trues is None:
        raise RuntimeError("Main process received None preds or labels from trainer.test().")

    mse_metric= MSEMetric(init_val=0.0)
    mae_metric= MAEMetric(init_val=0.0)

    if chunk_size is None:
        mse_metric.push(trues, preds)
        mae_metric.push(trues, preds)
    else:
        mse, mae= chunked_mse_mae(trues, preds, int(chunk_size))
        mse_metric.value= mse
        mae_metric.value= mae

    return mse_metric, mae_metric



def eval_forecast_horizons(trainer_obj, data_name, test_loader_96=None, test_loader_192=None,
                           test_loader_336=None, test_loader_720=None, dynamic_window=True):
    """
    Evaluate multiple forecast horizons and return (avg_mse, avg_mae), i.e., {96, 192, 336, 720}
    In distributed mode:
    - all ranks execute the same horizon sequence;
    - all ranks call test(), because test() uses distributed collectives;
    - only the main rank prints and accumulates final metric values.
    """
    avg_mse= []
    avg_mae= []

    def eval_one_horizon(horizon, loader, chunk_size=None):
        if loader is None:
            return
        # must happen on all ranks
        trainer_obj._print(f"\nForecast horizon: {horizon}")
        trainer_obj.set_forecast_horizon(horizon)
        mse_metric, mae_metric= get_metrics(trainer_obj, loader, dynamic_window, chunk_size)

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
    eval_one_horizon(720, test_loader_720, chunk_size=32)

    if not trainer_obj._is_main_process():
        return None, None
    if len(avg_mse) == 0:
        return float("nan"), float("nan")

    return np.mean(avg_mse).item(), np.mean(avg_mae).item()



def eval_reconstruction_quality(trainer_obj, data_name, test_loader, mask_ratio=None,
                                masked_out=True, inverse_transform=False, chunk_size=None):
    """
    Evaluate an SSL / MAE model by masked reconstruction and return (MSE, MAE).
    Sibling of eval_forecast_horizons, but for the reconstruction objective: there is no forecast
    horizon, so a single test loader is scored (the SSL factory builds one reconstruction loader,
    size_te = [block, block, 0]).
    - masked_out=True (default) scores only the masked positions -- the true MAE objective -- via
    MaeTrainer.test(masked_out=True), which returns preds/trues already restricted to those
    positions and mapped back to raw units.
    - masked_out=False scores the full reconstructed series (comparable to how the random-mask
    SSL path is currently evaluated).
    """
    trainer_obj._print(f"\n{data_name}")
    trainer_obj._print("\nReconstruction quality" + (" (masked positions only)" if masked_out else " (full series)"))

    # must run on all ranks (collectives inside test())
    preds, trues= trainer_obj.test(
        test_loader, inverse_transform=inverse_transform, mask_ratio=mask_ratio, masked_out=masked_out
    )

    if not trainer_obj._is_main_process():
        return None, None
    if preds is None or trues is None:
        raise RuntimeError("Main process received None preds or labels from trainer.test().")

    mse_metric= MSEMetric(init_val=0.0)
    mae_metric= MAEMetric(init_val=0.0)

    if chunk_size is None:
        mse_metric.push(trues, preds)
        mae_metric.push(trues, preds)
    else:
        mse, mae= chunked_mse_mae(trues, preds, int(chunk_size))
        mse_metric.value= mse
        mae_metric.value= mae

    print(mse_metric)
    print(mae_metric)
    trainer_obj._barrier()

    # MSEMetric/MAEMetric.value is a 0-dim tensor (non-chunked path) or a python float (chunked path);
    # return plain floats either way, mirroring eval_forecast_horizons' np.mean(...).item().
    mse_val= mse_metric.value
    mae_val= mae_metric.value
    mse_val= float(mse_val.item()) if hasattr(mse_val, "item") else float(mse_val)
    mae_val= float(mae_val.item()) if hasattr(mae_val, "item") else float(mae_val)

    return mse_val, mae_val
