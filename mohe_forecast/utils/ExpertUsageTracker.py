# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Expert Usage Tracker Utility
"""

import copy
import torch
from typing import Any, Dict, Optional



class ExpertUsageTracker:
    """
    Tracks global and layerwise expert-utilization statistics. The tracker stores CPU tensors
    and scalar histories so that statistics can be:
    - accumulated during training,
    - saved with checkpoints,
    - restored when resuming training,
    - plotted after training.
    """
    GLOBAL_KEYS= (
        "hard_fraction", "soft_fraction", "entropy", "dead_experts", "cv_hard", "cv_soft",
    )
    TENSOR_KEYS= (
        "hard_fraction", "soft_fraction",
    )
    SCALAR_KEYS= (
        "entropy", "dead_experts", "cv_hard", "cv_soft",
    )

    def __init__(self, n_experts:int, n_layers:Optional[int]=None):
        assert int(n_experts) >= 0, "n_experts must be non-negative"
        self.n_experts= int(n_experts)
        self.n_layers = n_layers

        self.global_step_history = self._new_metric_history()
        self.global_epoch_history= self._new_metric_history()

        self.layer_step_history: Dict[int, Dict[str, list]]= {}
        self.layer_epoch_history:Dict[int, Dict[str, list]]= {}

        self.reset_epoch()


    @classmethod
    def _new_metric_history(cls) -> Dict[str, list]:
        """
        Constructor for internal structures.
        """
        return {
            "hard_fraction": [],
            "soft_fraction": [],
            "entropy": [],
            "dead_experts": [],
            "cv_hard": [],
            "cv_soft": [],
        }


    def _new_global_epoch_accumulator(self) -> Dict[str, Any]:
        """
        Constructor for internal structures.
        """
        return {
            "sum_hard_fraction": torch.zeros(self.n_experts),
            "sum_soft_fraction": torch.zeros(self.n_experts),
            "sum_entropy": 0.0,
            "sum_dead_experts": 0.0,
            "sum_cv_hard": 0.0,
            "sum_cv_soft": 0.0,
            "num_updates": 0,
        }


    def _new_layer_epoch_accumulator(self) -> Dict[str, Any]:
        """
        Constructor for internal structures.
        """
        return {
            "sum_hard_fraction": torch.zeros(self.n_experts),
            "sum_soft_fraction": torch.zeros(self.n_experts),
            "sum_entropy": 0.0,
            "sum_dead_experts": 0.0,
            "sum_cv_hard": 0.0,
            "sum_cv_soft": 0.0,
            "num_updates": 0,
        }


    def _ensure_layer_storage(self, layer_id:int) -> None:
        """
        Constructor for internal structures.
        """
        layer_id= int(layer_id)

        if layer_id not in self.layer_step_history:
            self.layer_step_history[layer_id]= self._new_metric_history()

        if layer_id not in self.layer_epoch_history:
            self.layer_epoch_history[layer_id]= self._new_metric_history()


    @staticmethod
    def _tensor_to_cpu(x:torch.Tensor) -> torch.Tensor:
        """
        Conversion helper.
        """
        return x.detach().cpu()


    @staticmethod
    def _scalar_to_float(x:Any) -> float:
        """
        Conversion helper.
        """
        if torch.is_tensor(x):
            return float(x.detach().cpu())
        return float(x)


    def _extract_metrics(self, metrics:Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw metric dictionary into CPU tensors and Python floats.
        """
        return {
            "hard_fraction": self._tensor_to_cpu(metrics["hard_fraction"]).clone(),
            "soft_fraction": self._tensor_to_cpu(metrics["soft_fraction"]).clone(),
            "entropy": self._scalar_to_float(metrics["entropy"]),
            "dead_experts": self._scalar_to_float(metrics["dead_experts"]),
            "cv_hard": self._scalar_to_float(metrics["cv_hard"]),
            "cv_soft": self._scalar_to_float(metrics["cv_soft"]),
        }


    @staticmethod
    def _append_to_history(history:Dict[str, list], metrics:Dict[str, Any]) -> None:
        history["hard_fraction"].append(metrics["hard_fraction"].clone())
        history["soft_fraction"].append(metrics["soft_fraction"].clone())
        history["entropy"].append(metrics["entropy"])
        history["dead_experts"].append(metrics["dead_experts"])
        history["cv_hard"].append(metrics["cv_hard"])
        history["cv_soft"].append(metrics["cv_soft"])


    @staticmethod
    def _accumulate(acc:Dict[str, Any], metrics:Dict[str, Any]) -> None:
        acc["sum_hard_fraction"] += metrics["hard_fraction"]
        acc["sum_soft_fraction"] += metrics["soft_fraction"]
        acc["sum_entropy"] += metrics["entropy"]
        acc["sum_dead_experts"] += metrics["dead_experts"]
        acc["sum_cv_hard"] += metrics["cv_hard"]
        acc["sum_cv_soft"] += metrics["cv_soft"]
        acc["num_updates"] += 1


    @staticmethod
    def _average_accumulator(acc:Dict[str, Any]) -> Optional[Dict[str, Any]]:
        n= acc["num_updates"]
        if n == 0:
            return None

        return {
            "hard_fraction": acc["sum_hard_fraction"] / n,
            "soft_fraction": acc["sum_soft_fraction"] / n,
            "entropy": acc["sum_entropy"] / n,
            "dead_experts": acc["sum_dead_experts"] / n,
            "cv_hard": acc["sum_cv_hard"] / n,
            "cv_soft": acc["sum_cv_soft"] / n,
        }


    def reset_epoch(self) -> None:
        """
        Reset running accumulators for the current epoch.
        Does not erase stored histories.
        """
        self._global_epoch_acc= self._new_global_epoch_accumulator()
        self._layer_epoch_acc: Dict[int, Dict[str, Any]]= {}


    def update(self,
               global_metrics:Optional[Dict[str, Any]]=None,
               layer_metrics: Optional[Dict[int, Dict[str, Any]]]=None) -> None:
        """
        Update tracker with metrics from one training step.
        """
        if global_metrics is not None:
            metrics= self._extract_metrics(global_metrics)
            self._append_to_history(self.global_step_history, metrics)
            self._accumulate(self._global_epoch_acc, metrics)

        if layer_metrics is not None:
            for layer_id, raw_metrics in layer_metrics.items():
                layer_id= int(layer_id)
                self._ensure_layer_storage(layer_id)

                metrics= self._extract_metrics(raw_metrics)
                self._append_to_history(self.layer_step_history[layer_id], metrics)

                if layer_id not in self._layer_epoch_acc:
                    self._layer_epoch_acc[layer_id]= self._new_layer_epoch_accumulator()

                self._accumulate(self._layer_epoch_acc[layer_id], metrics)


    def finalize_epoch(self) -> Dict[str, Any]:
        """
        Finalize the current epoch, store epoch-level averages, and reset accumulators.
        """
        out= {"global": None, "layers": {},}

        global_epoch_metrics= self._average_accumulator(self._global_epoch_acc)
        if global_epoch_metrics is not None:
            self._append_to_history(self.global_epoch_history, global_epoch_metrics)
            out["global"]= global_epoch_metrics

        for layer_id, acc in self._layer_epoch_acc.items():
            layer_epoch_metrics= self._average_accumulator(acc)
            if layer_epoch_metrics is None:
                continue

            self._ensure_layer_storage(layer_id)
            self._append_to_history(self.layer_epoch_history[layer_id], layer_epoch_metrics)
            out["layers"][layer_id]= layer_epoch_metrics

        self.reset_epoch()
        return out


    @property
    def history(self) -> Dict[str, Any]:
        """
        Return a view of the history info.
        """
        return {
            "step_hard_fraction": self.global_step_history["hard_fraction"],
            "step_soft_fraction": self.global_step_history["soft_fraction"],
            "step_entropy": self.global_step_history["entropy"],
            "step_dead_experts": self.global_step_history["dead_experts"],
            "step_cv_hard": self.global_step_history["cv_hard"],
            "step_cv_soft": self.global_step_history["cv_soft"],

            "epoch_hard_fraction": self.global_epoch_history["hard_fraction"],
            "epoch_soft_fraction": self.global_epoch_history["soft_fraction"],
            "epoch_entropy": self.global_epoch_history["entropy"],
            "epoch_dead_experts": self.global_epoch_history["dead_experts"],
            "epoch_cv_hard": self.global_epoch_history["cv_hard"],
            "epoch_cv_soft": self.global_epoch_history["cv_soft"],

            "layer_step": self.layer_step_history,
            "layer_epoch": self.layer_epoch_history,
        }


    def to_serializable(self, compact:bool=False) -> Dict[str, Any]:
        """
        Return a checkpoint-safe representation of the tracker.
        - compact (bool): If True, removes step-level histories to reduce checkpoint size. Epoch
        histories and current epoch accumulators are preserved.
        """
        global_step_history= copy.deepcopy(self.global_step_history)
        layer_step_history = copy.deepcopy(self.layer_step_history)

        if compact:
            global_step_history= self._new_metric_history()
            layer_step_history= {}

        return {
            "class_name": self.__class__.__name__,
            "version": 1,

            "config": {"n_experts": self.n_experts, "n_layers": self.n_layers},

            "history": {
                "global_step": global_step_history,
                "global_epoch": copy.deepcopy(self.global_epoch_history),
                "layer_step": layer_step_history,
                "layer_epoch": copy.deepcopy(self.layer_epoch_history),
            },

            "running_state": {
                "global_epoch_acc": copy.deepcopy(self._global_epoch_acc),
                "layer_epoch_acc": copy.deepcopy(self._layer_epoch_acc),
            },
        }


    @classmethod
    def from_serializable(cls, state:Dict[str, Any]) -> "ExpertUsageTracker":
        """
        Build an ExpertUsageTracker from a serialized state.
        """
        config = state["config"]
        history= state["history"]
        running_state= state["running_state"]

        tracker= cls(n_experts=config["n_experts"], n_layers=config.get("n_layers", None),)
        tracker.global_step_history = copy.deepcopy(history["global_step"])
        tracker.global_epoch_history= copy.deepcopy(history["global_epoch"])
        tracker.layer_step_history  = copy.deepcopy(history["layer_step"])
        tracker.layer_epoch_history = copy.deepcopy(history["layer_epoch"])
        tracker._global_epoch_acc= copy.deepcopy(running_state["global_epoch_acc"])
        tracker._layer_epoch_acc = copy.deepcopy(running_state["layer_epoch_acc"])

        return tracker


    def load_serializable(self, state:Dict[str, Any], strict:bool=True) -> None:
        """
        Load serialized tracker state into an existing object.
        - state: Serialized tracker state from to_serializable().
        - strict: If True, enforce n_experts compatibility.
        """
        config = state["config"]
        history= state["history"]
        running_state= state["running_state"]

        if strict and int(config["n_experts"]) != self.n_experts:
            raise ValueError(
                f"n_experts mismatch: current tracker has {self.n_experts}, "
                f"serialized tracker has {config['n_experts']}."
            )

        self.n_experts= int(config["n_experts"])
        self.n_layers = config.get("n_layers", None)

        self.global_step_history = copy.deepcopy(history["global_step"])
        self.global_epoch_history= copy.deepcopy(history["global_epoch"])
        self.layer_step_history  = copy.deepcopy(history["layer_step"])
        self.layer_epoch_history = copy.deepcopy(history["layer_epoch"])
        self._global_epoch_acc= copy.deepcopy(running_state["global_epoch_acc"])
        self._layer_epoch_acc = copy.deepcopy(running_state["layer_epoch_acc"])


    def state_dict(self, compact:bool=False) -> Dict[str, Any]:
        """
        PyTorch-style alias for checkpointing.
        """
        return self.to_serializable(compact=compact)


    def load_state_dict(self, state_dict:Dict[str, Any], strict:bool=True) -> None:
        """
        PyTorch-style alias for restoring checkpointed tracker state.
        """
        self.load_serializable(state_dict, strict=strict)
