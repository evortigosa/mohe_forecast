# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Expert Usage Tracker Utility
"""

import copy
import torch



class ExpertUsageTracker:
    """
    Accumulates monitoring statistics across steps/epochs.
    Stores CPU tensors so you can save them later or plot directly.
    """

    def __init__(self, n_experts, n_layers=None):
        assert int(n_experts) >= 0, "n_experts must be non-negative"
        self.n_experts= int(n_experts)
        self.n_layers = n_layers
        self.history= {
            "step_hard_fraction": [],
            "step_soft_fraction": [],
            "step_entropy": [],
            "step_dead_experts": [],
            "step_cv_hard": [],
            "step_cv_soft": [],
            "epoch_hard_fraction": [],
            "epoch_soft_fraction": [],
            "epoch_entropy": [],
            "epoch_dead_experts": [],
            "epoch_cv_hard": [],
            "epoch_cv_soft": [],
            # layerwise history
            "layer_step": {},   # layer_id -> dict of lists
            "layer_epoch": {},  # layer_id -> dict of lists
        }
        self._sum_hard= torch.zeros(self.n_experts)
        self._sum_soft= torch.zeros(self.n_experts)
        self._sum_entropy= 0.0
        self._sum_dead= 0.0
        self._sum_cv_hard= 0.0
        self._sum_cv_soft= 0.0
        self._num_updates= 0
        # layerwise accumulators
        self._layer_epoch_acc= {}


    def ensure_layer_storage(self, layer_id):
        if layer_id not in self.history["layer_step"]:
            self.history["layer_step"][layer_id]= {
                "hard_fraction": [],
                "soft_fraction": [],
                "entropy": [],
                "dead_experts": [],
                "cv_hard": [],
                "cv_soft": [],
            }
        if layer_id not in self.history["layer_epoch"]:
            self.history["layer_epoch"][layer_id]= {
                "hard_fraction": [],
                "soft_fraction": [],
                "entropy": [],
                "dead_experts": [],
                "cv_hard": [],
                "cv_soft": [],
            }


    def reset_epoch(self):
        self._sum_hard= torch.zeros(self.n_experts)
        self._sum_soft= torch.zeros(self.n_experts)
        self._sum_entropy= 0.0
        self._sum_dead= 0.0
        self._sum_cv_hard= 0.0
        self._sum_cv_soft= 0.0
        self._num_updates= 0
        # layerwise accumulators
        self._layer_epoch_acc= {}


    def update(self, global_metrics=None, layer_metrics=None):
        if global_metrics is not None:
            hard= global_metrics["hard_fraction"].detach().cpu()
            soft= global_metrics["soft_fraction"].detach().cpu()
            entropy= float(global_metrics["entropy"].detach().cpu())
            dead= float(global_metrics["dead_experts"].detach().cpu())
            cv_hard= float(global_metrics["cv_hard"].detach().cpu())
            cv_soft= float(global_metrics["cv_soft"].detach().cpu())
            # step history
            self.history["step_hard_fraction"].append(hard.clone())
            self.history["step_soft_fraction"].append(soft.clone())
            self.history["step_entropy"].append(entropy)
            self.history["step_dead_experts"].append(dead)
            self.history["step_cv_hard"].append(cv_hard)
            self.history["step_cv_soft"].append(cv_soft)
            # epoch accumulators
            self._sum_hard += hard
            self._sum_soft += soft
            self._sum_entropy += entropy
            self._sum_dead += dead
            self._sum_cv_hard += cv_hard
            self._sum_cv_soft += cv_soft
            self._num_updates += 1

        if layer_metrics is not None:
            for layer_id, m in layer_metrics.items():
                self.ensure_layer_storage(layer_id)

                hard= m["hard_fraction"].detach().cpu()
                soft= m["soft_fraction"].detach().cpu()
                entropy= float(m["entropy"].detach().cpu())
                dead= float(m["dead_experts"].detach().cpu())
                cv_hard= float(m["cv_hard"].detach().cpu())
                cv_soft= float(m["cv_soft"].detach().cpu())
                # step history
                self.history["layer_step"][layer_id]["hard_fraction"].append(hard.clone())
                self.history["layer_step"][layer_id]["soft_fraction"].append(soft.clone())
                self.history["layer_step"][layer_id]["entropy"].append(entropy)
                self.history["layer_step"][layer_id]["dead_experts"].append(dead)
                self.history["layer_step"][layer_id]["cv_hard"].append(cv_hard)
                self.history["layer_step"][layer_id]["cv_soft"].append(cv_soft)
                # epoch accumulators
                if layer_id not in self._layer_epoch_acc:
                    self._layer_epoch_acc[layer_id] = {
                        "sum_hard": torch.zeros(self.n_experts),
                        "sum_soft": torch.zeros(self.n_experts),
                        "sum_entropy": 0.0,
                        "sum_dead": 0.0,
                        "sum_cv_hard": 0.0,
                        "sum_cv_soft": 0.0,
                        "num_updates": 0,
                    }
                acc = self._layer_epoch_acc[layer_id]
                acc["sum_hard"] += hard
                acc["sum_soft"] += soft
                acc["sum_entropy"] += entropy
                acc["sum_dead"] += dead
                acc["sum_cv_hard"] += cv_hard
                acc["sum_cv_soft"] += cv_soft
                acc["num_updates"] += 1


    def finalize_epoch(self):
        out= {"global": None, "layers": {}}

        if self._num_updates > 0:
            global_epoch_stats= {
                "hard_fraction": self._sum_hard / self._num_updates,
                "soft_fraction": self._sum_soft / self._num_updates,
                "entropy": self._sum_entropy / self._num_updates,
                "dead_experts": self._sum_dead / self._num_updates,
                "cv_hard": self._sum_cv_hard / self._num_updates,
                "cv_soft": self._sum_cv_soft / self._num_updates,
            }
            self.history["epoch_hard_fraction"].append(global_epoch_stats["hard_fraction"].clone())
            self.history["epoch_soft_fraction"].append(global_epoch_stats["soft_fraction"].clone())
            self.history["epoch_entropy"].append(global_epoch_stats["entropy"])
            self.history["epoch_dead_experts"].append(global_epoch_stats["dead_experts"])
            self.history["epoch_cv_hard"].append(global_epoch_stats["cv_hard"])
            self.history["epoch_cv_soft"].append(global_epoch_stats["cv_soft"])
            out["global"]= global_epoch_stats

        for layer_id, acc in self._layer_epoch_acc.items():
            if acc["num_updates"] == 0:
                continue
            layer_epoch_stats= {
                "hard_fraction": acc["sum_hard"] / acc["num_updates"],
                "soft_fraction": acc["sum_soft"] / acc["num_updates"],
                "entropy": acc["sum_entropy"] / acc["num_updates"],
                "dead_experts": acc["sum_dead"] / acc["num_updates"],
                "cv_hard": acc["sum_cv_hard"] / acc["num_updates"],
                "cv_soft": acc["sum_cv_soft"] / acc["num_updates"],
            }
            self.history["layer_epoch"][layer_id]["hard_fraction"].append(
                layer_epoch_stats["hard_fraction"].clone()
            )
            self.history["layer_epoch"][layer_id]["soft_fraction"].append(
                layer_epoch_stats["soft_fraction"].clone()
            )
            self.history["layer_epoch"][layer_id]["entropy"].append(layer_epoch_stats["entropy"])
            self.history["layer_epoch"][layer_id]["dead_experts"].append(layer_epoch_stats["dead_experts"])
            self.history["layer_epoch"][layer_id]["cv_hard"].append(layer_epoch_stats["cv_hard"])
            self.history["layer_epoch"][layer_id]["cv_soft"].append(layer_epoch_stats["cv_soft"])
            out["layers"][layer_id] = layer_epoch_stats

        self.reset_epoch()

        return out


    def state_dict(self, compact=False):
        """
        Return a checkpoint-safe representation of the tracker.
        """
        history= copy.deepcopy(self.history)
        if compact:
            history["step_hard_fraction"]= []
            history["step_soft_fraction"]= []
            history["step_entropy"]= []
            history["step_dead_experts"]= []
            history["step_cv_hard"]= []
            history["step_cv_soft"]= []
            history["layer_step"]= {}

        return {
            "n_experts": self.n_experts,
            "n_layers": self.n_layers,
            "history": history,
            "epoch_accumulators": {
                "sum_hard": self._sum_hard.clone(),
                "sum_soft": self._sum_soft.clone(),
                "sum_entropy": self._sum_entropy,
                "sum_dead": self._sum_dead,
                "sum_cv_hard": self._sum_cv_hard,
                "sum_cv_soft": self._sum_cv_soft,
                "num_updates": self._num_updates,
                "layer_epoch_acc": copy.deepcopy(self._layer_epoch_acc),
            },
        }


    def load_state_dict(self, state_dict):
        """
        Restore tracker state from checkpoint.
        """
        self.n_experts= state_dict["n_experts"]
        self.n_layers= state_dict.get("n_layers", None)
        self.history = copy.deepcopy(state_dict["history"])

        epoch_acc= state_dict["epoch_accumulators"]
        self._sum_hard= epoch_acc["sum_hard"].clone()
        self._sum_soft= epoch_acc["sum_soft"].clone()
        self._sum_entropy= epoch_acc["sum_entropy"]
        self._sum_dead= epoch_acc["sum_dead"]
        self._sum_cv_hard= epoch_acc["sum_cv_hard"]
        self._sum_cv_soft= epoch_acc["sum_cv_soft"]
        self._num_updates= epoch_acc["num_updates"]
        self._layer_epoch_acc= copy.deepcopy(epoch_acc["layer_epoch_acc"])
