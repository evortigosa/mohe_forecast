# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Muon Optimizer Setup
"""

import inspect
import torch
import torch.nn as nn
from collections import OrderedDict
from ..model.Normalization import RMSNorm, DynamicTanh, InstanceNorm, RevIN



class MultiOptimizer:
    """  --- WIP ---
    Lightweight wrapper that allows multiple PyTorch optimizers to be used through a single optimizer-like
    interface. Designed for:
    - AdamW on embeddings, heads, biases, norms, and other non-Muon params.
    - Muon on eligible 2D hidden-layer matrices.

    NOTE: The attribute 'self.param_groups' is a flattened list of references to the original
    parameter-group dictionaries owned by the internal optimizers. Therefore, code such as:
        optimizer.param_groups[0]["lr"]= new_lr
    modifies the actual internal optimizer group, preserving compatibility with custom learning-rate
    schedulers that directly read or write 'optimizer.param_groups".
    """

    def __init__(self, **optimizers):
        # preserve insertion order, i.e., the flattened 'param_groups' list follows the same order.
        self.optimizers= OrderedDict(
            (name, opt) for name, opt in optimizers.items() if opt is not None
        )
        if len(self.optimizers) == 0:
            raise ValueError("At least one optimizer must be provided.")

        self._refresh_param_groups()


    def _refresh_param_groups(self):
        """ Rebuild the flattened parameter-group references. """
        self.param_groups= []
        self.param_group_names= []
        self.param_group_to_optimizer= []

        for opt_name, opt in self.optimizers.items():
            for group_idx, group in enumerate(opt.param_groups):
                # store metadata directly in the parameter-group dictionary
                group.setdefault("optimizer_name", opt_name)
                group.setdefault("group_name", f"{opt_name}/group_{group_idx}")
                # append references, not copies
                self.param_groups.append(group)
                self.param_group_names.append(group["group_name"])
                self.param_group_to_optimizer.append(opt_name)


    def zero_grad(self, set_to_none=True):
        """ Clear gradients for all internal optimizers. """
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)


    def step(self, closure=None):
        """ Perform one optimization step for every internal optimizer. """
        loss= None
        for opt in self.optimizers.values():
            # in standard AdamW/Muon training loops closure is normally None
            if closure is None:
                opt.step()
            else:
                loss= opt.step(closure)

        return loss


    def state_dict(self):
        """ Return a checkpointable state dictionary for all internal optimizers. """
        return {
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "param_group_names": self.param_group_names,
        }


    def load_state_dict(self, state_dict):
        """ Load the state of each internal optimizer. """
        opt_states= state_dict["optimizers"]

        for name, opt_state in opt_states.items():
            if name not in self.optimizers:
                raise KeyError(f"Optimizer '{name}' not found in current MultiOptimizer.")
            self.optimizers[name].load_state_dict(opt_state)
        # rebuild flattened references after loading
        self._refresh_param_groups()


    def get_lrs(self):
        """ Return the learning rate of every flattened parameter group. """
        return {group["group_name"]: group["lr"] for group in self.param_groups}


    def get_lr(self, group_idx:int=0):
        """ Return the learning rate of one flattened parameter group. """
        return self.param_groups[group_idx]["lr"]


    def set_lr(self, lr, group_idx=None):
        """ Set the learning rate for one or all flattened parameter groups. """
        if group_idx is None:
            for group in self.param_groups:
                group["lr"]= lr
        else:
            assert isinstance(group_idx, int), "group_idx must be an integer."
            self.param_groups[group_idx]["lr"]= lr


    def get_group_value(self, key, group_idx:int=0, default=None):
        """ Retrieve an arbitrary field from a parameter group. """
        return self.param_groups[group_idx].get(key, default)


    def get_first_group_with_key(self, key):
        """ Return the first parameter group that contains a given key. """
        for group in self.param_groups:
            if key in group:
                return group
        return None


    def get_betas(self, default=None):
        """
        Return Adam-style beta coefficients from the first group that has them.
        Muon does not use AdamW-style beta coefficients, so this method returns 'default' if no
        internal optimizer exposes a "betas" field.
        """
        group= self.get_first_group_with_key("betas")
        if group is None:
            return default
        return group["betas"]


    def get_momentum(self, default=None):
        """
        Return the momentum value from the first group that has it. Useful for Muon or SGD-like
        optimizers. AdamW typically does not expose a "momentum" field in its parameter groups.
        """
        group= self.get_first_group_with_key("momentum")
        if group is None:
            return default
        return group["momentum"]


    def get_weight_decays(self):
        """ Return the weight decay assigned to each flattened parameter group. """
        return {group["group_name"]: group.get("weight_decay", None) for group in self.param_groups}


    def get_summary(self):
        """
        Return a nested dictionary indexed by readable parameter-group name of all optimizer
        parameter groups.
        """
        summary= {}
        for group in self.param_groups:
            name= group["group_name"]
            summary[name]= {
                "optimizer": group.get("optimizer_name", None),
                "lr": group.get("lr", None),
                "weight_decay": group.get("weight_decay", None),
                "betas": group.get("betas", None),
                "momentum": group.get("momentum", None),
                "nesterov": group.get("nesterov", None),
            }
        return summary



def setup_muon_optimizer(
    model, learning_rate:float, weight_decay:float, adamw_betas=(0.9, 0.95), adamw_eps=1e-10,
    muon_momentum=0.95, muon_nesterov=True, muon_ns_coefficients=(3.4445, -4.775, 2.0315),
    muon_eps=1e-10, muon_ns_steps=5, muon_adjust_lr_fn="match_rms_adamw",
    exclude_from_muon=("embed", "embedding", "pos_embed", "position", "pos_emb", "token", "head",
                       "lm_head", "output", "prediction", "forecast", "decoder",),
    verbose=False,
):
    """
    Setup optimizer using:
    - Muon for 2D hidden-layer weights.
    - AdamW for all remaining parameters.
    This method follows the intended Muon usage: hidden-layer 2D matrices are optimized by Muon,
    while biases, norms, embeddings, and heads are optimized by AdamW.
    """
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(
            "torch.optim.Muon is not available in your current PyTorch version. "
            "Please install a PyTorch version that provides torch.optim.Muon."
        )

    device= next(model.parameters()).device
    named_modules= dict(model.named_modules())

    muon_params= []
    adamw_decay_params= []
    adamw_nodecay_params= []

    muon_names= []
    adamw_decay_names= []
    adamw_nodecay_names= []

    def get_parent_module(param_name):
        if "." not in param_name:
            return model
        parent_name= param_name.rsplit(".", 1)[0]
        return named_modules.get(parent_name, None)

    def is_excluded_by_name(param_name):
        lname= param_name.lower()
        return any(key in lname for key in exclude_from_muon)

    def is_muon_candidate(param_name, param):
        """
        Muon should be used only for 2D hidden-layer matrices. We additionally exclude:
        - embeddings,
        - positional parameters,
        - output/prediction heads,
        - normalization/bias-like tensors.
        """
        if param.ndim != 2:
            return False

        if is_excluded_by_name(param_name):
            return False

        parent= get_parent_module(param_name)

        if isinstance(parent, nn.Embedding):
            return False

        if isinstance(parent, (
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm,
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.RMSNorm,
            RMSNorm, DynamicTanh, InstanceNorm, RevIN
        ),):
            return False

        return True

    param_dict= {name: param for name, param in model.named_parameters() if param.requires_grad}

    for name, param in param_dict.items():
        if is_muon_candidate(name, param):
            muon_params.append(param)
            muon_names.append(name)
        else:
            # AdamW policy: 2D remaining tensors are decayed, 1D tensors such as biases/norm
            # scales are not decayed. See model.setup_optimizer()
            if param.ndim >= 2:
                adamw_decay_params.append(param)
                adamw_decay_names.append(name)
            else:
                adamw_nodecay_params.append(param)
                adamw_nodecay_names.append(name)

    adamw_groups= [
        {"params": adamw_decay_params, "weight_decay": weight_decay,},
        {"params": adamw_nodecay_params, "weight_decay": 0.0,},
    ]
    fused_available= "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused= fused_available and device.type == "cuda"

    adamw_optimizer= None
    if len(adamw_decay_params) + len(adamw_nodecay_params) > 0:
        adamw_optimizer= torch.optim.AdamW(
            adamw_groups, lr=learning_rate, betas=adamw_betas, eps=adamw_eps, fused=use_fused,
        )

    muon_optimizer= None
    if len(muon_params) > 0:
        muon_optimizer= torch.optim.Muon(
            muon_params, lr=learning_rate, weight_decay=weight_decay, momentum=muon_momentum,
            nesterov=muon_nesterov, ns_coefficients=muon_ns_coefficients, eps=muon_eps,
            ns_steps=muon_ns_steps, adjust_lr_fn=muon_adjust_lr_fn,
        )
    # AdamW is the first in order to get param_groups[0]["betas"]
    optimizer= MultiOptimizer(adamw=adamw_optimizer, muon=muon_optimizer)

    if verbose:
        num_muon= sum(p.numel() for p in muon_params)
        num_adamw_decay= sum(p.numel() for p in adamw_decay_params)
        num_adamw_nodecay= sum(p.numel() for p in adamw_nodecay_params)
        print(f"[INFO] AdamW decayed tensors: {len(adamw_decay_params)}, with {num_adamw_decay:,} parameters")
        print(f"[INFO] AdamW non-decayed tensors: {len(adamw_nodecay_params)}, with {num_adamw_nodecay:,} parameters")
        print(f"[INFO] Using fused AdamW: {use_fused}")
        print(f"[INFO] Muon parameter tensors: {len(muon_params)}, with {num_muon:,} parameters")
        print(f"[INFO] Muon adjust_lr_fn: {muon_adjust_lr_fn}")
        print("\n[INFO] Muon parameters:")
        for n in muon_names:
            print(f"- {n}")
        print("\n[INFO] AdamW parameters:")
        for n in (adamw_decay_names + adamw_nodecay_names):
            print(f"- {n}")

    return optimizer
