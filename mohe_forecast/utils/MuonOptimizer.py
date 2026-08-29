# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Muon Optimizer Setup
"""

import inspect
import re
import warnings
import torch
import torch.nn as nn
from torch.optim import Optimizer
from collections import OrderedDict
from ..model.Normalization import RMSNorm, DynamicTanh, InstanceNorm, RevIN



class MultiOptimizer(Optimizer):
    """  --- WIP ---
    Lightweight wrapper that allows multiple PyTorch optimizers to be used through a single optimizer-like
    interface. It subclasses torch.optim.Optimizer so it is accepted by utilities that validate the optimizer
    type. Designed for:
    - AdamW on embeddings, heads, biases, norms, and other non-Muon params.
    - Muon on eligible 2D hidden-layer matrices.

    NOTE: The attribute 'self.param_groups' is a flattened list of references to the original
    parameter-group dictionaries owned by the internal optimizers. Therefore, code such as:
        optimizer.param_groups[0]["lr"]= new_lr
    modifies the actual internal optimizer group, preserving compatibility with custom learning-rate
    schedulers that directly read or write 'optimizer.param_groups'.
    """

    def __init__(self, **optimizers):
        opt_items= [(name, opt) for name, opt in optimizers.items() if opt is not None]
        if len(opt_items) == 0:
            raise ValueError("At least one optimizer must be provided.")
        # preserve insertion order, i.e., the flattened 'param_groups' list follows the same order.
        self.optimizers= OrderedDict(opt_items)
        # construction-time base LR of every group, in flattened order. This is the schedule's
        # reference ('initial_lr' for CosineLRDecay and every torch.optim.lr_scheduler).
        self._base_lrs= [
            float(group["lr"]) for opt in self.optimizers.values() for group in opt.param_groups
        ]

        # initialize the base Optimizer (state, hooks, and any version-specific attributes) with the union of
        # all internal parameters, then replace 'param_groups' with references to the internal optimizers'
        # own groups (_refresh_param_groups). The union is disjoint -- each parameter belongs to exactly one
        # internal optimizer -- no-duplicate-parameter requirement
        all_params= [
            p for opt in self.optimizers.values() for group in opt.param_groups for p in group["params"]
        ]
        super().__init__(all_params, defaults={})
        self._refresh_param_groups()


    def _refresh_param_groups(self):
        """ Rebuild the flattened parameter-group references. """
        self.param_groups= []
        self.param_group_names= []
        self.param_group_to_optimizer= []
        ref_lr= self._base_lrs[0]
        idx= 0

        for opt_name, opt in self.optimizers.items():
            for group_idx, group in enumerate(opt.param_groups):
                # assigned, not setdefault: these derive from the current construction and must
                # survive Optimizer.load_state_dict rebuilding the group dicts from a checkpoint
                group["optimizer_name"]= opt_name
                group["group_name"]= f"{opt_name}/group_{group_idx}"
                group["initial_lr"]= self._base_lrs[idx]                              # drives the schedule
                group["lr_scale"]= (self._base_lrs[idx] / ref_lr) if ref_lr else 1.0  # reporting only
                # append references, not copies
                self.param_groups.append(group)
                self.param_group_names.append(group["group_name"])
                self.param_group_to_optimizer.append(opt_name)
                idx += 1

        assert self.param_groups[0].get("lr_scale", 1.0) == 1.0, \
            "the first flattened group must be the LR reference"


    def zero_grad(self, set_to_none=True):
        """ Clear gradients for all internal optimizers. """
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)


    def step(self, closure=None):
        """
        Perform one optimization step for every internal optimizer. If a closure is provided it is
        evaluated only once to (re)compute the loss and populate gradients. Targets first-order
        optimizers (AdamW, Muon); it is not intended for closure-driven optimizers (i.e., LBFGS).
        """
        loss= None
        if closure is not None:
            with torch.enable_grad():
                loss= closure()
        for opt in self.optimizers.values():
            opt.step()

        return loss


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
                "lr_scale": group.get("lr_scale", None),
            }
        return summary


    def state_dict(self):
        """ Return a checkpointable state dictionary for all internal optimizers. """
        return {
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "param_group_names": list(self.param_group_names),
            "base_lrs": list(self._base_lrs),
        }


    def load_state_dict(self, state_dict, strict=True):
        """
        Load the state of each internal optimizer.
        - strict=True (default): every internal optimizer must have saved state and the flattened
        group layout must match the checkpoint. A partial restore is refused rather than silently
        leaving an optimizer's momentum / second-moment buffers at zero.
        - strict=False: restore what matches; returns the names that were not restored.
        """
        if "optimizers" not in state_dict:
            raise KeyError("MultiOptimizer.load_state_dict | malformed state: missing 'optimizers'.")
        opt_states= state_dict["optimizers"]

        unknown= [n for n in opt_states if n not in self.optimizers]
        missing= [n for n in self.optimizers if n not in opt_states]
        if unknown:
            raise KeyError(
                f"MultiOptimizer.load_state_dict | checkpoint defines optimizer(s) {unknown} not "
                f"present here (current: {list(self.optimizers)})."
            )
        if missing and strict:
            raise KeyError(
                f"MultiOptimizer.load_state_dict | no saved state for {missing}: their momentum / "
                f"second-moment buffers would restart from zero. Pass strict=False to accept that."
            )
        # the inner load_state_dict validates group sizes; comparing names first turns a routing
        # change into an actionable error instead of an opaque size mismatch
        saved_names= state_dict.get("param_group_names", None)
        if saved_names is not None and list(saved_names) != list(self.param_group_names):
            raise ValueError(
                "MultiOptimizer.load_state_dict | parameter-group layout differs from the checkpoint "
                f"(checkpoint={list(saved_names)}, current={list(self.param_group_names)}): the "
                "routing policy changed, so the saved state does not describe these groups."
            )
        # a deliberate LR change on resume is legitimate; announce it instead of inferring it
        saved_bases= state_dict.get("base_lrs", None)
        if saved_bases is not None and list(saved_bases) != list(self._base_lrs):
            warnings.warn(
                f"MultiOptimizer.load_state_dict | base LR policy changed "
                f"(checkpoint={list(saved_bases)}, current={list(self._base_lrs)}); keeping the "
                f"current configuration.", stacklevel=2
            )

        for name, opt_state in opt_states.items():
            self.optimizers[name].load_state_dict(opt_state)
        # rebuild flattened references (re-asserts optimizer_name / group_name / initial_lr / lr_scale)
        self._refresh_param_groups()

        return missing



# ----------------------------------------------------------------------------------------
def _name_segments(name):
    """ Split a parameter/module name into lowercase path segments on '.' and '_'. """
    return [seg for seg in re.split(r"[._]", name.lower()) if seg]



def _is_contiguous_subsequence(segments, pattern_tokens):
    """
    True if 'pattern_tokens' (a list of segments) appears as a contiguous run within 'segments'.
    For single-token patterns this reduces to exact segment membership.
    """
    m= len(pattern_tokens)
    if m == 0 or m > len(segments):
        return False
    for i in range(len(segments) - m + 1):
        if segments[i:i + m] == pattern_tokens:
            return True
    return False



def setup_muon_optimizer(
    model, learning_rate:float, weight_decay:float, adamw_betas=(0.9, 0.95), adamw_eps=1e-10,
    muon_momentum=0.95, muon_nesterov=True, muon_ns_coefficients=(3.4445, -4.775, 2.0315),
    muon_eps=1e-10, muon_ns_steps=5, muon_adjust_lr_fn="match_rms_adamw", muon_lr_scale=1.0,
    exclude_from_muon=("input_norm", "embed", "embedding", "pos_embed", "position", "pos_emb", "token", "head",
                       "lm_head", "output", "prediction", "forecast", "gating", "covariates",),
    verbose=False,
):
    """
    Build a MultiOptimizer that applies:
    - Muon to eligible 2D hidden-layer weight matrices.
    - AdamW to all remaining parameters.

    Scoping policy. A parameter is routed to Muon only if all the following hold: it is 2D (param.ndim == 2),
    its name is not matched by 'exclude_from_muon', and its parent module is neither an embedding nor a
    normalization layer. Everything else is routed to AdamW.

    Name matching. 'exclude_from_muon' is matched against the '.'/'_'-delimited path segments of each parameter
    name (case-insensitive), not as a raw substring. A single-token pattern matches when it equals a segment
    ('head' matches '...head.weight' but not the 'head' inside 'overhead'); a multi-token pattern (e.g. 'lm_head')
    matches only as a contiguous run of segments. Run once with verbose=True and read the printed Muon/AdamW split
    to confirm the routing for your actual module names.
    NOTE this is a secondary filter: embeddings and normalization layers are also caught structurally via their
    parent module type, independent of names.

    Weight-decay policy. The AdamW split is purely dimensionality-based: tensors with ndim >= 2 are weight-decayed
    and tensors with ndim < 2 (biases, norm scales) are not.
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

    # tokenize the exclusion patterns once
    exclude_token_seqs= [seq for seq in (_name_segments(p) for p in exclude_from_muon) if seq]

    def is_excluded_by_name(param_name):
        segments= _name_segments(param_name)
        return any(_is_contiguous_subsequence(segments, seq) for seq in exclude_token_seqs)

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
    # Muon and AdamW share one schedule shape; their magnitudes come from each group's own base LR.
    # Sharing the same base is only meaningful when Muon's update is rescaled to AdamW's RMS as an
    # orthogonalized update has RMS 1/sqrt(max(m,n)): ~45x smaller than AdamW's at the same lr for
    # a 2048x1024 matrix. torch.optim.Muon defaults adjust_lr_fn=None, which maps to "original".
    if (muon_optimizer is not None) and (adamw_optimizer is not None) \
            and (muon_adjust_lr_fn != "match_rms_adamw") and (muon_lr_scale == 1.0):
        raise ValueError(
            f"adjust_lr_fn={muon_adjust_lr_fn!r} does not match Muon's update RMS to AdamW's, so the "
            f"two cannot share a base learning rate. Use adjust_lr_fn='match_rms_adamw', or pass an "
            f"explicit muon_lr_scale (typically 10-50x)."
        )
    if (muon_optimizer is not None) and (muon_adjust_lr_fn == "match_rms_adamw") and (muon_lr_scale != 1.0):
        warnings.warn(
            f"adjust_lr_fn='match_rms_adamw' already rescales Muon's update to AdamW's RMS "
            f"(x0.2*sqrt(max(m,n))); muon_lr_scale={muon_lr_scale} is applied on top of it. That is a "
            f"deliberate ratio only if you intend it -- passing sqrt(d_ff) here would apply the RMS "
            f"correction twice.", stacklevel=2
        )
    # the per-group base LR is the single source of truth: CosineLRDecay and every
    # torch.optim.lr_scheduler read it as 'initial_lr'
    if (muon_optimizer is not None) and (muon_lr_scale != 1.0):
        for group in muon_optimizer.param_groups:
            group["lr"]= learning_rate * muon_lr_scale

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
