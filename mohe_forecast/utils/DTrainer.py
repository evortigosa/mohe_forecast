# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Distributed Trainer Class
"""

import torch
from .Trainer import Trainer



class DTrainer(Trainer):
    """
    Trainer class with distributed-specific logic using Lightning Fabric.
    - model creation, data-loader creation, and optimizer creation must happen after fabric.launch()
    - model, optimizer, and loaders must be provided on CPU
    See https://lightning.ai/docs/fabric/stable/
    """

    def __init__(self, fabric, **kwargs,) -> None:
        super().__init__(**kwargs)
        # create the Fabric object
        self.fabric= fabric
        self._setup()


    def _set_model(self):
        pass


    def _set_optimizer(self):
        if getattr(self, "optimizer", None) is None:
            return

        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key]= self.fabric.to_device(value)


    def _set_loader(self, loader):
        return self.fabric.setup_dataloaders(loader, move_to_device=False)


    def _move_to_device(self, x):
        if isinstance(x, torch.Tensor):
            return self.fabric.to_device(x)
        return x


    def _setup(self):
        self.device= self.fabric.device
        # setup model and optimizer
        assert self.model is not None, "Model must be provided."
        assert self.optimizer is not None, "Optimizer must be provided."
        self.model, self.optimizer= self.fabric.setup(self.model, self.optimizer)
        # setup dataloaders
        if self.train_loader is not None:
            self.train_loader= self._set_loader(self.train_loader)
        if self.val_loader is not None:
            self.val_loader= self._set_loader(self.val_loader)
        if self.test_loader is not None:
            self.test_loader= self._set_loader(self.test_loader)


    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model


    def _is_main_process(self) -> bool:
        return self.fabric.is_global_zero


    def _barrier(self):
        self.fabric.barrier()


    def _print(self, *args, **kwargs):
        self.fabric.print(*args, **kwargs)


    def _reduce_sum(self, x:torch.Tensor) -> torch.Tensor:
        return self.fabric.all_reduce(x, reduce_op="sum")


    def _gather_variable_batch(self, x:torch.Tensor) -> torch.Tensor:
        """
        Gather tensors with possibly different local batch sizes across ranks.
        Assumption:
        - the batch dimension is dim=0, and all non-batch dimensions are identical across ranks.
        Returns:
        - a tensor containing the concatenation of all valid local tensors from all ranks, on every rank.
        """
        if not hasattr(self, "fabric") or self.fabric.world_size == 1:
            # for single-GPU / non-distributed execution
            return x

        local_n= torch.tensor([x.size(0)], device=x.device, dtype=torch.long)
        # shape: [world_size, 1] -> [world_size]
        all_n= self.fabric.all_gather(local_n).reshape(-1)
        max_n= int(all_n.max().item())
        local_n_int= int(local_n.item())

        if local_n_int < max_n:
            pad_shape= list(x.shape)
            pad_shape[0]= max_n - local_n_int
            pad= torch.zeros(pad_shape, dtype=x.dtype, device=x.device,)
            x_pad= torch.cat([x, pad], dim=0)
        else:
            x_pad= x

        # shape: x_pad -> [max_n, ...]; gathered -> [world_size, max_n, ...]
        gathered= self.fabric.all_gather(x_pad)
        # if no rank dimension exists, return local valid part
        if gathered.ndim == x_pad.ndim:
            return gathered[:local_n_int]

        pieces= []
        for rank_idx, n in enumerate(all_n.tolist()):
            pieces.append(gathered[rank_idx, :int(n)])

        return torch.cat(pieces, dim=0)


    def _backward(self, loss, clip_grad=None):
        self.fabric.backward(loss)
        if clip_grad is not None:
            self.fabric.clip_gradients(self.model, self.optimizer, max_norm=clip_grad)


    def _reduce_moe_metrics(self, metrics):
        """
        Aggregate MoE routing diagnostics across distributed ranks. Reduce the sufficient statistics
        (token_counts, prob_mass, num_tokens) and recompute all derived quantities globally.
        - TODO: This is a WIP.
        """
        return metrics


    def _save(self, state_file, state_path):
        self.fabric.save(state_path, state_file)


    def _load(self, state_path, map_location="cpu", weights_only=True):
        return self.fabric.load(state_path)


    def train(self, *args, **kwargs) -> None:
        """
        - TODO: get_moe_metrics=True
        """
        finfo= (
            f"accelerator={self.fabric.accelerator.__class__.__name__.replace('Accelerator', '').lower()}, "
            f"devices={self.fabric.world_size}, precision={self.fabric._precision.precision}, "
            f"strategy={self.fabric.strategy.__class__.__name__.replace('Strategy', '').lower()}"
        )
        self._set_log("info", f"train | Fabric {finfo}")

        kwargs["use_bf16"]= None  # not necessary, fabric handles precision
        kwargs["get_moe_metrics"]= False
        super().train(*args, **kwargs)
