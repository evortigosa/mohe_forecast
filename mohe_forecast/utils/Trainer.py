# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Trainer Class
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import logging
from dataclasses import asdict
from .ExpertUsageTracker import ExpertUsageTracker
from ..model.TSFT import TSFTransformer
from tqdm import tqdm



class Trainer:
    """
    Trainer class for training the model.
    - train_loader, val_loader, test_loader are DataLoader objects with train, val, test data.
    - train_ds_scaler is a sklearn StandardScaler from train_loader.
    - scheduler (optional): a per-step learning rate scheduler (not per epoch).
    - early_stopping (optional): early stopping utility.
    - use_time_features (optional): whether to use time covariates.
    """

    def __init__(self, model, device, train_loader, train_ds_scaler, val_loader, test_loader,
                 criterion, optimizer, scheduler=None, aux_criterion=None, early_stopping=None,
                 use_time_features=False, do_validation=True, augmentation=None, checkpointing=True,
                 checkpoint_dir=None, filename=None, verbose=False, disable_tqdm=False) -> None:
        self.model= model
        self.device= torch.device(device)
        self.train_loader= train_loader
        self.val_loader= val_loader
        self.test_loader= test_loader
        self.criterion= criterion
        self.optimizer= optimizer
        self.scheduler= scheduler
        self.aux_criterion= aux_criterion
        self.early_stopping= early_stopping
        self.use_time_features= use_time_features
        self.do_validation= do_validation
        self.augmentation= augmentation
        self.checkpointing= checkpointing
        self.checkpoint_dir= checkpoint_dir if checkpoint_dir is not None else 'checkpoints'
        self.filename= filename if filename is not None else 'tsft_checkpoint'
        self.verbose= verbose
        self.disable_tqdm= disable_tqdm
        # track statistics
        self.train_losses= []
        self.val_losses= []
        self.lr_hist= []
        self.expert_traker= None
        # train_ds_scaler for denormalization
        self.train_ds_scaler= train_ds_scaler
        # --- minimal logging ---
        self._log= self._build_logger(f"{self.__class__.__name__}")


    def _set_model(self):
        """ Helper. Set the model on device. """
        if getattr(self, "model", None) is not None:
            self.model= self.model.to(self.device)


    def _set_optimizer(self):
        """ Helper. Set the optimizer on device. """
        if getattr(self, "optimizer", None) is None:
            return

        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key]= value.to(self.device)


    def _set_loader(self, loader):
        """ Helper. Just return loader. """
        return loader


    def _move_to_device(self, x):
        """ Helper. Return a tensor on device. """
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return x


    def _unwrap_model(self):
        """ Helper to ensure compatibility with distributed training. """
        return self.model


    def _is_main_process(self) -> bool:
        """ Helper to ensure compatibility with distributed training. """
        return True


    def _barrier(self):
        """ Helper to ensure compatibility with distributed training. """
        pass


    def _print(self, *args, **kwargs):
        """ Helper to ensure compatibility with distributed training. """
        print(*args, **kwargs)


    def _reduce_sum(self, x:torch.Tensor) -> torch.Tensor:
        """ Helper to ensure compatibility with distributed training. """
        return x


    def _gather_variable_batch(self, x:torch.Tensor) -> torch.Tensor:
        """ Helper to ensure compatibility with distributed training. """
        return x


    def _backward(self, loss, clip_grad=None):
        """
        Helper to ensure compatibility with distributed training.
        Runs a backward pass to get the gradients with (optional) clip_grad_norm_.
        """
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)


    def _reduce_moe_metrics(self, metrics):
        """ Helper to ensure compatibility with distributed training. """
        return metrics


    def _save(self, state_file, state_path):
        """ Helper. Just save state_file at state_path. """
        torch.save(state_file, state_path)


    def _load(self, state_path, map_location="cpu", weights_only=True):
        """ Helper. Just load and return a file from state_path. """
        return torch.load(state_path, map_location=map_location, weights_only=weights_only)


    def get_checkpoint_path(self, filename:str, checkpoint_dir:str) -> str:
        """
        Return the path to the checkpoint file.
        """
        if (filename is not None) and (checkpoint_dir is None):
            # assume that filename holds the complete checkpoint_path
            checkpoint_path= os.path.join(filename)
        elif (filename is None) and (checkpoint_dir is not None):
            # filename takes the default value
            checkpoint_path= os.path.join(checkpoint_dir, f'{self.filename}.pth')
        else:
            if (filename is None) and (checkpoint_dir is None):
                checkpoint_dir= self.checkpoint_dir
                filename= f'{self.filename}.pth'
            checkpoint_path= os.path.join(str(checkpoint_dir), str(filename))

        return str(checkpoint_path)


    def _build_logger(self, name):
        """
        Build a logger for the Trainer class.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        log_path= self.get_checkpoint_path(f'{self.filename}.log', self.checkpoint_dir)

        logger= logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate= False

        fmt= logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        # file handler
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == log_path:
                return logger
        fh= logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        return logger


    def _set_log(self, log_type:str, message:str) -> None:
        """
        Set a log entry. log_type can be "info", "warning", or "error".
        """
        if not self._is_main_process():
            return

        if log_type.lower() == "warning" or log_type.lower() == "error":
            self._log.warning(message)
            log_type= f"[{log_type.upper()}]"
        else:
            self._log.info(message)
            log_type= "[INFO]"

        if self.verbose:
            print(f"{log_type} {message}")


    @staticmethod
    def _format_dt(seconds) -> str:
        ms= seconds * 1000
        seconds, milliseconds= divmod(ms, 1000)
        minutes, seconds= divmod(seconds, 60)
        hours, minutes= divmod(minutes, 60)

        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


    def _reset_cuda_memory_stats(self, empty_cache=False, reset_peak_memory=False):
        """
        Empty cache (empty_cache=True) and reset the maximum peak GPU memory usage
        (reset_peak_memory=True).
        """
        if self.device.type == 'cuda':
            if empty_cache:
                torch.cuda.empty_cache()
            if reset_peak_memory:
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)


    def _get_cuda_memory_stats(self) -> float:
        """
        Return the maximum peak GPU memory usage in GB (sum across ranks).
        """
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            peak_vram= torch.cuda.max_memory_allocated(self.device)/1024**3
            peak_vram= torch.tensor(peak_vram, device=self.device)
            peak_vram= self._reduce_sum(peak_vram)
            return float(peak_vram.item())

        return 0.


    def _get_minibatch(self, batch, do_augmentation=False, return_index=False):
        """
        Minibatch construction from a DataLoader batch.
        """
        sample_index= None

        if self.use_time_features:
            if len(batch) == 5:
                data, target, data_time, target_time, sample_index= batch
            elif len(batch) == 4:
                data, target, data_time, target_time= batch
            else:
                raise ValueError(f"Unexpected batch length with time features: {len(batch)}")
        else:
            if len(batch) == 3:
                data, target, sample_index= batch
            elif len(batch) == 2:
                data, target= batch
            else:
                raise ValueError(f"Unexpected batch length without time features: {len(batch)}")
            data_time  = None
            target_time= None

        if do_augmentation and self.augmentation is not None:
            data= self.augmentation(data)

        data, target= self._move_to_device(data), self._move_to_device(target)
        data_time, target_time= self._move_to_device(data_time), self._move_to_device(target_time)

        if not return_index:
            return data, target, data_time, target_time

        if sample_index is None:
            raise RuntimeError("Distributed test requires sample indices, but the dataset did not return them.")
        return data, target, data_time, target_time, self._move_to_device(sample_index)


    def set_forecast_horizon(self, horizon:int):
        """ Helper to ensure a correct setting of n_outputs (forecast_horizon). """
        model= self._unwrap_model()
        if hasattr(model, "n_outputs"):
            model.n_outputs= horizon


    @torch.inference_mode()
    def test(self, test_loader=None, inverse_transform=False, dynamic_window=True):
        """
        Test the model on a test set. Safe for uneven last batches and correct under distributed sampler padding.
        Returns test predictions and test labels.
        """
        test_loader= self.test_loader if test_loader is None else self._set_loader(test_loader)
        assert test_loader is not None, "test_loader cannot refer to None"
        self.model.eval()
        # all logits and trues to feed external metrics
        all_logits, all_trues, all_indices= [], [], []

        if self.train_ds_scaler is not None and inverse_transform:
            scale_= self._move_to_device(torch.from_numpy(self.train_ds_scaler.scale_).float().view(1, -1, 1))
            mean_ = self._move_to_device(torch.from_numpy(self.train_ds_scaler.mean_).float().view(1, -1, 1))
        else:
            inverse_transform= False
            scale_= 1.0
            mean_ = 0.0

        start= time.time()
        p_bar= self.disable_tqdm or not self._is_main_process()
        self._reset_cuda_memory_stats(empty_cache=True, reset_peak_memory=True)
        model_ref= self._unwrap_model()

        for batch in tqdm(test_loader, desc='Testing', disable=p_bar):
            # --- minibatch construction ---
            data, target, data_time, target_time, sample_index= self._get_minibatch(batch, return_index=True)

            # --- forward pass and get logits ---
            if getattr(model_ref, "forecasting", False):
                logits= model_ref.forecast(
                    data, ts_mark=data_time, ts_mark_future=target_time, dynamic_window=dynamic_window
                )
                if inverse_transform:
                    # invert the scaling back to the original units
                    logits= logits * scale_ + mean_
                    target= target * scale_ + mean_
            else:
                logits, *_= self.model(data, ts_mark=data_time)

            logits_store= self._gather_variable_batch(logits.detach())
            target_store= self._gather_variable_batch(target.detach())
            index_store = self._gather_variable_batch(sample_index.detach().view(-1))
            if self._is_main_process():
                all_logits.append(logits_store.float().cpu())
                all_trues.append(target_store.float().cpu())
                all_indices.append(index_store.cpu())

        peak_vram= self._get_cuda_memory_stats()
        end= time.time()
        dt = self._format_dt(end - start)
        self._set_log("info", f"test | max GPU mem=%.2fGB | dt=%sms" % (peak_vram, dt),)

        if not self._is_main_process():
            return None, None
        if len(all_logits) == 0:
            raise RuntimeError("No predictions were collected during test().")

        preds= torch.cat(all_logits, dim=0)
        trues= torch.cat(all_trues, dim=0)
        indices= torch.cat(all_indices, dim=0).long()
        # gather sample indices too, then sort and deduplicate (may happen in distributed evaluations)
        order= torch.argsort(indices)
        indices= indices[order]
        preds= preds[order]
        trues= trues[order]
        # deduplicate preds and trues
        keep= torch.ones_like(indices, dtype=torch.bool)
        keep[1:]= indices[1:] != indices[:-1]
        preds= preds[keep]
        trues= trues[keep]

        return preds, trues


    @torch.inference_mode()
    def validate(self, val_criterion=nn.MSELoss(reduction='none')):
        """
        Validate the model on a validation set.
        """
        self.model.eval()
        val_criterion= val_criterion if val_criterion is not None else self.criterion
        val_loss= torch.tensor(0.0, device=self.device)
        n_samples= torch.tensor(0, device=self.device, dtype=torch.long)
        p_bar= self.disable_tqdm or not self._is_main_process()

        for batch in tqdm(self.val_loader, desc='Validating', disable=p_bar):
            # --- minibatch construction ---
            data, target, data_time, *_= self._get_minibatch(batch)

            # --- forward pass and get loss ---
            logits, *_= self.model(data, ts_mark=data_time)
            losses= val_criterion(logits, target)
            loss= torch.mean(losses)

            val_loss += loss.detach() * data.size(0)
            n_samples+= data.size(0)

        val_loss = self._reduce_sum(val_loss)
        n_samples= self._reduce_sum(n_samples)
        val_loss= val_loss / n_samples.clamp_min(1)

        return float(val_loss.item())


    @torch.inference_mode()
    def validate_bf16(self, val_criterion=nn.MSELoss(reduction='none')):
        """
        Validate the model on a validation set using bfloat16.
        """
        assert self.device.type == 'cuda', "BF16 training requires CUDA"

        self.model.eval()
        val_criterion= val_criterion if val_criterion is not None else self.criterion
        val_loss= 0.0
        n_samples= 0

        for batch in tqdm(self.val_loader, desc='Validating', disable=self.disable_tqdm):
            # --- minibatch construction ---
            data, target, data_time, *_= self._get_minibatch(batch)

            # --- forward pass and get loss ---
            # model defined as usual; model parameters kept as float32
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, *_= self.model(data, ts_mark=data_time)
                losses= val_criterion(logits, target)
                loss= torch.mean(losses)

                val_loss += float(loss.item()) * data.size(0)
                n_samples+= data.size(0)

        val_loss= val_loss / max(n_samples, 1)

        return val_loss


    def train_one_epoch(self, epoch, clip_grad=None, get_moe_metrics=False):
        """
        Train the model for one epoch, returning the training loss and learning rate.
        """
        self.model.train()
        train_loss= torch.tensor(0.0, device=self.device)
        n_samples= torch.tensor(0, device=self.device, dtype=torch.long)
        n_steps= 0
        epoch_lr= 0.0
        p_bar= self.disable_tqdm or not self._is_main_process()

        # --- training steps ---
        for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}", disable=p_bar):
            self.optimizer.zero_grad(set_to_none=True)

            # --- minibatch construction ---
            data, target, data_time, *_= self._get_minibatch(batch)
            padding_mask= None

            # --- forward pass and get loss ---
            logits, router_probs, *_= self.model(data, ts_mark=data_time)
            # compute training loss on the scaled data
            losses= self.criterion(logits, target)
            loss= torch.mean(losses)
            # sample‑weighted average loss
            train_loss += loss.detach() * data.size(0)
            n_samples += data.size(0)

            if self.aux_criterion is not None:
                aux_loss, global_metrics, layer_metrics= self.aux_criterion(
                    router_probs, padding_mask, get_moe_metrics
                )
                loss= loss + aux_loss

                if get_moe_metrics and (self.expert_traker is not None):
                    global_metrics= self._reduce_moe_metrics(global_metrics)
                    layer_metrics = self._reduce_moe_metrics(layer_metrics)
                    self.expert_traker.update(global_metrics, layer_metrics)

            # check loss finite
            if not torch.isfinite(loss).all():
                self._set_log("error",
                    f"train_one_epoch | non_finite_loss | epoch=%d | loss=%s" % (epoch, str(loss.detach().cpu()))
                )
                # best to raise early to see where it happened
                raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}: {loss.detach().cpu()}")

            # --- backward pass to calculate the gradients ---
            self._backward(loss, clip_grad)

            # --- update the parameters using the gradient ---
            self.optimizer.step()
            # per-step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_lr += self.optimizer.param_groups[0]['lr']
            n_steps += 1

        if self.augmentation is not None:
            self.augmentation.step_epoch()

        train_loss= self._reduce_sum(train_loss)
        n_samples = self._reduce_sum(n_samples)
        train_loss= train_loss / n_samples.clamp_min(1)
        epoch_lr  = epoch_lr / n_steps

        return float(train_loss.item()), epoch_lr


    def train_one_epoch_bf16(self, epoch, clip_grad=None, get_moe_metrics=False):
        """
        Train the model for one epoch using bfloat16, returning the training loss and learning rate.
        """
        assert self.device.type == 'cuda', "BF16 training requires CUDA"

        self.model.train()
        train_loss= 0.0
        n_samples= 0
        n_steps= 0
        epoch_lr= 0.0

        # --- training steps ---
        for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}", disable=self.disable_tqdm):
            self.optimizer.zero_grad(set_to_none=True)

            # --- minibatch construction ---
            data, target, data_time, *_= self._get_minibatch(batch)
            padding_mask= None

            # --- forward pass and get loss ---
            # model, optimizer defined as usual; model parameters kept as float32
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, router_probs, *_= self.model(data, ts_mark=data_time)
                # compute training loss on the scaled data
                losses= self.criterion(logits, target)
                loss= torch.mean(losses)
                # sample‑weighted average loss
                train_loss += float(loss.item()) * data.size(0)
                n_samples += data.size(0)

                if self.aux_criterion is not None:
                    aux_loss, global_metrics, layer_metrics= self.aux_criterion(
                        router_probs, padding_mask, get_moe_metrics
                    )
                    loss= loss + aux_loss

                    if get_moe_metrics and (self.expert_traker is not None):
                        self.expert_traker.update(global_metrics, layer_metrics)

            # check loss finite
            if not torch.isfinite(loss).all():
                self._set_log("error",
                    f"train_one_epoch_bf16 | non_finite_loss | epoch=%d | loss=%s" % (epoch, str(loss.detach().cpu()))
                )
                # best to raise early to see where it happened
                raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}: {loss.detach().cpu()}")

            # --- backward pass to calculate the gradients ---
            # gradients computed in BF16, but accumulation and params remain TF32
            # BF16 does not need loss scaling with torch.amp.GradScaler()
            self._backward(loss, clip_grad)

            # --- update the parameters using the gradient ---
            self.optimizer.step()
            # per-step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_lr += self.optimizer.param_groups[0]['lr']
            n_steps += 1

        if self.augmentation is not None:
            self.augmentation.step_epoch()

        train_loss= train_loss / n_samples
        epoch_lr  = epoch_lr / max(n_steps, 1)

        return train_loss, epoch_lr


    def train(self, epochs, eval_interval=1, use_bf16=False, clip_grad=None, get_moe_metrics=False) -> None:
        """
        Train the model for a specified number of epochs, performing validation and checkpointing.
        """
        train_info= f'use_bf16={use_bf16}, clip_grad={clip_grad}, get_moe_metrics={get_moe_metrics}'
        opt_info  = f'weight_decay={self.optimizer.param_groups[0]["weight_decay"]:.2e}, betas={self.optimizer.param_groups[0]["betas"]}'
        scheduler_info= self.scheduler.extra_repr() if self.scheduler is not None else "N/A"
        m_config= getattr(self._unwrap_model(), "config", None)
        model_info= f'model full config: {m_config}'
        self._set_log("info", f"train | {train_info} | optimizer info: {opt_info} | scheduler info: {scheduler_info} | {model_info}")

        self._reset_cuda_memory_stats(empty_cache=True)

        if get_moe_metrics and self.expert_traker is None:
            self.expert_traker= ExpertUsageTracker(m_config.n_experts, m_config.n_layer)

        best_val_loss= float('inf')
        best_epoch= -1
        val_loss= 0.
        did_validation= False

        for epoch in range(epochs):
            start= time.time()
            self._reset_cuda_memory_stats(reset_peak_memory=True)

            if self.expert_traker is not None:
                self.expert_traker.reset_epoch()

            if self.augmentation is not None:
                self.augmentation.set_epoch(epoch)

            if use_bf16 is True:
                train_loss, epoch_lr= self.train_one_epoch_bf16(epoch+1, clip_grad, get_moe_metrics)
            else:
                train_loss, epoch_lr= self.train_one_epoch(epoch+1, clip_grad, get_moe_metrics)
            self.train_losses.append(train_loss)
            self.lr_hist.append(epoch_lr)

            if self.do_validation and (epoch % eval_interval == 0 or epoch == epochs-1):
                if use_bf16 is True:
                    val_loss= self.validate_bf16()
                else:
                    val_loss= self.validate()
                did_validation= True
            else:
                did_validation= False
            self.val_losses.append(val_loss)

            if self.expert_traker is not None:
                self.expert_traker.finalize_epoch()

            peak_vram= self._get_cuda_memory_stats()
            end= time.time()
            dt = self._format_dt(end - start)
            val_loss_log= f'{val_loss:.6f}' if did_validation else 'N/A'  # did_validation is False
            self._set_log("info",
                f"train | epoch=%d/%d | train_loss=%.6f | val_loss=%s | lr=%.4e | max GPU mem=%.2fGB | dt=%sms" % \
                (epoch+1, epochs, train_loss, val_loss_log, epoch_lr, peak_vram, dt)
            )

            if did_validation:
                if val_loss < best_val_loss:
                    # Save model if it's the best so far
                    best_val_loss= val_loss
                    best_epoch= epoch
                    if self.checkpointing:
                        self.save_checkpoint(epoch, best_val_loss)

                if self.early_stopping is not None:
                    # Watches validation MSE and halts training if it hasn't improved
                    avg_val_loss= float(np.mean(self.val_losses))
                    if self.early_stopping(avg_val_loss, epoch+1):
                        self._set_log(
                            "warning", f"train | early_stopping_triggered | epoch=%d | avg_val_loss=%.6f" % \
                            (epoch+1, avg_val_loss)
                        )
                        break

        if did_validation:
            self._set_log("info", f"train | Best Validation Loss: %.6f | Epoch: %d" % (best_val_loss, best_epoch+1))
            # Save a final checkpoint only if the last epoch equals the best epoch.
            if best_epoch == epochs - 1 and self.checkpointing:
                self.save_checkpoint(best_epoch, best_val_loss)


    def save_checkpoint(self, epoch, best_val_loss) -> None:
        """
        Save the model checkpoint to disk, including training history.
        - save expert tracker (if not None): traker_path is {checkpoint_path}_expert_traker.pt
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path= self.get_checkpoint_path(f'{self.filename}.pth', self.checkpoint_dir)
        # construct checkpoint dictionary
        model= self._unwrap_model()
        checkpoint= {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config': asdict(model.config),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'lr_hist': self.lr_hist,
            'timestamp': time.time(),
        }
        try:
            self._save(checkpoint, checkpoint_path)
            # (optional) save full expert tracker history in a separate file
            if self.expert_traker is not None:
                traker_path= self.get_checkpoint_path(f'{self.filename}_expert_traker.pt', self.checkpoint_dir)
                self._save(self.expert_traker.state_dict(), traker_path)
            self._set_log(
                "info", f"save_checkpoint | epoch=%d | best_val_loss=%.6f | saved at %s" % \
                 (epoch+1, best_val_loss, checkpoint_path)
            )
        except Exception as e:
            self._set_log("error", f"save_checkpoint | Failed to save checkpoint: {e}")
            raise e


    @staticmethod
    def _strip_module_prefix(state_dict):
        """
        Ignore all 'inv_freq' keys (from previous versions of RoPE with persistent register_buffer),
        as well as remove a single leading 'module.' from keys if present.
        """
        state_dict= {k: v for k, v in state_dict.items() if "inv_freq" not in k}

        if any(k.startswith("module.") for k in state_dict.keys()):
            return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        return state_dict


    def load_checkpoint(self, filename=None, checkpoint_dir=None, restore_optimizer=False, restore_metadata=False) -> tuple:
        """
        This method loads the checkpoint from the given path and restores the model, optimizer
        (optional, when restore_optimizer=True), and training history (optional, when restore_metadata=True).
        - restore expert tracker (optional): expected traker_path is {checkpoint_path}_expert_traker.pt
        """
        checkpoint_path= self.get_checkpoint_path(filename, checkpoint_dir)

        if not os.path.exists(checkpoint_path):
            self._set_log("error", f"load_checkpoint | Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        # ensure model exists before loading (build_model will create it)
        if getattr(self, 'model', None) is None:
            self._set_log("error", "load_checkpoint | self.model is None: instantiate model before restoring state_dict")
            raise RuntimeError("self.model is None: instantiate model before restoring state_dict")

        try:
            checkpoint= self._load(checkpoint_path, map_location='cpu', weights_only=True)
            # restore model state into the real underlying module
            model= self._unwrap_model()
            model.load_state_dict(self._strip_module_prefix(checkpoint['model_state_dict']))
            # ensure model on target device
            self._set_model()
            # restore optimizer state
            if restore_optimizer:
                if getattr(self, "optimizer", None) is None:
                    self._set_log("warning",
                        "load_checkpoint | restore_optimizer=True but self.optimizer is None: skipping optimizer."
                    )
                elif "optimizer_state_dict" not in checkpoint:
                    self._set_log("warning",
                        "load_checkpoint | restore_optimizer=True but optimizer_state_dict is missing: skipping optimizer."
                    )
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self._set_optimizer()
                    if self.scheduler is not None:
                        self.scheduler.optimizer= self.optimizer

            # retrieve training metadata
            epoch= checkpoint.get("epoch", 0)
            best_val_loss= checkpoint.get('best_val_loss', float('inf'))
            _time= checkpoint.get("timestamp", "N/A")
            _time= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_time)) if _time != "N/A" else "N/A"
            if restore_metadata:
                self.train_losses= checkpoint.get('train_losses', [])
                self.val_losses= checkpoint.get('val_losses', [])
                self.lr_hist= checkpoint.get('lr_hist', [])
                self.expert_traker= None
                # load (optional) expert tracker history from a separate file
                root_path, _= os.path.splitext(checkpoint_path)
                traker_path= f'{root_path}_expert_traker.pt'
                # load (optional) expert tracker history from the checkpoint file (old)
                tracker= checkpoint.get('expert_traker', None)

                if os.path.exists(traker_path) or tracker is not None:
                    if os.path.exists(traker_path):
                        tracker= self._load(traker_path, map_location="cpu", weights_only=True)
                    m_config= getattr(model, "config", None)
                    if m_config is None:
                        self._set_log("warning",
                            "load_checkpoint | Cannot restore expert_traker because model.config is missing."
                        )
                    else:
                        self.expert_traker= ExpertUsageTracker(m_config.n_experts, m_config.n_layer)
                        self.expert_traker.load_state_dict(tracker)
                else:
                    self._set_log("warning", "load_checkpoint | No MoE usage history to load: skipping expert_traker restore.")

            self._barrier()
            self._set_log("info",
                "load_checkpoint | Checkpoint loaded from '%s'. Resuming training on '%s' with best validation loss of %.6f." % \
                (checkpoint_path, _time, best_val_loss)
            )
            return epoch, best_val_loss

        except Exception as e:
            self._set_log("error", f"load_checkpoint | Failed to load checkpoint from {checkpoint_path}: {e}")
            raise e


    def build_model(self, filename=None, checkpoint_dir=None):
        """
        Build and return a new model (on CPU) from a given checkpoint.
        NOTE: This is a pre-setup method. It builds a fresh model before the distributed trainer wraps it.
        """
        checkpoint_path= self.get_checkpoint_path(filename, checkpoint_dir)

        if not os.path.exists(checkpoint_path):
            self._set_log("error", f"build_model | Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint= self._load(checkpoint_path, map_location='cpu', weights_only=True)
            if 'config' not in checkpoint:
                self._set_log("error", f"build_model | Checkpoint does not contain a 'config' key")
                raise KeyError("Checkpoint does not contain a 'config' key to build the model")
            # build a fresh model from config (dict) hyperparameters
            config_args= checkpoint['config']
            if not isinstance(config_args, dict):
                self._set_log("error", f"build_model | checkpoint['config'] should be a dict of constructor kwargs")
                raise TypeError("checkpoint['config'] should be a dict of constructor kwargs")

            self._set_log("info", f"build_model | Building a new model with config: {config_args}")
            model= TSFTransformer(**config_args)

            return model

        except Exception as e:
            self._set_log("error", f"build_model | Failed to build and load checkpoint from {checkpoint_path}: {e}")
            raise e


    def _save_plot(self, plt_obj, file_name, as_pdf, info_message, tight=True) -> None:
        plots_dir= f"{self.checkpoint_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        save_path= self.get_checkpoint_path(file_name, plots_dir)

        if tight:
            plt_obj.tight_layout()
        if as_pdf:
            save_path= f'{save_path}.pdf'
            plt_obj.savefig(save_path, dpi=300, pad_inches=0.01, bbox_inches="tight")
        else:
            save_path= f'{save_path}.svg'
            plt_obj.savefig(save_path, pad_inches=0.01, bbox_inches="tight")

        self._set_log("info", f"{info_message}: {save_path}")


    def plot_results(self, cut_first_epoch=False, show_plot=True, save_charts=False, as_pdf=False,
                     file_name='training_results'):
        """
        Plot training metrics including training loss, validation loss, and learning rate history
        over epochs. Plots can be saved as checkpoint_dir/file_name.
        """
        method_name= "plot_results"

        if len(self.train_losses) == 0:
            info_message= "No training/validation history available to plot."
            self._set_log("warning", f"{method_name} | {info_message}")
            return

        epochs= range(1, len(self.train_losses) + 1)
        if cut_first_epoch:
            epochs= epochs[1:]
            train_losses= self.train_losses[1:]
            val_losses= self.val_losses[1:]
            lr_hist= self.lr_hist[1:]
        else:
            train_losses= self.train_losses
            val_losses= self.val_losses
            lr_hist= self.lr_hist

        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-')
        plt.plot(epochs, val_losses, label='Validation Loss', marker='o', linestyle='-')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, lr_hist, label='Learning Rate', marker='o', linestyle='-', color='tab:green')
        plt.title('Learning Rate History')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)

        if save_charts:
            self._save_plot(plt, file_name, as_pdf, f"{method_name} | Training charts were saved at")
        if show_plot:
            plt.show()
        plt.close()


    def plot_expert_routing_diagnostics(self, cut_first_epoch=False, show_plot=True, save_charts=False,
                                        as_pdf=False, file_name="expert_routing_diagnostics"):
        """
        MoE metrics. Plot global routing health metrics over epochs  (entropy, dead experts, and CV).
        Plots can be saved as checkpoint_dir/file_name.
        """
        method_name= "plot_expert_routing_diagnostics"

        entropy_hist, dead_hist, cv_hard_hist, cv_soft_hist= [], [], [], []
        if self.expert_traker is not None:
            entropy_hist= self.expert_traker.history["epoch_entropy"]
            dead_hist   = self.expert_traker.history["epoch_dead_experts"]
            cv_hard_hist= self.expert_traker.history["epoch_cv_hard"]
            cv_soft_hist= self.expert_traker.history["epoch_cv_soft"]

        if len(entropy_hist) == 0:
            info_message= "No routing diagnostic history available to plot."
            self._set_log("warning", f"{method_name} | {info_message}")
            return

        epochs= list(range(1, len(entropy_hist) + 1))
        if cut_first_epoch and len(epochs) > 1:
            epochs= epochs[1:]
            entropy_hist= entropy_hist[1:]
            dead_hist= dead_hist[1:]
            cv_hard_hist= cv_hard_hist[1:]
            cv_soft_hist= cv_soft_hist[1:]

        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, entropy_hist, label="Router Entropy", marker="o", linestyle="-")
        plt.plot(epochs, dead_hist, label="Dead Experts", marker="o", linestyle="-")
        plt.title("Routing Collapse Indicators")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, cv_hard_hist, label="CV Hard", marker="o", linestyle="-")
        plt.plot(epochs, cv_soft_hist, label="CV Soft", marker="o", linestyle="-")
        plt.title("Routing Imbalance")
        plt.xlabel("Epochs")
        plt.ylabel("Coefficient of Variation")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

        if save_charts:
            self._save_plot(plt, file_name, as_pdf, f"{method_name} | Routing diagnostic charts were saved at")
        if show_plot:
            plt.show()
        plt.close()


    def _get_expert_hist(self, is_global:bool, method_name:str):

        if is_global:
            hard_hist, soft_hist= [], []
            if self.expert_traker is not None:
                hard_hist= self.expert_traker.history["epoch_hard_fraction"]
                soft_hist= self.expert_traker.history["epoch_soft_fraction"]

            if len(hard_hist) == 0 or len(soft_hist) == 0:
                info_message= "No expert usage history available to plot."
                self._set_log("warning", f"{method_name} | {info_message}")
                return None, None

            return hard_hist, soft_hist

        layer_epoch= []
        if self.expert_traker is not None:
            layer_epoch= self.expert_traker.history["layer_epoch"]

        if len(layer_epoch) == 0:
            info_message= "No layerwise expert usage history available to plot."
            self._set_log("warning", f"{method_name} | {info_message}")
            return None

        return layer_epoch


    @staticmethod
    def _set_expert_usage_plot(hard_hist, soft_hist, cut_first_epoch:bool, layer_id=None):
        hard_hist= torch.stack(hard_hist).cpu()   # [n_epochs, E]
        soft_hist= torch.stack(soft_hist).cpu()   # [n_epochs, E]

        epochs= list(range(1, hard_hist.size(0) + 1))
        if cut_first_epoch and len(epochs) > 1:
            epochs= epochs[1:]
            hard_hist= hard_hist[1:]
            soft_hist= soft_hist[1:]

        n_experts= hard_hist.size(1)
        legend_ncol= min(n_experts, 8)
        handles= []
        labels = []
        fig, axes= plt.subplots(1, 2, figsize=(14, 5))

        # hard utilization
        for expert_id in range(n_experts):
            line,= axes[0].plot(
                epochs, hard_hist[:, expert_id].numpy(), marker="o", linestyle="-",
                label=f"Expert {expert_id}"
            )
            handles.append(line)
            labels.append(f"Expert {expert_id}")
        if layer_id is None:
            axes[0].set_title("Global Expert Hard Utilization")
        else:
            axes[0].set_title(f"Layer {layer_id} Hard Utilization")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Hard Fraction")
        axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

        # soft importance
        for expert_id in range(n_experts):
            axes[1].plot(
                epochs, soft_hist[:, expert_id].numpy(), marker="o", linestyle="-",
                label=f"Expert {expert_id}"
            )
        if layer_id is None:
            axes[1].set_title("Global Expert Soft Importance")
        else:
            axes[1].set_title(f"Layer {layer_id} Soft Importance")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Soft Fraction")
        axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

        fig.legend(
            handles, labels, loc="lower center", fancybox=True, ncol=legend_ncol, frameon=True,
            bbox_to_anchor=(0.5, -0.04), fontsize=9
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))

        return fig, axes


    def plot_expert_usage_global(self, cut_first_epoch=False, show_plot=True, save_charts=False,
                                 as_pdf=False, file_name="expert_usage_global"):
        """
        MoE metrics. Plot global expert hard/soft utilization over epochs.
        Plots can be saved as checkpoint_dir/file_name.
        """
        method_name= "plot_expert_usage_global"

        hard_hist, soft_hist= self._get_expert_hist(True, method_name)
        if hard_hist is None or soft_hist is None:
            return

        fig, axes= self._set_expert_usage_plot(hard_hist, soft_hist, cut_first_epoch)

        if save_charts:
            self._save_plot(fig, file_name, as_pdf, f"{method_name} | Expert usage charts were saved at")
        if show_plot:
            plt.show()
        plt.close(fig)


    def plot_expert_usage_layerwise(self, cut_first_epoch=False, show_plot=True, save_charts=False,
                                    as_pdf=False, file_name="expert_usage_layer"):
        """
        MoE metrics. Plot per-layer expert hard/soft utilization over epochs.
        Plots can be saved as checkpoint_dir/file_name. Saves one figure per MoE layer.
        """
        method_name= "plot_expert_usage_layerwise"

        layer_epoch= self._get_expert_hist(False, method_name)
        if layer_epoch is None:
            return

        for layer_id, layer_hist in layer_epoch.items():
            hard_hist= layer_hist["hard_fraction"]
            soft_hist= layer_hist["soft_fraction"]

            if len(hard_hist) == 0 or len(soft_hist) == 0:
                continue

            fig, axes= self._set_expert_usage_plot(hard_hist, soft_hist, cut_first_epoch, layer_id)

            if save_charts:
                file_name_curr= f"{file_name}_{layer_id}"
                info_message  = f"Layer {layer_id} expert usage charts were saved at"
                self._save_plot(fig, file_name_curr, as_pdf, f"{method_name} | {info_message}")
            if show_plot:
                plt.show()
            plt.close(fig)


    @staticmethod
    def _set_expert_usage_heatmap(hard_hist, soft_hist, cut_first_epoch:bool, layer_id=None):

        def set_sparse_epoch_ticks(ax, num_epochs, max_ticks=10):
            """
            helper to avoid crowded x ticks.
            """
            step= max(1, int(np.ceil(num_epochs / max_ticks)))
            ticks= list(range(0, num_epochs, step))
            labels= [str(t + 1) for t in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)

        hard_mat= torch.stack(hard_hist).cpu().numpy()  # [n_epochs, E]
        soft_mat= torch.stack(soft_hist).cpu().numpy()  # [n_epochs, E]

        if cut_first_epoch and hard_mat.shape[0] > 1:
            hard_mat= hard_mat[1:]
            soft_mat= soft_mat[1:]
        n_epochs, n_experts = hard_mat.shape
        # transpose so that experts are rows and epochs are columns.
        hard_plot= hard_mat.T  # [E, n_epochs]
        soft_plot= soft_mat.T  # [E, n_epochs]
        vmin= 0.0
        vmax= max(float(hard_plot.max()), float(soft_plot.max()))
        fig, axes= plt.subplots(1, 2, figsize=(16, 5.5))

        # hard utilization
        axes[0].imshow(
            hard_plot, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax,
        )
        if layer_id is None:
            axes[0].set_title("Global Expert Hard Utilization")
        else:
            axes[0].set_title(f"Layer {layer_id} Hard Utilization")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Expert ID")
        set_sparse_epoch_ticks(axes[0], n_epochs)
        axes[0].set_yticks(range(n_experts))
        axes[0].set_yticklabels(range(n_experts))

        # soft importance
        im1= axes[1].imshow(
            soft_plot, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax,
        )
        if layer_id is None:
            axes[1].set_title("Global Expert Soft Importance")
        else:
            axes[1].set_title(f"Layer {layer_id} Soft Importance")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Expert ID")
        set_sparse_epoch_ticks(axes[1], n_epochs)
        axes[1].set_yticks(range(n_experts))
        axes[1].set_yticklabels(range(n_experts))

        # one shared colorbar for both heatmaps
        fig.subplots_adjust(right=0.95, wspace=0.09)
        cbar_ax= fig.add_axes(rect=(0.96, 0.16, 0.010, 0.68))
        cbar= fig.colorbar(im1, cax=cbar_ax)
        cbar.set_label("Fraction")

        return fig, axes


    def plot_expert_usage_global_heatmap(self, cut_first_epoch=False, show_plot=True, save_charts=False,
                                         as_pdf=False, file_name="expert_usage_global_heatmap"):
        """
        Plot global expert-usage heatmaps for hard/soft utilization side by side, with experts
        on the y-axis and epochs on the x-axis. Plots can be saved as checkpoint_dir/file_name.
        """
        method_name= "plot_expert_usage_global_heatmap"

        hard_hist, soft_hist= self._get_expert_hist(True, method_name)
        if hard_hist is None or soft_hist is None:
            return

        fig, axes= self._set_expert_usage_heatmap(hard_hist, soft_hist, cut_first_epoch)

        if save_charts:
            info_message= "Expert usage heatmaps were saved at"
            self._save_plot(fig, file_name, as_pdf, f"{method_name} | {info_message}", tight=False)
        if show_plot:
            plt.show()
        plt.close(fig)


    def plot_expert_usage_layerwise_heatmap(self, cut_first_epoch=False, show_plot=True, save_charts=False,
                                            as_pdf=False, file_name="expert_usage_layer_heatmap"):
        """
        Plot per-layer expert-usage heatmaps for hard/soft utilization side by side, with experts
        on the y-axis and epochs on the x-axis. Plots can be saved as checkpoint_dir/file_name.
        Saves one figure per MoE layer.
        """
        method_name= "plot_expert_usage_layerwise_heatmap"

        layer_epoch= self._get_expert_hist(False, method_name)
        if layer_epoch is None:
            return

        for layer_id, layer_hist in layer_epoch.items():
            hard_hist= layer_hist["hard_fraction"]
            soft_hist= layer_hist["soft_fraction"]

            if len(hard_hist) == 0 or len(soft_hist) == 0:
                continue

            fig, axes= self._set_expert_usage_heatmap(hard_hist, soft_hist, cut_first_epoch, layer_id)

            if save_charts:
                file_name_curr= f"{file_name}_{layer_id}"
                info_message  = f"Layer {layer_id} expert usage heatmaps were saved at"
                self._save_plot(fig, file_name_curr, as_pdf, f"{method_name} | {info_message}", tight=False)
            if show_plot:
                plt.show()
            plt.close(fig)
