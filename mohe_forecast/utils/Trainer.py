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


    def _build_logger(self, name):
        """
        Build a logger for the Trainer class.
        """
        filename= f'{self.filename}.log'
        checkpoint_dir= self.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_path= os.path.abspath(os.path.join(checkpoint_dir, filename))

        logger= logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate= False

        fmt= logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
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


    def train_one_epoch(self, epoch, clip_grad=None, get_moe_metrics=False):
        """
        Train the model for one epoch, returning the training loss and learning rate.
        """
        self.model.train()
        train_loss= 0.0
        n_samples= 0
        n_steps= 0
        epoch_lr= 0.0

        # --- training steps ---
        for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}", disable=self.disable_tqdm):
            self.optimizer.zero_grad(set_to_none=True)

            # --- minibatch construction ---
            if self.use_time_features:
                data, target, data_time, _= batch
                data_time= data_time.to(self.device)
            else:
                data, target= batch
                data_time= None

            data  = data.to(self.device)
            target= target.to(self.device)
            padding_mask= None
            if self.augmentation is not None:
                data= (self.augmentation(data)).to(self.device)

            # --- forward pass and get loss ---
            logits, router_logits, *_= self.model(data, ts_mark=data_time)
            # compute training loss on the scaled data
            losses= self.criterion(logits, target)
            loss= torch.mean(losses)

            # sample‑weighted average loss
            train_loss += float(loss.item()) * data.size(0)
            n_samples += data.size(0)

            if self.aux_criterion is not None:
                aux_loss, global_metrics, layer_metrics= self.aux_criterion(
                    router_logits, padding_mask, get_moe_metrics
                )
                loss= loss + aux_loss

                if get_moe_metrics and self.expert_traker is not None:
                    self.expert_traker.update(global_metrics, layer_metrics)

            # check loss finite
            if not torch.isfinite(loss).all():
                self._log.warning(
                    "train_one_epoch | non_finite_loss | epoch=%d | loss=%s", epoch, str(loss.detach().cpu())
                )
                # best to raise early to see where it happened
                raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}: {loss}")

            # --- backward pass to calculate the gradients ---
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

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
        epoch_lr  = epoch_lr / n_steps

        return train_loss, epoch_lr


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
            if self.use_time_features:
                data, target, data_time, _= batch
                data_time= data_time.to(self.device)
            else:
                data, target= batch
                data_time= None

            data  = data.to(self.device)
            target= target.to(self.device)
            padding_mask= None
            if self.augmentation is not None:
                data= (self.augmentation(data)).to(self.device)

            # --- forward pass and get loss ---
            # model, optimizer defined as usual; model parameters kept as float32
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, router_logits, *_= self.model(data, ts_mark=data_time)
                # compute training loss on the scaled data
                losses= self.criterion(logits, target)
                loss= torch.mean(losses)

                # sample‑weighted average loss
                train_loss += float(loss.item()) * data.size(0)
                n_samples += data.size(0)

                if self.aux_criterion is not None:
                    aux_loss, global_metrics, layer_metrics= self.aux_criterion(
                        router_logits, padding_mask, get_moe_metrics
                    )
                    loss= loss + aux_loss

                    if get_moe_metrics and self.expert_traker is not None:
                        self.expert_traker.update(global_metrics, layer_metrics)

            # check loss finite
            if not torch.isfinite(loss).all():
                self._log.warning(
                    "train_one_epoch_bf16 | non_finite_loss | epoch=%d | loss=%s", epoch, str(loss.detach().cpu())
                )
                # best to raise early to see where it happened
                raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}: {loss}")

            # --- backward pass to calculate the gradients ---
            # gradients computed in BF16, but accumulation and params remain TF32
            # BF16 does not need loss scaling with torch.amp.GradScaler()
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

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
        epoch_lr  = epoch_lr / n_steps

        return train_loss, epoch_lr


    def validate(self, val_criterion=nn.MSELoss(reduction='none')):
        """
        Validate the model on a validation set.
        """
        self.model.eval()
        val_criterion= val_criterion if val_criterion is not None else self.criterion
        val_loss= 0.0
        n_samples= 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating', disable=self.disable_tqdm):
                # --- minibatch construction ---
                if self.use_time_features:
                    data, target, data_time, _= batch
                    data_time= data_time.to(self.device)
                else:
                    data, target= batch
                    data_time= None
                data  = data.to(self.device)
                target= target.to(self.device)

                # --- forward pass and get loss ---
                logits, *_= self.model(data, ts_mark=data_time)
                losses= val_criterion(logits, target)
                loss= torch.mean(losses)

                val_loss += float(loss.item()) * data.size(0)
                n_samples+= data.size(0)

        val_loss= val_loss / n_samples

        return val_loss


    def train(self, epochs, eval_interval=1, use_bf16=False, clip_grad=None, get_moe_metrics=False) -> None:
        """
        Train the model for a specified number of epochs, performing validation and checkpointing.
        """
        self._log.info(f"train | use_bf16={use_bf16}, clip_grad={clip_grad}, get_moe_metrics={get_moe_metrics}")
        self._log.info(f"train | Model full config: {self.model.config}")

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        if get_moe_metrics and self.expert_traker is None:
            self.expert_traker= ExpertUsageTracker(self.model.config.n_experts, self.model.config.n_layer)

        best_val_loss= float('inf')
        best_epoch= -1
        val_loss= 10.
        did_validation= False

        for epoch in range(epochs):
            start= time.time()
            if self.expert_traker is not None:
                self.expert_traker.reset_epoch()

            if use_bf16:
                train_loss, epoch_lr= self.train_one_epoch_bf16(epoch+1, clip_grad, get_moe_metrics)
            else:
                train_loss, epoch_lr= self.train_one_epoch(epoch+1, clip_grad, get_moe_metrics)
            self.train_losses.append(train_loss)
            self.lr_hist.append(epoch_lr)

            if self.do_validation and (epoch % eval_interval == 0 or epoch == epochs-1):
                val_loss= self.validate()
                did_validation= True
            else:
                did_validation= False
            self.val_losses.append(val_loss)

            if self.expert_traker is not None:
                self.expert_traker.finalize_epoch()

            end= time.time()
            dt = end - start

            if did_validation:
                self._log.info(
                    "train | epoch=%d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.3e | dt=%.2fs",
                    epoch + 1, epochs, train_loss, val_loss, epoch_lr, dt
                )
                if self.verbose:
                    print(f'Train loss: {train_loss:.4f}')
                    print(f'Valid loss: {val_loss:.4f} | epoch: {epoch + 1} | dt/epoch: {dt*1000:.2f}ms')
            else:  # did_validation is False
                self._log.info(
                    "train | epoch=%d/%d | train_loss=%.6f | val_loss not computed | lr=%.3e | dt=%.2fs",
                    epoch + 1, epochs, train_loss, epoch_lr, dt
                )
                if self.verbose:
                    print(f'Train loss: {train_loss:.4f} | epoch: {epoch + 1} | dt/epoch: {dt*1000:.2f}ms')

            if did_validation:
                if val_loss < best_val_loss:
                    # Save model if it's the best so far
                    best_val_loss= val_loss
                    best_epoch= epoch
                    if self.checkpointing:
                        self.save_checkpoint(epoch, best_val_loss)

                if self.early_stopping is not None:
                    # Watches validation MSE and halts training if it hasn't improved
                    avg_val_loss= np.mean(self.val_losses)
                    if self.early_stopping(avg_val_loss, epoch+1):
                        self._log.warning(
                            "train | early_stopping_triggered | epoch=%d | avg_val_loss=%.6f",
                            epoch + 1, avg_val_loss
                        )
                        if self.verbose:
                            print(f'[WARNING] Early stopping triggered during training at epoch {epoch+1}')
                        break

        if did_validation:
            self._log.info("train | Best Validation Loss: %.6f | Epoch: %d", best_val_loss, best_epoch + 1)
            if self.verbose:
                print(f'Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch + 1})')
            # Save a final checkpoint only if the last epoch equals the best epoch.
            if best_epoch == epochs - 1 and self.checkpointing:
                self.save_checkpoint(best_epoch, best_val_loss)


    def test(self, test_loader=None, test_criterion=nn.MSELoss(reduction='none'), inverse_transform=False):
        """
        Test the model on a test set.
        Returns the mean test loss, test predictions, and test labels.
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        test_loader= self.test_loader if test_loader is None else test_loader
        assert test_loader is not None, "test_loader cannot refer to None"

        if self.train_ds_scaler is not None:
            inverse_transform= inverse_transform
            scale_= torch.from_numpy(self.train_ds_scaler.scale_).float().view(1,-1,1).to(self.device)
            mean_ = torch.from_numpy(self.train_ds_scaler.mean_).float().view(1,-1,1).to(self.device)
        else:
            inverse_transform= False
            scale_= 1.
            mean_ = 0.

        self.model.eval()
        test_criterion= test_criterion if test_criterion is not None else self.criterion
        test_loss= 0.0
        n_samples= 0
        all_logits, all_trues= [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing', disable=self.disable_tqdm):
                # --- minibatch construction ---
                if self.use_time_features:
                    data, target, data_time, target_time= batch
                    data_time  = data_time.to(self.device)
                    target_time= target_time.to(self.device)
                else:
                    data, target= batch
                    data_time  = None
                    target_time= None
                data  = data.to(self.device)
                target= target.to(self.device)

                # --- forward pass and get loss ---
                if self.model.forecasting:
                    logits= self.model.forecast(data, ts_mark=data_time, ts_mark_future=target_time)
                    if inverse_transform:
                        # invert the scaling back to the original units
                        logits= logits * scale_ + mean_
                        target= target * scale_ + mean_
                else:
                    logits, *_= self.model(data, ts_mark=data_time)
                losses= test_criterion(logits, target)
                loss= torch.mean(losses)

                # --- register preds and trues ---
                test_loss += float(loss.item()) * data.size(0)
                n_samples += data.size(0)
                all_logits.append(logits.cpu())
                all_trues.append(target.cpu())

        test_loss= test_loss / n_samples

        return test_loss, torch.cat(all_logits, dim=0), torch.cat(all_trues, dim=0)


    def get_checkpoint_path(self, filename:str, checkpoint_dir:str):

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


    def save_checkpoint(self, epoch, best_val_loss) -> None:
        """
        Save the model checkpoint to disk, including training history.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path= self.get_checkpoint_path(f'{self.filename}.pth', self.checkpoint_dir)
        # construct checkpoint dictionary
        checkpoint= {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.model.config),
            'best_val_loss': best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'lr_hist': self.lr_hist,
            'expert_traker': self.expert_traker.state_dict() if self.expert_traker is not None else None,
            'timestamp': time.time(),
        }
        try:
            torch.save(checkpoint, checkpoint_path)
            self._log.info(
                "save_checkpoint | epoch=%d | best_val_loss=%.6f | saved at %s",
                epoch + 1, best_val_loss, checkpoint_path
            )
            if self.verbose:
                print(f"[INFO] Checkpoint saved at '{checkpoint_path}'")
        except Exception as e:
            self._log.warning("save_checkpoint | Failed to save checkpoint: %s", e)
            if self.verbose:
                print(f"[ERROR] Failed to save checkpoint: {e}")
            raise e


    @staticmethod
    def strip_module_prefix(state_dict):
        """
        Remove a single leading 'module.' from keys if present.
        """
        if any(k.startswith("module.") for k in state_dict.keys()):
            return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        return state_dict


    def load_checkpoint(self, filename=None, checkpoint_dir=None, restore_optimizer=False,
                        restore_metadata=False) -> tuple:
        """
        This method loads the checkpoint from the given path and restores the model, optimizer
        (optional, when restore_optimizer=True), and training history.
        - TODO: Fix dtype, device, and layout for optimizer state loading.
        """
        checkpoint_path= self.get_checkpoint_path(filename, checkpoint_dir)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        # ensure model exists before loading (build_model will create it)
        if getattr(self, 'model', None) is None:
            raise RuntimeError("self.model is None: instantiate model before restoring state_dict")

        try:
            checkpoint= torch.load(checkpoint_path, map_location=self.device)
            # restore model state
            self.model.load_state_dict(self.strip_module_prefix(checkpoint['model_state_dict']))
            # ensure model on target device
            if getattr(self, 'model', None) is not None:
                self.model.to(self.device)
            # restore optimizer state
            if restore_optimizer:
                if getattr(self, "optimizer", None) is None:
                    self._log.warning(
                        "load_checkpoint | Checkpoint contains optimizer state, but self.optimizer is None: skipping optimizer restore."
                    )
                    if self.verbose:
                        print("[WARNING] Checkpoint contains optimizer state, but self.optimizer is None: skipping optimizer restore.")
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if self.scheduler is not None:
                        self.scheduler.optimizer= self.optimizer

            # retrieve training metadata
            epoch= checkpoint['epoch']
            best_val_loss= checkpoint.get('best_val_loss', float('inf'))
            if restore_metadata:
                self.train_losses= checkpoint.get('train_losses', [])
                self.val_losses= checkpoint.get('val_losses', [])
                self.lr_hist= checkpoint.get('lr_hist', [])
                self.expert_traker= None
                tracker= checkpoint.get('expert_traker', None)
                if tracker is not None:
                    self.expert_traker= ExpertUsageTracker(self.model.config.n_experts, self.model.config.n_layer)
                    self.expert_traker.load_state_dict(tracker)

            self._log.info(
                "load_checkpoint | Checkpoint loaded from '%s'. Resuming training with best validation loss of %.4f.",
                checkpoint_path, best_val_loss
            )
            if self.verbose:
                print(f"[INFO] Checkpoint loaded from '{checkpoint_path}'. Resuming training with best validation loss of {best_val_loss:.4f}.")
            return epoch, best_val_loss

        except Exception as e:
            self._log.warning(
                "load_checkpoint | Failed to load checkpoint from %s: %s", checkpoint_path, e
            )
            if self.verbose:
                print(f"[ERROR] Failed to load checkpoint from {checkpoint_path}: {e}")
            raise e


    def build_model(self, filename=None, checkpoint_dir=None, restore_model=False, restore_optimizer=False,
                    restore_metadata=False) -> tuple:
        """
        Build a model from a given checkpoint.
        """
        checkpoint_path= self.get_checkpoint_path(filename, checkpoint_dir)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint= torch.load(checkpoint_path, map_location='cpu')
            if 'config' not in checkpoint:
                raise KeyError("Checkpoint does not contain a 'config' key to build the model")

            # build a fresh model from config (dict) hyperparameters
            config_args= checkpoint['config']
            if not isinstance(config_args, dict):
                raise TypeError("checkpoint['config'] should be a dict of constructor kwargs")

            self._log.info("build_model | Building a new model with config: %s", config_args)
            if self.verbose:
                print(f'[INFO] Building a new model with config: {config_args}')
            self.model= TSFTransformer(**config_args).to(self.device)

            epoch= 0
            best_val_loss= 0.0
            if restore_model:  # restore model state
                epoch, best_val_loss= self.load_checkpoint(
                    filename, checkpoint_dir, restore_optimizer, restore_metadata
                )

            return self.model, epoch, best_val_loss

        except Exception as e:
            self._log.warning(
                "build_model | Failed to build and load checkpoint from %s: %s", checkpoint_path, e
            )
            if self.verbose:
                print(f"[ERROR] Failed to build and load checkpoint from {checkpoint_path}: {e}")
            raise e


    def save_plot(self, plt_obj, file_name, as_pdf, method_name, info_message) -> None:
        plots_dir= f"{self.checkpoint_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        save_path= self.get_checkpoint_path(file_name, plots_dir)

        plt_obj.tight_layout()
        if as_pdf:
            save_path= f'{save_path}.pdf'
            plt_obj.savefig(save_path, dpi=300, pad_inches=0.01, bbox_inches="tight")
        else:
            save_path= f'{save_path}.svg'
            plt_obj.savefig(save_path, pad_inches=0.01, bbox_inches="tight")

        self._log.info(f"{method_name} | {info_message}: {save_path}")
        if self.verbose:
            print(f"[INFO] {info_message}: {save_path}")


    def plot_results(self, cut_first_epoch=False, show_plot=True, save_charts=False, as_pdf=False,
                     file_name='training_results'):
        """
        Plot training metrics including training loss, validation loss, and learning rate history
        over epochs. Plots can be saved as checkpoint_dir/file_name.
        """
        method_name= "plot_results"

        if len(self.train_losses) == 0:
            info_message= "No training/validation history available to plot."
            self._log.info(f"{method_name} | {info_message}")
            if self.verbose:
                print(f"[INFO] {info_message}")
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
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, lr_hist, label='Learning Rate', marker='o', linestyle='-', color='tab:green')
        plt.title('Learning Rate History')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        if save_charts:
            self.save_plot(plt, file_name, as_pdf, method_name, "Training charts were saved at")
        if show_plot:
            plt.show()
        plt.close()


    def plot_expert_usage_global(self, cut_first_epoch=False, show_plot=True, save_charts=False,
                                 as_pdf=False, file_name="expert_usage_global"):
        """
        MoE metrics. Plot global expert hard/soft utilization over epochs.
        Plots can be saved as checkpoint_dir/file_name.
        """
        method_name= "plot_expert_usage_global"

        hard_hist, soft_hist= [], []
        if self.expert_traker is not None:
            hard_hist= self.expert_traker.history["epoch_hard_fraction"]
            soft_hist= self.expert_traker.history["epoch_soft_fraction"]

        if len(hard_hist) == 0 or len(soft_hist) == 0:
            info_message= "No expert usage history available to plot."
            self._log.info(f"{method_name} | {info_message}")
            if self.verbose:
                print(f"[INFO] {info_message}")
            return

        hard_hist= torch.stack(hard_hist).cpu()   # [n_epochs, E]
        soft_hist= torch.stack(soft_hist).cpu()   # [n_epochs, E]

        epochs= list(range(1, hard_hist.size(0) + 1))
        if cut_first_epoch and len(epochs) > 1:
            epochs= epochs[1:]
            hard_hist= hard_hist[1:]
            soft_hist= soft_hist[1:]

        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        for expert_id in range(hard_hist.size(1)):
            plt.plot(
                epochs, hard_hist[:, expert_id].numpy(), marker="o", linestyle="-",
                label=f"Expert {expert_id}"
            )
        plt.title("Global Expert Hard Utilization")
        plt.xlabel("Epochs")
        plt.ylabel("Hard Fraction")
        plt.legend(ncol=2 if hard_hist.size(1) > 6 else 1, fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.subplot(1, 2, 2)
        for expert_id in range(soft_hist.size(1)):
            plt.plot(
                epochs, soft_hist[:, expert_id].numpy(), marker="o", linestyle="-",
                label=f"Expert {expert_id}"
            )
        plt.title("Global Expert Soft Importance")
        plt.xlabel("Epochs")
        plt.ylabel("Soft Fraction")
        plt.legend(ncol=2 if soft_hist.size(1) > 6 else 1, fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.7)

        if save_charts:
            self.save_plot(plt, file_name, as_pdf, method_name, "Expert usage charts were saved at")
        if show_plot:
            plt.show()
        plt.close()


    def plot_expert_routing_diagnostics(self, cut_first_epoch=False, show_plot=True, save_charts=False,
                                        as_pdf=False, file_name="expert_routing_diagnostics"):
        """
        MoE metrics. Plot global routing health metrics over epochs.
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
            self._log.info(f"{method_name} | {info_message}")
            if self.verbose:
                print(f"[INFO] {info_message}")
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
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, cv_hard_hist, label="CV Hard", marker="o", linestyle="-")
        plt.plot(epochs, cv_soft_hist, label="CV Soft", marker="o", linestyle="-")
        plt.title("Routing Imbalance")
        plt.xlabel("Epochs")
        plt.ylabel("Coefficient of Variation")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        if save_charts:
            self.save_plot(plt, file_name, as_pdf, method_name, "Routing diagnostic charts were saved at")
        if show_plot:
            plt.show()
        plt.close()


    def plot_expert_usage_layerwise(self, cut_first_epoch=False, show_plot=True, save_charts=False,
                                    as_pdf=False, file_name="expert_usage_layer"):
        """
        MoE metrics. Plot per-layer expert hard/soft utilization over epochs.
        Plots can be saved as checkpoint_dir/file_name. Saves one figure per MoE layer.
        """
        method_name= "plot_expert_usage_layerwise"

        layer_epoch= []
        if self.expert_traker is not None:
            layer_epoch= self.expert_traker.history["layer_epoch"]

        if len(layer_epoch) == 0:
            info_message= "No layerwise expert usage history available to plot."
            self._log.info(f"{method_name} | {info_message}")
            if self.verbose:
                print(f"[INFO] {info_message}")
            return

        for layer_id, layer_hist in layer_epoch.items():
            hard_hist= layer_hist["hard_fraction"]
            soft_hist= layer_hist["soft_fraction"]

            if len(hard_hist) == 0 or len(soft_hist) == 0:
                continue

            hard_hist= torch.stack(hard_hist).cpu()   # [n_epochs, E]
            soft_hist= torch.stack(soft_hist).cpu()   # [n_epochs, E]

            epochs= list(range(1, hard_hist.size(0) + 1))
            if cut_first_epoch and len(epochs) > 1:
                epochs= epochs[1:]
                hard_hist= hard_hist[1:]
                soft_hist= soft_hist[1:]

            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            for expert_id in range(hard_hist.size(1)):
                plt.plot(
                    epochs, hard_hist[:, expert_id].numpy(), marker="o", linestyle="-",
                    label=f"Expert {expert_id}"
                )
            plt.title(f"Layer {layer_id} Hard Utilization")
            plt.xlabel("Epochs")
            plt.ylabel("Hard Fraction")
            plt.legend(ncol=2 if hard_hist.size(1) > 6 else 1, fontsize=9)
            plt.grid(True, linestyle="--", alpha=0.7)

            plt.subplot(1, 2, 2)
            for expert_id in range(soft_hist.size(1)):
                plt.plot(
                    epochs, soft_hist[:, expert_id].numpy(), marker="o", linestyle="-",
                    label=f"Expert {expert_id}"
                )
            plt.title(f"Layer {layer_id} Soft Importance")
            plt.xlabel("Epochs")
            plt.ylabel("Soft Fraction")
            plt.legend(ncol=2 if soft_hist.size(1) > 6 else 1, fontsize=9)
            plt.grid(True, linestyle="--", alpha=0.7)

            if save_charts:
                file_name_curr= f"{file_name}_{layer_id}"
                info_message  = f"Layer {layer_id} expert usage chart was saved at"
                self.save_plot(plt, file_name_curr, as_pdf, method_name, info_message)
            if show_plot:
                plt.show()
            plt.close()


    def plot_expert_usage_heatmap(self, kind="hard", show_plot=True, save_charts=False, as_pdf=False,
                                  file_name="expert_usage_heatmap"):
        """
        MoE metrics. Plot a heatmap of global expert usage over epochs.
        - kind: 'hard' or 'soft'
        Plots can be saved as checkpoint_dir/file_name.
        """
        method_name= "plot_expert_usage_heatmap"

        hist= []
        title= ""
        if self.expert_traker is not None:
            if kind == "hard":
                hist = self.expert_traker.history["epoch_hard_fraction"]
                title= "Global Expert Hard Utilization Heatmap"
            elif kind == "soft":
                hist = self.expert_traker.history["epoch_soft_fraction"]
                title= "Global Expert Soft Importance Heatmap"
            else:
                raise ValueError("kind must be 'hard' or 'soft'")

        if len(hist) == 0:
            info_message= "No expert usage history available to plot."
            self._log.info(f"{method_name} | {info_message}")
            if self.verbose:
                print(f"[INFO] {info_message}")
            return

        mat= torch.stack(hist).cpu().numpy().transpose()  # [E, n_epochs]
        plt.figure(figsize=(10, 6))
        plt.imshow(mat, aspect="auto", interpolation="nearest")
        plt.colorbar(label=f"{kind.capitalize()} Fraction")
        plt.title(title)
        plt.xlabel("Expert ID")
        plt.ylabel("Epoch")

        if save_charts:
            file_name= f"{file_name}_{kind}"
            self.save_plot(plt, file_name, as_pdf, method_name, "Heatmap was saved at")
        if show_plot:
            plt.show()
        plt.close()
