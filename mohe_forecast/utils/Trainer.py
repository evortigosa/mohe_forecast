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
                 checkpoint_dir=None, filename=None, verbose=False) -> None:
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
        # track statistics
        self.train_losses= []
        self.val_losses= []
        self.lr_hist= []
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


    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch, returning the training loss and learning rate.
        """
        self.model.train()
        train_loss= 0.0
        n_samples= 0
        n_steps= 0
        epoch_lr= 0.0

        # --- training steps ---
        for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}"):
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

            if self.aux_criterion is not None:
                aux_loss= self.aux_criterion(router_logits, padding_mask)
                loss= loss + aux_loss

            # check loss finite
            if not torch.isfinite(loss).all():
                self._log.warning(
                    "train_one_epoch | non_finite_loss | epoch=%d | loss=%s", epoch, str(loss.detach().cpu())
                )
                # best to raise early to see where it happened
                raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}: {loss}")

            # --- backward pass to calculate the gradients ---
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # --- update the parameters using the gradient ---
            self.optimizer.step()

            # sample‑weighted average loss
            train_loss += float(loss.item()) * data.size(0)
            n_samples += data.size(0)

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


    def train_one_epoch_bf16(self, epoch):
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
        for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}"):
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

                if self.aux_criterion is not None:
                    aux_loss= self.aux_criterion(router_logits, padding_mask)
                    loss= loss + aux_loss

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
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # --- update the parameters using the gradient ---
            self.optimizer.step()

            # sample‑weighted average loss
            train_loss += float(loss.item()) * data.size(0)
            n_samples += data.size(0)

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
            for batch in tqdm(self.val_loader, desc='Validating'):
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


    def train(self, epochs, eval_interval=1, use_bf16=False) -> None:
        """
        Train the model for a specified number of epochs, performing validation and checkpointing.
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        best_val_loss= float('inf')
        best_epoch= -1
        val_loss= None

        for epoch in range(epochs):
            start= time.time()

            if use_bf16:
                train_loss, epoch_lr= self.train_one_epoch_bf16(epoch+1)
            else:
                train_loss, epoch_lr= self.train_one_epoch(epoch+1)
            self.train_losses.append(train_loss)
            self.lr_hist.append(epoch_lr)

            if self.do_validation and (epoch % eval_interval == 0 or epoch == epochs-1):
                val_loss= self.validate()
                self.val_losses.append(val_loss)
            else:
                val_loss= None

            end= time.time()
            dt = end - start

            if val_loss is not None:
                self._log.info(
                    "train | epoch=%d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.3e | dt=%.2fs",
                    epoch + 1, epochs, train_loss, val_loss, epoch_lr, dt
                )
                if self.verbose:
                    print(f'Train loss: {train_loss:.4f}')
                    print(f'Valid loss: {val_loss:.4f} | dt/epoch: {dt*1000:.2f}ms')
            else:  # val_loss is None
                self._log.info(
                    "train | epoch=%d/%d | train_loss=%.6f | val_loss not computed | lr=%.3e | dt=%.2fs",
                    epoch + 1, epochs, train_loss, epoch_lr, dt
                )
                if self.verbose:
                    print(f'Train loss: {train_loss:.4f} | dt/epoch: {dt*1000:.2f}ms')

            if val_loss is not None:
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
                        break

        if val_loss is not None:
            self._log.info("train | Best Validation Loss: %.6f | Epoch: %d", best_val_loss, best_epoch + 1)
            if self.verbose:
                print(f'Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch + 1})')
            # Save a final checkpoint only if the last epoch equals the best epoch.
            if best_epoch == epochs - 1 and self.checkpointing:
                self.save_checkpoint(best_epoch, best_val_loss)


    def test(self, test_loader=None, inverse_transform=False, test_criterion=nn.MSELoss(reduction='none')):
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

        self.model.eval()
        test_criterion= test_criterion if test_criterion is not None else self.criterion
        test_loss= 0.0
        n_samples= 0
        all_logits, all_trues= [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
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


    def save_checkpoint(self, epoch, best_val_loss) -> None:
        """
        Save the model checkpoint to disk, including training history.
        """
        checkpoint_dir= self.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path= os.path.join(checkpoint_dir, f'{self.filename}.pth')
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
        - TODO: Fix dtype, device, and layout for optimizer state.
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
            checkpoint_path= os.path.join(checkpoint_dir, filename)

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
            # retrieve training metadata
            epoch= checkpoint['epoch']
            best_val_loss= checkpoint.get('best_val_loss', float('inf'))
            if restore_metadata:
                self.train_losses= checkpoint.get('train_losses', [])
                self.val_losses= checkpoint.get('val_losses', [])
                self.lr_hist= checkpoint.get('lr_hist', [])

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


    def build_model(self, filename=None, checkpoint_dir=None, restore_optimizer=False,
                    restore_metadata=False) -> tuple:
        """
        Build a model from a given checkpoint.
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
            checkpoint_path= os.path.join(checkpoint_dir, filename)

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
            # restore model state
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


    def plot_results(self, cut_first_epoch=True, save_charts=False, file_name='training_results') -> None:
        """
        Plot training metrics including training loss, validation loss, and learning rate history
        over epochs. The results can be saved as file_name.pdf.
        """
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

        plt.tight_layout()
        if save_charts:
            plt.savefig(f'{file_name}.pdf', pad_inches=0.01, bbox_inches="tight")
        plt.show()
        plt.close()
