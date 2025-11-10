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
    - TODO: switch raw prints by loggings.
    """

    def __init__(self, model, device, train_loader, train_ds_scaler, val_loader, test_loader,
                 criterion, optimizer, scheduler=None, aux_criterion=None, early_stopping=None,
                 use_time_features=False, do_validation=True, augmentation=None, checkpointing=True,
                 checkpoint_dir=None, filename=None, verbose=True) -> None:
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
        # train_ds_scaler normalization info
        self.scale_= torch.from_numpy(train_ds_scaler.scale_).float().view(1,-1,1).to(self.device)
        self.mean_ = torch.from_numpy(train_ds_scaler.mean_).float().view(1,-1,1).to(self.device)


    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch, returning the training loss and learning rate.
        """
        self.model.train()
        train_loss= 0.0
        epoch_lr= 0.0

        # --- training steps ---
        for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}"):
            self.optimizer.zero_grad()

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
            logits, router_logits= self.model(data, ts_mark=data_time)
            # compute training loss on the scaled data
            losses= self.criterion(logits, target)
            loss= torch.mean(losses)

            if self.aux_criterion is not None:
                aux_loss= self.aux_criterion(router_logits, padding_mask)
                loss= loss + aux_loss
            # sample‑weighted average loss
            train_loss += float(loss.item()) * data.size(0)

            # --- backward pass to calculate the gradients ---
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # --- update the parameters using the gradient ---
            self.optimizer.step()
            # per-step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_lr += self.optimizer.param_groups[0]['lr']

        if self.augmentation is not None:
            self.augmentation.step_epoch()
        # transform per-step values in per-epoch averages
        train_loss /= len(self.train_loader)
        epoch_lr /= len(self.train_loader)

        return train_loss, epoch_lr


    def train_one_epoch_bf16(self, epoch):
        """
        Train the model for one epoch using bfloat16, returning the training loss and learning rate.
        """
        assert self.device.type == 'cuda', "BF16 training requires CUDA"
        self.model.train()
        train_loss= 0.0
        epoch_lr= 0.0
        # track if any NaN/inf is observed
        any_nonfinite_loss= False

        # --- training steps ---
        for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}"):
            self.optimizer.zero_grad()

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
                logits, router_logits= self.model(data, ts_mark=data_time)
                # compute training loss on the scaled data
                losses= self.criterion(logits, target)
                loss= torch.mean(losses)

                if self.aux_criterion is not None:
                    aux_loss= self.aux_criterion(router_logits, padding_mask)
                    loss= loss + aux_loss

            # check loss finite
            if not torch.isfinite(loss).all():
                any_nonfinite_loss= True
                # best to raise early to see where it happened
                raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}: {loss}")

            # sample‑weighted average loss
            train_loss += float(loss.item()) * data.size(0)

            # --- backward pass to calculate the gradients ---
            # gradients computed in bfloat16, but accumulation and params remain TF32
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # --- update the parameters using the gradient ---
            self.optimizer.step()
            # per-step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_lr += self.optimizer.param_groups[0]['lr']

        if self.augmentation is not None:
            self.augmentation.step_epoch()
        # transform per-step values in per-epoch averages
        train_loss /= len(self.train_loader)
        epoch_lr /= len(self.train_loader)

        return train_loss, epoch_lr


    def validate(self, val_criterion=nn.MSELoss(reduction='none')):
        """
        Validate the model on a validation set.
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.model.eval()
        val_criterion= val_criterion if val_criterion is not None else self.criterion
        val_loss= 0.0

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
                logits, _= self.model(data, ts_mark=data_time)
                losses= val_criterion(logits, target)
                loss= torch.mean(losses)

                val_loss += float(loss.item()) * data.size(0)

        val_loss /= len(self.val_loader)

        return val_loss


    def train(self, epochs, eval_interval=1, use_bf16=False) -> None:
        """
        Train the model for a specified number of epochs, performing validation and checkpointing.
        """
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

            if self.verbose:
                if val_loss is not None:
                    print(f'Train loss: {train_loss:.4f}')
                    print(f'Valid loss: {val_loss:.4f} | dt/epoch: {dt*1000:.2f}ms')
                else:  # val_loss is None
                    print(f'Train loss: {train_loss:.4f} | dt/epoch: {dt*1000:.2f}ms')
                print(f'======= Epoch {epoch+1} Completed =======')

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
                        break

        if val_loss is not None:
            print(f'Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch + 1})')
            # Save save a final checkpoint only if the last epoch equals the best epoch.
            if best_epoch == epochs - 1 and self.checkpointing:
                self.save_checkpoint(epochs, best_val_loss)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


    def test(self, test_loader=None, inverse_transform=False, test_criterion=nn.MSELoss(reduction='none')):
        """
        Test the model on a test set.
        Returns the mean test loss, test predictions, and test labels.
        """
        test_loader= self.test_loader if test_loader is None else test_loader
        assert test_loader is not None, "test_loader cannot refer to None"

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.model.eval()
        test_criterion= test_criterion if test_criterion is not None else self.criterion
        test_loss= 0.0
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
                        logits= logits * self.scale_ + self.mean_
                        target= target * self.scale_ + self.mean_
                else:
                    logits, _= self.model(data, ts_mark=data_time)
                losses= test_criterion(logits, target)
                loss= torch.mean(losses)

                # --- register preds and trues ---
                test_loss += float(loss.item()) * data.size(0)
                all_logits.append(logits.cpu())
                all_trues.append(target.cpu())

        test_loss /= len(test_loader)

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
            if self.verbose:
                print(f"[INFO] Checkpoint saved at '{checkpoint_path}'")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")


    @staticmethod
    def strip_module_prefix(state_dict):
        """
        Remove a single leading 'module.' from keys if present.
        """
        if any(k.startswith("module.") for k in state_dict.keys()):
            return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        return state_dict


    def load_checkpoint(self, checkpoint_path, restore_optimizer=False, restore_metadata=False) -> tuple:
        """
        This method loads the checkpoint from the given path and restores the model, optimizer
        (optional, when restore_optimizer=True), and training history.
        - TODO: Fix dtype, device, and layout for optimizer state.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        # ensure model exists before loading (build_model will create it)
        if getattr(self, 'model', None) is None:
            raise RuntimeError("self.model is None: instantiate model before restoring state_dict")

        try:
            checkpoint= torch.load(checkpoint_path, map_location=self.device)
            # Restore model state
            self.model.load_state_dict(self.strip_module_prefix(checkpoint['model_state_dict']))
            # ensure model on target device
            if getattr(self, 'model', None) is not None:
                self.model.to(self.device)
            # Restore optimizer state
            if restore_optimizer:
                if getattr(self, "optimizer", None) is None:
                    print("[INFO] Checkpoint contains optimizer state, but self.optimizer is None: skipping optimizer restore.")
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Retrieve training metadata
            epoch= checkpoint['epoch']
            best_val_loss= checkpoint.get('best_val_loss', float('inf'))
            if restore_metadata:
                self.train_losses= checkpoint.get('train_losses', [])
                self.val_losses= checkpoint.get('val_losses', [])
                self.lr_hist= checkpoint.get('lr_hist', [])

            print(f"[INFO] Checkpoint loaded from '{checkpoint_path}'. Resuming training with best validation loss of {best_val_loss:.4f}.")
            return epoch, best_val_loss

        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint from {checkpoint_path}: {e}")
            raise e


    def build_model(self, checkpoint_path, restore_optimizer=False, restore_metadata=False) -> tuple:
        """
        Build a model from a given checkpoint.
        """
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

            print(f'[INFO] Building a model with config: {config_args}')
            self.model= TSFTransformer(**config_args).to(self.device)
            # restore model state
            epoch, best_val_loss= self.load_checkpoint(checkpoint_path, restore_optimizer, restore_metadata)
            return self.model, epoch, best_val_loss

        except Exception as e:
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
