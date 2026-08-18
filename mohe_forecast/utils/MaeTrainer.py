# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
MaeTrainer Class -- SSL MAE (distributed) training for MohetsMAE.
"""

import time
import torch
from contextlib import contextmanager
from tqdm import tqdm
from .Trainer import Trainer
from .DTrainer import DTrainer



class MaeTrainer(Trainer):
    """
    Self-supervised MAE training for MohetsMAE. Extends Trainer, overriding only what SSL requires: the
    two training loops, the two validation loops, and test(). Everything else -- the epoch loop, early
    stopping, checkpointing, history, plotting and the distributed hooks -- is inherited unchanged.
    --- Differences from Trainer ---
    - The model computes its own loss: MohetsMAE.forward returns (loss, logits, mask, enc_router,
      dec_router), where loss is the masked-reconstruction loss over the removed patches. There is no
      supervised target: the loader's seq_y is ignored and the target is derived from seq_x itself.
    - forward()'s mask_ratio default (0.75) overwrites the encoder's configured ratio on every call
      (forward_encoder -> set_mask_ratio), so this trainer always passes mask_ratio explicitly.
    - validate() / test() measure masked reconstruction, not forecasting. Masks are random, so with
      deterministic_eval=True the RNG is frozen for the evaluation passes: every epoch -- and every model
      -- is scored on the same masks, which makes the val curve, early stopping and cross-model
      comparisons paired instead of mask-noise-limited. The training RNG stream is saved and restored, so
      training stochasticity is untouched.
    - test() returns the reconstructed and original series, or only their masked positions.
    - the decoder has its own load-balancing criterion (dec_aux_criterion), because LoadBalancingLoss
      binds n_experts/top_k at construction and the decoder geometry differs from the encoder's.
    """

    def __init__(self, model, device, train_loader, train_ds_scaler, val_loader, test_loader,
                 criterion, optimizer, scheduler=None, aux_criterion=None, early_stopping=None,
                 use_time_features=False, do_validation=True, augmentation=None, checkpointing=True,
                 checkpoint_dir=None, filename=None, verbose=False, disable_tqdm=False,
                 # --- MAE specific ---
                 dec_aux_criterion=None, mask_ratio=None, eval_mask_ratio=None,
                 deterministic_eval=True, eval_seed=1234) -> None:
        super().__init__(
            model=model, device=device, train_loader=train_loader, train_ds_scaler=train_ds_scaler,
            val_loader=val_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, aux_criterion=aux_criterion, early_stopping=early_stopping,
            use_time_features=use_time_features, do_validation=do_validation, augmentation=augmentation,
            checkpointing=checkpointing, checkpoint_dir=checkpoint_dir, filename=filename,
            verbose=verbose, disable_tqdm=disable_tqdm,
        )
        if criterion is not None and getattr(criterion, "reduction", None) != "none":
            raise ValueError(
                "MaeTrainer requires an element-wise criterion: MohetsMAE.forward_loss applies "
                ".mean(dim=-1) per patch before masking. Use reduction='none', or criterion=None."
            )

        # forward()'s mask_ratio default would silently override the configured ratio: resolve it once
        model_ref= self._unwrap_model()
        self.mask_ratio= float(model_ref.mask_ratio) if mask_ratio is None else float(mask_ratio)
        self.eval_mask_ratio= self.mask_ratio if eval_mask_ratio is None else float(eval_mask_ratio)
        assert 0.0 < self.mask_ratio < 1.0, "mask_ratio must be in (0, 1)"
        assert 0.0 < self.eval_mask_ratio < 1.0, "eval_mask_ratio must be in (0, 1)"

        self.deterministic_eval= bool(deterministic_eval)
        self.eval_seed= int(eval_seed)

        # the decoder gets its own balancing loss. Leave it None for a dense decoder side
        self.dec_aux_criterion= dec_aux_criterion


    @contextmanager
    def _frozen_eval_rng(self):
        """
        Freeze the RNG for an evaluation pass (identical masks every epoch / every model), then restore the
        training RNG stream so training stochasticity is completely unaffected.
        """
        if not self.deterministic_eval:
            yield
            return

        cpu_state  = torch.get_rng_state()
        cuda_states= torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(self.eval_seed)   # seeds CPU and all CUDA devices
            yield
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


    @torch.inference_mode()
    def test(self, test_loader=None, inverse_transform=False, dynamic_window=True, mask_ratio=None,
             masked_out=False) -> tuple:
        """
        Reconstruct a held-out test set. Absorbs the original signature.
        - preds: the reconstructed series, in the original data units;
        - trues: the input series (which is the reconstruction target), same units.
        When masked_out=False, returns preds, trues so Metrics(preds, trues) scores the full series, while
        masked_out=True means Metrics(preds[mask], trues[mask]) scores only the masked positions -- the
        actual MAE objective.
        The decoder predicts per-patch normalized patches (norm_pix) whenever input_norm is enabled, so the
        reconstruction is mapped back to raw units with the ground-truth per-patch stats (see patch_stats /
        unpatchify). As in the reference MAE, the reconstruction is therefore only defined up to each
        patch's mean and stdev, which are taken from the target.
        """
        test_loader= self.test_loader if test_loader is None else self._set_loader(test_loader)
        assert test_loader is not None, "test_loader cannot refer to None"
        self.model.eval()
        # all reconstructions and inputs to feed external metrics
        all_preds, all_trues, all_indices= [], [], []
        all_masks = [] if masked_out else None
        recon_loss= torch.tensor(0.0, device=self.device)
        n_samples = torch.tensor(0, device=self.device, dtype=torch.long)

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
        model_ref  = self._unwrap_model()
        patch_width= model_ref.encoder.t_embedding.patch_width
        mask_ratio = self.eval_mask_ratio if mask_ratio is None else float(mask_ratio)

        with self._frozen_eval_rng():
            for batch in tqdm(test_loader, desc='Testing', disable=p_bar):
                # --- minibatch construction (SSL: seq_y is ignored; seq_x is the target) ---
                data, _, data_time, _, sample_index= self._get_minibatch(batch, return_index=True)

                # --- forward pass: masked reconstruction ---
                loss, logits, mask, *_= self.model(
                    data, mask_ratio=mask_ratio, ts_mark=data_time, criterion=self.criterion
                )
                recon_loss+= loss.detach() * data.size(0)
                n_samples += data.size(0)

                # --- rebuild the series in the original units ---
                if model_ref.input_norm is not None:
                    # logits live in per-patch (norm_pix) space: invert those stats, not the instance stats
                    p_mean, p_stdev= model_ref.patch_stats(data)
                    preds= model_ref.unpatchify(logits, p_mean=p_mean, p_stdev=p_stdev)
                else:
                    preds= model_ref.unpatchify(logits)
                trues= data

                if inverse_transform:
                    # invert the dataset scaling back to the original units
                    preds= preds * scale_ + mean_
                    trues= trues * scale_ + mean_

                # (B*C, P) at patch resolution -> (B, C, T) at time resolution. carried as uint8: the
                # gather has only ever handled float/long, and bool support is backend-dependent
                if masked_out:
                    B, C, T= trues.size()
                    mask_t= mask.detach().unsqueeze(-1).expand(-1, -1, patch_width).reshape(B, C, T)
                    mask_store= self._gather_variable_batch(mask_t.to(torch.uint8))

                preds_store= self._gather_variable_batch(preds.detach())
                trues_store= self._gather_variable_batch(trues.detach())
                index_store= self._gather_variable_batch(sample_index.detach().view(-1))
                if self._is_main_process():
                    all_preds.append(preds_store.float().cpu())
                    all_trues.append(trues_store.float().cpu())
                    all_indices.append(index_store.cpu())
                    if masked_out:
                        all_masks.append(mask_store.cpu())

        recon_loss= self._reduce_sum(recon_loss)
        n_samples = self._reduce_sum(n_samples)
        recon_loss= float((recon_loss / n_samples.clamp_min(1)).item())

        peak_vram= self._get_cuda_memory_stats()
        end= time.time()
        dt = self._format_dt(end - start)
        self._set_log("info",
            f"test | masked_recon_loss=%.6f | mask_ratio=%.2f | max GPU mem=%.2fGB | dt=%sms" % \
            (recon_loss, mask_ratio, peak_vram, dt)
        )

        if not self._is_main_process():
            return None, None
        if len(all_preds) == 0:
            raise RuntimeError("No reconstructions were collected during test().")

        # order and dedup depend solely on the (small) index vector, so the final gather
        # index can be computed before materializing preds/trues
        indices= torch.cat(all_indices, dim=0).long()
        all_indices.clear()
        order= torch.argsort(indices)
        indices= indices[order]                          # sample ids in sorted order
        keep= torch.ones_like(indices, dtype=torch.bool)
        keep[1:]= indices[1:] != indices[:-1]            # drop duplicate ids from distributed padding
        # sel folds sort + dedup into a single gather
        sel= order[keep]
        del order, keep, indices

        preds= torch.cat(all_preds, dim=0)
        all_preds.clear()
        preds= preds.index_select(0, sel)

        trues= torch.cat(all_trues, dim=0)
        all_trues.clear()
        trues= trues.index_select(0, sel)

        if masked_out:
            mask= torch.cat(all_masks, dim=0)
            all_masks.clear()
            mask= mask.index_select(0, sel).bool()
            del sel

            return preds[mask], trues[mask]
        del sel

        return preds, trues


    @torch.inference_mode()
    def validate(self, val_criterion=None):
        """
        Masked-reconstruction loss on the validation set. The aux loss is excluded: it is a training
        regularizer, not a measure of reconstruction quality.
        """
        self.model.eval()
        val_criterion= self.criterion if val_criterion is None else val_criterion
        val_loss= torch.tensor(0.0, device=self.device)
        n_samples= torch.tensor(0, device=self.device, dtype=torch.long)
        p_bar= self.disable_tqdm or not self._is_main_process()

        with self._frozen_eval_rng():
            for batch in tqdm(self.val_loader, desc='Validating', disable=p_bar):
                # --- minibatch construction ---
                data, _, data_time, *_= self._get_minibatch(batch)

                # --- forward pass and get loss ---
                loss, *_= self.model(
                    data, mask_ratio=self.eval_mask_ratio, ts_mark=data_time, criterion=val_criterion
                )
                val_loss += loss.detach() * data.size(0)
                n_samples+= data.size(0)

        val_loss = self._reduce_sum(val_loss)
        n_samples= self._reduce_sum(n_samples)
        val_loss = val_loss / n_samples.clamp_min(1)

        return float(val_loss.item())


    @torch.inference_mode()
    def validate_bf16(self, val_criterion=None):
        """
        Masked-reconstruction loss on the validation set using bfloat16.
        """
        assert self.device.type == 'cuda', "BF16 training requires CUDA"

        self.model.eval()
        val_criterion= self.criterion if val_criterion is None else val_criterion
        val_loss= torch.tensor(0.0, device=self.device)
        n_samples= torch.tensor(0, device=self.device, dtype=torch.long)

        with self._frozen_eval_rng():
            for batch in tqdm(self.val_loader, desc='Validating', disable=self.disable_tqdm):
                # --- minibatch construction ---
                data, _, data_time, *_= self._get_minibatch(batch)

                # --- forward pass and get loss ---
                # autocast wraps only the forward pass and the loss computation
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss, *_= self.model(
                        data, mask_ratio=self.eval_mask_ratio, ts_mark=data_time, criterion=val_criterion
                    )

                val_loss += loss.detach().float() * data.size(0)
                n_samples+= data.size(0)

        val_loss= val_loss / n_samples.clamp_min(1)

        return float(val_loss.item())


    def train_one_epoch(self, epoch, clip_grad=None, get_moe_metrics=False):
        """
        Train the MAE for one epoch, returning the reconstruction loss and the learning rate.
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

            # --- minibatch construction (SSL: the target is derived from seq_x) ---
            data, _, data_time, *_= self._get_minibatch(batch)
            padding_mask= None

            # --- forward pass: the model returns the masked-reconstruction loss ---
            loss, _logits, _mask, enc_router_probs, dec_router_probs= self.model(
                data, mask_ratio=self.mask_ratio, ts_mark=data_time, criterion=self.criterion
            )
            # sample-weighted average of the task loss (the aux loss is excluded from the report)
            train_loss+= loss.detach() * data.size(0)
            n_samples += data.size(0)

            if self.aux_criterion is not None:
                aux_loss, global_metrics, layer_metrics= self.aux_criterion(
                    enc_router_probs, padding_mask, get_moe_metrics
                )
                loss= loss + aux_loss

                if get_moe_metrics and (self.expert_traker is not None):
                    global_metrics= self._reduce_moe_metrics(global_metrics)
                    layer_metrics = self._reduce_moe_metrics(layer_metrics)
                    self.expert_traker.update(global_metrics, layer_metrics)

            # decoder MoE: optional, for a full MoE MAE. It needs its own criterion (the decoder
            # geometry differs from the encoder's), and no routing diagnostics are collected here.
            # with a dense decoder, dec_router_probs is all None and this contributes 0
            if self.dec_aux_criterion is not None:
                dec_aux_loss, _, _= self.dec_aux_criterion(dec_router_probs)
                loss= loss + dec_aux_loss

            # check loss finite -- collectively, so that under DDP every rank raises together
            not_finite= (~torch.isfinite(loss).all()).to(torch.long)
            not_finite= self._reduce_sum(not_finite)
            if int(not_finite.item()) > 0:
                self._set_log("error",
                    f"train_one_epoch | non_finite_loss | epoch=%d | loss=%s" % (epoch, str(loss.detach().cpu()))
                )
                # best to raise early to see where it happened
                raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}: {loss.detach().cpu()}")

            # --- backward pass to calculate the gradients ---
            self._backward(loss, clip_grad)

            # --- update the parameters using the gradient ---
            self.optimizer.step()
            epoch_lr += self.optimizer.param_groups[0]['lr']
            # per-step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            n_steps += 1

        if self.augmentation is not None:
            self.augmentation.step_epoch()

        train_loss= self._reduce_sum(train_loss)
        n_samples = self._reduce_sum(n_samples)
        train_loss= train_loss / n_samples.clamp_min(1)
        epoch_lr  = epoch_lr / max(n_steps, 1)

        return float(train_loss.item()), epoch_lr


    def train_one_epoch_bf16(self, epoch, clip_grad=None, get_moe_metrics=False):
        """
        Train the MAE for one epoch using bfloat16 (single device: Fabric owns distributed precision).
        """
        assert self.device.type == 'cuda', "BF16 training requires CUDA"

        self.model.train()
        train_loss= torch.tensor(0.0, device=self.device)
        n_samples= torch.tensor(0, device=self.device, dtype=torch.long)
        n_steps= 0
        epoch_lr= 0.0

        # --- training steps ---
        for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}", disable=self.disable_tqdm):
            self.optimizer.zero_grad(set_to_none=True)

            # --- minibatch construction (SSL: the target is derived from seq_x) ---
            data, _, data_time, *_= self._get_minibatch(batch)
            padding_mask= None
            global_metrics= layer_metrics= None

            # --- forward pass and get loss ---
            # model, optimizer defined as usual; model parameters kept as float32
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss, _logits, _mask, enc_router_probs, dec_router_probs= self.model(
                    data, mask_ratio=self.mask_ratio, ts_mark=data_time, criterion=self.criterion
                )
                task_loss= loss.detach()  # pre-aux loss: this is what gets reported

                if self.aux_criterion is not None:
                    aux_loss, global_metrics, layer_metrics= self.aux_criterion(
                        enc_router_probs, padding_mask, get_moe_metrics
                    )
                    loss= loss + aux_loss

                # decoder MoE: optional, no diagnostics.
                if self.dec_aux_criterion is not None:
                    dec_aux_loss, _, _= self.dec_aux_criterion(dec_router_probs)
                    loss= loss + dec_aux_loss

            # sample-weighted average loss
            train_loss+= task_loss.float() * data.size(0)
            n_samples += data.size(0)

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
            epoch_lr += self.optimizer.param_groups[0]['lr']
            # per-step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            n_steps += 1

        if self.augmentation is not None:
            self.augmentation.step_epoch()

        train_loss= train_loss / n_samples.clamp_min(1)
        epoch_lr  = epoch_lr / max(n_steps, 1)

        return float(train_loss.item()), epoch_lr



class MaeDTrainer(DTrainer, MaeTrainer):
    """
    Distributed SSL MAE training for MohetsMAE using Lightning Fabric machinery.

    --- Why this class is (almost) empty ---
    MaeTrainer overrides only the SSL logic (train_one_epoch, the two validate variants, test, and
    __init__), and every one of those methods is written against the Trainer hook contract
    (_move_to_device, _backward, _reduce_sum, _gather_variable_batch, _is_main_process, _set_loader,
    _unwrap_model, _reduce_moe_metrics). DTrainer overrides exactly those hooks -- and nothing MaeTrainer
    touches. The two parents therefore override disjoint method sets, and cooperative inheritance
    composes them with no glue code:

        MaeDTrainer -> DTrainer -> MaeTrainer -> Trainer     (the MRO)

    - distribution hooks, _setup, checkpoint IO (_save/_load) ......... resolve to DTrainer
    - SSL loops, masked test(), _frozen_eval_rng, MAE __init__ ........ resolve to MaeTrainer
    - the epoch loop, early stopping, checkpointing, plots ............ inherited from Trainer
    - train() ......................................................... DTrainer's wrapper: it logs the
      Fabric setup, forces use_bf16=None (Fabric owns precision), then falls through the MRO to
      Trainer.train, whose self.train_one_epoch / self.validate dispatch to the MAE loops.

    The __init__ chain is: MaeDTrainer(fabric, **kwargs) -> DTrainer.__init__ -> MaeTrainer.__init__
    (which consumes the MAE-specific kwargs) -> Trainer.__init__; then, back in DTrainer.__init__,
    _setup() swaps self.device for fabric.device and wraps the model, optimizer and loaders. When
    MaeTrainer.__init__ resolves mask_ratio via _unwrap_model(), _bare_model does not exist yet and the
    hook falls back to the still-bare CPU module, which is exactly the right object to read it from.

    --- Usage contract (same as DTrainer) ---
    Build the model, optimizer and loaders on CPU after fabric.launch(), and pass device='cpu' (it is
    replaced by fabric.device during _setup). The MAE-specific arguments (dec_aux_criterion, mask_ratio,
    eval_mask_ratio, deterministic_eval, eval_seed) travel through **kwargs to MaeTrainer, and the
    criterion must be element-wise (reduction='none'), as in the single-device MaeTrainer.

    --- Distributed notes specific to the MAE ---
    - Deterministic evaluation still holds per rank: _frozen_eval_rng seeds every rank with the same
      eval_seed, and the val/test samplers are not shuffled, so each rank scores its own fixed shard
      under identical masks every epoch. The reduced val loss is therefore deterministic, and early
      stopping and per-epoch comparisons stay paired. Comparisons across runs are paired as long as the
      world size is unchanged (re-sharding the data realigns which mask lands on which sample).
    - Masks travel through the gather as uint8 (bool all_gather is backend-dependent); test() upcasts
      back to bool on the main rank only.
    - The collective schedule is data-independent: masked_out is a uniform call argument, the finite
      check all-reduces on every step, and the per-batch gather order inside test() is fixed, so all
      ranks always issue the same collectives in the same order.
    - test() returns (None, None) on non-main ranks -- Metrics.eval_reconstruction_quality already
      guards for that, and must be called on all ranks (there are collectives inside test()).
    """

    def __init__(self, fabric, **kwargs) -> None:
        super().__init__(fabric, **kwargs)


    def train_one_epoch_bf16(self, *args, **kwargs):
        """
        The manual-autocast loop is a single-device tool: it performs no cross-rank reductions, so under
        world_size > 1 it would silently report per-rank losses. Precision under Fabric belongs to the
        Fabric(precision=...) plugin with the standard train_one_epoch; train() already routes there.
        """
        if self.fabric.world_size > 1:
            raise RuntimeError(
                "train_one_epoch_bf16 is single-device: use Fabric(precision='bf16-mixed') with the "
                "standard training loop instead (train() already forces use_bf16=None)."
            )
        return super().train_one_epoch_bf16(*args, **kwargs)


    def validate_bf16(self, *args, **kwargs):
        """
        Single-device only, for the same reason as train_one_epoch_bf16 (no cross-rank reductions).
        """
        if self.fabric.world_size > 1:
            raise RuntimeError(
                "validate_bf16 is single-device: use Fabric(precision='bf16-mixed') with the standard "
                "validate() instead."
            )
        return super().validate_bf16(*args, **kwargs)
