"""
UNSIR (Unlearning via Noise-based Impair-Repair) method.

Based on: "Fast Yet Effective Machine Unlearning" (Tarun et al., TNNLS 2024)
https://github.com/vikram2000b/Fast-Machine-Unlearning

The approach has three stages:
  1. **Noise optimisation**: For each forget class, learn an error-maximising
     noise tensor whose images maximise the model's cross-entropy loss on
     that class (gradient ascent on the noise, model frozen).
  2. **Impair**: Fine-tune the model for a few epochs on a mixture of noisy
     images (labelled with a dummy class index) and a retain sample, which
     degrades the model's knowledge of the forget classes.
  3. **Repair**: Fine-tune the model briefly on the retain set only, to
     recover generalisation on the remaining classes.
"""

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from src.models.ClassificationModel import ClassificationModel
from src.utils.dataset_loader import ClassificationDataset

from .baseMethod import BaseMethod
from .utils import should_save_checkpoint


# ── Learnable noise module (as in the original repo) ──────────────────
class _Noise(nn.Module):
    """A trainable noise tensor of shape (batch_size, C, H, W)."""

    def __init__(self, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise


class _RepeatedNoiseDataset(Dataset):
    """Wraps noise tensors and virtually repeats them `num_repeats` times.

    Only one physical copy per class is stored, keeping memory usage at
    ``noise_batch_size × C × H × W`` instead of
    ``noise_batch_size × num_repeats × C × H × W``.
    """

    def __init__(self, noise_tensors, num_repeats, label=0):
        # noise_tensors: list of (N_i, C, H, W) CPU tensors, one per class
        self.data = torch.cat(noise_tensors, dim=0) if noise_tensors else torch.empty(0)
        self.num_repeats = num_repeats
        self.base_len = len(self.data)
        self.label = label

    def __len__(self):
        return self.base_len * self.num_repeats

    def __getitem__(self, idx):
        return self.data[idx % self.base_len], self.label


class UNSIRMethod(BaseMethod):

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)

    # ------------------------------------------------------------------ #
    def setup(self, train_X=None, train_y=None,
              retain_X=None, retain_y=None,
              forget_X=None, forget_y=None,
              transform_train=None, transform_retain=None,
              transform_forget=None):
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Apply freeze if configured
        freeze_mode = self.config.get('freeze', 'nofreeze')
        if freeze_mode == 'freeze':
            self.model.freeze()
        elif freeze_mode == 'embs':
            self.model.emb_freeze()

        return train_X, train_y, retain_X, retain_y, forget_X, forget_y

    # ------------------------------------------------------------------ #
    def unlearn(self, train_X=None, train_y=None,
                retain_X=None, retain_y=None,
                forget_X=None, forget_y=None,
                transform_train=None, transform_retain=None,
                transform_forget=None):

        device = self.model.device
        cfg = self.config
        forbidden_labels = cfg.get('forbidden_labels', [])

        noise_batch_size = cfg.get('unsir_noise_batch_size', 256)
        noise_epochs = cfg.get('unsir_noise_epochs', 5)
        noise_steps = cfg.get('unsir_noise_steps', 8)
        noise_lr = cfg.get('unsir_noise_lr', 0.1)
        noise_reg = cfg.get('unsir_noise_reg', 0.1)
        num_noise_batches = cfg.get('unsir_num_noise_batches', 20)
        impair_epochs = cfg.get('unsir_impair_epochs', 1)
        impair_lr = cfg.get('unsir_impair_lr', 0.02)
        repair_epochs = cfg.get('unsir_repair_epochs', 1)
        repair_lr = cfg.get('unsir_repair_lr', 0.01)

        # Determine image shape from a single forget sample
        sample_ds = ClassificationDataset(forget_X[:1], forget_y[:1], transforms=transform_forget)
        sample_img, _ = sample_ds[0]
        img_shape = sample_img.shape  # (C, H, W)

        # ──────────────────────────────────────────────────────────────
        # Stage 1: Optimise error-maximising noise per forget class
        # ──────────────────────────────────────────────────────────────
        self.model.eval()
        noises = {}

        for cls in forbidden_labels:
            print(f"[UNSIR] Optimising noise for class {cls} ...")
            noise_module = _Noise(noise_batch_size, *img_shape).to(device)
            opt_noise = torch.optim.Adam(noise_module.parameters(), lr=noise_lr)

            for ep in range(noise_epochs):
                total_loss = []
                for _ in range(noise_steps):
                    inputs = noise_module()
                    labels = torch.full((noise_batch_size,), cls, dtype=torch.long, device=device)
                    outputs = self.model(inputs)
                    # Gradient ascent: *maximise* CE → negate it
                    loss = -F.cross_entropy(outputs, labels) + noise_reg * torch.mean(
                        torch.sum(inputs.pow(2), dim=[1, 2, 3])
                    )
                    opt_noise.zero_grad()
                    loss.backward()
                    opt_noise.step()
                    total_loss.append(loss.item())
                print(f"  epoch {ep}: loss = {np.mean(total_loss):.4f}")

            noises[cls] = noise_module

        # Forget validation loader for early stopping
        forget_train_x, forget_val_x, forget_train_y, forget_val_y = \
            train_test_split(forget_X, forget_y, test_size=0.2,
                             stratify=forget_y, random_state=42)
        val_loader_forget = DataLoader(
            ClassificationDataset(forget_val_x, forget_val_y,
                                  transforms=transform_forget),
            batch_size=cfg.get('val_batch_size', 64),
            shuffle=False, num_workers=0)

        # Early stopping state (shared across impair + repair)
        early_stop_state = {
            'succed_saved': False,
            'best_forget_acc': float('inf'),
            'consecutive_low_count': 0,
        }
        global_epoch = 0

        # ──────────────────────────────────────────────────────────────
        # Stage 2: Impair – train on noisy + retain data
        # ──────────────────────────────────────────────────────────────
        print("[UNSIR] Impair step ...")

        # Build noisy dataset: label all noisy images with a *wrong* class.
        # Original paper uses class_num=0 for all noise batches.
        # We keep only ONE copy of the learned noise per class and repeat
        # virtually via _RepeatedNoiseDataset to avoid huge RAM usage.
        noise_tensors = []
        for cls in forbidden_labels:
            noise_tensors.append(noises[cls]().cpu().detach())

        noisy_ds = _RepeatedNoiseDataset(noise_tensors, num_noise_batches, label=0)

        # Retain dataset – lazy loading; transforms applied on-the-fly in
        # __getitem__, so the full 224×224 tensors are NEVER held in RAM.
        retain_ds = ClassificationDataset(retain_X, retain_y, transforms=transform_retain)

        # Combine and create DataLoader
        combined_ds = ConcatDataset([noisy_ds, retain_ds])
        combined_loader = DataLoader(combined_ds, batch_size=cfg.get('train_batch_size', 256),
                                     shuffle=True, num_workers=0, drop_last=False)

        self.model.train()
        opt_impair = torch.optim.Adam(self.model.parameters(), lr=impair_lr)

        for ep in range(impair_epochs):
            running_loss = 0.0
            running_acc = 0
            total = 0
            for batch_data in tqdm(combined_loader, desc=f"Impair ep {ep+1}/{impair_epochs}"):
                inputs, labels = batch_data
                if isinstance(inputs, (list, tuple)):
                    inputs = torch.stack(inputs)
                if isinstance(labels, (list, tuple)):
                    labels = torch.tensor(labels)
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                opt_impair.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                opt_impair.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                running_acc += (preds == labels).sum().item()
                total += inputs.size(0)

            print(f"  Impair ep {ep+1}: loss={running_loss/total:.4f}, acc={running_acc*100/total:.1f}%")

            # Forget validation after impair epoch
            self.model.eval()
            correct_f, total_f = 0, 0
            with torch.no_grad():
                for imgs, tgt in val_loader_forget:
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    out = self.model(imgs)
                    correct_f += (out.argmax(1) == tgt).sum().item()
                    total_f += tgt.size(0)
            forget_acc = correct_f / max(total_f, 1)
            print(f"  Impair ep {ep+1}: forget_acc={forget_acc:.4f}")

            checkpoints_to_save, stop = should_save_checkpoint(
                global_epoch, forget_acc, cfg.get('num_classes', 10), cfg, early_stop_state)
            for ckpt_name, reason in checkpoints_to_save:
                print(f"Saving checkpoint at impair ep {ep+1}: {reason}", flush=True)
                self.save_results(global_epoch, 0, filename=ckpt_name)
            global_epoch += 1
            if stop:
                print(f"🛑 Early stopping during impair at ep {ep+1}", flush=True)
                break

        # ──────────────────────────────────────────────────────────────
        # Stage 3: Repair – fine-tune on retain data only
        # ──────────────────────────────────────────────────────────────
        print("[UNSIR] Repair step ...")

        heal_loader = DataLoader(retain_ds, batch_size=cfg.get('train_batch_size', 256),
                                 shuffle=True, num_workers=0, drop_last=False)

        opt_repair = torch.optim.Adam(self.model.parameters(), lr=repair_lr)

        for ep in range(repair_epochs):
            running_loss = 0.0
            running_acc = 0
            total = 0
            self.model.train()
            for batch_data in tqdm(heal_loader, desc=f"Repair ep {ep+1}/{repair_epochs}"):
                inputs, labels = batch_data
                if isinstance(inputs, (list, tuple)):
                    inputs = torch.stack(inputs)
                if isinstance(labels, (list, tuple)):
                    labels = torch.tensor(labels)
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                opt_repair.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                opt_repair.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                running_acc += (preds == labels).sum().item()
                total += inputs.size(0)

            print(f"  Repair ep {ep+1}: loss={running_loss/total:.4f}, acc={running_acc*100/total:.1f}%")

            # Forget validation after repair epoch
            self.model.eval()
            correct_f, total_f = 0, 0
            with torch.no_grad():
                for imgs, tgt in val_loader_forget:
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    out = self.model(imgs)
                    correct_f += (out.argmax(1) == tgt).sum().item()
                    total_f += tgt.size(0)
            forget_acc = correct_f / max(total_f, 1)
            print(f"  Repair ep {ep+1}: forget_acc={forget_acc:.4f}")

            checkpoints_to_save, stop = should_save_checkpoint(
                global_epoch, forget_acc, cfg.get('num_classes', 10), cfg, early_stop_state)
            for ckpt_name, reason in checkpoints_to_save:
                print(f"Saving checkpoint at repair ep {ep+1}: {reason}", flush=True)
                self.save_results(global_epoch, 0, filename=ckpt_name)
            global_epoch += 1
            if stop:
                print(f"🛑 Early stopping during repair at ep {ep+1}", flush=True)
                break

        # ── Save ─────────────────────────────────────────────────────
        # Save hyperparameters
        os.makedirs(self.config.get('hyperparameters_path', self.config['checkpoint_path']), exist_ok=True)
        hp = {k: v for k, v in self.config.items()
              if k.startswith('unsir_') or k in ('unlearning_method',)}
        with open(os.path.join(self.config.get('hyperparameters_path', self.config['checkpoint_path']),
                               "hyperparameters.yaml"), 'w') as f:
            yaml.dump(hp, f)

        # Final checkpoint (if no best was found during training)
        if not early_stop_state['succed_saved'] and early_stop_state['best_forget_acc'] == float('inf'):
            self.save_results(0, 0)

        return self.model

    # ------------------------------------------------------------------ #
    def save_results(self, epoch, global_step, filename=None):
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": self.model.state_dict(),
        }
        os.makedirs(self.config["checkpoint_path"], exist_ok=True)
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:02}.pt"
        torch.save(
            checkpoint,
            os.path.join(self.config["checkpoint_path"], filename),
        )
        print(f"✅ Checkpoint guardado en {self.config['checkpoint_path']}/{filename}")
