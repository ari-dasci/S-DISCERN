"""
Boundary Expanding unlearning method.

Based on: "Boundary Unlearning: Rapid Forgetting of Deep Networks via
Shifting the Decision Boundary" (Chen et al., 2023)
https://github.com/... (Boundary-Unlearning-Code / expand_exp.py)

Idea: widen the classifier by adding an extra "dustbin" class, train the
forget samples to be mapped to that new class, and then prune the extra
class back, so the original model no longer classifies forget data correctly.
"""

import copy
import os

import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset_loader import ClassificationDataset

from .baseMethod import BaseMethod
from .utils import should_save_checkpoint


class BoundaryExpandMethod(BaseMethod):

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)

    def setup(self, train_X=None, train_y=None,
              retain_X=None, retain_y=None,
              forget_X=None, forget_y=None,
              transform_train=None, transform_retain=None,
              transform_forget=None):

        # Apply freeze if configured
        freeze_mode = self.config.get('freeze', 'nofreeze')
        if freeze_mode == 'freeze':
            self.model.freeze()
        elif freeze_mode == 'embs':
            self.model.emb_freeze()

        return train_X, train_y, retain_X, retain_y, forget_X, forget_y

    def save_results(self, epoch, global_step, filename=None):
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": self.model.state_dict(),
        }
        os.makedirs(self.config["checkpoint_path"], exist_ok=True)
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:02}.pt"
        torch.save(checkpoint,
                    os.path.join(self.config["checkpoint_path"], filename))
        print(f"✅ Checkpoint guardado en {self.config['checkpoint_path']}/{filename}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _init_params(module):
        """Kaiming-uniform init for Linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ #
    def unlearn(self, train_X=None, train_y=None,
                retain_X=None, retain_y=None,
                forget_X=None, forget_y=None,
                transform_train=None, transform_retain=None,
                transform_forget=None):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = self.model._num_classes
        dustbin_class = num_classes  # new class index

        # ── 1. Build widened classifier (num_classes + 1) ──────────────
        old_classifier = self.model.classifier  # nn.Sequential(ReLU, Dropout, Linear, TemperatureScaling)
        old_linear = old_classifier[-2]          # last nn.Linear (before TemperatureScaling)
        in_features = old_linear.in_features

        widen_linear = nn.Linear(in_features, num_classes + 1)
        self._init_params(widen_linear)

        # Copy existing weights into the widened layer
        with torch.no_grad():
            widen_linear.weight[:num_classes] = old_linear.weight.clone()
            widen_linear.bias[:num_classes] = old_linear.bias.clone()

        # Replace the classifier head temporarily, keeping TemperatureScaling last
        widen_classifier = nn.Sequential(*list(old_classifier[:-2]), widen_linear, old_classifier[-1])
        self.model.classifier = widen_classifier
        self.model.to(device)

        # ── 2. Train forget samples → dustbin class ───────────────────
        forget_ds = ClassificationDataset(forget_X, forget_y,
                                          transforms=transform_forget)
        forget_loader = DataLoader(forget_ds,
                                   batch_size=self.config["train_batch_size"],
                                   shuffle=True, num_workers=4)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(),
                                     lr=self.config.get("boundary_lr", 1e-5),
                                     momentum=0.9)

        os.makedirs(self.config["hyperparameters_path"], exist_ok=True)
        hp = {"boundary_lr": self.config.get("boundary_lr", 1e-5),
              "max_epoch": self.config["max_epoch"]}
        with open(os.path.join(self.config["hyperparameters_path"],
                               "hyperparameters.yml"), "w") as f:
            yaml.dump(hp, f, default_flow_style=False)

        # Forget validation loader for early stopping
        forget_train_x, forget_val_x, forget_train_y, forget_val_y = \
            train_test_split(forget_X, forget_y, test_size=0.2,
                             stratify=forget_y, random_state=42)
        val_loader_forget = DataLoader(
            ClassificationDataset(forget_val_x, forget_val_y,
                                  transforms=transform_forget),
            batch_size=self.config.get("val_batch_size", 64),
            shuffle=False, num_workers=4)

        n_step = 0
        print("Starting BoundaryExpand unlearning (widening phase)", flush=True)

        # Early stopping state
        early_stop_state = {
            'succed_saved': False,
            'best_forget_acc': float('inf'),
            'consecutive_low_count': 0,
        }
        stopped_early = False

        for epoch in range(self.config["max_epoch"]):
            self.model.train()
            for images, _ in tqdm(forget_loader, total=len(forget_loader)):
                n_step += 1
                images = images.to(device)
                target_labels = torch.full((images.size(0),), dustbin_class,
                                           dtype=torch.long, device=device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, target_labels)
                loss.backward()
                optimizer.step()

            # Forget validation (against original labels, not dustbin)
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, tgt in val_loader_forget:
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    out = self.model(imgs)
                    # Only consider the original num_classes logits
                    preds = out[:, :num_classes].argmax(dim=1)
                    correct += (preds == tgt).sum().item()
                    total += tgt.size(0)
            forget_acc = correct / max(total, 1)
            print(f"Epoch {epoch} (widen)  forget_acc={forget_acc:.4f}", flush=True)

            # Check early stopping (but save checkpoints after pruning)
            _, stop = should_save_checkpoint(epoch, forget_acc, num_classes, self.config, early_stop_state)
            if stop:
                print(f"🛑 Early stopping at epoch {epoch} (widening phase)", flush=True)
                stopped_early = True
                break

        # ── 3. Prune the dustbin class ────────────────────────────────
        widen_linear_trained = self.model.classifier[-2]
        pruned_linear = nn.Linear(in_features, num_classes)
        with torch.no_grad():
            pruned_linear.weight.copy_(widen_linear_trained.weight[:num_classes])
            pruned_linear.bias.copy_(widen_linear_trained.bias[:num_classes])

        pruned_classifier = nn.Sequential(
            *list(self.model.classifier[:-2]), pruned_linear, self.model.classifier[-1])
        self.model.classifier = pruned_classifier
        self.model._num_classes = num_classes
        self.model.to(device)
        print("BoundaryExpand: dustbin class pruned", flush=True)

        # Save named checkpoints based on achieved thresholds
        if early_stop_state['succed_saved']:
            self.save_results(self.config["max_epoch"] - 1, n_step, filename="checkpoint_succed.pt")
        if early_stop_state['best_forget_acc'] < float('inf'):
            self.save_results(self.config["max_epoch"] - 1, n_step, filename="checkpoint_best.pt")
        self.save_results(self.config["max_epoch"] - 1, n_step)

        return self.model
