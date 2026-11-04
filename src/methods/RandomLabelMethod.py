"""
Random Label unlearning method.

Idea: re-label the forget set with random (incorrect) class labels and
fine-tune the model on the combined (re-labelled forget + optional retain
subset) data.  This causes the model to "unlearn" the correct mapping for
the forget samples while preserving retain performance.
"""

import os
import random as pyrandom

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset_loader import ClassificationDataset
from src.utils.load_data import number_dataset_classes

from .baseMethod import BaseMethod
from .utils import should_save_checkpoint


class RandomLabelMethod(BaseMethod):

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)

    # ------------------------------------------------------------------ #
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
        torch.save(checkpoint,
                    os.path.join(self.config["checkpoint_path"], filename))
        print(f"✅ Checkpoint guardado en {self.config['checkpoint_path']}/{filename}")

    # ------------------------------------------------------------------ #
    def unlearn(self, train_X=None, train_y=None,
                retain_X=None, retain_y=None,
                forget_X=None, forget_y=None,
                transform_train=None, transform_retain=None,
                transform_forget=None):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = self.config["num_classes"]
        retain_fraction = self.config.get("rl_retain_fraction", 0.0)

        # ── Build randomly-relabeled forget set ───────────────────────
        forget_y_random = []
        if not self.config.get('use_forbidden_labels', False):
            for orig_label in forget_y:
                # Choose a label ≠ original
                candidates = [c for c in range(num_classes) if c != orig_label]
                forget_y_random.append(pyrandom.choice(candidates))

        else:
            candidates = [c for c in range(num_classes) if c not in self.config['forbidden_labels']]
            for orig_label in forget_y:
                # Choose a label from allowed candidates
                forget_y_random.append(pyrandom.choice(candidates))
        forget_y_random = np.array(forget_y_random)

        # ── Optionally include a subset of retain data ────────────────
        if retain_fraction > 0 and retain_X is not None and len(retain_X) > 0:
            n_retain_subset = max(1, int(len(retain_X) * retain_fraction))
            idx = pyrandom.sample(range(len(retain_X)), n_retain_subset)
            retain_sub_X = [retain_X[i] for i in idx]
            retain_sub_y = np.array([retain_y[i] for i in idx])

            combined_X = list(forget_X) + list(retain_sub_X)
            combined_y = np.concatenate([forget_y_random, retain_sub_y])
            combined_transform = transform_forget  # same transform for all
        else:
            combined_X = list(forget_X)
            combined_y = forget_y_random
            combined_transform = transform_forget

        train_ds = ClassificationDataset(combined_X, combined_y,
                                         transforms=combined_transform)
        train_loader = DataLoader(train_ds,
                                  batch_size=self.config["train_batch_size"],
                                  shuffle=True, num_workers=4)

        # Validation (forget split)
        forget_train_x, forget_val_x, forget_train_y, forget_val_y = \
            train_test_split(forget_X, forget_y, test_size=0.2,
                             stratify=forget_y, random_state=42)
        val_loader = DataLoader(
            ClassificationDataset(forget_val_x, forget_val_y,
                                  transforms=transform_forget),
            batch_size=self.config["val_batch_size"],
            shuffle=False, num_workers=4)

        self.model.to(device)

        # Save hyper-parameters
        os.makedirs(self.config["hyperparameters_path"], exist_ok=True)
        hp = {
            "rl_retain_fraction": retain_fraction,
            "max_epoch": self.config["max_epoch"],
            "num_classes": num_classes,
        }
        with open(os.path.join(self.config["hyperparameters_path"],
                               "hyperparameters.yml"), "w") as f:
            yaml.dump(hp, f, default_flow_style=False)

        n_step = 0
        print("Starting RandomLabel unlearning", flush=True)
        
        # Early stopping state
        early_stop_state = {
            'succed_saved': False,
            'best_forget_acc': float('inf'),
            'consecutive_low_count': 0,
        }
        
        for epoch in range(self.config["max_epoch"]):
            self.model.train()
            for images, target in tqdm(train_loader,
                                       total=len(train_loader)):
                n_step += 1
                images, target = images.to(device), target.to(device)
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.model.criterion(logits, target)
                loss.backward()
                self.optimizer.step()

            # Validation
            self.model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, tgt in val_loader:
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    out = self.model(imgs)
                    val_loss += self.model.criterion(out, tgt).item()
                    correct += (out.argmax(1) == tgt).sum().item()
                    total += tgt.size(0)
            
            forget_acc = correct / max(total, 1)
            print(f"Epoch {epoch}  val_loss={val_loss / max(len(val_loader), 1):.4f}  "
                  f"forget_acc={forget_acc:.4f}", flush=True)

            # Check if checkpoint should be saved / early stopping
            checkpoints_to_save, stop = should_save_checkpoint(epoch, forget_acc, self.config['num_classes'], self.config, early_stop_state)
            for ckpt_name, reason in checkpoints_to_save:
                print(f"Saving checkpoint at epoch {epoch}: {reason}", flush=True)
                self.save_results(epoch, n_step, filename=ckpt_name)
            if stop:
                print(f"🛑 Early stopping at epoch {epoch}", flush=True)
                break

        return self.model
