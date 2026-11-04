"""
Boundary Shrink unlearning method.

Based on: "Boundary Unlearning: Rapid Forgetting of Deep Networks via
Shifting the Decision Boundary" (Chen et al., 2023)
https://github.com/... (Boundary-Unlearning-Code)

Idea: generate adversarial examples from the forget set using FGSM so
that they cross the decision boundary.  Then train the model to classify
those adversarial examples with the *adversarial* predicted labels, effectively
shrinking the decision boundary around the forget class.
"""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset_loader import ClassificationDataset

from .baseMethod import BaseMethod
from .utils import should_save_checkpoint


# ── Simple FGSM attack ──────────────────────────────────────────────────
class _FGSM:
    """Fast Gradient Sign Method adversarial attack."""

    def __init__(self, model, eps, norm=True, random_start=False, device="cuda"):
        self.model = model
        self.eps = eps
        self.norm = norm
        self.random_start = random_start
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def perturb(self, x, y, target_y=None, model=None, device=None):
        model = model or self.model
        device = device or self.device
        x_adv = x.clone().detach().requires_grad_(True).to(device)
        y = y.to(device)

        output = model(x_adv)
        loss = self.criterion(output, y)
        loss.backward()

        x_adv = x_adv + self.eps * x_adv.grad.sign()
        if self.norm:
            x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv.detach()


class BoundaryShrinkMethod(BaseMethod):

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)
        self.original_model = None

    def setup(self, train_X=None, train_y=None,
              retain_X=None, retain_y=None,
              forget_X=None, forget_y=None,
              transform_train=None, transform_retain=None,
              transform_forget=None):

        self.original_model = copy.deepcopy(self.model)
        self.original_model.eval()
        for p in self.original_model.parameters():
            p.requires_grad = False

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

    def unlearn(self, train_X=None, train_y=None,
                retain_X=None, retain_y=None,
                forget_X=None, forget_y=None,
                transform_train=None, transform_retain=None,
                transform_forget=None):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        eps = self.config.get("boundary_eps", 0.1)

        forget_ds = ClassificationDataset(forget_X, forget_y,
                                          transforms=transform_forget)
        forget_loader = DataLoader(forget_ds,
                                   batch_size=self.config["train_batch_size"],
                                   shuffle=True, num_workers=4)

        # Forget validation loader for early stopping
        forget_train_x, forget_val_x, forget_train_y, forget_val_y = \
            train_test_split(forget_X, forget_y, test_size=0.2,
                             stratify=forget_y, random_state=42)
        val_loader_forget = DataLoader(
            ClassificationDataset(forget_val_x, forget_val_y,
                                  transforms=transform_forget),
            batch_size=self.config["val_batch_size"],
            shuffle=False, num_workers=4)

        self.model.to(device)
        self.original_model.to(device)
        adv = _FGSM(self.original_model, eps, norm=True,
                     random_start=False, device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(),
                                     lr=self.config.get("boundary_lr", 1e-5),
                                     momentum=0.9)

        os.makedirs(self.config["hyperparameters_path"], exist_ok=True)
        hp = {"boundary_eps": eps,
              "boundary_lr": self.config.get("boundary_lr", 1e-5),
              "max_epoch": self.config["max_epoch"]}
        with open(os.path.join(self.config["hyperparameters_path"],
                               "hyperparameters.yml"), "w") as f:
            yaml.dump(hp, f, default_flow_style=False)

        n_step = 0
        print("Starting BoundaryShrink unlearning", flush=True)

        # Early stopping state
        early_stop_state = {
            'succed_saved': False,
            'best_forget_acc': float('inf'),
            'consecutive_low_count': 0,
        }

        for epoch in range(self.config["max_epoch"]):
            self.model.train()
            for images, target in tqdm(forget_loader,
                                       total=len(forget_loader)):
                n_step += 1
                images, target = images.to(device), target.to(device)

                # Generate adversarial examples using the frozen original model
                self.original_model.eval()
                x_adv = adv.perturb(images, target)

                # Get the boundary label from the original model
                with torch.no_grad():
                    adv_logits = self.original_model(x_adv)
                    boundary_labels = adv_logits.argmax(dim=1)

                # Train the student model to output the boundary labels
                self.model.train()
                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, boundary_labels)
                loss.backward()
                optimizer.step()

            # Forget validation
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, tgt in val_loader_forget:
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    out = self.model(imgs)
                    correct += (out.argmax(1) == tgt).sum().item()
                    total += tgt.size(0)
            forget_acc = correct / max(total, 1)
            print(f"Epoch {epoch}  forget_acc={forget_acc:.4f}", flush=True)

            # Check if checkpoint should be saved / early stopping
            checkpoints_to_save, stop = should_save_checkpoint(epoch, forget_acc, self.config['num_classes'], self.config, early_stop_state)
            for ckpt_name, reason in checkpoints_to_save:
                print(f"Saving checkpoint at epoch {epoch}: {reason}", flush=True)
                self.save_results(epoch, n_step, filename=ckpt_name)
            if stop:
                print(f"🛑 Early stopping at epoch {epoch}", flush=True)
                break

        return self.model
