"""
SFU – Source-Free Unlearning method.

Based on: "Towards Source-Free Machine Unlearning" (Ahmed et al., CVPR 2025)
https://github.com/UCR-Vision-and-Learning-Group/source-free-unlearning

Idea: linearise the last layers of the model, compute the Hessian–vector
product (or diagonal Hessian) on the retain data, and use Newton-step style
weight correction to remove the influence of the forget set.

Our adaptation: we approximate the Hessian diagonal on the retain set,
compute the gradient of the loss w.r.t. the forget set, and apply a
Newton-like weight update:  θ' = θ + H_retain^{-1} * ∇L_forget
This effectively subtracts the influence of the forget data.
"""

import copy
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset_loader import ClassificationDataset

from .baseMethod import BaseMethod


class SFUMethod(BaseMethod):

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
    def save_results(self, epoch, global_step):
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": self.model.state_dict(),
        }
        os.makedirs(self.config["checkpoint_path"], exist_ok=True)
        torch.save(checkpoint,
                    os.path.join(self.config["checkpoint_path"],
                                 f"checkpoint_epoch_{epoch:02}.pt"))
        print(f"✅ Checkpoint guardado en {self.config['checkpoint_path']}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_hessian_diag(model, data_loader, device):
        """Approximate diagonal Hessian using Hutchinson's estimator."""
        hess_diag = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                     if p.requires_grad}
        criterion = nn.CrossEntropyLoss()
        n_samples = 0

        model.eval()
        for images, target in tqdm(data_loader, desc="Hessian diag"):
            images, target = images.to(device), target.to(device)
            model.zero_grad()
            output = model(images)
            loss = criterion(output, target)
            grads = torch.autograd.grad(loss, [p for p in model.parameters()
                                                if p.requires_grad],
                                         create_graph=True)
            for (n, p), g in zip(
                    ((n, p) for n, p in model.named_parameters() if p.requires_grad),
                    grads):
                # Hutchinson diagonal: E[v * (Hv)] with v ~ Rademacher
                v = torch.randint_like(g, 0, 2) * 2.0 - 1.0
                Hv = torch.autograd.grad(g, p, grad_outputs=v,
                                          retain_graph=True)[0]
                hess_diag[n] += (v * Hv).detach() * images.size(0)
            n_samples += images.size(0)

        for n in hess_diag:
            hess_diag[n] /= max(n_samples, 1)
            hess_diag[n] = hess_diag[n].abs()  # take absolute value for stability
        return hess_diag

    @staticmethod
    def _compute_gradient(model, data_loader, device):
        """Average gradient of the loss on a given dataset."""
        grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                 if p.requires_grad}
        criterion = nn.CrossEntropyLoss()
        n_samples = 0

        model.eval()
        for images, target in tqdm(data_loader, desc="Gradient"):
            images, target = images.to(device), target.to(device)
            model.zero_grad()
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grads[n] += p.grad.detach() * images.size(0)
            n_samples += images.size(0)

        for n in grads:
            grads[n] /= max(n_samples, 1)
        return grads

    # ------------------------------------------------------------------ #
    def unlearn(self, train_X=None, train_y=None,
                retain_X=None, retain_y=None,
                forget_X=None, forget_y=None,
                transform_train=None, transform_retain=None,
                transform_forget=None):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        sfu_damping = self.config.get("sfu_damping", 1e-4)
        sfu_scale = self.config.get("sfu_scale", 1.0)
        sfu_hessian_data = self.config.get("sfu_hessian_data", "retain")

        # Data loaders
        retain_loader = DataLoader(
            ClassificationDataset(retain_X, retain_y,
                                  transforms=transform_retain),
            batch_size=self.config["train_batch_size"],
            shuffle=True, num_workers=4)

        forget_loader = DataLoader(
            ClassificationDataset(forget_X, forget_y,
                                  transforms=transform_forget),
            batch_size=self.config["train_batch_size"],
            shuffle=False, num_workers=4)

        # Save hyper-parameters
        os.makedirs(self.config["hyperparameters_path"], exist_ok=True)
        hp = {"sfu_damping": sfu_damping, "sfu_scale": sfu_scale,
              "sfu_hessian_data": sfu_hessian_data}
        with open(os.path.join(self.config["hyperparameters_path"],
                               "hyperparameters.yml"), "w") as f:
            yaml.dump(hp, f, default_flow_style=False)

        # 1. Compute diagonal Hessian on the chosen split
        if sfu_hessian_data == "forget":
            print("SFU: computing Hessian diagonal on FORGET set …", flush=True)
            hess_diag = self._compute_hessian_diag(self.model, forget_loader, device)
        else:
            print("SFU: computing Hessian diagonal on RETAIN set …", flush=True)
            hess_diag = self._compute_hessian_diag(self.model, retain_loader, device)

        # 2. Compute gradient on forget set
        print("SFU: computing gradient on forget set …", flush=True)
        grad_forget = self._compute_gradient(self.model, forget_loader, device)

        # 3. Newton-like update:  θ' = θ + scale * H_retain^{-1} * ∇L_forget
        print("SFU: applying weight update …", flush=True)
        freeze_mode = self.config.get('freeze', 'nofreeze')

        with torch.no_grad():
            for n, p in self.model.named_parameters():
                # When freeze is active, only modify classifier parameters
                if freeze_mode != 'nofreeze' and not n.startswith('classifier.'):
                    continue

                if n in hess_diag and n in grad_forget:
                    h_inv = 1.0 / (hess_diag[n] + sfu_damping)
                    delta = sfu_scale * h_inv * grad_forget[n]
                    p.add_(delta)

        print("SFU: unlearning completed", flush=True)
        self.save_results(0, 0)
        return self.model
