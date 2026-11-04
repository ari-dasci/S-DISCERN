"""
Fisher Forgetting / Variational Forgetting method.

Based on: "Eternal Sunshine of the Spotless Net: Selective Forgetting in
Deep Networks" (Golatkar, Achille & Soatto, CVPR 2020)

Two modes (controlled by config key ``fisher_mode``):

* ``"retain"`` (default) — compute the FIM diagonal on the *retain* set.
  Parameters important for retain data receive less noise.

* ``"subtract"`` — compute the FIM diagonal on the *full training* set,
  then subtract the FIM diagonal of the *forget* set:
      F_retain_approx = F_train - F_forget
  This avoids a pass over the (potentially large) retain set and gives
  an unbiased approximation when retain ∪ forget = train.
"""

import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset_loader import ClassificationDataset
from .baseMethod import BaseMethod
from .utils import compute_fisher


class FisherForgettingMethod(BaseMethod):

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)
        self.fisher_diag = None

    # ------------------------------------------------------------------ #
    def setup(self, train_X=None, train_y=None,
              retain_X=None, retain_y=None,
              forget_X=None, forget_y=None,
              transform_train=None, transform_retain=None,
              transform_forget=None):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

        fisher_mode = self.config.get("fisher_mode", "retain")
        fisher_path = self.config.get("fisher_path", None)

        if fisher_mode == "subtract":
            # F_retain ≈ F_train − F_forget
            # Only the train Fisher is cached (fisher_path); the forget
            # Fisher is always recomputed because it varies per experiment.

            print("Computing Fisher on full train set…", flush=True)
            if fisher_path is not None and os.path.exists(fisher_path):
                print(f"  Loading cached train Fisher from {fisher_path}", flush=True)
                fisher_train = torch.load(fisher_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            else:
                train_loader = DataLoader(
                    ClassificationDataset(train_X, train_y,
                                          transforms=transform_train),
                    batch_size=self.config["train_batch_size"],
                    shuffle=True, num_workers=4)
                fisher_train = compute_fisher(self.model, train_loader)
                if fisher_path is not None:
                    os.makedirs(os.path.dirname(fisher_path) or ".", exist_ok=True)
                    torch.save(fisher_train, fisher_path)
                    print(f"  Train Fisher saved to {fisher_path}", flush=True)

            print("Computing Fisher on forget set…", flush=True)
            forget_loader = DataLoader(
                ClassificationDataset(forget_X, forget_y,
                                      transforms=transform_forget),
                batch_size=self.config["train_batch_size"],
                shuffle=True, num_workers=4)
            fisher_forget = compute_fisher(self.model, forget_loader)

            # Subtract and clamp to 0 (never negative diagonal)
            self.fisher_diag = {
                n: torch.clamp(fisher_train[n] - fisher_forget[n], min=0.0)
                for n in fisher_train
            }

        else:  # default: "retain"
            # Compute Fisher on the *retain* set so we know which parameters
            # are important for retaining performance.
            retain_loader = DataLoader(
                ClassificationDataset(retain_X, retain_y,
                                      transforms=transform_retain),
                batch_size=self.config["train_batch_size"],
                shuffle=True, num_workers=4)

            print("Computing Fisher on retain set…", flush=True)
            self.fisher_diag = compute_fisher(self.model, retain_loader,
                                              path=fisher_path)

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
    def unlearn(self, train_X=None, train_y=None,
                retain_X=None, retain_y=None,
                forget_X=None, forget_y=None,
                transform_train=None, transform_retain=None,
                transform_forget=None):

        if self.fisher_diag is None:
            raise ValueError("setup() must be called before unlearn().")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        noise_scale = self.config.get("ff_noise_scale", 1.0)
        ff_alpha = self.config.get("ff_alpha", 1e-6)  # damping to avoid division by zero
        fisher_mode = self.config.get("fisher_mode", "retain")

        # Save hyper-parameters
        os.makedirs(self.config["hyperparameters_path"], exist_ok=True)
        hp = {
            "ff_noise_scale": noise_scale,
            "ff_alpha": ff_alpha,
            "fisher_mode": fisher_mode,
        }
        with open(os.path.join(self.config["hyperparameters_path"],
                               "hyperparameters.yml"), "w") as f:
            yaml.dump(hp, f, default_flow_style=False)

        print(f"Applying Fisher Forgetting (noise injection, mode={fisher_mode})", flush=True)

        # Add noise inversely proportional to the Fisher diagonal:
        #   θ' = θ + noise_scale * N(0, 1 / (F + α))
        freeze_mode = self.config.get('freeze', 'nofreeze')

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # When freeze is active, only modify classifier parameters
                if freeze_mode == 'freeze' and not name.startswith('classifier.'):
                    continue

                if freeze_mode == 'embs' and not self.model.is_emb_param(name):  # classifier + last backbone stage
                    continue

                if name in self.fisher_diag:
                    fisher_val = self.fisher_diag[name].to(device)
                    # Variance of noise = noise_scale / (fisher + alpha)
                    noise_var = noise_scale / (fisher_val + ff_alpha)
                    noise = torch.randn_like(param) * noise_var.sqrt()
                    param.add_(noise)

        print("Fisher Forgetting completed – noise injected", flush=True)
        self.save_results(0, 0)

        return self.model
