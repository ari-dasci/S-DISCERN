"""
SCRUB (Stochastic Gradient-Descent / Ascent based UnleaRning with
knowledge-distillation Bounding) method.

Based on: "SCRUB: Fast and Private Unlearning" (Kurmanji et al., 2023)
https://github.com/meghdadk/SCRUB

Idea: teacher-student KD.  Maximise loss on the forget set (gradient
ascent) for the first *msteps* epochs, then minimise on the retain
set using KL-divergence from the frozen teacher.  Optionally apply
SWA-style model smoothing.
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


class _DistillKL(nn.Module):
    """KL-divergence distillation loss with temperature scaling."""

    def __init__(self, T=4.0):
        super().__init__()
        self.T = T

    def forward(self, student_logits, teacher_logits):
        p_s = F.log_softmax(student_logits / self.T, dim=1)
        p_t = F.softmax(teacher_logits / self.T, dim=1)
        return F.kl_div(p_s, p_t, reduction="batchmean") * (self.T ** 2)


class SCRUBMethod(BaseMethod):

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)
        self.teacher = None

    # ------------------------------------------------------------------ #
    def setup(self, train_X=None, train_y=None,
              retain_X=None, retain_y=None,
              forget_X=None, forget_y=None,
              transform_train=None, transform_retain=None,
              transform_forget=None):

        # Teacher = frozen copy of the original (pre-unlearning) model
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

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

        # Hyper-parameters from config
        scrub_alpha = self.config.get("scrub_alpha", 0.5)
        scrub_gamma = self.config.get("scrub_gamma", 1.0)
        scrub_msteps = self.config.get("scrub_msteps", 3)
        scrub_kd_T = self.config.get("scrub_kd_T", 4.0)
        scrub_smoothing = self.config.get("scrub_smoothing", 0.0)
        scrub_sstart = self.config.get("scrub_sstart", 10)

        # Data loaders
        retain_ds = ClassificationDataset(retain_X, retain_y,
                                          transforms=transform_retain)
        retain_loader = DataLoader(retain_ds,
                                   batch_size=self.config["train_batch_size"],
                                   shuffle=True, num_workers=4)

        forget_ds = ClassificationDataset(forget_X, forget_y,
                                          transforms=transform_forget)
        forget_loader = DataLoader(forget_ds,
                                   batch_size=self.config["train_batch_size"],
                                   shuffle=True, num_workers=4)

        # Validation
        forget_train_x, forget_val_x, forget_train_y, forget_val_y = \
            train_test_split(forget_X, forget_y, test_size=0.2,
                             stratify=forget_y, random_state=42)
        val_loader = DataLoader(
            ClassificationDataset(forget_val_x, forget_val_y,
                                  transforms=transform_forget),
            batch_size=self.config["val_batch_size"],
            shuffle=False, num_workers=4)

        # Models to device
        self.model.to(device)
        self.teacher.to(device)

        # Losses
        criterion_cls = nn.CrossEntropyLoss()
        criterion_kd = _DistillKL(T=scrub_kd_T)

        # Optional SWA smoothing
        if scrub_smoothing > 0:
            def avg_fn(avg_p, model_p, num_avg):
                return (1 - scrub_smoothing) * avg_p + scrub_smoothing * model_p
            swa_model = torch.optim.swa_utils.AveragedModel(
                self.model, avg_fn=avg_fn)
        else:
            swa_model = None

        # Save hyper-parameters
        os.makedirs(self.config["hyperparameters_path"], exist_ok=True)
        hp = {
            "scrub_alpha": scrub_alpha,
            "scrub_gamma": scrub_gamma,
            "scrub_msteps": scrub_msteps,
            "scrub_kd_T": scrub_kd_T,
            "scrub_smoothing": scrub_smoothing,
            "scrub_sstart": scrub_sstart,
            "max_epoch": self.config["max_epoch"],
        }
        with open(os.path.join(self.config["hyperparameters_path"],
                               "hyperparameters.yml"), "w") as f:
            yaml.dump(hp, f, default_flow_style=False)

        n_step = 0
        print("Starting SCRUB unlearning", flush=True)

        # Early stopping state
        early_stop_state = {
            'succed_saved': False,
            'best_forget_acc': float('inf'),
            'consecutive_low_count': 0,
        }

        for epoch in range(self.config["max_epoch"]):

            # ── Maximise step (gradient ascent on forget set) ────────
            if epoch < scrub_msteps:
                self.model.train()
                for images, target in tqdm(forget_loader,
                                           total=len(forget_loader),
                                           desc=f"E{epoch} maximize"):
                    n_step += 1
                    images, target = images.to(device), target.to(device)
                    self.optimizer.zero_grad()

                    student_out = self.model(images)
                    with torch.no_grad():
                        teacher_out = self.teacher(images)

                    # Ascent: maximise classification + KD on forget data
                    loss = -(scrub_alpha * criterion_cls(student_out, target)
                             + (1 - scrub_alpha) * criterion_kd(student_out, teacher_out))
                    loss *= scrub_gamma
                    loss.backward()
                    self.optimizer.step()

            # ── Minimise step (normal KD on retain set) ──────────────
            self.model.train()
            for images, target in tqdm(retain_loader,
                                       total=len(retain_loader),
                                       desc=f"E{epoch} minimize"):
                n_step += 1
                images, target = images.to(device), target.to(device)
                self.optimizer.zero_grad()

                student_out = self.model(images)
                with torch.no_grad():
                    teacher_out = self.teacher(images)

                loss = (scrub_alpha * criterion_cls(student_out, target)
                        + (1 - scrub_alpha) * criterion_kd(student_out, teacher_out))
                loss.backward()
                self.optimizer.step()

            # SWA update
            if swa_model is not None and epoch >= scrub_sstart:
                swa_model.update_parameters(self.model)

            # Forget validation
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, tgt in val_loader:
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

        # If using SWA, copy averaged parameters back
        if swa_model is not None:
            # Update batch-norm statistics
            retain_loader_bn = DataLoader(retain_ds,
                                          batch_size=self.config["train_batch_size"],
                                          shuffle=True, num_workers=4)
            torch.optim.swa_utils.update_bn(retain_loader_bn, swa_model, device)
            self.model.load_state_dict(
                {k.replace("module.", ""): v
                 for k, v in swa_model.state_dict().items()})

        return self.model
