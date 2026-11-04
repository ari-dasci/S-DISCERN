"""
BadTeacher (Incompetent Teacher) unlearning method.

Based on: "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks
using an Incompetent Teacher" (Chundawat et al., 2022)
https://github.com/vikram2000b/bad-teaching-unlearning

The idea: use a randomly-initialised "incompetent" teacher for forget samples
and the original (competent) teacher for retain samples.  A student model is
distilled with KL-divergence so that it learns to mimic the incompetent
teacher on the forget set and the competent teacher on the retain set.
"""

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.ClassificationModel import ClassificationModel
from src.utils.dataset_loader import ClassificationDataset

from .baseMethod import BaseMethod
from .utils import should_save_checkpoint


# ── helper dataset that interleaves forget (label=1) / retain (label=0) ──
class _UnLearningData(Dataset):
    """Wraps forget + retain data.  __getitem__ returns (image, is_forget)."""

    def __init__(self, forget_X, forget_y, retain_X, retain_y,
                 transform_forget=None, transform_retain=None):
        super().__init__()
        self.forget_ds = ClassificationDataset(forget_X, forget_y, transforms=transform_forget)
        self.retain_ds = ClassificationDataset(retain_X, retain_y, transforms=transform_retain)
        self.forget_len = len(self.forget_ds)
        self.retain_len = len(self.retain_ds)

    def __len__(self):
        return self.forget_len + self.retain_len

    def __getitem__(self, idx):
        if idx < self.forget_len:
            x, _ = self.forget_ds[idx]
            return x, 1  # 1 → forget
        else:
            x, _ = self.retain_ds[idx - self.forget_len]
            return x, 0  # 0 → retain


class _ForgetOnlyData(Dataset):
    """Wraps only forget data. All samples marked as forget (is_forget=1)."""

    def __init__(self, forget_X, forget_y, transform_forget=None):
        super().__init__()
        self.forget_ds = ClassificationDataset(forget_X, forget_y, transforms=transform_forget)

    def __len__(self):
        return len(self.forget_ds)

    def __getitem__(self, idx):
        x, _ = self.forget_ds[idx]
        return x, 1  # 1 → forget (incompetent teacher)


class BadTeacherMethod(BaseMethod):

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)
        self.competent_teacher = None
        self.incompetent_teacher = None

    # ------------------------------------------------------------------ #
    def setup(self, train_X=None, train_y=None,
              retain_X=None, retain_y=None,
              forget_X=None, forget_y=None,
              transform_train=None, transform_retain=None,
              transform_forget=None):

        # Competent teacher = frozen copy of the original model
        self.competent_teacher = deepcopy(self.model)
        self.competent_teacher.eval()
        for p in self.competent_teacher.parameters():
            p.requires_grad = False

        # Incompetent teacher = randomly-initialised model with same arch
        self.incompetent_teacher = ClassificationModel(
            self.model._base_model,
            self.model._num_classes,
            self.model._hyperparameters,
            classes_weights=self.model.classes_weights,
            extra_layers=self.model._extra_layers,
        )
        self.incompetent_teacher.eval()
        for p in self.incompetent_teacher.parameters():
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
    @staticmethod
    def _unlearner_loss(student_logits, is_forget_labels,
                        competent_logits, incompetent_logits,
                        temperature):
        """KL between student and teacher mix (incompetent on forget, competent on retain)."""
        labels = is_forget_labels.unsqueeze(1).float()
        teacher_soft = (labels * F.softmax(incompetent_logits / temperature, dim=1)
                        + (1 - labels) * F.softmax(competent_logits / temperature, dim=1))
        student_log_soft = F.log_softmax(student_logits / temperature, dim=1)
        return F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")

    # ------------------------------------------------------------------ #
    def unlearn(self, train_X=None, train_y=None,
                retain_X=None, retain_y=None,
                forget_X=None, forget_y=None,
                transform_train=None, transform_retain=None,
                transform_forget=None):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        temperature = self.config.get("bt_temperature", 1.0)
        use_retain = self.config.get("bt_use_retain", True)

        if use_retain:
            # ── Mode WITH retain: interleave forget + retain ──────────
            unlearn_ds = _UnLearningData(
                forget_X, forget_y, retain_X, retain_y,
                transform_forget=transform_forget,
                transform_retain=transform_retain,
            )
        else:
            # ── Mode WITHOUT retain: only forget data ─────────────────
            # All samples are labelled as 1 (forget) → incompetent teacher only
            unlearn_ds = _ForgetOnlyData(
                forget_X, forget_y, transform_forget=transform_forget,
            )

        unlearn_loader = DataLoader(unlearn_ds,
                                    batch_size=self.config["train_batch_size"],
                                    shuffle=True, num_workers=4)

        # Validation on forget split
        forget_train_x, forget_val_x, forget_train_y, forget_val_y = \
            train_test_split(forget_X, forget_y, test_size=0.2,
                             stratify=forget_y, random_state=42)
        val_loader = DataLoader(
            ClassificationDataset(forget_val_x, forget_val_y,
                                  transforms=transform_forget),
            batch_size=self.config["val_batch_size"],
            shuffle=False, num_workers=4)

        # Move models
        self.model.to(device)
        self.competent_teacher.to(device)
        self.incompetent_teacher.to(device)

        # Save hyper-parameters
        os.makedirs(self.config["hyperparameters_path"], exist_ok=True)
        hp = {"bt_temperature": temperature,
              "bt_use_retain": use_retain,
              "max_epoch": self.config["max_epoch"]}
        with open(os.path.join(self.config["hyperparameters_path"],
                               "hyperparameters.yml"), "w") as f:
            yaml.dump(hp, f, default_flow_style=False)

        n_step = 0
        mode_str = "with retain" if use_retain else "forget-only (no retain)"
        print(f"Starting BadTeacher unlearning ({mode_str})", flush=True)
        
        # Early stopping state
        early_stop_state = {
            'succed_saved': False,
            'best_forget_acc': float('inf'),
            'consecutive_low_count': 0,
        }
        
        for epoch in range(self.config["max_epoch"]):
            self.model.train()
            for images, is_forget in tqdm(unlearn_loader,
                                          total=len(unlearn_loader)):
                n_step += 1
                images = images.to(device)
                is_forget = is_forget.to(device)

                self.optimizer.zero_grad()
                student_logits = self.model(images)
                with torch.no_grad():
                    competent_logits = self.competent_teacher(images)
                    incompetent_logits = self.incompetent_teacher(images)

                loss = self._unlearner_loss(student_logits, is_forget,
                                           competent_logits,
                                           incompetent_logits,
                                           temperature)
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
