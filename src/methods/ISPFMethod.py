"""
ISPF (Inhibited Synthetic PostFilter) unlearning method.

Based on: "Toward Efficient Data-Free Unlearning" (Lin et al., AAAI 2025)
https://github.com/ChildEden/ISPF

ISPF is a **data-free** unlearning method built on top of Data-Free Knowledge
Distillation (DFKD).  The key ideas:
  1. **Inhibited Synthetic (IS)**: When training the generator, the adversarial
     KL-divergence loss is sign-flipped for the forget-class logits, which
     *inhibits* the generator from producing forget-related information.
  2. **PostFilter (PF)**: During the KD step, teacher logits for the forget
     classes are redistributed to the retain classes so the student never
     learns forget knowledge.

Simplified integration: we embed a lightweight generator inside this method
and perform alternating (1) generator update and (2) student distillation
steps.  A separate generator architecture module is *not* required.
"""

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ClassificationModel import ClassificationModel
from src.utils.dataset_loader import ClassificationDataset
from .baseMethod import BaseMethod
from .utils import should_save_checkpoint


# ── Lightweight convolutional generator ──────────────────────────────
class _Generator(nn.Module):
    """Simple DCGAN-style generator: z → (C, H, W).

    Works for 32×32 images (CIFAR-10/100).  For larger resolutions the
    architecture should be scaled accordingly.
    """

    def __init__(self, nz=256, ngf=64, nc=3, img_size=32):
        super().__init__()
        self.nz = nz
        self.img_size = img_size

        if img_size == 32:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf), nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        else:
            # Fallback for arbitrary sizes – use adaptive pipeline
            self.main = nn.Sequential(
                nn.Linear(nz, ngf * 4 * 4 * 4),
                nn.ReLU(True),
            )
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf), nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
            self._use_linear = True
        self._use_linear = getattr(self, '_use_linear', False)

    def forward(self, z):
        if self._use_linear:
            x = self.main(z)
            x = x.view(z.size(0), -1, 4, 4)
            x = self.deconv(x)
            return F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        else:
            return self.main(z.view(z.size(0), self.nz, 1, 1))


class ISPFMethod(BaseMethod):

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)
        self.teacher = None
        self.student = None
        self.generator = None

    # ------------------------------------------------------------------ #
    def setup(self, train_X=None, train_y=None,
              retain_X=None, retain_y=None,
              forget_X=None, forget_y=None,
              transform_train=None, transform_retain=None,
              transform_forget=None):

        device = self.model.device

        # Teacher = frozen copy of the original model
        self.teacher = deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Student = new randomly-initialised model with same architecture
        self.student = ClassificationModel(
            self.model._base_model,
            self.model._num_classes,
            self.model._hyperparameters,
            classes_weights=self.model.classes_weights,
            extra_layers=self.model._extra_layers
        ).to(device)

        # Apply freeze if configured (on the student that will be trained)
        freeze_mode = self.config.get('freeze', 'nofreeze')
        if freeze_mode == 'freeze':
            self.student.freeze()
        elif freeze_mode == 'embs':
            self.student.emb_freeze()

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
        num_classes = self.model._num_classes

        nz = cfg.get('ispf_nz', 256)
        ngf = cfg.get('ispf_ngf', 64)
        g_steps = cfg.get('ispf_g_steps', 5)
        kd_steps = cfg.get('ispf_kd_steps', 50)
        lr_g = cfg.get('ispf_lr_g', 1e-3)
        lr_s = cfg.get('ispf_lr_s', 0.05)
        synthesis_bs = cfg.get('ispf_synthesis_bs', 256)
        epochs = cfg.get('max_epoch', 200)
        kd_T = cfg.get('ispf_kd_T', 1.0)
        pf_threshold = cfg.get('ispf_pf_threshold', 0.01)

        # Weights for generator loss components
        w_adv = cfg.get('ispf_adv', 1.0)
        w_oh = cfg.get('ispf_oh', 1.0)
        w_bn = cfg.get('ispf_bn', 1.0)

        # Determine image shape
        nc = 3
        img_size = 32
        if cfg.get('dataset', 'CIFAR10') in ('CIFAR10', 'CIFAR100'):
            img_size = 32
        elif cfg.get('dataset') in ('DTD',):
            img_size = 224

        # ViT requires 224×224 input regardless of the native dataset resolution
        is_vit = 'vit' in self.model._base_model.lower()
        if is_vit:
            img_size = 224
            # ViT + 224×224 images = huge activations; reduce batch to fit in GPU memory
            if synthesis_bs > 64:
                print(f"[ISPF] Reducing synthesis_bs {synthesis_bs} → 64 for ViT (memory savings)")
                synthesis_bs = 64

        # Use mixed precision to halve activation memory for large models
        use_amp = is_vit

        # Free the original model reference — teacher & student are all we need
        self.model.to('cpu')
        torch.cuda.empty_cache()

        # Initialise generator
        self.generator = _Generator(nz=nz, ngf=ngf, nc=nc, img_size=img_size).to(device)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))

        # Student optimizer
        opt_s = torch.optim.SGD(self.student.parameters(), lr=lr_s,
                                weight_decay=1e-4, momentum=0.9)
        sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=epochs, eta_min=2e-4)

        # Criterion for KD
        kl_criterion = _KLDiv(T=kd_T)

        # BN hooks on teacher for DeepInversion regularisation
        bn_hooks = []
        bn_losses = []
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_hooks.append(_BNHook(m, bn_losses))

        unlearn_set = set(forbidden_labels)

        # Forget validation loader for early stopping
        forget_train_x, forget_val_x, forget_train_y, forget_val_y = \
            train_test_split(forget_X, forget_y, test_size=0.2,
                             stratify=forget_y, random_state=42)
        val_loader_forget = DataLoader(
            ClassificationDataset(forget_val_x, forget_val_y,
                                  transforms=transform_forget
                                  if transform_forget is not None
                                  else transform_train),
            batch_size=cfg.get('val_batch_size', 64),
            shuffle=False, num_workers=4)

        # Early stopping state
        early_stop_state = {
            'succed_saved': False,
            'best_forget_acc': float('inf'),
            'consecutive_low_count': 0,
        }

        # ──────────────────────────────────────────────────────────────
        # Training loop
        # ──────────────────────────────────────────────────────────────
        for epoch in tqdm(range(epochs), desc="[ISPF] Epochs"):

            # ---- 1. Generator step(s) ----
            self.student.eval()
            self.generator.train()
            for _ in range(g_steps):
                opt_g.zero_grad()
                z = torch.randn(synthesis_bs, nz, device=device)
                fake = self.generator(z)

                # Clear BN losses *before* the single teacher forward pass
                # so hooks fire and collect stats in one pass (not two).
                bn_losses.clear()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    t_out = self.teacher(fake)
                    s_out = self.student(fake)

                # One-hot loss
                loss_oh = F.cross_entropy(t_out, t_out.max(1)[1])

                # BN regularisation (already collected from the single pass above)
                loss_bn = sum(bn_losses) if bn_losses else torch.tensor(0., device=device)

                # Adversarial loss: Inhibited Synthetic (IS)
                # Standard: - KL(student || teacher)
                # IS: flip sign on forget-class KL columns
                kld = F.kl_div(F.log_softmax(s_out, dim=1),
                               F.softmax(t_out, dim=1),
                               reduction='none')
                kld_is = -kld.clone()
                for c in unlearn_set:
                    kld_is[:, c] = kld[:, c]  # negate back → positive for forget
                loss_adv = kld_is.sum() / s_out.size(0)

                loss_g = w_bn * loss_bn + w_oh * loss_oh + w_adv * loss_adv
                loss_g.backward()
                opt_g.step()

            # ---- 2. KD step(s) ----
            torch.cuda.empty_cache()
            self.student.train()
            self.generator.eval()
            for _ in range(kd_steps):
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                    z = torch.randn(synthesis_bs, nz, device=device)
                    fake = self.generator(z).detach()
                    t_out = self.teacher(fake).detach()

                    # PostFilter (PF): redistribute forget logits to retain
                    t_prob = torch.softmax(t_out, dim=1)
                    retaining_idx = torch.ones_like(t_out, dtype=torch.bool)
                    forget_idx = torch.zeros_like(t_out, dtype=torch.bool)
                    for c in unlearn_set:
                        retaining_idx[:, c] = False
                        forget_idx[:, c] = True

                    # Find high-confidence forget predictions
                    need_redist = (t_prob > pf_threshold) & forget_idx
                    need_redist_logits = torch.zeros_like(t_out)
                    need_redist_logits[need_redist] = t_out[need_redist]
                    # Move mass from forget to retain
                    min_logits = t_out.min(dim=1, keepdim=True)[0]
                    min_matrix = min_logits.expand_as(t_out)
                    min_matrix_masked = torch.zeros_like(t_out)
                    min_matrix_masked[need_redist] = min_matrix[need_redist]
                    delta = (need_redist_logits - min_matrix_masked).sum(dim=1, keepdim=True)
                    n_retain_classes = num_classes - len(unlearn_set)
                    redist = delta / max(n_retain_classes, 1)
                    t_out[retaining_idx] += redist.expand_as(t_out)[retaining_idx]
                    t_out[need_redist] = min_matrix[need_redist]

                    # Filter: skip samples whose argmax is a forget class
                    preds = t_out.max(1)[1]
                    mask = torch.ones(t_out.size(0), dtype=torch.bool, device=device)
                    for c in unlearn_set:
                        mask &= (preds != c)

                    if mask.sum() == 0:
                        continue
                    fake = fake[mask]
                    t_out = t_out[mask]

                with torch.cuda.amp.autocast(enabled=use_amp):
                    s_out = self.student(fake)
                    loss_s = kl_criterion(s_out, t_out)
                opt_s.zero_grad()
                loss_s.backward()
                opt_s.step()

            sched_s.step()

            # Forget validation for early stopping
            self.student.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, tgt in val_loader_forget:
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    out = self.student(imgs)
                    correct += (out.argmax(1) == tgt).sum().item()
                    total += tgt.size(0)
            forget_acc = correct / max(total, 1)

            if (epoch + 1) % 50 == 0:
                print(f"  epoch {epoch+1}/{epochs}  loss_g={loss_g.item():.4f}  loss_s={loss_s.item():.4f}  forget_acc={forget_acc:.4f}")

            checkpoints_to_save, stop = should_save_checkpoint(
                epoch, forget_acc, num_classes, cfg, early_stop_state)
            if checkpoints_to_save:
                # Copy student to model temporarily for saving
                self.model.to(device)
                self.model.load_state_dict(self.student.state_dict())
                for ckpt_name, reason in checkpoints_to_save:
                    print(f"Saving checkpoint at epoch {epoch}: {reason}", flush=True)
                    self.save_results(epoch, 0, filename=ckpt_name)
                self.model.to('cpu')
                torch.cuda.empty_cache()
            if stop:
                print(f"🛑 Early stopping at epoch {epoch}", flush=True)
                break

        # Remove BN hooks
        for h in bn_hooks:
            h.remove()

        # Copy student weights back to self.model
        self.model.to(device)
        self.model.load_state_dict(self.student.state_dict())

        # Save hyperparameters
        hp_dir = self.config.get('hyperparameters_path', self.config['checkpoint_path'])
        os.makedirs(hp_dir, exist_ok=True)
        hp = {k: v for k, v in self.config.items()
              if k.startswith('ispf_') or k in ('unlearning_method',)}
        with open(os.path.join(hp_dir, "hyperparameters.yaml"), 'w') as f:
            yaml.dump(hp, f)

        # Save generator (optional, for debugging/analysis)
        if self.generator is not None:
            gen_ckpt = os.path.join(self.config['checkpoint_path'], "generator.pt")
            torch.save(self.generator.state_dict(), gen_ckpt)

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


# ── Helper modules ────────────────────────────────────────────────────
class _KLDiv(nn.Module):
    """KL-divergence distillation loss with temperature scaling."""

    def __init__(self, T=1.0):
        super().__init__()
        self.T = T

    def forward(self, student_logits, teacher_logits):
        p = F.log_softmax(student_logits / self.T, dim=1)
        q = F.softmax(teacher_logits / self.T, dim=1)
        return F.kl_div(p, q, reduction='batchmean') * (self.T ** 2)


class _BNHook:
    """Hook to collect BatchNorm feature-distribution loss (DeepInversion)."""

    def __init__(self, module, loss_list):
        self.loss_list = loss_list
        self.hook = module.register_forward_hook(self._hook_fn)
        self.module = module

    def _hook_fn(self, module, inp, out):
        mean = inp[0].mean(dim=[0, 2, 3])
        var = inp[0].var(dim=[0, 2, 3], unbiased=False)
        r_feature = torch.norm(module.running_mean - mean, 2) + \
                    torch.norm(module.running_var - var, 2)
        self.loss_list.append(r_feature)

    def remove(self):
        self.hook.remove()
