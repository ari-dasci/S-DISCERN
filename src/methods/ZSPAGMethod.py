"""
ZS-PAG (Zero-Shot Proxy Adversarial data Generation) unlearning method.

Paper: "Zero-shot machine unlearning with proxy adversarial data generation"
https://arxiv.org/abs/2507.21738

No retain set is required — only the forget set is accessed.
The method proceeds in three sequential stages:

  Stage 1 – Proxy Adversarial Data Generation
    For each forget sample x_i, targeted PGD pushes it across the nearest
    class boundary into the region of the remaining class with the
    second-highest model logit.  The resulting set D_adv proxies the true
    remaining-class distribution without accessing the retain set.

  Stage 2 – Influence-Based Pseudo-label Optimisation
    Soft labels y_pl are found for each proxy sample to minimise the
    influence that fine-tuning on (x_adv, y_pl) would have on the remaining-
    class loss.  Two modes are supported (zspag_pl_mode):
      'simple'    – model predictions on x_adv with forget-class zeroed out.
      'influence' – gradient descent on the full influence objective using a
                    diagonal-Fisher Hessian approximation and finite-difference
                    Jacobian-vector products.

  Stage 3 – Orthogonal Projection Unlearning
    For each linear layer l, the feature subspace S_l spanned by D_adv
    activations is computed via SVD.  All gradient updates are projected to
    be orthogonal to S_l  (P_l = I − S_l S_l^T), guaranteeing that remaining-
    class representations remain undistorted while the forget-class signal is
    erased.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils.dataset_loader import ClassificationDataset
from .baseMethod import BaseMethod
from .utils import should_save_checkpoint


class ZSPAGMethod(BaseMethod):
    """ZS-PAG: zero-shot unlearning via proxy adversarial data generation."""

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)
        # Populated during setup()
        self.adv_images = None        # Tensor (N, C, H, W) – proxy adversarial samples
        self.pseudo_labels = None     # Tensor (N, num_classes) – soft labels for D_adv
        # Maps linear-layer name → S_l tensor of shape (in_features, k)
        # Used for projected-gradient update: g_proj = g - (g @ S) @ S.T
        self.subspaces = {}
        self._forget_classes = []
        self.setup_done = False

    # ================================================================== #
    # Stage 1 – Proxy Adversarial Data Generation                        #
    # ================================================================== #

    def _target_label(self, logits, forget_classes):
        """Return per-sample adversarial target: highest-confidence non-forget class.

        The paper targets the class with the second-highest overall logit, which
        is the highest logit among remaining (non-forget) classes.
        """
        masked = logits.clone()
        for c in forget_classes:
            masked[:, c] = -float('inf')
        return masked.argmax(dim=1)

    def _pgd_attack(self, images, target_labels):
        """Run targeted PGD attack: minimise CE(f(x+δ), y_target) s.t. ||δ||_∞ ≤ ε.

        ZS-PAG Stage 1 – push each forget sample into the nearest remaining-class
        region as defined by the current model's decision boundary.
        """
        eps     = self.config.get('zspag_pgd_eps', 0.03)
        alpha   = self.config.get('zspag_pgd_alpha', eps / 10.0)
        n_steps = self.config.get('zspag_pgd_steps', 20)

        x_adv = images.clone().detach()
        # Random initialisation within the ε-ball (standard PGD)
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)

        for _ in range(n_steps):
            x_adv = x_adv.detach().requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), target_labels)
            loss.backward()
            with torch.no_grad():
                # Minimise CE towards target → step in -sign(grad) direction
                x_adv = x_adv - alpha * x_adv.grad.sign()
                delta = (x_adv - images).clamp(-eps, eps)
                x_adv = (images + delta).clamp(0.0, 1.0)

        return x_adv.detach()

    def _generate_adversarial_proxies(self, forget_X, forget_y, transform_forget):
        """Stage 1: generate D_adv from D_forget via targeted PGD.

        For each forget sample, solves min_δ L(x+δ, y_target; θ) s.t. ||δ||_∞ ≤ ε
        where y_target is the highest-confidence remaining class (Eq. 1–2 in paper).
        The resulting adversarial samples reside in remaining-class regions and
        serve as a zero-shot proxy for the retain set.
        """
        device = next(self.model.parameters()).device
        n_adv   = self.config.get('zspag_n_adv', 100)  # max proxy samples to generate

        loader = DataLoader(
            ClassificationDataset(forget_X, forget_y, transforms=transform_forget),
            batch_size=self.config['train_batch_size'],
            shuffle=False, num_workers=4,
        )

        all_adv, all_targets = [], []
        self.model.eval()
        collected = 0

        for images, _ in tqdm(loader, desc="ZS-PAG Stage 1: adversarial proxy generation"):
            if collected >= n_adv:
                break
            images = images.to(device)
            with torch.no_grad():
                logits = self.model(images)
            targets = self._target_label(logits, self._forget_classes).to(device)
            x_adv = self._pgd_attack(images, targets)
            all_adv.append(x_adv.cpu())
            all_targets.append(targets.cpu())
            collected += images.size(0)

        self.adv_images = torch.cat(all_adv, dim=0)[:n_adv]
        # Store hard labels (used as fallback and for subspace computation)
        self._adv_hard_labels = torch.cat(all_targets, dim=0)[:n_adv]
        print(f"  Generated {len(self.adv_images)} adversarial proxy samples.", flush=True)

    # ================================================================== #
    # Stage 2 – Influence-Based Pseudo-label Optimisation                #
    # ================================================================== #

    def _pseudo_labels_simple(self):
        """Simple pseudo-labels: model predictions on x_adv with forget-class zeroed.

        This mode captures the key idea of the pseudo-label optimisation — assign
        probability mass only to remaining classes — without the expensive Hessian
        computation.  Serves as a fast baseline for Stage 2.
        """
        device = next(self.model.parameters()).device
        loader = DataLoader(
            TensorDataset(self.adv_images),
            batch_size=self.config['train_batch_size'],
            shuffle=False, num_workers=0,
        )
        self.model.eval()
        parts = []
        with torch.no_grad():
            for (imgs,) in loader:
                logits = self.model(imgs.to(device))
                soft = F.softmax(logits, dim=1)
                for c in self._forget_classes:
                    soft[:, c] = 0.0
                # Renormalise after zeroing forget class probability
                soft = soft / soft.sum(dim=1, keepdim=True).clamp(min=1e-8)
                parts.append(soft.cpu())
        self.pseudo_labels = torch.cat(parts, dim=0)

    def _pseudo_labels_influence(self):
        """Full influence-based pseudo-label optimisation (Definition 1 / Eq. 10).

        Minimises: (1/N) ∇_θ L(D_adv)^T H^{-1} ∇_θ ℓ(x_i, y_pl) + λ·KL(y_pl || y_f)
        where H is approximated by the diagonal empirical Fisher over D_adv, and
        the Jacobian-vector product J_θ(x_i) h is computed via finite differences.

        The KL divergence from y_f (smoothed forget-class label) prevents the
        pseudo-labels from collapsing back to the original forget-class prediction.
        """
        device  = next(self.model.parameters()).device
        n_steps = self.config.get('zspag_pl_steps', 10)
        lr_pl   = self.config.get('zspag_pl_lr', 0.01)
        lam     = self.config.get('zspag_pl_lambda', 1.0)
        damping = self.config.get('zspag_fisher_damping', 1e-3)
        jvp_eps = self.config.get('zspag_jvp_eps', 1e-3)
        num_cls = self.config['num_classes']

        adv_loader = DataLoader(
            TensorDataset(self.adv_images, self._adv_hard_labels),
            batch_size=self.config['train_batch_size'],
            shuffle=False, num_workers=0,
        )

        self.model.eval()

        # ── (a) Compute average gradient g = ∇_θ L(D_adv) ────────────
        # Used as the numerator in the influence direction h = H^{-1} g
        param_list = [p for p in self.model.parameters() if p.requires_grad]
        g_avg = [torch.zeros_like(p) for p in param_list]
        n_total = 0
        for imgs, tgts in adv_loader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            self.model.zero_grad()
            F.cross_entropy(self.model(imgs), tgts).backward()
            for i, p in enumerate(param_list):
                if p.grad is not None:
                    g_avg[i] += p.grad.data.clone() * imgs.size(0)
            n_total += imgs.size(0)
        for i in range(len(g_avg)):
            g_avg[i] /= max(n_total, 1)

        # ── (b) Diagonal Fisher F_ii ≈ E[(∂ℓ/∂θ_i)^2] ────────────────
        # Cap at 64 samples to keep setup cost manageable
        diag_F = [torch.zeros_like(p) for p in param_list]
        n_fish = 0
        for imgs, tgts in adv_loader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            for img, tgt in zip(imgs, tgts):
                self.model.zero_grad()
                F.cross_entropy(self.model(img.unsqueeze(0)),
                                tgt.unsqueeze(0)).backward()
                for i, p in enumerate(param_list):
                    if p.grad is not None:
                        diag_F[i] += p.grad.data.pow(2)
                n_fish += 1
                if n_fish >= 64:
                    break
            if n_fish >= 64:
                break
        for i in range(len(diag_F)):
            diag_F[i] = diag_F[i] / max(n_fish, 1) + damping

        # ── (c) Influence direction h = H^{-1} g ≈ g / F_diag ────────
        h = [g_avg[i] / diag_F[i] for i in range(len(g_avg))]

        # ── (d) Gradient-descent on y_pl per batch ────────────────────
        # ∂Δ/∂y_pl = -v  where v = J_θ(x_batch) h  (JVP via finite diff)
        # KL(y_pl || y_smooth_forget) gradient: log(y_pl_k) - log(y_f_k) + 1
        all_pl = []
        for imgs, _ in adv_loader:
            imgs = imgs.to(device)
            bs = imgs.size(0)

            # Initialise y_pl as model prediction with forget class zeroed
            with torch.no_grad():
                soft_init = F.softmax(self.model(imgs), dim=1)
                for c in self._forget_classes:
                    soft_init[:, c] = 0.0
                soft_init = (soft_init /
                             soft_init.sum(dim=1, keepdim=True).clamp(min=1e-8))

            y_pl = soft_init.clone()  # (bs, num_cls) – will be updated in-place

            # Smooth forget-label reference for KL (prevent collapse to forget)
            y_forget_smooth = torch.full((bs, num_cls),
                                         1e-8 / max(num_cls - 1, 1),
                                         device=device)
            for c in self._forget_classes:
                y_forget_smooth[:, c] = 1.0 / len(self._forget_classes)
            y_forget_smooth = (y_forget_smooth /
                               y_forget_smooth.sum(dim=1, keepdim=True))

            for _ in range(n_steps):
                # JVP via forward finite differences: v = (logits(θ+εh) - logits(θ)) / ε
                with torch.no_grad():
                    logits_0 = self.model(imgs)
                    for p, hi in zip(param_list, h):
                        p.data.add_(jvp_eps * hi)
                    logits_eps = self.model(imgs)
                    for p, hi in zip(param_list, h):
                        p.data.sub_(jvp_eps * hi)
                v = (logits_eps - logits_0) / jvp_eps  # (bs, num_cls)

                # Influence gradient: -(1/N) v  (minimise influence → step in +v)
                grad_infl = -v / max(n_total, 1)  # (bs, num_cls)

                # KL(y_pl || y_forget_smooth) gradient w.r.t. y_pl
                # ∂KL/∂y_pl_k = log(y_pl_k / y_f_k) + 1
                kl_grad = (torch.log(y_pl.clamp(min=1e-8))
                           - torch.log(y_forget_smooth.clamp(min=1e-8))
                           + 1.0)

                # Combined gradient step
                y_pl = y_pl - lr_pl * (grad_infl + lam * kl_grad)
                # Project onto probability simplex via softmax renormalisation
                y_pl = F.softmax(y_pl, dim=1)
                # Enforce zero probability on forget classes
                for c in self._forget_classes:
                    y_pl[:, c] = 0.0
                y_pl = (y_pl / y_pl.sum(dim=1, keepdim=True).clamp(min=1e-8))

            all_pl.append(y_pl.cpu())

        self.pseudo_labels = torch.cat(all_pl, dim=0)

    def _compute_pseudo_labels(self):
        """Stage 2 dispatcher: route to simple or influence pseudo-label computation."""
        mode = self.config.get('zspag_pl_mode', 'simple')
        print(f"  ZS-PAG Stage 2: pseudo-label mode = '{mode}'", flush=True)
        if mode == 'influence':
            self._pseudo_labels_influence()
        else:
            self._pseudo_labels_simple()
        print(f"  Pseudo-labels ready for {len(self.pseudo_labels)} samples.", flush=True)

    # ================================================================== #
    # Stage 3a – Subspace Identification                                 #
    # ================================================================== #

    def _compute_subspaces(self):
        """Stage 3a: compute per-layer orthogonal subspaces S_l from D_adv activations.

        For each linear layer l the method (Eq. 3–4 in paper):
          1. Collects input activations A_l ∈ R^{N × in} from D_adv.
          2. SVD(A_l) → V_l whose columns span the remaining-class input subspace.
          3. Selects top-k singular vectors capturing sv_threshold of variance.
          4. S_l ∈ R^{in × k} is stored for projected-gradient updates in unlearn().

        The projection is applied as: g_proj = g - (g @ S_l) @ S_l.T
        which zeros out gradient components that would change remaining-class
        representations.
        """
        device      = next(self.model.parameters()).device
        sv_thresh   = self.config.get('zspag_sv_threshold', 0.99)

        activations = {}  # layer_name → list of input-activation tensors

        # Register forward hooks on all Linear modules
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                def _hook(mod, inp, out, n=name):
                    # inp[0]: (batch, in_features)
                    activations.setdefault(n, []).append(inp[0].detach().cpu())
                hooks.append(module.register_forward_hook(_hook))

        self.model.eval()
        adv_loader = DataLoader(
            TensorDataset(self.adv_images),
            batch_size=self.config['train_batch_size'],
            shuffle=False, num_workers=0,
        )
        with torch.no_grad():
            for (imgs,) in adv_loader:
                self.model(imgs.to(device))

        for h in hooks:
            h.remove()

        # SVD per layer → orthonormal basis S_l
        for layer_name, acts in activations.items():
            A = torch.cat(acts, dim=0)          # (N, in_features)
            A = A - A.mean(dim=0, keepdim=True)  # centre activations
            if A.shape[0] < 2:
                continue
            try:
                _, S_vals, Vt = torch.linalg.svd(A, full_matrices=False)
            except Exception:
                continue
            # Select fewest singular vectors capturing ≥ sv_threshold variance
            energy = (S_vals ** 2).cumsum(0) / (S_vals ** 2).sum().clamp(min=1e-12)
            k = int((energy < sv_thresh).sum().item()) + 1
            k = min(k, Vt.shape[0], A.shape[1])
            # S_l: columns are the k basis vectors in input space
            S_l = Vt[:k].T  # (in_features, k) – orthonormal columns
            self.subspaces[layer_name] = S_l   # stored on CPU, moved to device in unlearn

        print(f"  Subspaces computed for {len(self.subspaces)} linear layers.", flush=True)

    # ================================================================== #
    # Public API                                                          #
    # ================================================================== #

    def setup(self,
              train_X=None, train_y=None,
              retain_X=None, retain_y=None,
              forget_X=None, forget_y=None,
              transform_train=None, transform_retain=None,
              transform_forget=None):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self._forget_classes = sorted(set(int(c) for c in forget_y))

        # Stage 1 – generate adversarial proxies from D_forget
        self._generate_adversarial_proxies(forget_X, forget_y, transform_forget)

        # Stage 2 – compute soft pseudo-labels for the proxies
        self._compute_pseudo_labels()

        # Stage 3a – compute per-layer subspaces for orthogonal projection
        self._compute_subspaces()

        # Optional layer freezing (same interface as other methods)
        freeze_mode = self.config.get('freeze', 'nofreeze')
        if freeze_mode == 'freeze':
            self.model.freeze()
        elif freeze_mode == 'embs':
            self.model.emb_freeze()

        self.setup_done = True
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
        torch.save(checkpoint, os.path.join(self.config["checkpoint_path"], filename))
        print(f"✅ Checkpoint guardado en {self.config['checkpoint_path']}/{filename}")

    def unlearn(self,
                train_X=None, train_y=None,
                retain_X=None, retain_y=None,
                forget_X=None, forget_y=None,
                transform_train=None, transform_retain=None,
                transform_forget=None):
        """Stage 3b: fine-tune on D_adv with projected gradients.

        For each training step the gradient update at each linear layer l is
        projected onto the subspace orthogonal to S_l:
            g_proj = g - (g @ S_l) @ S_l^T
        This prevents remain-class representations from drifting (Eq. 5 / Thm. 1).
        """
        if not self.setup_done:
            raise ValueError("setup() must be called before unlearn().")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        # Move subspaces to device for efficient projection
        subspaces_dev = {name: S.to(device) for name, S in self.subspaces.items()}

        # Map parameter names for quick lookup during gradient projection
        name_to_param = dict(self.model.named_parameters())

        adv_loader = DataLoader(
            TensorDataset(self.adv_images, self.pseudo_labels),
            batch_size=self.config['train_batch_size'],
            shuffle=True, num_workers=0,
        )

        # Validation: forget set (20 % holdout for monitoring)
        forget_train_x, forget_val_x, forget_train_y, forget_val_y = \
            train_test_split(forget_X, forget_y, test_size=0.2,
                             stratify=forget_y, random_state=42)
        val_loader = DataLoader(
            ClassificationDataset(forget_val_x, forget_val_y,
                                  transforms=transform_forget),
            batch_size=self.config['val_batch_size'],
            shuffle=False, num_workers=4,
        )

        # Save hyperparameters
        os.makedirs(self.config['hyperparameters_path'], exist_ok=True)
        hp = {
            'zspag_pgd_eps':       self.config.get('zspag_pgd_eps', 0.03),
            'zspag_pgd_alpha':     self.config.get('zspag_pgd_alpha', 0.003),
            'zspag_pgd_steps':     self.config.get('zspag_pgd_steps', 20),
            'zspag_n_adv':         self.config.get('zspag_n_adv', 100),
            'zspag_pl_mode':       self.config.get('zspag_pl_mode', 'simple'),
            'zspag_pl_steps':      self.config.get('zspag_pl_steps', 10),
            'zspag_pl_lr':         self.config.get('zspag_pl_lr', 0.01),
            'zspag_pl_lambda':     self.config.get('zspag_pl_lambda', 1.0),
            'zspag_sv_threshold':  self.config.get('zspag_sv_threshold', 0.99),
            'max_epoch':           self.config['max_epoch'],
        }
        with open(os.path.join(self.config['hyperparameters_path'],
                               'hyperparameters.yml'), 'w') as f:
            yaml.dump(hp, f, default_flow_style=False)

        n_step = 0
        print("Starting ZS-PAG unlearning (Stage 3b)", flush=True)

        early_stop_state = {
            'succed_saved': False,
            'best_forget_acc': float('inf'),
            'consecutive_low_count': 0,
        }

        for epoch in range(self.config['max_epoch']):
            self.model.train()
            for adv_imgs, soft_tgts in tqdm(adv_loader, desc=f"E{epoch} ZS-PAG"):
                n_step += 1
                adv_imgs  = adv_imgs.to(device)
                soft_tgts = soft_tgts.to(device)

                self.optimizer.zero_grad()
                logits = self.model(adv_imgs)

                # CE with soft pseudo-labels: − Σ_k y_pl_k · log(softmax_k)
                loss = -(soft_tgts * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                loss.backward()

                # Projected-gradient update (Eq. 5 in paper):
                # For each layer l: g_proj = g − (g @ S_l) @ S_l^T
                # This zeroes gradient components that lie within the remaining-
                # class input subspace, preserving those representations.
                for layer_name, S_l in subspaces_dev.items():
                    weight_key = layer_name + '.weight'
                    if weight_key in name_to_param:
                        p = name_to_param[weight_key]
                        if p.grad is not None:
                            # g: (out, in), S_l: (in, k)
                            g = p.grad.data
                            p.grad.data = g - (g @ S_l) @ S_l.T

                self.optimizer.step()

            # Validation on forget split
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, tgt in val_loader:
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    pred = self.model(imgs).argmax(dim=1)
                    correct += (pred == tgt).sum().item()
                    total   += tgt.size(0)
            forget_acc = correct / max(total, 1)
            print(f"Epoch {epoch}  forget_acc={forget_acc:.4f}", flush=True)

            checkpoints_to_save, stop = should_save_checkpoint(
                epoch, forget_acc, self.config['num_classes'],
                self.config, early_stop_state,
            )
            for ckpt_name, reason in checkpoints_to_save:
                print(f"Saving checkpoint at epoch {epoch}: {reason}", flush=True)
                self.save_results(epoch, n_step, filename=ckpt_name)
            if stop:
                print(f"🛑 Early stopping at epoch {epoch}", flush=True)
                break

        return self.model
