"""
DistributionAdapter: adapts per-class distribution parameters to embedding
drift introduced by the model during ICU unlearning.

For each retain class, m fixed samples are generated from the original
distribution P_i^0 at initialization. Optionally (adapt_current_forget=True),
the class being forgotten in the current ICU step is treated the same way —
classes forgotten in earlier ICU steps are always excluded and stay frozen.
After each training epoch an affine
transport map  T_t(z) = R_t z + t  is estimated via ridge regression from
the pairs {(f(x_j), g_t(x_j))} of base / current model embeddings on the
observed (retain) data. The original samples are propagated to
z̃ = R_t z + t and the same distribution family is refit by closed-form
MLE / method-of-moments (no full pipeline rerun).

Supported families: gaussian, laplace, vmf, kent, tstudent, uniform.

Usage inside DistributionalMethod.unlearn():
    adapter = DistributionAdapter(forbidden_labels, device, n_samples, ridge_alpha)
    adapter.initialize(base_model, observed_loader, dist_infos)
    for epoch in training_loop:
        # ... one epoch of training ...
        updated_dist_infos, metrics = adapter.step(current_model)
        self.dist_infos = updated_dist_infos
"""

import math
import numpy as np
import torch
from copy import deepcopy
from scipy import stats as sp_stats
from scipy.special import digamma
from scipy.optimize import brentq
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from .DistributionalMethod import (
    DistributionInfo,
    sample_from_distribution,
    _fit_kent_from_vmf,
    _vmf_log_likelihood,
    _tstudent_loglikelihood,
)


# ---------------------------------------------------------------------------
# Radial distribution refit
# Select the best of Normal / LogNormal / Gamma by log-likelihood.
# All three have closed-form MLE or method-of-moments (Choi & Wette 1969).
# ---------------------------------------------------------------------------

def _refit_radial(norms: np.ndarray) -> dict:
    """Fit and select the best radial (scalar norm) distribution."""
    pos = norms[norms > 0] if np.any(norms > 0) else norms.copy()
    if len(pos) < 3:
        pos = norms.copy()

    best_ll = -np.inf
    best: dict = None

    # ---- Normal -----------------------------------------------------------
    mu_n = float(pos.mean())
    sigma_n = max(float(pos.std(ddof=1 if len(pos) > 1 else 0)), 1e-12)
    ll_n = float(sp_stats.norm.logpdf(pos, loc=mu_n, scale=sigma_n).sum())
    if ll_n > best_ll:
        best_ll = ll_n
        best = {"type": "normal", "mean": mu_n, "std": sigma_n}

    # ---- LogNormal --------------------------------------------------------
    log_pos = np.log(np.clip(pos, 1e-12, None))
    mu_ln = float(log_pos.mean())
    sigma_ln = max(float(log_pos.std(ddof=1 if len(pos) > 1 else 0)), 1e-12)
    ll_ln = float(sp_stats.lognorm.logpdf(pos, s=sigma_ln, scale=math.exp(mu_ln)).sum())
    if ll_ln > best_ll:
        best_ll = ll_ln
        best = {"type": "lognormal", "mu": mu_ln, "sigma": sigma_ln}

    # ---- Gamma — Choi & Wette (1969) closed-form moments ------------------
    m1 = float(pos.mean())
    s = math.log(m1 + 1e-30) - float(log_pos.mean())
    if s > 1e-12:
        alpha = (3.0 - s + math.sqrt(max((s - 3.0) ** 2 + 24.0 * s, 0.0))) / (12.0 * s)
        alpha = max(alpha, 1e-6)
        scale_g = m1 / alpha
        ll_g = float(sp_stats.gamma.logpdf(pos, a=alpha, scale=scale_g).sum())
        if ll_g > best_ll:
            best = {"type": "gamma", "shape": float(alpha), "loc": 0.0, "scale": float(scale_g)}

    return best or {
        "type": "normal",
        "mean": float(norms.mean()),
        "std": max(float(norms.std()), 1e-12),
    }


# ---------------------------------------------------------------------------
# t-Student EM refit
# ---------------------------------------------------------------------------

def _refit_tstudent_em(z: np.ndarray, n_iter: int = 7) -> tuple:
    """Fit multivariate t-Student to z via EM (ν solved by brentq each step)."""
    n, d = z.shape
    eps = 1e-6

    mu = z.mean(axis=0)
    diff = z - mu
    cov = diff.T @ diff / n + eps * np.eye(d)
    nu = 4.0

    for _ in range(n_iter):
        # ---- E-step: per-sample weights ----
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov = cov + eps * np.eye(d)
            cov_inv = np.linalg.inv(cov)

        diff = z - mu
        delta = np.sum((diff @ cov_inv) * diff, axis=1)   # (N,)
        w = (nu + d) / (nu + delta)                        # (N,)

        # ---- M-step: mu, cov ----
        w_sum = float(w.sum())
        mu = (w[:, None] * z).sum(axis=0) / w_sum
        diff = z - mu
        cov = (w[:, None] * diff).T @ diff / n + eps * np.eye(d)

        # ---- M-step: nu (brentq) ----
        log_w = np.log(np.clip(w, 1e-300, None))
        mean_lw_w = float(np.mean(log_w - w))

        def _nu_eq(nu_val: float) -> float:
            return (
                math.log(nu_val / 2.0 + 1e-30)
                - float(digamma(nu_val / 2.0))
                + 1.0
                + mean_lw_w
                + float(digamma((nu_val + d) / 2.0))
                - math.log((nu_val + d) / 2.0 + 1e-30)
            )

        try:
            nu = float(brentq(_nu_eq, 0.1, 100.0))
        except (ValueError, RuntimeError):
            pass   # keep current nu

    return mu, cov, nu


# ---------------------------------------------------------------------------
# Per-family closed-form refit
# ---------------------------------------------------------------------------

def _refit_distribution(info: DistributionInfo, z_tilde: np.ndarray) -> DistributionInfo:
    """Refit the *same* distribution family to the propagated samples z_tilde."""
    n, d = z_tilde.shape
    dt = info.dist_type
    eps = 1e-6

    # ---- Gaussian --------------------------------------------------------
    if dt == "gaussian":
        mu = z_tilde.mean(axis=0)
        cov = np.cov(z_tilde, rowvar=False) if n > 1 else np.eye(d)
        cov = np.atleast_2d(cov) + eps * np.eye(d)
        try:
            ll = float(sp_stats.multivariate_normal.logpdf(z_tilde, mean=mu, cov=cov).sum())
        except Exception:
            ll = -np.inf
        return DistributionInfo("gaussian", {"mean": mu, "cov": cov}, ll,
                                info.bic, info.decided_by)

    # ---- Laplace (independent per dimension) -----------------------------
    elif dt == "laplace":
        locs = np.median(z_tilde, axis=0)
        scales = np.mean(np.abs(z_tilde - locs), axis=0)
        scales = np.clip(scales, eps, None)
        ll = float(sum(
            -n * math.log(2.0 * scales[j])
            - float(np.sum(np.abs(z_tilde[:, j] - locs[j]))) / scales[j]
            for j in range(d)
        ))
        return DistributionInfo("laplace", {"locs": locs, "scales": scales}, ll,
                                info.bic, info.decided_by)

    # ---- vMF / Kent (directional + radial) --------------------------------
    elif dt in ("vmf", "kent"):
        norms = np.linalg.norm(z_tilde, axis=1)
        norms = np.clip(norms, eps, None)
        unit_vecs = z_tilde / norms[:, None]

        # Directional: vMF MLE (Banerjee et al. 2005)
        mean_dir = unit_vecs.mean(axis=0)
        R_bar = float(np.linalg.norm(mean_dir))
        mu_dir = mean_dir / max(R_bar, eps)
        kappa = R_bar * (d - R_bar ** 2) / max(1.0 - R_bar ** 2, eps)
        kappa = max(kappa, 1e-6)

        # Radial: choose best family by LL
        norm_dist = _refit_radial(norms)

        # Compute combined LL (directional + radial)
        ll_dir = _vmf_log_likelihood(unit_vecs, mu_dir, kappa)
        nd = norm_dist
        if nd["type"] == "normal":
            ll_rad = float(sp_stats.norm.logpdf(
                norms, loc=nd["mean"], scale=nd["std"]).sum())
        elif nd["type"] == "lognormal":
            ll_rad = float(sp_stats.lognorm.logpdf(
                norms, s=nd["sigma"], scale=math.exp(nd["mu"])).sum())
        else:  # gamma
            ll_rad = float(sp_stats.gamma.logpdf(
                norms, a=nd["shape"], scale=nd["scale"]).sum())
        ll = ll_dir + ll_rad

        new_params = {"mu": mu_dir, "kappa": kappa, "norm_dist": norm_dist}

        # For Kent: re-check Kent vs vMF on the propagated samples
        if dt == "kent":
            kent_type, kent_extra = _fit_kent_from_vmf(z_tilde, mu_dir, kappa, norm_dist)
            if kent_type == "kent":
                new_params["tangent_cov"] = kent_extra["tangent_cov"]
                new_params["tangent_mean"] = kent_extra["tangent_mean"]
                return DistributionInfo("kent", new_params, ll, info.bic, info.decided_by)
        return DistributionInfo("vmf", new_params, ll, info.bic, info.decided_by)

    # ---- t-Student (EM, ~7 iterations) ------------------------------------
    elif dt == "tstudent":
        mu, cov, nu = _refit_tstudent_em(z_tilde)
        ll = float(_tstudent_loglikelihood(z_tilde, mu, cov, nu))
        return DistributionInfo("tstudent", {"mean": mu, "cov": cov, "df": nu}, ll,
                                info.bic, info.decided_by)

    # ---- Uniform (new bounding box) ---------------------------------------
    elif dt == "uniform":
        mins = z_tilde.min(axis=0)
        maxs = z_tilde.max(axis=0)
        widths = np.clip(maxs - mins, eps, None)
        ll = -n * float(np.sum(np.log(widths)))
        return DistributionInfo("uniform", {"mins": mins, "maxs": maxs}, ll,
                                info.bic, info.decided_by)

    # ---- Fallback: refit as Gaussian -------------------------------------
    else:
        mu = z_tilde.mean(axis=0)
        cov = (np.cov(z_tilde, rowvar=False) if n > 1 else np.eye(d)) + eps * np.eye(d)
        try:
            ll = float(sp_stats.multivariate_normal.logpdf(z_tilde, mean=mu, cov=cov).sum())
        except Exception:
            ll = -np.inf
        return DistributionInfo("gaussian", {"mean": mu, "cov": cov}, ll,
                                info.bic, info.decided_by)


# ---------------------------------------------------------------------------
# Analytical parameter propagation under affine map T(z) = z @ R_t.T + t_t
# ---------------------------------------------------------------------------

def _propagate_params_analytically(
    info: DistributionInfo,
    R_t: np.ndarray,
    t_t: np.ndarray,
    sigma_min: float,
    sigma_mean: float,
) -> DistributionInfo:
    """Propagate distribution parameters analytically under T(z) = z @ R_t.T + t_t.

    No sampling — closed-form update for each family.

    Parameters
    ----------
    R_t : (D, D)  linear part of the affine map (Ridge coef_)
    t_t : (D,)    translation (Ridge intercept_)
    sigma_min : minimum singular value of R_t  (for vMF kappa scaling)
    sigma_mean : mean singular value of R_t    (for radial distribution scaling)
    """
    dt = info.dist_type
    eps = 1e-6
    D = R_t.shape[0]

    # ---- Gaussian ---------------------------------------------------------
    if dt == "gaussian":
        mu = info.params["mean"]
        cov = info.params["cov"]
        mu_new = mu @ R_t.T + t_t
        cov_new = R_t @ cov @ R_t.T + eps * np.eye(D)
        return DistributionInfo("gaussian", {"mean": mu_new, "cov": cov_new},
                                info.ll, info.bic, info.decided_by)

    # ---- Laplace (independent per dimension) ------------------------------
    elif dt == "laplace":
        locs = info.params["locs"]
        scales = info.params["scales"]
        locs_new = locs @ R_t.T + t_t
        # b_k' = sum_l |R_kl| * b_l  — exact for independent Laplace under linear map
        scales_new = np.clip(np.abs(R_t) @ scales, eps, None)
        return DistributionInfo("laplace", {"locs": locs_new, "scales": scales_new},
                                info.ll, info.bic, info.decided_by)

    # ---- t-Student --------------------------------------------------------
    elif dt == "tstudent":
        mu = info.params["mean"]
        cov = info.params["cov"]
        nu = info.params["df"]
        mu_new = mu @ R_t.T + t_t
        cov_new = R_t @ cov @ R_t.T + eps * np.eye(D)
        # Degrees of freedom are invariant under affine transforms
        return DistributionInfo("tstudent", {"mean": mu_new, "cov": cov_new, "df": nu},
                                info.ll, info.bic, info.decided_by)

    # ---- vMF / Kent (directional + radial) --------------------------------
    elif dt in ("vmf", "kent"):
        mu_dir = info.params["mu"]
        kappa = info.params["kappa"]
        norm_dist = info.params.get("norm_dist")

        # Directional: apply only the linear part R_t to mu_dir (no translation —
        # translation is a Euclidean shift, not a rotation on the sphere).
        mu_new = mu_dir @ R_t.T
        mu_new = mu_new / max(float(np.linalg.norm(mu_new)), eps)

        # Concentration: sphere maps to ellipsoid under R_t; most conservative
        # direction shrinks by sigma_min.  κ' ≈ κ * σ_min(R).
        kappa_new = max(kappa * sigma_min, eps)

        # Radial distribution: scale by sigma_mean, shift mean by ||t|| (first-order)
        t_norm = float(np.linalg.norm(t_t))
        norm_dist_new = None
        if norm_dist is not None:
            nd = norm_dist
            sm = max(sigma_mean, eps)
            if nd["type"] == "normal":
                norm_dist_new = {
                    "type": "normal",
                    "mean": sm * nd["mean"] + t_norm,
                    "std": max(sm * nd["std"], eps),
                }
            elif nd["type"] == "lognormal":
                norm_dist_new = {
                    "type": "lognormal",
                    "mu": nd["mu"] + math.log(sm),
                    "sigma": nd["sigma"],
                }
            elif nd["type"] == "gamma":
                norm_dist_new = {
                    "type": "gamma",
                    "shape": nd["shape"],
                    "loc": nd.get("loc", 0.0),
                    "scale": nd["scale"] / sm,
                }

        new_params = {
            "mu": mu_new,
            "kappa": kappa_new,
            "norm_dist": norm_dist_new if norm_dist_new is not None else norm_dist,
        }
        # Always simplify to vmf (Kent tangent-covariance transform is not closed-form)
        return DistributionInfo("vmf", new_params, info.ll, info.bic, info.decided_by)

    # ---- Uniform: no clean closed-form — return unchanged -----------------
    return info


# ---------------------------------------------------------------------------
# Monitoring helper: mean log-likelihood of numpy samples under a fitted dist
# ---------------------------------------------------------------------------

def _mean_loglik_np(info: DistributionInfo, z: np.ndarray) -> float:
    """Mean per-sample log-likelihood of z under info (monitoring only)."""
    dt = info.dist_type
    p = info.params
    try:
        if dt == "gaussian":
            return float(
                np.mean(sp_stats.multivariate_normal.logpdf(z, mean=p["mean"], cov=p["cov"]))
            )
        elif dt == "laplace":
            total = sum(
                float(sp_stats.laplace.logpdf(z[:, j], loc=p["locs"][j],
                                               scale=p["scales"][j]).mean())
                for j in range(z.shape[1])
            )
            return total / z.shape[1]
        elif dt in ("vmf", "kent"):
            norms = np.linalg.norm(z, axis=1, keepdims=True)
            unit = z / np.clip(norms, 1e-12, None)
            return float(_vmf_log_likelihood(unit, p["mu"], p["kappa"]) / len(z))
        elif dt == "tstudent":
            return float(
                _tstudent_loglikelihood(z, p["mean"], p["cov"], p["df"]) / len(z)
            )
        elif dt == "uniform":
            widths = np.clip(p["maxs"] - p["mins"], 1e-12, None)
            return float(-np.sum(np.log(widths)))
    except Exception:
        pass
    return float("nan")


# ---------------------------------------------------------------------------
# DistributionAdapter
# ---------------------------------------------------------------------------

class DistributionAdapter:
    """
    Adapts per-class distribution parameters to embedding drift during ICU
    unlearning. Should only be used when embunfreeze != 'freeze'.

    Lifecycle (one ICU step):
        adapter = DistributionAdapter(forbidden_labels, device, n_samples, ridge_alpha)
        adapter.initialize(base_model, observed_loader, dist_infos)
        for epoch in training_loop:
            <train one epoch>
            updated_dist_infos, metrics = adapter.step(current_model)
            method.dist_infos = updated_dist_infos
        # After loading best checkpoint, do one final alignment:
        final_dist_infos, _ = adapter.step(checkpoint_model)
    """

    def __init__(
        self,
        forbidden_labels,
        device,
        n_samples: int = 500,
        ridge_alpha: float = 1e-3,
        pca_variance: float = None,
        transforms_param: bool = False,
        current_forget_labels=None,
        adapt_current_forget: bool = False,
    ):
        self.forbidden_labels = set(int(x) for x in forbidden_labels)
        self.device = device
        self.n_samples = n_samples
        self.ridge_alpha = ridge_alpha
        self.pca_variance = pca_variance
        self.transforms_param = transforms_param

        # Class(es) being forgotten in *this* ICU step (subset of forbidden_labels).
        # Classes forgotten in previous steps stay in forbidden_labels but not here,
        # so they remain frozen even when adapt_current_forget is True.
        self.current_forget_labels = (
            set(int(x) for x in current_forget_labels) if current_forget_labels else set()
        )
        self.adapt_current_forget = adapt_current_forget

        # Set in initialize()
        self.base_model = None                  # frozen copy of the base model
        self.forget_loader = None
        self.extra_loaders: list = []           # optional DataLoaders from other datasets
        self.dist_infos_0: dict = None          # P_i^0 — never modified
        self.original_samples: dict = {}        # class_id -> [ndarray per sub-cluster]
        self.prev_R_t: np.ndarray = None
        self.rng = np.random.default_rng(42)

    # ------------------------------------------------------------------

    def _is_adaptable(self, c) -> bool:
        """Whether class `c`'s distribution should be adapted this step.

        Retain classes are always adaptable. Classes in forbidden_labels are
        excluded, except the current-step forget class(es) when
        adapt_current_forget is enabled — classes forgotten in earlier ICU
        steps stay frozen either way.
        """
        c = int(c)
        if c not in self.forbidden_labels:
            return True
        return self.adapt_current_forget and c in self.current_forget_labels

    # ------------------------------------------------------------------

    def initialize(self, base_model, forget_loader, dist_infos: dict, extra_loaders=None):
        """
        Save base model copy and loaders; generate fixed distribution samples.

        Must be called once before the training loop. Embeddings are NOT
        pre-computed here — both base and current embeddings are computed
        together in step() from the same loader pass, guaranteeing correspondence
        regardless of whether the loader shuffles.

        Parameters
        ----------
        base_model : ClassificationModel
            Model at the start of this ICU step. A deepcopy is kept as reference.
        forget_loader : DataLoader
            DataLoader of forget images. Used in step() to estimate T_t.
        dist_infos : dict
            dist_infos from DistributionalMethod (possibly overridden from
            the previous ICU step's adapted distributions).
        extra_loaders : list[DataLoader], optional
            Additional DataLoaders from other datasets, also used for T_t estimation.
        """
        self.dist_infos_0 = deepcopy(dist_infos)
        self.base_model = deepcopy(base_model)
        self.base_model.eval()
        self.forget_loader = forget_loader
        self.extra_loaders = extra_loaders or []

        # Quick single-batch forward to get embedding dimension for fallback
        emb_dim = None
        with torch.no_grad():
            for images, _ in forget_loader:
                if isinstance(images, (list, tuple)):
                    images = torch.stack(images)
                _, embs = self.base_model(images[:1].to(self.device).float(), True)
                emb_dim = embs.shape[1]
                break

        if emb_dim is None:
            print("  [ICU adapter] Warning: forget_loader is empty — adapter disabled.")
            return

        # Generate m fixed original samples from P_i^0 for each adaptable class
        # (retain classes, plus the current-step forget class if requested)
        for c, infos in dist_infos.items():
            if not self._is_adaptable(c) or not infos:
                continue
            self.original_samples[c] = []
            for info, intra_frac in infos:
                n_sub = max(int(self.n_samples * intra_frac), 10)
                try:
                    samples_t = sample_from_distribution(
                        info, n_sub, device="cpu", rng=self.rng
                    )
                    self.original_samples[c].append(samples_t.numpy())
                except Exception as exc:
                    print(f"  [ICU adapter] Could not sample class {c}: {exc}")
                    self.original_samples[c].append(np.zeros((n_sub, emb_dim)))

        n_extra = len(self.extra_loaders)
        n_current_forget = sum(
            1 for c in self.original_samples if int(c) in self.current_forget_labels
        )
        n_retain = len(self.original_samples) - n_current_forget
        print(
            f"  [ICU adapter] Initialized: {n_retain} retain classes, "
            f"{n_current_forget} current-forget classes"
            f"{' (adapted)' if self.adapt_current_forget else ''}, "
            f"{n_extra} extra loaders.",
            flush=True,
        )

    # ------------------------------------------------------------------

    def step(self, current_model) -> tuple:
        """
        Estimate T_t, propagate original samples, refit distributions.

        Parameters
        ----------
        current_model : ClassificationModel
            Model after training the current epoch (g_t).

        Returns
        -------
        updated_dist_infos : dict
            New dist_infos ready to replace method.dist_infos.
        metrics : dict
            {'R_norm_diff': float, 'avg_loglik': float}
        """
        if self.base_model is None or not self.original_samples:
            empty = deepcopy(self.dist_infos_0) if self.dist_infos_0 else {}
            return empty, {"R_norm_diff": float("nan"), "avg_loglik": float("nan")}

        # ---- Collect paired embeddings (f(x_i), g_t(x_i)) in a single pass ----
        # Iterating each loader once guarantees correspondence regardless of shuffle.
        def _collect_pairs(loader):
            f_list, gt_list = [], []
            self.base_model.eval()
            current_model.eval()
            with torch.no_grad():
                for images, _ in loader:
                    if isinstance(images, (list, tuple)):
                        images = torch.stack(images)
                    images = images.to(self.device).float()
                    _, f_embs = self.base_model(images, True)
                    _, gt_embs = current_model(images, True)
                    f_list.append(f_embs.cpu().numpy())
                    gt_list.append(gt_embs.cpu().numpy())
            if not f_list:
                return None, None
            return np.concatenate(f_list, axis=0), np.concatenate(gt_list, axis=0)

        f_all, gt_all = _collect_pairs(self.forget_loader)
        if f_all is None:
            empty = deepcopy(self.dist_infos_0) if self.dist_infos_0 else {}
            return empty, {"R_norm_diff": float("nan"), "avg_loglik": float("nan")}

        for extra_loader in self.extra_loaders:
            f_ex, gt_ex = _collect_pairs(extra_loader)
            if f_ex is not None:
                f_all = np.vstack([f_all, f_ex])
                gt_all = np.vstack([gt_all, gt_ex])

        # ---- Estimate T_t: gt_all ≈ f_all @ R_t.T + t_t ----
        # Optional PCA preprocessing: project to the K directions with highest
        # variance in f_all (discarding uninformative/near-zero dimensions that
        # make the system ill-conditioned). The map is estimated in PCA space and
        # the transport is applied via encode → transform → decode.
        if self.pca_variance is not None:
            pca = PCA(n_components=self.pca_variance)  # float → sklearn selects K to explain pca_variance of variance
            f_pca = pca.fit_transform(f_all)   # (N, K)
            gt_pca = pca.transform(gt_all)     # (N, K) — projected into f's PCA basis

            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True, solver="cholesky")
            ridge.fit(f_pca, gt_pca)
            R_pca = ridge.coef_    # (K, K)
            t_pca = ridge.intercept_  # (K,)

            V   = pca.components_  # (K, D)
            mu  = pca.mean_        # (D,)

            # Full-space equivalents for monitoring (‖R-I‖_F, ‖ΔR‖_F)
            R_t = V.T @ R_pca @ V                          # (D, D)  approx
            t_t = mu + t_pca @ V - mu @ V.T @ R_pca.T @ V  # (D,)   approx

            # Efficient per-sample transport (avoids storing D×D)
            def _transport(z: np.ndarray) -> np.ndarray:
                z_enc = (z - mu) @ V.T        # (m, K)
                z_trf = z_enc @ R_pca.T + t_pca  # (m, K)
                return z_trf @ V + mu         # (m, D)
        else:  # no PCA — full-space ridge
            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True, solver="cholesky")
            ridge.fit(f_all, gt_all)
            R_t = ridge.coef_       # (D, D)
            t_t = ridge.intercept_  # (D,)

            def _transport(z: np.ndarray) -> np.ndarray:
                return z @ R_t.T + t_t

        # ---- Monitor ||R_t - R_{t-1}||_F,  ||R_t - I||_F,  ||t_t|| ----
        R_norm_diff = (
            float(np.linalg.norm(R_t - self.prev_R_t, ord="fro"))
            if self.prev_R_t is not None
            else float("nan")
        )
        self.prev_R_t = R_t.copy()
        R_drift_from_identity = float(np.linalg.norm(R_t - np.eye(R_t.shape[0]), ord="fro"))
        t_t_norm = float(np.linalg.norm(t_t))

        # ---- Singular values for analytical propagation (computed once per step) ----
        # Use R_pca (smaller) when PCA is active; otherwise the full R_t.
        _sigma_min = _sigma_max = _sigma_mean = _condition_number = float("nan")
        if self.transforms_param:
            _svd_mat = R_pca if self.pca_variance is not None else R_t
            _sing_vals = np.linalg.svd(_svd_mat, compute_uv=False)
            _sigma_min  = float(_sing_vals[-1])
            _sigma_max  = float(_sing_vals[0])
            _sigma_mean = float(np.mean(_sing_vals))
            _condition_number = _sigma_max / max(_sigma_min, 1e-12)

        # ---- Propagate original samples and refit each sub-cluster ----
        new_dist_infos: dict = {}
        logliks_adapted: list = []   # LL of transported samples under new dist (should be high)
        logliks_original: list = []  # LL of original samples under new dist (measures mismatch)
        # Euclidean metrics (meaningful for gaussian / laplace / tstudent / uniform)
        centroid_shifts: list = []
        std_ratios: list = []
        # Directional metrics (meaningful for vmf / kent)
        angular_shifts_deg: list = []  # angle between original and adapted mean direction (degrees)
        kappa_ratios: list = []        # kappa_adapted / kappa_orig (concentration change)
        # Transport displacement: mean ||z_tilde - z_orig|| per sample
        displacements: list = []

        for c, infos in self.dist_infos_0.items():
            if not self._is_adaptable(c) or c not in self.original_samples:
                new_dist_infos[c] = infos
                continue

            new_infos = []
            for sub_idx, (info, intra_frac) in enumerate(infos):
                z_orig  = self.original_samples[c][sub_idx]    # (m, D)
                z_tilde = _transport(z_orig)                   # (m, D)

                # Mean displacement per sample (how much the transport moves embeddings)
                displacements.append(float(np.linalg.norm(z_tilde - z_orig, axis=1).mean()))

                if self.transforms_param:
                    new_info = _propagate_params_analytically(
                        info, R_t, t_t, _sigma_min, _sigma_mean
                    )
                else:
                    new_info = _refit_distribution(info, z_tilde)
                new_infos.append((new_info, intra_frac))

                # LL of transported samples under adapted distribution
                ll_adapted = _mean_loglik_np(new_info, z_tilde)
                if np.isfinite(ll_adapted):
                    logliks_adapted.append(ll_adapted)

                # LL of original (non-transported) samples under adapted distribution
                # — measures how wrong the new dist would be without transport
                ll_original = _mean_loglik_np(new_info, z_orig)
                if np.isfinite(ll_original):
                    logliks_original.append(ll_original)

                # ---- Distribution-aware drift metrics ----
                dt = info.dist_type
                if dt in ("vmf", "kent"):
                    # For directional distributions: angular shift of mean direction
                    # and kappa ratio (concentration change).
                    mu_orig = info.params.get("mu")
                    mu_new  = new_info.params.get("mu")
                    if mu_orig is not None and mu_new is not None:
                        cos_sim = float(np.clip(np.dot(mu_orig, mu_new) /
                                                (np.linalg.norm(mu_orig) * np.linalg.norm(mu_new) + 1e-12),
                                                -1.0, 1.0))
                        angular_shifts_deg.append(math.degrees(math.acos(cos_sim)))
                    k_orig = info.params.get("kappa", 1.0)
                    k_new  = new_info.params.get("kappa", 1.0)
                    if k_orig > 1e-12:
                        kappa_ratios.append(float(k_new) / float(k_orig))
                else:
                    # For euclidean distributions: L2 centroid shift and std ratio
                    shift = float(np.linalg.norm(z_tilde.mean(axis=0) - z_orig.mean(axis=0)))
                    centroid_shifts.append(shift)
                    std_orig = float(np.sqrt(np.mean(np.var(z_orig, axis=0))))
                    std_new  = float(np.sqrt(np.mean(np.var(z_tilde, axis=0))))
                    if std_orig > 1e-12:
                        std_ratios.append(std_new / std_orig)

            new_dist_infos[c] = new_infos

        # Preserve keys that were not in original_samples (e.g. forbidden classes)
        for c in self.dist_infos_0:
            if c not in new_dist_infos:
                new_dist_infos[c] = self.dist_infos_0[c]

        def _arr_stats(lst):
            if not lst:
                return float("nan"), float("nan"), float("nan")
            a = np.array(lst)
            return float(np.mean(a)), float(np.std(a)), float(np.max(a))

        cs_mean, cs_std, cs_max = _arr_stats(centroid_shifts)
        sr_mean, sr_std, _      = _arr_stats(std_ratios)
        as_mean, as_std, as_max = _arr_stats(angular_shifts_deg)
        kr_mean, kr_std, _      = _arr_stats(kappa_ratios)

        disp_mean, disp_std, _ = _arr_stats(displacements)

        metrics = {
            "R_norm_diff":          R_norm_diff,
            "R_drift":              R_drift_from_identity,
            "t_norm":               t_t_norm,
            "sigma_min":            _sigma_min,
            "condition_number":     _condition_number,
            "mean_displacement":    disp_mean,
            "std_displacement":     disp_std,
            "avg_ll_adapted":       float(np.mean(logliks_adapted))  if logliks_adapted  else float("nan"),
            "avg_ll_original":      float(np.mean(logliks_original)) if logliks_original else float("nan"),
            # Euclidean (gaussian / laplace / tstudent / uniform)
            "mean_centroid_shift":  cs_mean,
            "std_centroid_shift":   cs_std,
            "max_centroid_shift":   cs_max,
            "mean_std_ratio":       sr_mean,
            "std_std_ratio":        sr_std,
            # Directional (vmf / kent)
            "mean_angular_shift_deg": as_mean,
            "std_angular_shift_deg":  as_std,
            "max_angular_shift_deg":  as_max,
            "mean_kappa_ratio":       kr_mean,
            "std_kappa_ratio":        kr_std,
        }
        return new_dist_infos, metrics
