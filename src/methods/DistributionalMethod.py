"""
DistributionalMethod: Adaptive distribution fitting per group (class or cluster).

Instead of assuming Gaussian noise for all groups, this method follows a
structured pipeline to find the best-fitting distribution for each group:

1. Henze-Zirkler multivariate normality test
2. Shell analysis + Rayleigh test for von Mises-Fisher
3. Uniformity test
4. BIC duel among Gaussian, t-Student, vMF, Laplace, Uniform

If clustering is enabled, groups that fail the BIC acceptance threshold are
recursively split until the cluster size falls below sqrt(N) or the embedding
dimensionality.
"""

import os
import math
import inspect
import warnings
from copy import deepcopy
import random as pyrandom

import yaml
import numpy as np
from tqdm import tqdm
from scipy import stats as sp_stats
from scipy.special import gammaln, iv as besseli, ive  # modified Bessel of 1st kind
from scipy.stats import yeojohnson, yeojohnson_normmax
from sklearn.cluster import KMeans
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.model_selection import train_test_split
from pingouin import multivariate_normality

from src.models.ClassificationModel import ExtractCLS
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal

from src.utils import ClassificationDataset
from .baseMethod import BaseMethod
from .utils import (
    _n_params_cov,
    _auto_select_n_components,
    _constrain_covariance,
    _n_params_cov_from_precision,
    should_save_checkpoint,
)

from src.models.ADINgenerativeModel import ADINGenerativeModel

# ---------------------------------------------------------------------------
# Helpers shared with NOISYMethod
# ---------------------------------------------------------------------------

class ConditionalTransform:
    """Apply transform only when the input is NOT a 1-D embedding tensor."""

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, x):
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            if self.transform is not None:
                return self.transform(x)
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x


def collate_fn_mixed(batch):
    images, labels = zip(*batch)
    labels = torch.stack(
        [torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in labels]
    )
    try:
        images = torch.stack(images)
    except RuntimeError:
        images = list(images)
    return images, labels


# ===================================================================
#  Distribution fitting utilities
# ===================================================================

# ---- Radio dispersion coefficient (Cr) --------------------------------

def _radio_dispersion(data: np.ndarray) -> float:
    """Cr = std(norms) / mean(norms). Low Cr ≈ data on a shell."""
    norms = np.linalg.norm(data, axis=1)
    mean_norm = np.mean(norms)
    if mean_norm < 1e-12:
        return float("inf")
    return float(np.std(norms) / mean_norm)


# ---- Covariance constraint helpers (imported from .utils) -------------
# _n_params_cov, _auto_select_n_components, _constrain_covariance,
# _n_params_cov_from_precision are now in methods/utils.py.


def _fit_gaussian_glasso(data: np.ndarray, eps: float = 1e-6):
    """
    Fit Gaussian with Graphical Lasso (sparse precision) using real data.
    Returns (mean, cov, log_likelihood, n_params).
    """
    n, d = data.shape
    mean = data.mean(axis=0)
    centred = data - mean
    try:
        gl = GraphicalLassoCV(cv=min(3, max(2, n // 10)), max_iter=300,
                              assume_centered=True)
        gl.fit(centred)
        cov = gl.covariance_
        precision = gl.precision_
        n_cov_params = _n_params_cov_from_precision(precision)
    except Exception:
        # Fallback to diagonal
        cov = np.diag(np.maximum(np.var(centred, axis=0), eps))
        n_cov_params = d
    try:
        ll = float(
            sp_stats.multivariate_normal.logpdf(data, mean=mean, cov=cov).sum()
        )
    except Exception:
        ll = -np.inf
    n_params = d + n_cov_params
    return mean, cov, ll, n_params


# ---- Henze-Zirkler normality test -------------------------------------

def _hz_test(data: np.ndarray, alpha: float = 0.05):
    """Return (is_normal: bool, p_value: float)."""
    n, d = data.shape
    if n <= d + 1:
        return False, 0.0
    try:
        res = multivariate_normality(data, alpha=alpha)
        if isinstance(res, (list, tuple)):
            if len(res) >= 3:
                return bool(res[2]), float(res[1])
            return float(res[1]) > alpha, float(res[1])
        return False, 0.0
    except Exception:
        return False, 0.0


# ---- von Mises-Fisher helpers -----------------------------------------

def _estimate_kappa_approx(R_bar: float, d: int) -> float:
    """Approximate MLE of concentration κ for vMF (Banerjee et al. 2005)."""
    kappa = R_bar * (d - R_bar ** 2) / (1.0 - R_bar ** 2 + 1e-12)
    return max(kappa, 1e-6)


def _vmf_log_norm_const(d: int, kappa: float) -> float:
    """Log of the normalisation constant C_d(κ) for vMF.

    Uses the exponentially-scaled Bessel function ``ive`` to avoid
    overflow when κ is large:  log I_v(κ) = log(ive(v,κ)) + κ.
    """
    half_d = d / 2.0
    v = half_d - 1.0
    # C_d(κ) = κ^{d/2-1} / ((2π)^{d/2} I_{d/2-1}(κ))
    log_num = v * math.log(kappa + 1e-30)
    # Numerically stable log of the Bessel function via ive
    ive_val = ive(v, kappa)
    if ive_val > 0:
        log_bessel = math.log(ive_val) + kappa
    else:
        # Fallback: direct computation (may be -inf for extreme values)
        log_bessel = math.log(besseli(v, kappa) + 1e-300)
    log_denom = half_d * math.log(2.0 * math.pi) + log_bessel
    return log_num - log_denom


def _vmf_log_likelihood(data_unit: np.ndarray, mu: np.ndarray, kappa: float) -> float:
    """Total log-likelihood under vMF(mu, κ)."""
    n, d = data_unit.shape
    log_C = _vmf_log_norm_const(d, kappa)
    dots = data_unit @ mu  # (n,)
    return float(n * log_C + kappa * dots.sum())


def _fit_vmf(data: np.ndarray):
    """
    MLE for von Mises-Fisher (directional component only).
    Returns (mu, kappa, log_likelihood, n_params).

    The log-likelihood is computed on the **unit sphere** (directional
    component).  The radial (norm) distribution is handled separately by
    ``_fit_norm_distribution``.
    """
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    data_unit = data / norms
    n, d = data_unit.shape
    mean_dir = data_unit.mean(axis=0)
    R_bar = np.linalg.norm(mean_dir)
    if R_bar < 1e-12:
        mu = np.zeros(d)
        mu[0] = 1.0
    else:
        mu = mean_dir / R_bar
    kappa = _estimate_kappa_approx(R_bar, d)
    ll = _vmf_log_likelihood(data_unit, mu, kappa)
    n_params = d  # mu lives on S^{d-1} so (d-1) free params + 1 for kappa = d
    return mu, kappa, ll, n_params


# ---- Kent (Fisher-Bingham) helpers ------------------------------------

def _householder_matrix(mu: np.ndarray) -> np.ndarray:
    """Householder reflection matrix H such that H @ mu = e1 and H @ e1 = mu.

    Since Householder reflections are self-inverse (H = H^T = H^{-1}),
    the same matrix rotates in both directions.
    """
    d = len(mu)
    e1 = np.zeros(d)
    e1[0] = 1.0
    u = e1 - mu
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-12:
        return np.eye(d)  # mu ≈ e1, no rotation needed
    u = u / u_norm
    return np.eye(d) - 2.0 * np.outer(u, u)


def _stereographic_proj(data_sphere: np.ndarray) -> np.ndarray:
    """Stereographic projection from south pole for points near north pole.

    Maps x in S^{d-1} to y in R^{d-1} via:
        y_i = x_i / (1 + x_0)   for i = 1, ..., d-1

    Points near the north pole (x_0 ~ 1) map to y ~ 0.
    """
    x0 = data_sphere[:, 0:1]  # (n, 1)
    denom = 1.0 + x0  # (n, 1)
    denom = np.clip(denom, 1e-12, None)
    return data_sphere[:, 1:] / denom  # (n, d-1)


def _inv_stereographic_proj(data_tangent: np.ndarray) -> np.ndarray:
    """Inverse stereographic projection: R^{d-1} -> S^{d-1}.

    Given y in R^{d-1}, returns x in S^{d-1}:
        x_0 = (1 - ||y||^2) / (1 + ||y||^2)
        x_i = 2 y_i / (1 + ||y||^2)   for i >= 1
    """
    y_sq_norm = np.sum(data_tangent ** 2, axis=1, keepdims=True)  # (n, 1)
    denom = 1.0 + y_sq_norm  # (n, 1)
    x0 = (1.0 - y_sq_norm) / denom  # (n, 1)
    x_rest = 2.0 * data_tangent / denom  # (n, p)
    return np.hstack([x0, x_rest])  # (n, d)


def _test_sphericity(eigenvalues: np.ndarray, n_samples: int,
                     p: int, alpha: float = 0.05) -> tuple:
    """Test if eigenvalues are approximately equal (sphericity test).

    Returns (is_spherical, p_value).
    - is_spherical=True  -> data consistent with vMF (isotropic)
    - is_spherical=False -> Kent is more appropriate (anisotropic)

    When n_samples <= p the sample covariance is singular and no reliable
    sphericity test exists -> assume spherical (keep vMF).
    When p < n_samples < p + 0.1*p: uses John's sphericity test.
    Otherwise: likelihood ratio test  n*p*log(AM/GM)  vs  chi2(df).
    """
    eigenvalues = np.clip(eigenvalues, 1e-30, None)  # avoid log(0)

    if n_samples <= p:
        # Not enough samples for a reliable covariance estimate -> keep vMF
        return True, 1.0

    if n_samples < p + 0.1 * p:
        # ---- John's sphericity test ------------------------------------
        trace_sq = np.sum(eigenvalues) ** 2
        trace_of_sq = np.sum(eigenvalues ** 2)
        if trace_sq < 1e-30 or p <= 1:
            return True, 1.0
        ratio = p * trace_of_sq / trace_sq - 1.0
        T = n_samples * p * (p + 1) / (2.0 * max(p - 1, 1)) * ratio
        df_john = max(p * (p + 1) // 2 - 1, 1)
        p_value = 1.0 - sp_stats.chi2.cdf(T, df=df_john)
        return p_value > alpha, float(p_value)
    else:
        # ---- Likelihood ratio test (Bartlett-corrected) ---------------
        # Raw: n*p*log(AM/GM).  Bartlett correction factor avoids
        # over-rejection in moderate dimensions (Anderson 2003).
        # df = p*(p+1)/2 - 1  (standard LRT sphericity dof).
        am = np.mean(eigenvalues)
        gm = np.exp(np.mean(np.log(eigenvalues)))
        if gm < 1e-30 or am < 1e-30:
            return False, 0.0
        correction = 1.0 - (2.0 * p * p + p + 2) / (6.0 * p * max(n_samples - 1, 1))
        correction = max(correction, 0.1)  # safety floor
        stat = (n_samples - 1) * p * np.log(am / gm) * correction
        df = max(p * (p + 1) // 2 - 1, 1)
        p_value = 1.0 - sp_stats.chi2.cdf(stat, df=df)
        return p_value > alpha, float(p_value)


def _fit_kent_from_vmf(data: np.ndarray, mu: np.ndarray, kappa: float,
                       norm_dist: dict, alpha: float = 0.05):
    """Test whether data is better described by Kent than vMF.

    Process
    -------
    1. Rotate data so mu aligns with north pole (e1) via Householder.
    2. Stereographic projection to tangent plane R^{d-1}.
    3. Compute covariance (Ledoit-Wolf if singular).
    4. Sphericity test on eigenvalues of covariance.
    5. If not spherical -> Kent; else keep vMF.

    Returns
    -------
    dist_type : str
        ``"vmf"`` or ``"kent"``
    extra_params : dict
        Empty for vMF; ``{"tangent_cov", "tangent_mean"}`` for Kent.
    """
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    data_unit = data / norms
    n, d = data_unit.shape
    p = d - 1  # tangent plane dimension

    if p < 2:
        # In 2D (circle), Kent reduces to vMF
        return "vmf", {}

    # Step 1: Householder rotation to align mu with e1
    H = _householder_matrix(mu)
    rotated = (H @ data_unit.T).T  # (n, d) — now mean direction ~ e1

    # Step 2: Stereographic projection
    projected = _stereographic_proj(rotated)  # (n, p)

    # Step 3: Covariance in tangent plane
    use_ledoit_wolf = False
    try:
        cov_tangent = np.cov(projected, rowvar=False)
        if cov_tangent.ndim < 2:
            cov_tangent = np.array([[cov_tangent]])
        eigvals = np.linalg.eigvalsh(cov_tangent)
        if np.any(eigvals <= 0) or np.any(~np.isfinite(eigvals)):
            raise np.linalg.LinAlgError("Singular covariance")
    except (np.linalg.LinAlgError, ValueError):
        use_ledoit_wolf = True

    if use_ledoit_wolf:
        try:
            lw = LedoitWolf(assume_centered=False)
            lw.fit(projected)
            cov_tangent = lw.covariance_
            eigvals = np.linalg.eigvalsh(cov_tangent)
        except Exception:
            return "vmf", {}

    # Step 4: Sphericity test
    is_spherical, p_val = _test_sphericity(eigvals, n, p, alpha=alpha)

    if is_spherical:
        print(f"  [Kent test] Sphericity NOT rejected (p={p_val:.4f}) -> keeping vMF")
        return "vmf", {}
    else:
        print(f"  [Kent test] Sphericity REJECTED (p={p_val:.4f}) -> upgrading to Kent")
        # Ensure positive-definite for future sampling
        try:
            np.linalg.cholesky(cov_tangent)
        except np.linalg.LinAlgError:
            try:
                lw = LedoitWolf(assume_centered=False)
                lw.fit(projected)
                cov_tangent = lw.covariance_
            except Exception:
                cov_tangent = cov_tangent + 1e-6 * np.eye(p)

        return "kent", {
            "tangent_cov": cov_tangent,
            "tangent_mean": projected.mean(axis=0),
        }


def _sample_kent(mu, tangent_cov, tangent_mean, n_samples, rng):
    """Sample directions from Kent (Fisher-Bingham) on S^{d-1}.

    1. Sample from N(tangent_mean, tangent_cov) in tangent plane.
    2. Inverse stereographic projection -> S^{d-1} near north pole.
    3. Householder rotation to align north pole back to mu.

    Returns unit vectors of shape (n_samples, d).
    """
    d = len(mu)
    p = d - 1

    # Step 1: Sample in tangent plane
    try:
        samples_tangent = rng.multivariate_normal(
            mean=tangent_mean, cov=tangent_cov, size=n_samples
        )
    except np.linalg.LinAlgError:
        cov_reg = tangent_cov + 1e-6 * np.eye(p)
        samples_tangent = rng.multivariate_normal(
            mean=tangent_mean, cov=cov_reg, size=n_samples
        )

    # Step 2: Inverse stereographic projection
    unit_samples = _inv_stereographic_proj(samples_tangent)  # (n, d) on S^{d-1}

    # Step 3: Householder rotation from north pole frame to mu frame
    H = _householder_matrix(mu)
    rotated = (H @ unit_samples.T).T  # (n, d)

    # Normalize for numerical safety
    norms_out = np.linalg.norm(rotated, axis=1, keepdims=True)
    norms_out = np.clip(norms_out, 1e-12, None)
    return rotated / norms_out


# ---- Rayleigh test for directional uniformity --------------------------

def _rayleigh_test(data: np.ndarray) -> float:
    """
    Rayleigh test on the unit sphere.
    Returns a p-value; small p → data is NOT uniformly distributed on the
    sphere → consistent with vMF.
    """
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    data_unit = data / norms
    n, d = data_unit.shape
    R_bar = np.linalg.norm(data_unit.mean(axis=0))
    # test statistic
    stat = n * d * R_bar ** 2
    # under H0 (uniform on sphere) stat ~ chi2(d)
    p_value = 1.0 - sp_stats.chi2.cdf(stat, df=d)
    return float(p_value)


# ---- Norm distribution fitting for vMF --------------------------------

def _fit_norm_distribution(norms_1d: np.ndarray, alpha: float = 0.05):
    """Fit a distribution to the scalar norms of the data vectors.

    Tries Normal (Shapiro-Wilk), then LogNormal, then Gamma.
    If none passes the test, defaults to LogNormal (always a reasonable
    choice for positive-valued norms).

    Returns a dict with keys:
        - "type": "normal" | "lognormal" | "gamma"
        - parameters of the fitted distribution
        - "ll": log-likelihood of the fit
        - "n_params": number of free parameters
    """
    n = len(norms_1d)

    # Shapiro-Wilk helper (max 5000 samples for scipy limitation)
    def _shapiro_p(x):
        try:
            _, p = sp_stats.shapiro(x[:5000] if len(x) > 5000 else x)
            return p
        except Exception:
            return 0.0

    # ---- Try Normal ------------------------------------------------------
    mu_n = float(np.mean(norms_1d))
    sigma_n = float(np.std(norms_1d, ddof=1))
    sigma_n = max(sigma_n, 1e-12)

    if _shapiro_p(norms_1d) >= alpha:
        ll = float(sp_stats.norm.logpdf(norms_1d, loc=mu_n, scale=sigma_n).sum())
        print(f"  [norm dist] Normal accepted (p={_shapiro_p(norms_1d):.4f})")
        return {
            "type": "normal",
            "mean": mu_n,
            "std": sigma_n,
            "ll": ll,
            "n_params": 2,
        }

    # ---- Try LogNormal ---------------------------------------------------
    pos_norms = norms_1d[norms_1d > 0]
    if len(pos_norms) >= 3:
        log_norms = np.log(pos_norms)
        mu_ln = float(np.mean(log_norms))
        sigma_ln = float(np.std(log_norms, ddof=1))
        sigma_ln = max(sigma_ln, 1e-12)

        if _shapiro_p(log_norms) >= alpha:
            ll = float(sp_stats.lognorm.logpdf(
                pos_norms, s=sigma_ln, scale=np.exp(mu_ln)
            ).sum())
            print(f"  [norm dist] LogNormal accepted (p={_shapiro_p(log_norms):.4f})")
            return {
                "type": "lognormal",
                "mu": mu_ln,
                "sigma": sigma_ln,
                "ll": ll,
                "n_params": 2,
            }

    # ---- Try Gamma -------------------------------------------------------
    if len(pos_norms) >= 3:
        try:
            a_fit, loc_fit, scale_fit = sp_stats.gamma.fit(pos_norms)
            gamma_ll = float(sp_stats.gamma.logpdf(
                pos_norms, a=a_fit, loc=loc_fit, scale=scale_fit
            ).sum())
            # Anderson-Darling test via goodness_of_fit (Monte Carlo)
            gof_result = sp_stats.goodness_of_fit(
                sp_stats.gamma, pos_norms, statistic='ad',
                n_mc_samples=500,
            )
            p_ad = gof_result.pvalue
            if p_ad >= alpha:
                print(f"  [norm dist] Gamma accepted (AD p={p_ad:.4f}, "
                      f"shape={a_fit:.4f}, loc={loc_fit:.4f}, scale={scale_fit:.4f})")
                return {
                    "type": "gamma",
                    "shape": float(a_fit),
                    "loc": float(loc_fit),
                    "scale": float(scale_fit),
                    "ll": gamma_ll,
                    "n_params": 3,
                }
        except Exception:
            pass

    # ---- Fallback: LogNormal (default) -----------------------------------
    # If no test passed, use LogNormal as a safe default for positive norms
    if len(pos_norms) < 3:
        pos_norms = norms_1d  # should not happen but guard
    log_norms = np.log(np.clip(pos_norms, 1e-12, None))
    mu_ln = float(np.mean(log_norms))
    sigma_ln = float(np.std(log_norms, ddof=1))
    sigma_ln = max(sigma_ln, 1e-12)
    ll = float(sp_stats.lognorm.logpdf(
        pos_norms, s=sigma_ln, scale=np.exp(mu_ln)
    ).sum())
    print(f"  [norm dist] Fallback to LogNormal (no test passed)")
    return {
        "type": "lognormal",
        "mu": mu_ln,
        "sigma": sigma_ln,
        "ll": ll,
        "n_params": 2,
    }


# ---- Multivariate Laplace (independent coords) MLE --------------------

def _fit_laplace(data: np.ndarray):
    """
    Fit product-of-independent-Laplace (location-scale per dimension).
    Returns (locs, scales, log_likelihood, n_params).
    """
    n, d = data.shape
    locs = np.median(data, axis=0)
    scales = np.mean(np.abs(data - locs), axis=0)
    scales = np.clip(scales, 1e-12, None)
    # log-likelihood = sum_i sum_j [ -log(2*b_j) - |x_{ij} - loc_j|/b_j ]
    ll = 0.0
    for j in range(d):
        ll += -n * math.log(2.0 * scales[j]) - np.sum(np.abs(data[:, j] - locs[j])) / scales[j]
    n_params = 2 * d  # loc + scale per dimension
    return locs, scales, ll, n_params


def _sample_laplace(locs, scales, n_samples, rng=None):
    """Sample from product-of-independent Laplace."""
    if rng is None:
        rng = np.random.default_rng()
    d = len(locs)
    samples = np.empty((n_samples, d))
    for j in range(d):
        samples[:, j] = rng.laplace(loc=locs[j], scale=scales[j], size=n_samples)
    return samples


# ---- Multivariate Gaussian MLE ----------------------------------------

def _fit_gaussian(data: np.ndarray, eps: float = 1e-6,
                  covariance_type: str = "full",
                  n_pca_components: int = None):
    """
    Returns (mean, cov, log_likelihood, n_params).

    covariance_type : {"full", "diagonal", "spherical", "lasso", "ledoit_wolf", "pca"}
        Controls the structural constraint applied to the covariance.
        When "lasso", uses GraphicalLassoCV on the real data for a sparse
        precision matrix with properly counted effective parameters.
        When "ledoit_wolf", uses Ledoit-Wolf optimal shrinkage estimator.
    """
    if covariance_type == "lasso":
        return _fit_gaussian_glasso(data, eps=eps)
    n, d = data.shape
    mean = data.mean(axis=0)
    cov = np.cov(data, rowvar=False)
    if cov.ndim < 2:
        cov = np.array([[cov]])
    cov = _constrain_covariance(cov, covariance_type, d,
                                n_pca_components=n_pca_components,
                                n_samples=n, eps=eps)
    try:
        ll = float(
            sp_stats.multivariate_normal.logpdf(data, mean=mean, cov=cov).sum()
        )
    except Exception:
        ll = -np.inf
    n_params = d + _n_params_cov(d, covariance_type, n_pca_components)
    return mean, cov, ll, n_params


# ---- Multivariate t-Student MLE (low-rank approx) ---------------------

def _fit_tstudent(data: np.ndarray, max_iter: int = 50, eps: float = 1e-6,
                  covariance_type: str = "full",
                  n_pca_components: int = None):
    """
    EM-based MLE for multivariate t-distribution.
    Returns (mean, cov, df, log_likelihood, n_params).

    covariance_type : {"full", "diagonal", "spherical", "lasso", "ledoit_wolf", "pca"}
        The constraint is applied at every M-step so that the
        degrees-of-freedom (sparsity) remain consistent with the shape
        matrix complexity.
        For "lasso", GraphicalLassoCV is applied to the weighted residuals
        at each M-step.
        For "ledoit_wolf", Ledoit-Wolf shrinkage is applied at each M-step.
    """
    n, d = data.shape
    # initialise
    mu = data.mean(axis=0)
    diff = data - mu
    scatter_init = (diff.T @ diff) / n
    cov = _constrain_covariance(scatter_init, covariance_type, d,
                                n_pca_components=n_pca_components,
                                n_samples=n, eps=eps)
    df = 5.0  # starting guess
    _lasso_precision = None  # track precision for lasso n_params count

    for _ in range(max_iter):
        # E-step: weights
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov += eps * np.eye(d)
            cov_inv = np.linalg.inv(cov)

        diff = data - mu
        maha = np.sum((diff @ cov_inv) * diff, axis=1)  # (n,)
        w = (df + d) / (df + maha)  # (n,)

        # M-step
        w_sum = w.sum()
        mu = (w[:, None] * data).sum(axis=0) / w_sum
        diff = data - mu

        if covariance_type == "lasso":
            # For lasso: fit GraphicalLassoCV on sqrt(w)-weighted residuals
            weighted_residuals = diff * np.sqrt(w[:, None])
            try:
                gl = GraphicalLassoCV(cv=min(3, max(2, n // 10)),
                                      max_iter=200, assume_centered=True)
                gl.fit(weighted_residuals)
                cov = gl.covariance_
                _lasso_precision = gl.precision_
            except Exception:
                scatter = (diff * w[:, None]).T @ diff / n
                cov = np.diag(np.maximum(np.diag(scatter), eps))
                _lasso_precision = None
        else:
            scatter = (diff * w[:, None]).T @ diff / n
            cov = _constrain_covariance(scatter, covariance_type, d,
                                        n_pca_components=n_pca_components,
                                        n_samples=n, eps=eps)

        # update df via simple grid search (cheap)
        best_ll = -np.inf
        best_df = df
        for candidate_df in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]:
            ll_c = _tstudent_loglikelihood(data, mu, cov, candidate_df)
            if ll_c > best_ll:
                best_ll = ll_c
                best_df = candidate_df
        df = best_df

    ll = _tstudent_loglikelihood(data, mu, cov, df)
    if covariance_type == "lasso" and _lasso_precision is not None:
        n_cov_params = _n_params_cov_from_precision(_lasso_precision)
    else:
        n_cov_params = _n_params_cov(d, covariance_type, n_pca_components)
    n_params = d + n_cov_params + 1
    return mu, cov, df, ll, n_params


def _tstudent_loglikelihood(data, mu, cov, df):
    n, d = data.shape
    try:
        cov_inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return -np.inf
    except np.linalg.LinAlgError:
        return -np.inf
    diff = data - mu
    maha = np.sum((diff @ cov_inv) * diff, axis=1)
    ll = (
        n * gammaln((df + d) / 2.0)
        - n * gammaln(df / 2.0)
        - n * d / 2.0 * math.log(df * math.pi)
        - n * 0.5 * logdet
        - ((df + d) / 2.0) * np.log(1.0 + maha / df).sum()
    )
    return float(ll)


# ---- Uniform on bounding box MLE --------------------------------------

def _fit_uniform(data: np.ndarray):
    """
    Fit a d-dimensional uniform on an axis-aligned bounding box.
    Returns (mins, maxs, log_likelihood, n_params).
    """
    n, d = data.shape
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    widths = maxs - mins
    widths = np.clip(widths, 1e-12, None)
    log_vol = np.sum(np.log(widths))
    ll = -n * log_vol  # each point has pdf = 1/volume
    n_params = 2 * d  # min + max per dimension
    return mins, maxs, ll, n_params


def _sample_uniform(mins, maxs, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    d = len(mins)
    return rng.uniform(low=mins, high=maxs, size=(n_samples, d))


# ---- Yeo-Johnson transformed diagonal Gaussian ---------------------------

def _inv_yeojohnson_single(y: np.ndarray, lmbda: float) -> np.ndarray:
    """Inverse Yeo-Johnson transform for a single dimension.

    The forward Yeo-Johnson transform is defined piecewise:
        If x >= 0:
            y = ((x+1)^λ − 1) / λ         if λ ≠ 0
            y = log(x+1)                   if λ = 0
        If x < 0:
            y = −((−x+1)^(2−λ) − 1)/(2−λ) if λ ≠ 2
            y = −log(−x+1)                 if λ = 2

    We invert each branch analytically.
    """
    x = np.empty_like(y)
    # Positive branch: y >= 0 when original x >= 0
    pos = y >= 0
    neg = ~pos

    # Invert positive branch
    if np.any(pos):
        yp = y[pos]
        if abs(lmbda) < 1e-14:
            # Clamp argument to avoid Inf from exp of very large values
            x[pos] = np.expm1(np.clip(yp, None, 80.0))
        else:
            # The forward YJ maps x>=0 to a bounded range when lmbda<0,
            # so Gaussian samples can fall outside the valid inverse domain.
            # Clamp the base to stay positive and avoid NaN from
            # np.power(negative, fractional).
            base = yp * lmbda + 1.0
            base = np.clip(base, 1e-10, None)
            x[pos] = np.power(base, 1.0 / lmbda) - 1.0

    # Invert negative branch
    if np.any(neg):
        yn = y[neg]
        if abs(lmbda - 2.0) < 1e-14:
            x[neg] = 1.0 - np.exp(np.clip(-yn, None, 80.0))
        else:
            # Same issue: for lmbda>2 the valid range is bounded,
            # clamp the base to avoid NaN.
            base = -(2.0 - lmbda) * yn + 1.0
            base = np.clip(base, 1e-10, None)
            x[neg] = 1.0 - np.power(base, 1.0 / (2.0 - lmbda))
    return x


def _yeojohnson_logjacobian(data: np.ndarray, lambdas: np.ndarray) -> float:
    """Sum of log|dy/dx| across all samples and dimensions.

    For the Yeo-Johnson transform:
        dy/dx = (x+1)^{λ-1}   if x >= 0
        dy/dx = (-x+1)^{1-λ}  if x < 0
    so  log|dy/dx| = (λ-1)*log(x+1) or (1-λ)*log(-x+1).
    """
    n, d = data.shape
    total = 0.0
    for j in range(d):
        col = data[:, j]
        lam = lambdas[j]
        pos = col >= 0
        neg = ~pos
        if np.any(pos):
            total += (lam - 1.0) * np.sum(np.log1p(col[pos]))
        if np.any(neg):
            total += (1.0 - lam) * np.sum(np.log1p(-col[neg]))
    return total


def _fit_yeojohnson(data: np.ndarray):
    """Fit per-dimension Yeo-Johnson transform + diagonal Gaussian.

    For each of the *d* dimensions:
      1. Find the optimal Yeo-Johnson lambda via MLE.
      2. Apply the transform.
    Then fit a diagonal Gaussian (mean, variance per dimension) on the
    transformed data.

    Returns (lambdas, means, stds, log_likelihood, n_params) where
    log_likelihood is corrected with the Jacobian of the transform so
    that it is comparable with likelihoods of other distributions in the
    *original* data space.
    """
    n, d = data.shape
    lambdas = np.empty(d)
    transformed = np.empty_like(data)

    for j in range(d):
        col = data[:, j]
        # scipy.stats.yeojohnson returns (transformed, lambda)
        transformed[:, j], lambdas[j] = yeojohnson(col)

    means = transformed.mean(axis=0)
    stds = transformed.std(axis=0, ddof=0)
    stds = np.clip(stds, 1e-12, None)

    # Log-likelihood in the transformed space (diagonal Gaussian)
    ll_transformed = 0.0
    for j in range(d):
        ll_transformed += sp_stats.norm.logpdf(
            transformed[:, j], loc=means[j], scale=stds[j]
        ).sum()

    # Jacobian correction: p_original(x) = p_transformed(y) * |dy/dx|
    # so  log p_original = log p_transformed + sum log|dy/dx|
    log_jac = _yeojohnson_logjacobian(data, lambdas)
    ll = ll_transformed + log_jac

    # Parameters: d lambdas + d means + d variances = 3d
    n_params = 3 * d
    return lambdas, means, stds, float(ll), n_params


# ---- Uniformity test (simple KS-based) ---------------------------------

def _uniformity_test(data: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Quick marginal uniformity check: if ALL marginals pass KS-uniform test
    over [min, max], accept as Uniform.
    """
    n, d = data.shape
    for j in range(d):
        col = data[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            return False
        normalised = (col - lo) / (hi - lo)
        stat, p = sp_stats.kstest(normalised, "uniform")
        if p < alpha:
            return False
    return True


# ---- BIC ---------------------------------------------------------------

def _bic(ll: float, n_params: int, n_samples: int) -> float:
    return -2.0 * ll + n_params * math.log(n_samples)


def _aic(ll: float, n_params: int) -> float:
    """Akaike Information Criterion."""
    return -2.0 * ll + 2.0 * n_params


def _aicc(ll: float, n_params: int, n_samples: int) -> float:
    """Corrected AIC — better for small-sample / high-dimensional settings."""
    aic_val = _aic(ll, n_params)
    denom = n_samples - n_params - 1
    if denom > 0:
        return aic_val + 2.0 * n_params * (n_params + 1) / denom
    return float("inf")


def _compute_criterion(ll: float, n_params: int, n_samples: int,
                       criterion: str = "bic") -> float:
    """Evaluate an information criterion.  Supported: bic, aic, aicc."""
    if criterion == "aic":
        return _aic(ll, n_params)
    elif criterion == "aicc":
        return _aicc(ll, n_params, n_samples)
    # default: BIC
    return _bic(ll, n_params, n_samples)


def _n_params_for(dist_type: str, d: int,
                  covariance_type: str = "full",
                  n_pca_components: int = None) -> int:
    """Number of free parameters for a distribution family in *d* dims."""
    if dist_type == "gaussian":
        return d + _n_params_cov(d, covariance_type, n_pca_components)
    elif dist_type == "tstudent":
        return d + _n_params_cov(d, covariance_type, n_pca_components) + 1
    elif dist_type == "vmf":
        return d + 1 + 2                        # mu + kappa + norm_dist (2 params)
    elif dist_type == "kent":
        p_tangent = d - 1
        return d + 1 + p_tangent * (p_tangent + 1) // 2 + 2  # mu+kappa + tangent_cov + norm_dist
    elif dist_type == "laplace":
        return 2 * d                            # loc + scale (per dim)
    elif dist_type == "uniform":
        return 2 * d                            # lo + hi (per dim)
    elif dist_type == "yeojohnson":
        return 3 * d                            # lambda + mean + var (per dim)
    else:                                       # fallback → Gaussian
        return d + _n_params_cov(d, covariance_type, n_pca_components)


def _partition_criterion(groups, d: int, n_total: int,
                         criterion: str = "bic",
                         covariance_type: str = "full",
                         n_pca_components: int = None) -> float:
    """
    Information criterion for a partition viewed as a *k*-component mixture.

    Uses ``log(n_total)`` globally (instead of ``log(n_i)`` per cluster)
    and adds *k − 1* mixing-weight parameters so that models with more
    clusters are penalised more heavily — preventing heavy-tailed unimodal
    distributions from being shattered into many small Gaussians.

    Supported criteria: "bic" (default), "aic", "aicc".
    """
    total_ll = 0.0
    total_params = 0
    k = len(groups)
    for info, _idx in groups:
        if not np.isfinite(info.ll):
            return np.inf
        total_ll += info.ll
        total_params += _n_params_for(info.dist_type, d,
                                       covariance_type, n_pca_components)
    if k > 1:
        total_params += k - 1          # mixing weights
    if n_total < 2:
        return np.inf
    return _compute_criterion(total_ll, total_params, n_total, criterion)


# backward-compatible alias
def _partition_bic(groups, d: int, n_total: int) -> float:
    return _partition_criterion(groups, d, n_total, "bic")


# =====================================================================
# Distribution info container
# =====================================================================

class DistributionInfo:
    """Stores the fitted distribution type and parameters for one group."""

    # valid types: "gaussian", "vmf", "kent", "uniform", "laplace", "tstudent", "yeojohnson"
    def __init__(self, dist_type: str, params: dict, ll: float, bic: float,
                 decided_by: str = "bic"):
        self.dist_type = dist_type
        self.params = params          # distribution-specific parameters
        self.ll = ll
        self.bic = bic
        self.decided_by = decided_by  # "test" or "bic"

    def __repr__(self):
        return (f"DistributionInfo(type={self.dist_type}, decided_by={self.decided_by}, "
                f"ll={self.ll:.2f}, bic={self.bic:.2f})")


# =====================================================================
# Pipeline: fit best distribution for one group
# =====================================================================

def fit_best_distribution(
    data: np.ndarray,
    alpha_hz: float = 0.05,
    cr_threshold: float = 0.10,
    alpha_rayleigh: float = 0.05,
    alpha_uniform: float = 0.05,
    bic_penalty_factor: float = 1.0,
    device=None,
    covariance_type: str = "full",
    n_pca_components: int = None,
    model_selection: str = "bic",
):
    """
    Applies the structured pipeline on *data* (n × d numpy array) and returns
    a DistributionInfo describing the best distribution.

    Parameters
    ----------
    covariance_type : {"full", "diagonal", "spherical", "ledoit_wolf", "pca"}
        Structural constraint on the covariance / shape matrix.
    n_pca_components : int or None
        Number of PCA components when covariance_type="pca".
        Auto-selected (95 % variance) when None.
    model_selection : {"bic", "aic", "aicc"}
        Information criterion used for the distribution duel.

    Steps
    -----
    1. HZ multivariate normality test → Gaussian
    2. Yeo-Johnson per-dim normality test → yeojohnson
    3. Shell analysis (Cr) + Rayleigh → vMF
    4. Marginal uniformity test → Uniform
    5. Criterion duel among candidates
    """
    n, d = data.shape
    _cov_kw = dict(covariance_type=covariance_type,
                   n_pca_components=n_pca_components)

    # ----- Step 1: Henze-Zirkler ------------------------------------------
    is_normal, p_hz = _hz_test(data, alpha=alpha_hz)
    if is_normal:
        mean, cov, ll, n_params = _fit_gaussian(data, **_cov_kw)
        crit_val = _compute_criterion(ll, n_params, n, model_selection)
        params = {"mean": mean, "cov": cov}
        return DistributionInfo("gaussian", params, ll, crit_val, decided_by="test")

    # ----- Step 2: Yeo-Johnson per-dimension normality --------------------
    #  Disabled: the inverse Yeo-Johnson can map Gaussian samples to very
    #  large values in the original space (especially for |λ|≈0 or λ≈2),
    #  causing enormous synthetic logits and unstable training losses.
    #  Re-enable once a bounded-sampling strategy is in place.
    #
    # try:
    #     yj_lambdas, yj_means, yj_stds, yj_ll, yj_np = _fit_yeojohnson(data)
    #     n_sw = min(n, 5000)
    #     transformed_check = np.empty((n_sw, d))
    #     for _j in range(d):
    #         transformed_check[:, _j] = yeojohnson(data[:n_sw, _j], lmbda=yj_lambdas[_j])
    #     all_normal = True
    #     for _j in range(d):
    #         _stat, _p = sp_stats.shapiro(transformed_check[:, _j])
    #         if _p < alpha_hz:
    #             all_normal = False
    #             break
    #     if all_normal:
    #         crit_val = _compute_criterion(yj_ll, yj_np, n, model_selection)
    #         params = {"lambdas": yj_lambdas, "means": yj_means, "stds": yj_stds}
    #         return DistributionInfo("yeojohnson", params, yj_ll, crit_val,
    #                                 decided_by="test")
    # except Exception:
    #     pass

    # ----- Step 3: Shell / vMF -------------------------------------------
    cr = _radio_dispersion(data)
    if cr < cr_threshold:
        p_ray = _rayleigh_test(data)
        if p_ray < alpha_rayleigh:
            norms_vec = np.linalg.norm(data, axis=1)
            norms_vec = np.clip(norms_vec, 1e-12, None)
            unit_vectors = data / norms_vec[:, None]
            rbar = float(np.linalg.norm(unit_vectors.mean(axis=0)))
            if rbar > 0.7:
                # Estimate vMF parameters
                mu, kappa, ll_dir, n_params_dir = _fit_vmf(data)
                # Fit norm distribution (Normal, LogNormal, Gamma; fallback LogNormal)
                norm_dist = _fit_norm_distribution(norms_vec, alpha=alpha_rayleigh)
                # Total ll = directional ll + norm ll
                ll = ll_dir + norm_dist["ll"]
                n_params = n_params_dir + norm_dist["n_params"]
                crit_val = _compute_criterion(ll, n_params, n, model_selection)
                params = {
                    "mu": mu,
                    "kappa": kappa,
                    "norm_dist": norm_dist,
                }

                # ----- Test vMF vs Kent (Fisher-Bingham) -----------------
                kent_type, kent_extra = _fit_kent_from_vmf(
                    data, mu, kappa, norm_dist, alpha=alpha_rayleigh
                )
                if kent_type == "kent":
                    params["tangent_cov"] = kent_extra["tangent_cov"]
                    params["tangent_mean"] = kent_extra["tangent_mean"]
                    # Kent has additional tangent covariance parameters
                    p_tangent = len(kent_extra["tangent_mean"])
                    n_params_kent = (
                        n_params_dir
                        + p_tangent * (p_tangent + 1) // 2
                        + norm_dist["n_params"]
                    )
                    crit_val_kent = _compute_criterion(
                        ll, n_params_kent, n, model_selection
                    )
                    print(f"[KENT ACCEPTED] Cr={cr:.4f}, p_ray={p_ray:.4f}, "
                          f"Rbar={rbar:.4f}, kappa={kappa:.4f}, "
                          f"norm_dist={norm_dist['type']}")
                    return DistributionInfo("kent", params, ll, crit_val_kent,
                                            decided_by="test")

                print(f"[vMF ACCEPTED] Cr={cr:.4f}, p_ray={p_ray:.4f}, "
                      f"Rbar={rbar:.4f}, kappa={kappa:.4f}, "
                      f"norm_dist={norm_dist['type']}")
                return DistributionInfo("vmf", params, ll, crit_val,
                                        decided_by="test")
            else:
                print(f"[vMF REJECTED] Mean resultant length too low: Rbar={rbar:.4f} < 0.7. "
                      f"Cr={cr:.4f}, p_ray={p_ray:.4f}")
        else:
            print(f"[vMF REJECTED] Rayleigh test failed: p_value={p_ray:.4f} >= {alpha_rayleigh}. "
                  f"Directional distribution is too uniform. Cr={cr:.4f}")
    else:
        print(f"[vMF REJECTED] Radio dispersion too high: Cr={cr:.4f} >= {cr_threshold}. "
              f"Data not concentrated on spherical shell.")

    # ----- Step 4: Uniformity test ---------------------------------------
    if _uniformity_test(data, alpha=alpha_uniform):
        mins, maxs, ll, n_params = _fit_uniform(data)
        crit_val = _compute_criterion(ll, n_params, n, model_selection)
        params = {"mins": mins, "maxs": maxs}
        return DistributionInfo("uniform", params, ll, crit_val, decided_by="test")

    # ----- Step 5: criterion duel (catch-all) ----------------------
    candidates = {}

    # Gaussian
    g_mean, g_cov, g_ll, g_np = _fit_gaussian(data, **_cov_kw)
    candidates["gaussian"] = (g_ll, g_np, {"mean": g_mean, "cov": g_cov})

    # t-Student (same covariance constraint applied at every EM M-step)
    try:
        t_mean, t_cov, t_df, t_ll, t_np = _fit_tstudent(data, **_cov_kw)
        candidates["tstudent"] = (t_ll, t_np, {"mean": t_mean, "cov": t_cov, "df": t_df})
    except Exception:
        pass

    # NOTE: vMF is intentionally NOT included in the criterion duel, by
    # design choice — even though it *could* be made comparable the same way
    # Kent is below, it is kept out on purpose.

    # Kent (Fisher-Bingham) — only enters the duel when the data shows
    # genuine anisotropy on the sphere (sphericity test rejected against the
    # vMF fit). Unlike the excluded vMF candidate above, Kent's ll here is
    # NOT just the directional (unit-sphere) component: it is
    # ll = ll_directional (Kent shape on S^{d-1}) + ll_radial (a fitted
    # Normal/LogNormal/Gamma distribution over the vector *norms*), i.e. the
    # radius of each embedding is explicitly modelled too. That direction +
    # norm decomposition is what makes its total ll comparable in R^d to the
    # Gaussian/t-Student/Laplace/Uniform candidates below.
    try:
        k_mu, k_kappa, k_ll_dir, k_np_dir = _fit_vmf(data)
        k_norms = np.clip(np.linalg.norm(data, axis=1), 1e-12, None)
        k_norm_dist = _fit_norm_distribution(k_norms, alpha=alpha_rayleigh)
        kent_type, kent_extra = _fit_kent_from_vmf(
            data, k_mu, k_kappa, k_norm_dist, alpha=alpha_rayleigh
        )
        if kent_type == "kent":
            k_ll_direction_plus_norm = k_ll_dir + k_norm_dist["ll"]
            p_tangent = len(kent_extra["tangent_mean"])
            k_np = k_np_dir + p_tangent * (p_tangent + 1) // 2 + k_norm_dist["n_params"]
            candidates["kent"] = (
                k_ll_direction_plus_norm, k_np,
                {
                    "mu": k_mu,
                    "kappa": k_kappa,
                    "norm_dist": k_norm_dist,  # radial (norm) component, fitted explicitly
                    "tangent_cov": kent_extra["tangent_cov"],
                    "tangent_mean": kent_extra["tangent_mean"],
                },
            )
    except Exception:
        pass

    # Laplace
    try:
        l_locs, l_scales, l_ll, l_np = _fit_laplace(data)
        candidates["laplace"] = (l_ll, l_np, {"locs": l_locs, "scales": l_scales})
    except Exception:
        pass

    # Uniform
    u_mins, u_maxs, u_ll, u_np = _fit_uniform(data)
    candidates["uniform"] = (u_ll, u_np, {"mins": u_mins, "maxs": u_maxs})

    # Yeo-Johnson transformed diagonal Gaussian — disabled (see Step 2 comment).
    # try:
    #     yj_lambdas, yj_means, yj_stds, yj_ll, yj_np = _fit_yeojohnson(data)
    #     candidates["yeojohnson"] = (
    #         yj_ll, yj_np,
    #         {"lambdas": yj_lambdas, "means": yj_means, "stds": yj_stds},
    #     )
    # except Exception:
    #     pass

    # Select by chosen criterion
    best_name, best_crit, best_params = None, np.inf, None
    best_ll = -np.inf
    for name, (ll, np_, params) in candidates.items():
        c = _compute_criterion(ll, np_, n, model_selection) * bic_penalty_factor
        if c < best_crit:
            best_crit = c
            best_name = name
            best_params = params
            best_ll = ll

    # ---- Confidence check -----------------------------------------------
    decided = "bic"
    if best_name not in ("gaussian", "vmf") and "gaussian" in candidates:
        g_ll_c, g_np_c, _ = candidates["gaussian"]
        g_crit = _compute_criterion(g_ll_c, g_np_c, n, model_selection) * bic_penalty_factor
        dim_scale = max(1.0, math.log(max(d, 2)))
        if g_crit - best_crit > 10.0 * dim_scale:
            decided = "bic_confident"

    return DistributionInfo(best_name, best_params, best_ll, best_crit,
                            decided_by=decided)


# =====================================================================
# Recursive cluster subdivision
# =====================================================================

def _fit_cluster_recursive(
    data: np.ndarray,
    initial_k: int = 1,
    min_cluster_size: int = None,
    dim: int = None,
    alpha_hz: float = 0.05,
    cr_threshold: float = 0.10,
    max_iterations: int = 20,
    covariance_type: str = "full",
    n_pca_components: int = None,
    model_selection: str = "bic",
):
    """
    Iteratively fit distributions to clusters of *data* using a pool-based
    approach with criterion-guided splitting.

    Algorithm
    ---------
    0. Fit the best distribution to the entire pending pool (baseline).
       If accepted by a statistical test → return immediately.
    1. Try splitting the pending pool into *k* clusters.
    2. Compute the total criterion of the new partition (sum over clusters).
       - If criterion ≥ previous baseline → splitting hurts.
         Roll back to the previous partition and stop.
       - If criterion improved → adopt the new partition.
    3. Clusters accepted by statistical test are finalised permanently.
       The remaining (criterion-decided) clusters are merged and become the
       pending pool for the next round (k = n_remaining + 1).
    4. Repeat until:
       a. All clusters are accepted by test, OR
       b. Splitting does not improve criterion (early stop), OR
       c. Cluster sizes fall below sqrt(N) or the embedding dim, OR
       d. max_iterations is reached.

    Returns a list of (DistributionInfo, points_indices) tuples, where
    indices refer to rows of the original *data* array.
    """
    n, d = data.shape
    if min_cluster_size is None:
        min_cluster_size = max(int(math.sqrt(n)), d)
    if dim is None:
        dim = d

    _dist_kw = dict(covariance_type=covariance_type,
                    n_pca_components=n_pca_components,
                    model_selection=model_selection)
    _crit_kw = dict(criterion=model_selection,
                    covariance_type=covariance_type,
                    n_pca_components=n_pca_components)

    # ---- Initial baseline: fit the whole data as a single cluster -------
    pending_indices = np.arange(n)
    initial_info = fit_best_distribution(
        data, alpha_hz=alpha_hz, cr_threshold=cr_threshold, **_dist_kw,
    )

    if initial_info.decided_by in ("test", "bic_confident"):
        # Accepted by statistical test or by confident criterion win – done
        return [(initial_info, pending_indices)]

    # The best single-cluster fit is our baseline partition.
    accepted = []                                       # finalised groups
    prev_partition = [(initial_info, pending_indices.copy())]
    prev_pcrit = _partition_criterion(prev_partition, d, n, **_crit_kw)

    current_k = 2  # first split attempt: 2 clusters

    for _iteration in range(max_iterations):
        pending_data = data[pending_indices]
        n_pending = len(pending_indices)

        if n_pending < 2:
            accepted.extend(prev_partition)
            prev_partition = []
            break

        # --- Cluster the pending pool ------------------------------------
        actual_k = min(current_k, n_pending)
        if actual_k < 1:
            actual_k = 1
        kmeans = KMeans(
            n_clusters=actual_k, random_state=42, n_init=10
        ).fit(pending_data)

        # Fit each cluster
        this_round = []

        for cid in range(actual_k):
            local_mask = kmeans.labels_ == cid
            cluster_data = pending_data[local_mask]
            cluster_orig_idx = pending_indices[local_mask]
            n_c = len(cluster_data)

            if n_c < 2:
                mean = cluster_data.mean(axis=0) if n_c > 0 else np.zeros(d)
                cov = np.eye(d) * 1e-4
                info = DistributionInfo(
                    "gaussian", {"mean": mean, "cov": cov},
                    -np.inf, np.inf, "fallback",
                )
                this_round.append((info, cluster_orig_idx))
                continue

            info = fit_best_distribution(
                cluster_data, alpha_hz=alpha_hz, cr_threshold=cr_threshold,
                **_dist_kw,
            )
            this_round.append((info, cluster_orig_idx))

        # --- Compare partition criterion (global penalty) ----------------
        new_pcrit = _partition_criterion(this_round, d, n_pending, **_crit_kw)

        if (np.isfinite(new_pcrit) and np.isfinite(prev_pcrit)
                and new_pcrit >= prev_pcrit):
            # Splitting made it worse (or equal) → keep previous partition
            accepted.extend(prev_partition)
            prev_partition = []
            break

        # --- Splitting improved! -----------------------------------------
        _ACCEPTED_KINDS = {"test", "bic_confident", "fallback"}
        confident_accepted = [
            (info, idx) for info, idx in this_round
            if info.decided_by in _ACCEPTED_KINDS
        ]
        bic_decided = [
            (info, idx) for info, idx in this_round
            if info.decided_by not in _ACCEPTED_KINDS
        ]

        accepted.extend(confident_accepted)

        if len(bic_decided) == 0:
            prev_partition = []
            break

        prev_partition = bic_decided
        pending_indices = np.concatenate([idx for _, idx in bic_decided])
        n_remaining = len(pending_indices)
        prev_pcrit = _partition_criterion(bic_decided, d, n_remaining,
                                          **_crit_kw)

        new_k = len(bic_decided) + 1
        new_min_size = max(int(math.sqrt(n_remaining)), dim)

        if n_remaining // new_k < new_min_size:
            accepted.extend(bic_decided)
            prev_partition = []
            break

        current_k = new_k

    else:
        if prev_partition:
            accepted.extend(prev_partition)

    return accepted


# =====================================================================
# Sampling from fitted distributions
# =====================================================================

def _to_tensor(x, device):
    """Convert numpy array or existing tensor to float32 on *device*."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=device)


def sample_from_distribution(info: DistributionInfo, n_samples: int,
                             device="cpu", rng=None):
    """
    Sample *n_samples* points from the distribution described by *info*.
    Returns a torch Tensor of shape (n_samples, d).

    Uses GPU-native PyTorch distributions for Gaussian, t-Student, Laplace
    and Uniform.  vMF falls back to numpy (rejection sampling).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    p = info.params
    dt = info.dist_type

    if dt == "gaussian":
        mean_t = _to_tensor(p["mean"], device)
        cov_t = _to_tensor(p["cov"], device)
        dist = MultivariateNormal(loc=mean_t, covariance_matrix=cov_t)
        return dist.sample((n_samples,))

    elif dt == "tstudent":
        # X = mu + Z * sqrt(df / chi2(df))  — fully on device
        d = len(p["mean"])
        mean_t = _to_tensor(p["mean"], device)
        cov_t = _to_tensor(p["cov"], device)
        df_t = torch.tensor(p["df"], dtype=torch.float32, device=device)
        mvn = MultivariateNormal(
            loc=torch.zeros(d, device=device), covariance_matrix=cov_t
        )
        z = mvn.sample((n_samples,))
        chi2_dist = torch.distributions.Chi2(df=df_t)
        chi2_samples = chi2_dist.sample((n_samples,)).unsqueeze(1)
        scaling = torch.sqrt(df_t / chi2_samples)
        return mean_t + z * scaling

    elif dt == "vmf":
        # Sample directions on the unit sphere from vMF,
        # then multiply by norms sampled from the fitted norm distribution.
        d = len(p["mu"])
        kappa = p["kappa"]
        unit_samples = _sample_vmf(p["mu"], kappa, n_samples, rng)  # (n, d) on S^{d-1}

        norm_dist = p.get("norm_dist")
        if norm_dist is not None and norm_dist.get("type") == "normal":
            sampled_norms = rng.normal(
                loc=norm_dist["mean"], scale=norm_dist["std"], size=n_samples
            )
            sampled_norms = np.clip(sampled_norms, 1e-6, None)  # norms must be positive
        elif norm_dist is not None and norm_dist.get("type") == "lognormal":
            sampled_norms = rng.lognormal(
                mean=norm_dist["mu"], sigma=norm_dist["sigma"], size=n_samples
            )
        elif norm_dist is not None and norm_dist.get("type") == "gamma":
            sampled_norms = rng.gamma(
                shape=norm_dist["shape"], scale=norm_dist["scale"], size=n_samples
            ) + norm_dist.get("loc", 0.0)
            sampled_norms = np.clip(sampled_norms, 1e-6, None)
        else:
            # Legacy fallback: use scalar mean_norm
            mean_norm = p.get("mean_norm", 1.0)
            sampled_norms = np.full(n_samples, mean_norm)

        samples = unit_samples * sampled_norms[:, None]
        return torch.tensor(samples, dtype=torch.float32, device=device)

    elif dt == "kent":
        # Sample directions from Kent (anisotropic on sphere),
        # then multiply by sampled norms (same radial model as vMF).
        d = len(p["mu"])
        tangent_cov = p["tangent_cov"]
        tangent_mean = p["tangent_mean"]
        unit_samples = _sample_kent(p["mu"], tangent_cov, tangent_mean,
                                    n_samples, rng)

        norm_dist = p.get("norm_dist")
        if norm_dist is not None and norm_dist.get("type") == "normal":
            sampled_norms = rng.normal(
                loc=norm_dist["mean"], scale=norm_dist["std"], size=n_samples
            )
            sampled_norms = np.clip(sampled_norms, 1e-6, None)
        elif norm_dist is not None and norm_dist.get("type") == "lognormal":
            sampled_norms = rng.lognormal(
                mean=norm_dist["mu"], sigma=norm_dist["sigma"], size=n_samples
            )
        elif norm_dist is not None and norm_dist.get("type") == "gamma":
            sampled_norms = rng.gamma(
                shape=norm_dist["shape"], scale=norm_dist["scale"], size=n_samples
            ) + norm_dist.get("loc", 0.0)
            sampled_norms = np.clip(sampled_norms, 1e-6, None)
        else:
            mean_norm = p.get("mean_norm", 1.0)
            sampled_norms = np.full(n_samples, mean_norm)

        samples = unit_samples * sampled_norms[:, None]
        return torch.tensor(samples, dtype=torch.float32, device=device)

    elif dt == "laplace":
        locs_t = _to_tensor(p["locs"], device)
        scales_t = _to_tensor(p["scales"], device)
        dist = torch.distributions.Laplace(loc=locs_t, scale=scales_t)
        return dist.sample((n_samples,))

    elif dt == "uniform":
        mins_t = _to_tensor(p["mins"], device)
        maxs_t = _to_tensor(p["maxs"], device)
        dist = torch.distributions.Uniform(low=mins_t, high=maxs_t)
        return dist.sample((n_samples,))

    elif dt == "yeojohnson":
        # Sample from diagonal Gaussian in transformed space,
        # then invert Yeo-Johnson per dimension.
        lambdas = p["lambdas"]  # (d,) numpy
        means = p["means"]      # (d,) numpy
        stds = p["stds"]        # (d,) numpy
        d = len(means)
        # Clamp Gaussian samples to ±4σ to stay within the valid domain
        # of the inverse Yeo-Johnson (which is bounded for some λ values).
        z = rng.standard_normal((n_samples, d))
        z = np.clip(z, -4.0, 4.0)
        transformed_samples = z * stds[np.newaxis, :] + means[np.newaxis, :]
        # Invert Yeo-Johnson per dimension
        original_samples = np.empty_like(transformed_samples)
        for j in range(d):
            original_samples[:, j] = _inv_yeojohnson_single(
                transformed_samples[:, j], lambdas[j]
            )
        # Sanitise: replace any residual NaN/Inf with the column mean
        # (from the fitted distribution) and clamp to float32-safe range.
        for j in range(d):
            col = original_samples[:, j]
            bad = ~np.isfinite(col)
            if np.any(bad):
                col_mean = np.nanmean(col) if np.any(~bad) else 0.0
                col[bad] = col_mean
                original_samples[:, j] = col
        original_samples = np.clip(original_samples, -1e6, 1e6)
        return torch.tensor(original_samples, dtype=torch.float32, device=device)

    else:
        raise ValueError(f"Unknown distribution type: {dt}")


def _sample_vmf(mu: np.ndarray, kappa: float, n: int, rng):
    """
    Sample from vMF(mu, kappa) using Wood's algorithm (1994).
    """
    d = len(mu)
    # Householder to align mu with e1
    result = np.zeros((n, d))
    for i in range(n):
        # sample w ~ f(w) ∝ (1-w²)^{(d-3)/2} exp(κ w), w ∈ [-1,1]
        w = _sample_w(d, kappa, rng)
        # sample v uniformly on S^{d-2}
        v = rng.standard_normal(d - 1)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            v = np.zeros(d - 1)
            v[0] = 1.0
        else:
            v = v / v_norm
        # x = sqrt(1-w²)*v , w  in the frame where mu = e_1
        x_local = np.concatenate(([w], np.sqrt(max(1.0 - w * w, 0.0)) * v))
        result[i] = _householder_rotate(x_local, mu)
    return result


def _sample_w(d, kappa, rng):
    """Sample scalar w for the Wood algorithm.

    If Wood's rejection loop does not accept within 10000 iterations
    (can happen when κ ≫ d), falls back to the Gamma approximation
    1 − w ≈ Gamma((d−1)/2, 1/κ).
    """
    dim_m1 = d - 1
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa ** 2 + dim_m1 ** 2)) / dim_m1
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + dim_m1 * math.log(1.0 - x0 ** 2 + 1e-30)

    for _ in range(10000):
        z = rng.beta(dim_m1 / 2.0, dim_m1 / 2.0)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        t = kappa * w + dim_m1 * math.log(1.0 - x0 * w + 1e-30) - c
        if t >= math.log(rng.random() + 1e-30):
            return w
    # Fallback: Gamma approximation for high κ
    offset = rng.gamma((d - 1) / 2.0, 1.0 / kappa)
    return float(np.clip(1.0 - offset, -1.0, 1.0))


def _householder_rotate(x, mu):
    """Rotate x so that e1 maps to mu via Householder reflection."""
    d = len(mu)
    e1 = np.zeros(d)
    e1[0] = 1.0
    u = e1 - mu
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-12:
        return x
    u = u / u_norm
    return x - 2.0 * np.dot(u, x) * u


# =====================================================================
# DistributionalMethod
# =====================================================================

class DistributionalMethod(BaseMethod):
    """
    Unlearning method that adaptively fits the best statistical distribution
    to each class / cluster of embeddings and uses those distributions to
    generate synthetic retain data.
    """

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)
        self.using_robust = None

        # Per-class distribution info.  When clustering is enabled each entry
        # is a *list* of (DistributionInfo, intra_fraction) tuples.
        self.dist_infos = None          # dict  class_id -> DistributionInfo | list[...]
        self.percentages = None         # inter-class percentages (tensor)
        self.intra_percentage = None    # list of tensors (only with clustering)
        self.num_samples_original = None

        # Gaussian-compatible attributes kept for the `update_centroids` /
        # `update_covariance` code path (identical to NOISYMethod).
        self.centroids = None
        self.covariance = None
        self.pre_centroids = None
        self.pre_covariance = None
        self.gaussian_noise = None      # used only for Gaussian sub-groups


        self.generator = None
        self.noisy_adder = None

        self.lambda_margin = 0.0
        self.ref_model = None  # frozen copy of model before unlearning, used by 'delete' loss

        # Gradient subspace projection (optional, computed in setup())
        # Shape (emb_dim, k): orthonormal basis of forget subspace in embedding space
        self._grad_proj_classifier_S = None
        # Shape (backbone_dim, k): orthonormal basis in backbone feature space (extra_layers only)
        self._grad_proj_emb_S = None

        # Classifier anchor: frozen copy of retain-class rows (shape: n_retain × emb_dim)
        self._anchor_retain_weights = None
        self._anchor_retain_indices = None  # LongTensor of retain class indices

        # Logit-profile regularizer: per-retain-class reference logits/softmax,
        # snapshotted once in setup() from the originally generated synthetic
        # samples (see --logit-profile-lambda / --logit-profile-mode).
        # Shape (num_classes, num_classes); rows for forget classes are unused.
        self._logit_profile_ref_logits = None
        self._logit_profile_ref_softmax = None

        self.config["retain_labels"] = [i for i in range(self.config["num_classes"]) if i not in self.config["forbidden_labels"]]

    # ----- helpers identical to NOISYMethod -------------------------------

    @property
    def device(self):
        return next(self.model.parameters()).device

    @staticmethod
    def _nanmax_compat(x):
        """Compatibility wrapper for torch.nanmax in older PyTorch versions."""
        if hasattr(torch, "nanmax"):
            return torch.nanmax(x)
        valid = x[~torch.isnan(x)]
        if valid.numel() == 0:
            return torch.tensor(float("nan"), dtype=x.dtype, device=x.device)
        return torch.max(valid)

    @staticmethod
    def _nanmin_compat(x):
        """Compatibility wrapper for torch.nanmin in older PyTorch versions."""
        if hasattr(torch, "nanmin"):
            return torch.nanmin(x)
        valid = x[~torch.isnan(x)]
        if valid.numel() == 0:
            return torch.tensor(float("nan"), dtype=x.dtype, device=x.device)
        return torch.min(valid)

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

    # ----- Lift centroids / covariance to pre-embedding space -------------

    def calculate_minimun_pre_data(self):
        """Identical to NOISYMethod – lifts Gaussian centroids/cov to
        pre-embedding space when `embunfreeze` is enabled."""
        if self.centroids is None:
            return
        if not self.model._extra_layers:
            print("⚠️  Skipping analytical centroid/cov lifting (no linear embedding layer).")
            return
        head = self.model.get_embs_layer()
        W = head.weight.detach()
        b = head.bias.detach()
        W_pinv = torch.linalg.pinv(W)

        aux_centroids = [None] * self.config["num_classes"]
        aux_cov = [None] * self.config["num_classes"]

        if self.config["update_covariance"]:
            I = torch.eye(W.shape[1], device=W.device)
            P_null = I - W_pinv @ W
            invisible = P_null @ (I * 1e-8) @ P_null.t()

        if self.config["clustering"]:
            for i in range(self.config["num_classes"]):
                aux_centroids[i] = []
                aux_cov[i] = []
                for j in range(len(self.centroids[i])):
                    if self.config["update_centroids"]:
                        X = self.centroids[i][j].unsqueeze(1)
                        aux_centroids[i].append((W_pinv @ (X - b)).squeeze(1))
                    if self.config["update_covariance"]:
                        cur = self.covariance[i][j]
                        vis = W_pinv @ cur @ W_pinv.t()
                        aux_cov[i].append(vis + invisible)
        else:
            for i in range(self.config["num_classes"]):
                if self.config["update_centroids"]:
                    X = self.centroids[i].unsqueeze(1)
                    aux_centroids[i] = (W_pinv @ (X - b)).squeeze(1)
                if self.config["update_covariance"]:
                    cur = self.covariance[i]
                    vis = W_pinv @ cur @ W_pinv.t()
                    aux_cov[i] = vis + invisible

        if self.config["update_centroids"]:
            self.pre_centroids = aux_centroids
        if self.config["update_covariance"]:
            self.pre_covariance = aux_cov

    def calculate_minimun_preEmbeddings(self, syntetic_embeddings, labels=None):
        """Map embeddings back to the pre-embedding space.

        * extra_layers=True  → analytical pseudo-inverse of the linear head.
        * extra_layers=False → gradient-based optimisation that balances
          Euclidean closeness to the target and distribution-specific fit,
          with a cosine scheduler  α : 1 → 0  (Euclidean first, then
          distribution).
        """
        if self.model._extra_layers:
            # Analytical inversion via pseudoinverse (linear embedding layer)
            head = self.model.get_embs_layer()
            W = head.weight.detach()
            b = head.bias.detach()
            return (torch.linalg.pinv(W) @ (syntetic_embeddings.t() - b.unsqueeze(1))).t()
        else:
            # Optimisation-based: model has no linear embedding layer
            return self._optimize_preembeddings(syntetic_embeddings, labels)

    # -----------------------------------------------------------------
    #  Optimisation-based pre-embedding generation  (no extra layers)
    # -----------------------------------------------------------------

    def _compute_inverse_embs_generator(self, syntetic_embeddings, labels=None):
        device = syntetic_embeddings.device
        dtype = syntetic_embeddings.dtype

        # Keep labels optional while preserving DataLoader compatibility.
        if labels is None:
            labels = torch.full(
                (syntetic_embeddings.shape[0],),
                -1,
                dtype=torch.long,
                device=device,
            )
        else:
            labels = labels.to(device=device)

        noisy_dim = max(1, int(syntetic_embeddings.shape[1] * 0.1))
        self.noisy_adder = torch.nn.Sequential( 
            torch.nn.Linear(noisy_dim, syntetic_embeddings.shape[1], bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(syntetic_embeddings.shape[1], syntetic_embeddings.shape[1], bias=True)
        ).to(device=device, dtype=dtype)

        def _default_inverse_input_shape(base_model: str):
            if base_model == "resnet18":
                # Input to layer4 for torchvision resnet18 with 224x224 images.
                return (256, 14, 14)
            if base_model == "efficientnet_v2_s":
                # Input to features[-1] for torchvision efficientnet_v2_s with 384x384 images.
                return (256, 12, 12)
            if base_model == "swin_t":
                # Input to features[7] for torchvision swin_t with 224x224 images (HWC format).
                return (7, 7, 768)
            if base_model == "vit_b_16":
                hidden = self.model._hyperparameters["embedding_size"]
                return (197, hidden)
            raise ValueError(f"Unsupported base model for inverse input shape: {base_model}")

        inverse_builder = self.model.get_inverse_embs_layer
        signature = inspect.signature(inverse_builder)
        if "input_shape" in signature.parameters:
            inverse_embs_layer = inverse_builder(
                input_shape=_default_inverse_input_shape(self.model._base_model)
            )
        else:
            # Backward compatibility with previous model wrappers.
            inverse_embs_layer = inverse_builder()

        self.generator = ADINGenerativeModel(inverse_embs_layer, syntetic_embeddings.shape[1]).to(device=device, dtype=dtype)

        self.generator.train()
        for param in self.generator.parameters():
            param.requires_grad_(True)
        
        self.noisy_adder.train()
        for param in self.noisy_adder.parameters():
            param.requires_grad_(True)

        # Infer actual pre-embedding shape produced by generator so InfoQ receives
        # the correct dimensionality before the final linear layer.
        with torch.no_grad():
            probe_emb = syntetic_embeddings[:1].to(device=device, dtype=dtype)
            probe_noisy = self.noisy_adder(torch.zeros(1, noisy_dim, device=device, dtype=dtype))
            probe_pre = self.generator(probe_emb, probe_noisy)

        if self.model._base_model == "vit_b_16":
            if probe_pre.dim() != 3:
                raise ValueError(
                    f"For vit_b_16, generator must output [B, T, D], got shape {tuple(probe_pre.shape)}"
                )
            infoq_in_dim = int(probe_pre.shape[-1])
            infoQ = torch.nn.Sequential(
                ExtractCLS(),
                torch.nn.Linear(infoq_in_dim, noisy_dim),
            )
        else:
            if probe_pre.dim() == 4:
                infoq_in_dim = int(probe_pre.shape[1])
                infoQ = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten(1),
                    torch.nn.Linear(infoq_in_dim, noisy_dim),
                )
            elif probe_pre.dim() == 2:
                infoq_in_dim = int(probe_pre.shape[1])
                infoQ = torch.nn.Sequential(
                    torch.nn.Linear(infoq_in_dim, noisy_dim),
                )
            else:
                raise ValueError(
                    f"Unsupported generator output shape for non-ViT infoQ: {tuple(probe_pre.shape)}"
                )

        infoQ = infoQ.to(device=device, dtype=dtype)
        self._inverse_infoQ = infoQ
        self._inverse_gen_debug = {
            "noisy_dim": noisy_dim,
            "probe_pre_shape": tuple(probe_pre.shape),
            "infoQ_input_dim": infoq_in_dim,
            "infoQ_output_dim": noisy_dim,
        }

        best_loss = float("inf")
        trainable_params = list(self.generator.parameters()) + list(self.noisy_adder.parameters()) + list(infoQ.parameters())
        
        
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

        batch_size = int(self.config.get("preemb_generator_batch_size", 64))


        embeddings_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(syntetic_embeddings, labels),
            batch_size=batch_size, shuffle=True
        )

        head = self.model.get_embs_layer() # Final layer, not necessarily linear
        head_params_grad = {}
        head.eval()
        
        for n, p in head.named_parameters():
            head_params_grad[n] = p.requires_grad
            p.requires_grad_(False)

        # Pre-compute per-class best DistributionInfo (avoids repeated work)
        class_info_map = {}  # class_id -> DistributionInfo
        if labels is not None and self.dist_infos is not None:
            for cid in torch.unique(labels).tolist():
                if cid < 0 or cid >= self.config["num_classes"]:
                    continue
                infos = self.dist_infos.get(cid, [])
                if isinstance(infos, list) and len(infos) > 0:
                    class_info_map[cid] = max(infos, key=lambda t: t[1])[0]
                elif hasattr(infos, "dist_type"):
                    class_info_map[cid] = infos

        beta_max  = self.config.get("preemb_beta_max", 0.3)
        n_epochs = int(self.config.get("preemb_generator_epochs", 100))
        history = {
            "loss_total": [],
            "loss_euclidean": [],
            "loss_infoQ": [],
            "loss_dist": [],
            "beta": [],
        }

        best_model = None
        for i in tqdm(range(n_epochs), total=n_epochs):
            epoch_total = 0.0
            epoch_euclidean = 0.0
            epoch_infoq = 0.0
            epoch_dist = 0.0
            n_batches = 0
            for (emb_batch, label_batch) in embeddings_loader:
                noisy_batch = torch.randn(len(emb_batch), noisy_dim, device=device, dtype=dtype)
                noisy_batch = torch.nn.functional.normalize(noisy_batch, p=2, dim=1) # Project onto the sphere
                noisy_batch = noisy_batch.to(device=device, dtype=dtype, non_blocking=True)
                
                emb_batch = emb_batch.to(device=device, dtype=dtype, non_blocking=True)
                label_batch = label_batch.to(device=device, dtype=torch.long, non_blocking=True)

                optimizer.zero_grad()
                t = i / max(n_epochs - 1, 1)
                beta = beta_max * 0.5 * (1.0 - math.cos(math.pi * t))

                noisy_emb = self.noisy_adder(noisy_batch)

                gen_pre_emb = self.generator(emb_batch, noisy_emb)

                if gen_pre_emb.device != emb_batch.device:
                    raise RuntimeError(
                        f"Device mismatch: gen_pre_emb on {gen_pre_emb.device}, emb_batch on {emb_batch.device}"
                    )


                # InfoQ predicts the latent noise code (noisy_dim), not the
                # projected embedding-space perturbation (emb_dim).
                infoQ_loss = F.mse_loss(noisy_batch, infoQ(gen_pre_emb), reduction='mean')
                gen_emb = head(gen_pre_emb)

                # ---- Euclidean reconstruction loss (vectorised) ----
                loss_euclidean = torch.norm(gen_emb - emb_batch, p=2, dim=1)
                loss_euclidean = loss_euclidean.mean()

                # ---- Distribution-specific loss (vectorised per class) -------
                loss_dist = torch.tensor(0.0, device=device, dtype=dtype)
                n_dist = 0
                if class_info_map:
                    for cid, info in class_info_map.items():
                        mask = (label_batch == cid)
                        if not mask.any():
                            continue
                        batch = gen_emb[mask]  # (K, D_emb)
                        nll = -self._loglik_batch(batch, info)  # (K,)
                        loss_dist = loss_dist + nll.sum()
                        n_dist += batch.shape[0]
                    if n_dist > 0:
                        loss_dist = loss_dist / n_dist


                
                loss = loss_euclidean + infoQ_loss + loss_dist

                loss.backward()
                optimizer.step()

                epoch_total += float(loss.detach().item())
                epoch_euclidean += float(loss_euclidean.detach().item())
                epoch_infoq += float(infoQ_loss.detach().item())
                epoch_dist += float(loss_dist.detach().item())
                n_batches += 1

            if n_batches > 0:
                history["loss_total"].append(epoch_total / n_batches)
                history["loss_euclidean"].append(epoch_euclidean / n_batches)
                history["loss_infoQ"].append(epoch_infoq / n_batches)
                history["loss_dist"].append(epoch_dist / n_batches)
                history["beta"].append(float(beta))
            else:
                history["loss_total"].append(float("nan"))
                history["loss_euclidean"].append(float("nan"))
                history["loss_infoQ"].append(float("nan"))
                history["loss_dist"].append(float("nan"))
                history["beta"].append(float(beta))

            if best_loss < epoch_total:
                best_loss = epoch_total
                best_model = {
                    "generator": self.generator.state_dict(),
                    "noisy_adder": self.noisy_adder.state_dict()
                }

            print(f"Epoch {i+1}/{n_epochs} | Total Loss: {epoch_total / n_batches:.6f} | Euclidean Loss: {epoch_euclidean / n_batches:.6f} | InfoQ Loss: {epoch_infoq / n_batches:.6f} | Distribution Loss: {epoch_dist / n_batches:.6f} | Beta: {beta:.6f}", flush=True)
        for n, p in head.named_parameters():
            p.requires_grad_(head_params_grad[n])

        self._inverse_gen_training_history = history

        # self.generator.load_state_dict(best_model["generator"])
        # self.noisy_adder.load_state_dict(best_model["noisy_adder"])

        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad_(False)
        
        self.noisy_adder.eval()
        for param in self.noisy_adder.parameters():
            param.requires_grad_(False)

    def _optimize_preembeddings(self, syntetic_embeddings, labels):
        """Optimise pre-embeddings so they approximate the target embeddings
        while fitting the class distribution.

        Two modes depending on the model architecture:

        * ``extra_layers=True``  – ``get_embs_layer()`` is a Linear.
          z lives in the pre-embedding space (dim = ``in_features``);
          the loss compares ``head(z)`` to x_target.

        * ``extra_layers=False`` – there is no separate linear projection.
          The embedding IS the pre-embedding (same dim).  z is optimised
          directly in embedding space.

        Loss = L_euclid + β · L_dist

          * L_euclid = MSE(z_emb, x)                    – reconstruction
          * L_dist   = −mean_loglik(z_emb | class_dist)  – distribution

        Schedule (cosine):
          * β : 0 → β_max   (distributional gains importance gradually)

        Classification is **not** part of the loss; it is verified
        post-hoc after optimisation and logged for diagnostics.

        Parameters
        ----------
        syntetic_embeddings : Tensor (N, D_emb)
            Target synthetic embeddings.
        labels : Tensor (N,) or None
            Original class ids (0 … num_classes-1).

        Returns
        -------
        Tensor (N, D_z)  – optimised pre-embeddings.
        """
        device = syntetic_embeddings.device
        dtype = syntetic_embeddings.dtype
        n_samples, emb_dim = syntetic_embeddings.shape
        x_target = syntetic_embeddings.detach().clone()


        # --- Determine whether we have a linear head or not ---------------
        head = None
        head_was_training = False
        head_params_grad = {}

        head = self.model.get_embs_layer() # Final layer, not necessarily linear
        
        head.eval()
        
        z_dim = self.model.get_input_dim_of_sequential()
        for n, p in head.named_parameters():
            head_params_grad[n] = p.requires_grad
            p.requires_grad_(False)

        # --- Random initialisation of z -----------------------------------
        if self.generator is None:
            if self.model._base_model == "vit_b_16" and not self.model._extra_layers:
                # ViT espera (Batch, Sequence, Dim) -> (N, 197, 768)
                z = torch.randn(n_samples, 197, z_dim, device=device, dtype=dtype)

            elif (self.model._base_model in ["resnet18", "efficientnet_v2_s"]) and not self.model._extra_layers:
                # These models expect (Batch, Channels, H, W)
                spatial_res = 14 if self.model._base_model == "resnet18" else 12
                z = torch.randn(n_samples, z_dim, spatial_res, spatial_res, device=device, dtype=dtype)

            elif self.model._base_model == "swin_t" and not self.model._extra_layers:
                # Swin-T espera (Batch, H, W, Channels) -> (N, 7, 7, 768)
                z = torch.randn(n_samples, 7, 7, z_dim, device=device, dtype=dtype)

            else:
                # If extra_layers=True the block is linear and takes flat vectors [N, z_dim]
                z = torch.randn(n_samples, z_dim, device=device, dtype=dtype)
        
        else:
            noise = torch.randn(n_samples, max(1, emb_dim * 0.1), device=device, dtype=dtype)
            noise = torch.nn.functional.normalize(noise, p=2, dim=1) # Project onto the sphere
            noise = noise.to(device=device, dtype=dtype, non_blocking=True)
            noisy_emb = self.noisy_adder(noise)

            z = self.generator(syntetic_embeddings, noisy_emb)
            

        z.requires_grad_(True) # Needed to optimise z

        # --- Configurable hyper-parameters --------------------------------
        n_steps   = self.config.get("preemb_optim_steps", 1000)
        lr        = self.config.get("preemb_optim_lr", 5e-3)
        beta_max  = self.config.get("preemb_beta_max", 0.3)

        optimizer = torch.optim.Adam([z], lr=lr)

        # Pre-compute per-class best DistributionInfo (avoids repeated work)
        class_info_map = {}  # class_id -> DistributionInfo
        if labels is not None and self.dist_infos is not None:
            for cid in torch.unique(labels).tolist():
                if cid < 0 or cid >= self.config["num_classes"]:
                    continue
                infos = self.dist_infos.get(cid, [])
                if isinstance(infos, list) and len(infos) > 0:
                    class_info_map[cid] = max(infos, key=lambda t: t[1])[0]
                elif hasattr(infos, "dist_type"):
                    class_info_map[cid] = infos

        loader_z_x = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(z, x_target, labels),
            batch_size=1000, shuffle=False
        )

        for step in range(n_steps):
            total_loss = 0.0
            total_ecuclidean = 0.0
            total_dist = 0.0
            loader_step = 0
            for (z_batch, x_target_batch, labels_batch) in loader_z_x:
                optimizer.zero_grad()

                # Cosine schedule for β: 0 → β_max
                t = step / max(n_steps - 1, 1)
                beta = beta_max * 0.5 * (1.0 - math.cos(math.pi * t))

                # ---- Map z to embedding space ----
                z_emb = head(z_batch)      # Linear: (N, z_dim) → (N, D_emb)

                # ---- Euclidean reconstruction loss (vectorised) ----
                loss_euclidean = torch.norm(z_emb - x_target_batch, p=2, dim=1)
                loss_euclidean = loss_euclidean.mean()

                # ---- Distribution-specific loss (vectorised per class) -------
                loss_dist = torch.tensor(0.0, device=device, dtype=dtype)
                n_dist = 0
                if class_info_map and labels_batch is not None:
                    for cid, info in class_info_map.items():
                        mask = (labels_batch == cid)
                        if not mask.any():
                            continue
                        batch = z_emb[mask]  # (K, D_emb)
                        nll = -self._loglik_batch(batch, info)  # (K,)
                        loss_dist = loss_dist + nll.sum()
                        n_dist += batch.shape[0]
                    if n_dist > 0:
                        loss_dist = loss_dist / n_dist

                loss = loss_euclidean + beta * loss_dist
                total_loss += float(loss.detach().item())
                total_ecuclidean += float(loss_euclidean.detach().item())
                total_dist += float(loss_dist.detach().item())
                loss.backward()
                optimizer.step()
                print(f"Loader step {loader_step} | Loss: {loss.item():.6f} | Euclidean: {loss_euclidean.item():.6f} | Dist: {loss_dist.item():.6f} | Beta: {beta:.6f}", flush=True)
            if step % 50 == 0 or step == n_steps - 1:
                ld = total_dist.item() if isinstance(total_dist, torch.Tensor) else total_dist
                print(
                    f"  PreEmb optim step {step}/{n_steps}: "
                    f"β={beta:.3f} "
                    f"L_euclid={total_ecuclidean/len(loader_z_x):.6f}  "
                    f"L_dist={ld/len(loader_z_x):.6f}  "
                    f"total={total_loss/len(loader_z_x):.6f}"
                )

        # --- Post-hoc classification check (no gradient, diagnostic only) -
        if labels is not None:
            classifier = self.model.classifier
            with torch.no_grad():
                z_emb_final = head(z)
                logits = classifier(z_emb_final)
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                n_correct = (preds == labels).sum().item()
            print(
                f"  PreEmb classification check: "
                f"{n_correct}/{n_samples} correct ({acc*100:.1f}%)"
            )

        # Restore frozen modules

        for n, p in head.named_parameters():
            p.requires_grad_(head_params_grad[n])

        return z.detach()

    def actualizeCentroidsAndCovariance(self):
        """Push pre-embedding centroids/cov through the current linear head."""
        if not self.model._extra_layers:
            return  # Not applicable without linear embedding layer
        if self.pre_centroids is not None:
            with torch.no_grad():
                head = self.model.get_embs_layer()
                W = head.weight.detach()
                b = head.bias.detach()
                if self.config["clustering"]:
                    for i in range(self.config["num_classes"]):
                        for j in range(len(self.pre_centroids[i])):
                            X_pre = self.pre_centroids[i][j].unsqueeze(1)
                            self.centroids[i][j] = (W @ X_pre + b.unsqueeze(1)).squeeze(1)
                else:
                    for i in range(self.config["num_classes"]):
                        X_pre = self.pre_centroids[i].unsqueeze(1)
                        self.centroids[i] = (W @ X_pre + b.unsqueeze(1)).squeeze(1)

        if self.pre_covariance is not None:
            with torch.no_grad():
                head = self.model.get_embs_layer()
                W = head.weight.detach()
                if self.config["clustering"]:
                    for i in range(self.config["num_classes"]):
                        for j in range(len(self.pre_covariance[i])):
                            X_pre = self.pre_covariance[i][j]
                            self.covariance[i][j] = W @ X_pre @ W.t()
                else:
                    for i in range(self.config["num_classes"]):
                        X_pre = self.pre_covariance[i]
                        self.covariance[i] = W @ X_pre @ W.t()

    def actualizeGaussianNoise(self):
        """Re-build MultivariateNormal samplers for Gaussian sub-groups."""
        if self.centroids is None:
            return
        if self.config["clustering"]:
            self.gaussian_noise = {i: [] for i in range(self.config["num_classes"])}
            for i in range(self.config["num_classes"]):
                for j in range(len(self.centroids[i])):
                    self.gaussian_noise[i].append(
                        MultivariateNormal(
                            loc=self.centroids[i][j],
                            covariance_matrix=self.covariance[i][j],
                        )
                    )
        else:
            self.gaussian_noise = {}
            for i in range(self.config["num_classes"]):
                self.gaussian_noise[i] = MultivariateNormal(
                    loc=self.centroids[i],
                    covariance_matrix=self.covariance[i],
                )

    # =====================================================================
    #  Gradient subspace projection helpers
    # =====================================================================

    def _compute_grad_subspaces(self, forget_X, forget_y, transform_forget,
                                retain_X=None, retain_y=None, transform_retain=None):
        """Compute SVD-based projection matrices for gradient subspace restriction.

        Two variants (controlled by config):
          grad_proj_forget_variance (float 0-1): keep top-k forget singular vectors
            capturing this fraction of forget embedding variance.
          grad_proj_retain_ortho (bool): compute retain subspace dimension, express
            as fraction k_retain/emb_dim, use that as variance threshold for forget.

        Results stored in:
          self._grad_proj_classifier_S  (emb_dim, k) – for classifier[-1] gradient
          self._grad_proj_emb_S         (backbone_dim, k) – for embedding layer gradient
        """
        variance_threshold = self.config.get('grad_proj_forget_variance')
        retain_ortho = self.config.get('grad_proj_retain_ortho', False)
        proj_layers = self.config.get('grad_proj_layers', 'embeddings')

        if variance_threshold is None and not retain_ortho:
            return

        device = self.device

        def _svd_subspace(feats_list, threshold):
            """SVD of stacked activations; return top-k orthonormal columns as (dim, k)."""
            A = torch.cat(feats_list, dim=0).float()
            A = A - A.mean(dim=0, keepdim=True)
            if A.shape[0] < 2:
                return None
            try:
                _, S_vals, Vt = torch.linalg.svd(A, full_matrices=False)
            except Exception:
                return None
            energy = (S_vals ** 2).cumsum(0) / (S_vals ** 2).sum().clamp(min=1e-12)
            k = int((energy < threshold).sum().item()) + 1
            k = min(k, Vt.shape[0])
            return Vt[:k].T  # (dim, k)

        # ── Collect forget embeddings (backbone output = input to classifier) ────
        forget_loader = DataLoader(
            ClassificationDataset(forget_X, forget_y, transforms=ConditionalTransform(transform_forget)),
            batch_size=self.config['train_batch_size'],
            shuffle=False, num_workers=4, collate_fn=collate_fn_mixed,
        )

        emb_feats = []        # backbone output (= input to classifier[-1])
        backbone_feats = []   # input to embedding linear (extra_layers only)

        hook_handle = None
        if proj_layers in ('embeddings', 'both') and self.model._extra_layers:
            emb_linear = self.model.get_embs_layer()  # Linear(backbone_dim, emb_size)
            def _hook(module, inp, out, buf=backbone_feats):
                buf.append(inp[0].detach().cpu())
            hook_handle = emb_linear.register_forward_hook(_hook)

        self.model.eval()
        with torch.no_grad():
            for images, _ in tqdm(forget_loader, desc="GradProj: forget embeddings"):
                if not isinstance(images, torch.Tensor):
                    continue
                embs = self.model._model(images.to(device))
                emb_feats.append(embs.cpu())

        if hook_handle is not None:
            hook_handle.remove()

        if not emb_feats:
            print("  [GradProj] No forget embeddings collected — skipping.", flush=True)
            return

        # ── Determine variance threshold ──────────────────────────────────────
        if retain_ortho:
            if not retain_X:
                print("  [GradProj] retain_ortho requires retain data — skipping.", flush=True)
                return
            retain_loader = DataLoader(
                ClassificationDataset(
                    retain_X, retain_y,
                    transforms=ConditionalTransform(transform_retain),
                ),
                batch_size=self.config['train_batch_size'],
                shuffle=False, num_workers=4, collate_fn=collate_fn_mixed,
            )
            retain_emb_feats = []
            with torch.no_grad():
                for images, _ in tqdm(retain_loader, desc="GradProj: retain embeddings"):
                    if not isinstance(images, torch.Tensor):
                        continue
                    embs = self.model._model(images.to(device))
                    retain_emb_feats.append(embs.cpu())

            if not retain_emb_feats:
                print("  [GradProj] No retain embeddings collected — skipping.", flush=True)
                return

            R = torch.cat(retain_emb_feats, dim=0).float()
            R = R - R.mean(dim=0, keepdim=True)
            try:
                _, r_vals, _ = torch.linalg.svd(R, full_matrices=False)
            except Exception:
                print("  [GradProj] SVD of retain embeddings failed — skipping.", flush=True)
                return
            r_energy = (r_vals ** 2).cumsum(0) / (r_vals ** 2).sum().clamp(min=1e-12)
            k_retain = int((r_energy < 0.99).sum().item()) + 1
            k_retain = min(k_retain, len(r_vals))
            emb_dim = R.shape[1]
            threshold = float(k_retain) / emb_dim
            print(
                f"  [GradProj] retain_ortho: k_retain={k_retain}/{emb_dim} "
                f"→ variance_threshold={threshold:.4f}", flush=True
            )
        else:
            threshold = variance_threshold

        # ── Build subspaces ───────────────────────────────────────────────────
        if proj_layers in ('classifier', 'both') or (
            proj_layers == 'embeddings' and not self.model._extra_layers
        ):
            S = _svd_subspace(emb_feats, threshold)
            if S is not None:
                self._grad_proj_classifier_S = S.to(device)
                print(
                    f"  [GradProj] Classifier subspace: k={S.shape[1]}/{S.shape[0]}, "
                    f"threshold={threshold:.4f}", flush=True
                )
            if proj_layers == 'embeddings' and not self.model._extra_layers:
                print(
                    "  [GradProj] 'embeddings' target requested but model has no "
                    "extra linear projection — applied to classifier instead.", flush=True
                )

        if proj_layers in ('embeddings', 'both') and self.model._extra_layers:
            if backbone_feats:
                S = _svd_subspace(backbone_feats, threshold)
                if S is not None:
                    self._grad_proj_emb_S = S.to(device)
                    print(
                        f"  [GradProj] Embedding subspace: k={S.shape[1]}/{S.shape[0]}, "
                        f"threshold={threshold:.4f}", flush=True
                    )

    def _apply_grad_projection(self):
        """Project parameter gradients onto the forget subspace (call after backward()).

        For each configured layer:
          g_proj = (g @ S) @ S.T   keeps only components along S's columns.
        """
        proj_layers = self.config.get('grad_proj_layers', 'embeddings')

        if self._grad_proj_classifier_S is not None and proj_layers in ('classifier', 'both'):
            fc = self.model.classifier[-2]
            if fc.weight.grad is not None:
                S = self._grad_proj_classifier_S
                g = fc.weight.grad.data  # (num_classes, emb_dim)
                fc.weight.grad.data = (g @ S) @ S.T

        if (self._grad_proj_emb_S is not None
                and proj_layers in ('embeddings', 'both')
                and self.model._extra_layers):
            emb_layer = self.model.get_embs_layer()  # Linear when extra_layers=True
            if hasattr(emb_layer, 'weight') and emb_layer.weight.grad is not None:
                S = self._grad_proj_emb_S
                g = emb_layer.weight.grad.data  # (emb_size, backbone_dim)
                emb_layer.weight.grad.data = (g @ S) @ S.T

        # Fallback: 'embeddings' with no extra_layers uses classifier subspace
        if (self._grad_proj_classifier_S is not None
                and proj_layers == 'embeddings'
                and not self.model._extra_layers):
            fc = self.model.classifier[-2]
            if fc.weight.grad is not None:
                S = self._grad_proj_classifier_S
                g = fc.weight.grad.data
                fc.weight.grad.data = (g @ S) @ S.T

    # =====================================================================
    #  Core: fit distributions per class (possibly with recursive clustering)
    # =====================================================================

    def _fit_distributions_for_class(self, embeddings: np.ndarray, class_id: int):
        """
        Apply the distribution-fitting pipeline to one class's embeddings.

        When clustering is enabled, performs recursive subdivision until every
        sub-cluster has an acceptable fit (or hits the minimum size).

        Returns
        -------
        list of (DistributionInfo, np.ndarray-of-indices)
        """
        n, d = embeddings.shape
        _kw = dict(
            covariance_type=self.config.get("covariance_type", "full"),
            n_pca_components=self.config.get("n_pca_components", None),
            model_selection=self.config.get("model_selection", "bic"),
        )

        if not self.config["clustering"]:
            info = fit_best_distribution(embeddings, **_kw)
            return [(info, np.arange(n))]

        # --- Clustering path: recursive subdivision -----------------------
        initial_k = 1
        results = _fit_cluster_recursive(
            embeddings,
            initial_k=initial_k,
            min_cluster_size=max(int(math.sqrt(n)), d),
            dim=d,
            **_kw,
        )
        return results

    def _collect_embeddings(self, dataloader):
        """Run model in eval mode and collect embeddings per class."""
        class_embeddings = {i: [] for i in range(self.config["num_classes"])}
        total_samples = 0
        total_per_class = [0] * self.config["num_classes"]

        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Collecting embeddings"):
                images = images.to(self.model.device).float()
                _, embs = self.model(images, True)
                total_samples += labels.size(0)
                for emb, lbl in zip(embs, labels):
                    c = lbl.item()
                    total_per_class[c] += 1
                    class_embeddings[c].append(emb.cpu().numpy())

        return class_embeddings, total_samples, total_per_class

    def _build_distribution_stats(self, class_embeddings, total_samples, total_per_class):
        """
        For each class, run the distribution-fitting pipeline and populate
        self.dist_infos, self.centroids, self.covariance, self.percentages,
        and self.intra_percentage.
        """
        num_classes = self.config["num_classes"]
        eps = 1e-6

        self.dist_infos = {}
        self.centroids = {i: [] for i in range(num_classes)}
        self.covariance = {i: [] for i in range(num_classes)}
        pct_inter = [0.0] * num_classes
        pct_intra = [[] for _ in range(num_classes)]
        ns_per_class = [[] for _ in range(num_classes)]

        for i in tqdm(range(num_classes), desc="Fitting distributions"):
            embs = np.array(class_embeddings[i])
            if len(embs) == 0:
                self.dist_infos[i] = []
                continue

            pct_inter[i] = total_per_class[i] / total_samples
            results = self._fit_distributions_for_class(embs, i)

            class_infos = []
            for info, idx in results:
                n_cluster = len(idx)
                ns_per_class[i].append(n_cluster)
                intra_frac = n_cluster / max(1, total_per_class[i])
                pct_intra[i].append(intra_frac)
                class_infos.append((info, intra_frac))

                # Build centroid / covariance for update-path compatibility
                cluster_data = embs[idx]
                if info.dist_type == "gaussian":
                    centroid = torch.tensor(info.params["mean"], dtype=torch.float32, device=self.device)
                    cov = torch.tensor(info.params["cov"], dtype=torch.float32, device=self.device)
                elif info.dist_type == "tstudent":
                    centroid = torch.tensor(info.params["mean"], dtype=torch.float32, device=self.device)
                    cov = torch.tensor(info.params["cov"], dtype=torch.float32, device=self.device)
                elif info.dist_type == "vmf":
                    mu = info.params["mu"]
                    norm_dist = info.params.get("norm_dist")
                    if norm_dist is not None and norm_dist.get("type") == "normal":
                        mean_norm = norm_dist["mean"]
                    elif norm_dist is not None and norm_dist.get("type") == "lognormal":
                        # Mean of LogNormal = exp(mu + sigma²/2)
                        mean_norm = math.exp(norm_dist["mu"] + 0.5 * norm_dist["sigma"] ** 2)
                    else:
                        mean_norm = info.params.get("mean_norm", 1.0)
                    centroid = torch.tensor(mu * mean_norm, dtype=torch.float32, device=self.device)
                    # For vMF we keep a rough covariance (isotropic scaled)
                    d = len(mu)
                    cov_np = np.cov(cluster_data, rowvar=False) + eps * np.eye(d)
                    cov = torch.tensor(cov_np, dtype=torch.float32, device=self.device)
                elif info.dist_type == "kent":
                    mu = info.params["mu"]
                    norm_dist = info.params.get("norm_dist")
                    if norm_dist is not None and norm_dist.get("type") == "normal":
                        mean_norm = norm_dist["mean"]
                    elif norm_dist is not None and norm_dist.get("type") == "lognormal":
                        mean_norm = math.exp(norm_dist["mu"] + 0.5 * norm_dist["sigma"] ** 2)
                    else:
                        mean_norm = info.params.get("mean_norm", 1.0)
                    centroid = torch.tensor(mu * mean_norm, dtype=torch.float32, device=self.device)
                    # For Kent we keep a rough covariance (from the data)
                    d = len(mu)
                    cov_np = np.cov(cluster_data, rowvar=False) + eps * np.eye(d)
                    cov = torch.tensor(cov_np, dtype=torch.float32, device=self.device)
                elif info.dist_type == "laplace":
                    centroid = torch.tensor(info.params["locs"], dtype=torch.float32, device=self.device)
                    d = len(info.params["locs"])
                    # Laplace var = 2 * b^2
                    var_diag = 2.0 * info.params["scales"] ** 2
                    cov = torch.tensor(np.diag(var_diag) + eps * np.eye(d), dtype=torch.float32, device=self.device)
                elif info.dist_type == "uniform":
                    mid = (info.params["mins"] + info.params["maxs"]) / 2.0
                    centroid = torch.tensor(mid, dtype=torch.float32, device=self.device)
                    d = len(mid)
                    widths = info.params["maxs"] - info.params["mins"]
                    var_diag = widths ** 2 / 12.0  # Uniform variance
                    cov = torch.tensor(np.diag(var_diag) + eps * np.eye(d), dtype=torch.float32, device=self.device)
                else:
                    centroid = torch.tensor(cluster_data.mean(axis=0), dtype=torch.float32, device=self.device)
                    d = cluster_data.shape[1]
                    cov = torch.tensor(np.cov(cluster_data, rowvar=False) + eps * np.eye(d),
                                       dtype=torch.float32, device=self.device)

                self.centroids[i].append(centroid)
                self.covariance[i].append(cov)

            self.dist_infos[i] = class_infos

        # Convert to non-clustering format when clustering is disabled
        if not self.config["clustering"]:
            for i in range(num_classes):
                if len(self.centroids[i]) == 1:
                    self.centroids[i] = self.centroids[i][0]
                    self.covariance[i] = self.covariance[i][0]

        self.percentages = torch.tensor(pct_inter, dtype=torch.float32)
        self.intra_percentage = [torch.tensor(p, dtype=torch.float32) for p in pct_intra]

        # num_samples_original
        if self.config["clustering"]:
            max_k = max((len(lst) for lst in ns_per_class), default=0)
            ns_tensor = torch.zeros(
                (num_classes, max(max_k, 1)), dtype=torch.float32, device=self.device
            )
            for i, lst in enumerate(ns_per_class):
                for j, val in enumerate(lst):
                    ns_tensor[i, j] = float(val)
            self.num_samples_original = ns_tensor
        else:
            self.num_samples_original = torch.tensor(
                [total_per_class[i] for i in range(num_classes)],
                dtype=torch.float32,
                device=self.device,
            )

    # =====================================================================
    #  Synthetic embedding generation  (distribution-aware)
    # =====================================================================

    def generate_syntetic_embeddings(self, n_samples, forbidden_labels):
        """
        Generate synthetic embeddings using the fitted distributions.

        Batches all samples that share the same (class, sub-group) and
        generates them in a single call to sample_from_distribution,
        which uses GPU-native PyTorch distributions for massive speedup.
        """
        if self.percentages is None or self.dist_infos is None:
            raise RuntimeError(
                "_build_distribution_stats must be called before generate_syntetic_embeddings()"
            )

        seed = self.config["seed"]
        probs = self.percentages.to(self.device).float()
        rng = np.random.default_rng(seed)

        # Assign class labels (filtering forbidden)
        syntetic_labels = []
        while len(syntetic_labels) < n_samples:
            torch.manual_seed(seed)
            candidates = torch.multinomial(
                probs, num_samples=n_samples - len(syntetic_labels), replacement=True
            )
            for lbl in candidates.tolist():
                if lbl not in forbidden_labels:
                    syntetic_labels.append(lbl)
            seed += 1

        syntetic_labels = torch.tensor(syntetic_labels, dtype=torch.long, device=self.device)

        # Group samples by (class_id, sub_group) for batched generation
        sample_groups = {}  # (class_id, sub_idx) -> list of output positions
        for pos, class_id in enumerate(syntetic_labels.tolist()):
            class_infos = self.dist_infos[class_id]
            if self.config["clustering"] and len(class_infos) > 1:
                intra_probs = self.intra_percentage[class_id].to(self.device).float()
                sub_idx = torch.multinomial(intra_probs, num_samples=1).item()
            else:
                sub_idx = 0
            key = (class_id, sub_idx)
            if key not in sample_groups:
                sample_groups[key] = []
            sample_groups[key].append(pos)

        # Determine embedding dimension from first available distribution
        emb_dim = None
        for cid in range(self.config["num_classes"]):
            infos = self.dist_infos.get(cid, [])
            if infos:
                info0, _ = infos[0]
                p = info0.params
                for k in ("mean", "mu", "locs", "mins"):
                    if k in p:
                        emb_dim = len(p[k])
                        break
            if emb_dim is not None:
                break

        # Batch-sample per group and scatter into pre-allocated output
        syntetic_embeddings = torch.empty(
            n_samples, emb_dim, dtype=torch.float32, device=self.device
        )
        for (class_id, sub_idx), positions in sample_groups.items():
            info, _ = self.dist_infos[class_id][sub_idx]
            n_group = len(positions)
            samples = sample_from_distribution(
                info, n_group, device=self.device, rng=rng
            )
            syntetic_embeddings[positions] = samples

        return syntetic_embeddings, syntetic_labels + self.config["num_classes"]

    # =====================================================================
    #  Logit-profile regularizer (retain-only)
    # =====================================================================

    def _compute_logit_profile_reference(self, forbidden_labels, n_samples=None):
        """Snapshot the per-retain-class mean classifier logits/softmax on a
        batch of freshly generated synthetic samples.

        This is computed once, in setup(), before any unlearning step, and
        is used as the fixed reference profile for the logit-profile
        regularizer (see --logit-profile-lambda / --logit-profile-mode):
        during training the logits (or softmax) that the model assigns to
        synthetic samples of a retain class are pulled towards this
        reference so the classifier's output shape for that class does not
        drift while unlearning the forget classes.
        """
        if n_samples is None:
            n_samples = self.config.get("syntetic_samples", 500)
            if n_samples is None or n_samples == -1:
                n_samples = 500

        ref_embeddings, ref_labels = self.generate_syntetic_embeddings(n_samples, forbidden_labels)
        orig_labels = ref_labels - self.config["num_classes"]

        num_classes = self.config["num_classes"]
        ref_logits = torch.zeros(num_classes, num_classes, device=self.device)
        ref_softmax = torch.zeros(num_classes, num_classes, device=self.device)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            logits = self.model.classifier(ref_embeddings)
            probs = F.softmax(logits, dim=1)
            for cid in orig_labels.unique().tolist():
                mask = orig_labels == cid
                ref_logits[cid] = logits[mask].mean(dim=0)
                ref_softmax[cid] = probs[mask].mean(dim=0)
        if was_training:
            self.model.train()

        self._logit_profile_ref_logits = ref_logits
        self._logit_profile_ref_softmax = ref_softmax
        print(
            f"  [LogitProfile] Reference profile computed from {n_samples} "
            f"originally generated synthetic retain samples.", flush=True
        )

    def _logit_profile_loss(self, logits, orig_labels):
        """MSE between the current logits/softmax on synthetic retain
        samples and the per-class reference profile computed in
        ``_compute_logit_profile_reference``.

        ``orig_labels`` must already be the original (un-shifted) class ids
        and must only contain retain classes — the caller is responsible
        for that, since this regularizer is retain-only by design.
        """
        if self._logit_profile_ref_logits is None or logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device)
        mode = self.config.get("logit_profile_mode", "logits")
        if mode == "softmax":
            values = F.softmax(logits, dim=1)
            reference = self._logit_profile_ref_softmax[orig_labels]
        else:
            values = logits
            reference = self._logit_profile_ref_logits[orig_labels]
        return F.mse_loss(values, reference)

    # =====================================================================
    #  Post-hoc forget-temperature calibration
    # =====================================================================

    def _calibrate_forget_temperature(self, val_loader_forget):
        """Calibrate the classifier's TemperatureScaling so that, at
        inference, logits on *real* forgotten-class samples match, on
        average, the logits the classifier was actually trained against for
        that class: the synthetic forgotten-class samples.

        This direction matters: only real images are ever fed to the model
        at inference/evaluation time (synthetic embeddings are purely a
        training-time device), so it's the real-image logits that must be
        rescaled to the classifier's synthetic-trained scale — not the
        other way round.

        Called once, after the unlearning loop finishes (see
        --forget-temperature-calibration / --forget-temperature-always-apply).
        Never called when self_forget=True: that mode never sees real
        forgotten-class data to calibrate against.

        With --forget-temperature-monotonic-decrease, the newly calibrated
        temperature only replaces the current one (already loaded from
        checkpoint_best.pt / carried over from a previous ICU round) when it
        is strictly lower — i.e. a stronger correction. This guards against a
        single noisy round (small real forget sample) undoing the more
        aggressive calibration accumulated over previous rounds.

        Uses ``val_loader_forget`` (the held-out real forget split already
        built in ``unlearn()``) — note this can be a small sample for some
        datasets (e.g. ~50 images/class on CIFAR-100), so the resulting
        mean logit is noisier than it would be with more data.

        The classifier's TemperatureScaling divides logits by ``T``
        (``x / T``); to make real logits look like ``scale * real`` at
        inference we therefore store ``T = 1 / scale``.
        """
        forbidden_labels = self.config["forbidden_labels"]
        if not forbidden_labels:
            print("  [ForgetTempCalib] No forbidden labels — skipping.", flush=True)
            return

        temp_module = self.model.classifier[-1]
        was_active = bool(temp_module.active.item())
        was_training = self.model.training
        current_temperature = temp_module.temperature.data.clone()

        # Compare *raw* logits (bypass any temperature already applied).
        temp_module.active.fill_(False)
        self.model.eval()

        num_classes = self.config["num_classes"]
        label_to_pos = {lbl: i for i, lbl in enumerate(forbidden_labels)}
        real_sums = torch.zeros(len(forbidden_labels), num_classes, device=self.device)
        real_counts = torch.zeros(len(forbidden_labels), device=self.device)
        with torch.no_grad():
            for images, target in val_loader_forget:
                images = images.to(self.device)
                target = target.to(self.device)
                logits = self.model(images)
                for lbl, pos in label_to_pos.items():
                    mask = target == lbl
                    if mask.any():
                        real_sums[pos] += logits[mask].sum(dim=0)
                        real_counts[pos] += mask.sum()

        if not (real_counts > 0).any():
            print("  [ForgetTempCalib] No real forgotten-class samples found — skipping.", flush=True)
            temp_module.active.fill_(was_active)
            if was_training:
                self.model.train()
            return

        n_synth = max(self.config.get("forget_syntetic_samples", 500), 50 * len(forbidden_labels))
        synth_embeddings, synth_labels = self.generate_syntetic_embeddings(
            n_synth, self.config["retain_labels"]
        )
        synth_orig_labels = synth_labels - num_classes
        synth_sums = torch.zeros_like(real_sums)
        synth_counts = torch.zeros_like(real_counts)
        with torch.no_grad():
            synth_logits = self.model.classifier(synth_embeddings)
        for lbl, pos in label_to_pos.items():
            mask = synth_orig_labels == lbl
            if mask.any():
                synth_sums[pos] += synth_logits[mask].sum(dim=0)
                synth_counts[pos] += mask.sum()

        valid = (real_counts > 0) & (synth_counts > 0)
        if not valid.any():
            print("  [ForgetTempCalib] No overlapping real/synthetic forgotten classes — skipping.", flush=True)
            temp_module.active.fill_(was_active)
            if was_training:
                self.model.train()
            return

        real_mean = (real_sums[valid] / real_counts[valid].unsqueeze(1)).flatten()
        synth_mean = (synth_sums[valid] / synth_counts[valid].unsqueeze(1)).flatten()

        # Least-squares scalar minimising ||scale * real_mean - synth_mean||^2
        # (real logits are the ones rescaled at inference, towards the
        # synthetic-trained scale — see docstring).
        denom = torch.dot(real_mean, real_mean).clamp_min(1e-8)
        scale_raw = torch.dot(real_mean, synth_mean) / denom

        # Safety clamp: a handful of real forget samples can make this
        # least-squares fit noisy, occasionally blowing up (or collapsing
        # towards 0, e.g. when dot(real_mean, synth_mean) <= 0), which then
        # massively amplifies every logit at inference (softmax/entropy
        # overflow risk downstream). Bound scale so the resulting
        # temperature = 1/scale stays within [MIN_TEMPERATURE, MAX_TEMPERATURE].
        # clamp() alone would leave a NaN/inf scale_raw as NaN, so fall back
        # to scale=1.0 (no correction) first in that case.
        MIN_TEMPERATURE, MAX_TEMPERATURE = 0.001, 100.0
        min_scale, max_scale = 1.0 / MAX_TEMPERATURE, 1.0 / MIN_TEMPERATURE
        if not torch.isfinite(scale_raw):
            print(
                f"  [ForgetTempCalib] scale_raw is non-finite ({scale_raw.item()}) "
                "— falling back to scale=1.0 (no correction).", flush=True
            )
            scale_raw = torch.ones_like(scale_raw)
        scale = scale_raw.clamp(min_scale, max_scale)
        if abs(scale_raw.item() - scale.item()) > 1e-6:
            print(
                f"  [ForgetTempCalib] scale_raw={scale_raw.item():.4f} clamped to "
                f"{scale.item():.4f} (safety range for temperature: "
                f"[{MIN_TEMPERATURE}, {MAX_TEMPERATURE}]).", flush=True
            )

        new_temperature = (1.0 / scale).view(1)
        monotonic_decrease = self.config.get("forget_temperature_monotonic_decrease", False)
        if monotonic_decrease and new_temperature.item() >= current_temperature.item():
            print(
                f"  [ForgetTempCalib] New temperature ({new_temperature.item():.4f}) is not "
                f"lower than the current one ({current_temperature.item():.4f}) — "
                "keeping the current temperature (--forget-temperature-monotonic-decrease).",
                flush=True,
            )
            temp_module.temperature.data = current_temperature
        else:
            temp_module.temperature.data = new_temperature

        always_apply = self.config.get("forget_temperature_always_apply", True)
        temp_module.active.fill_(always_apply)

        if was_training:
            self.model.train()

        print(
            f"  [ForgetTempCalib] scale={scale.item():.4f}  "
            f"new_temperature={new_temperature.item():.4f}  "
            f"temperature={temp_module.temperature.item():.4f}  "
            f"active={bool(temp_module.active.item())}",
            flush=True,
        )

    # =====================================================================
    #  Distribution-aware distance loss
    # =====================================================================

    def _get_all_dist_types(self):
        """Return set of distribution types across all classes/sub-groups,
        treating 'vmf' and 'kent' as a single family 'vmf_kent'."""
        types = set()
        for c in range(self.config["num_classes"]):
            infos = self.dist_infos[c]
            if not isinstance(infos, list):
                infos = [(infos, 1.0)]
            for info, _ in infos:
                t = info.dist_type
                if t in ("vmf", "kent"):
                    types.add("vmf_kent")
                else:
                    types.add(t)
        return types

    # ---------- per-distribution log-likelihood (torch, differentiable) ---

    @staticmethod
    def _loglik_gaussian(x, info):
        """Normalized (per-dimension) log-likelihood under a Gaussian."""
        mean = torch.as_tensor(info.params["mean"], dtype=x.dtype, device=x.device)
        cov = torch.as_tensor(info.params["cov"], dtype=x.dtype, device=x.device)
        d = mean.shape[0]
        diff = x - mean
        cov_inv = torch.inverse(cov)
        maha = diff @ cov_inv @ diff
        sign, logdet = torch.linalg.slogdet(cov)
        ll = -0.5 * (maha + logdet + d * math.log(2.0 * math.pi))
        return ll / d  # normalize by dimensionality

    @staticmethod
    def _loglik_tstudent(x, info):
        """Normalized log-likelihood under a multivariate t-Student."""
        mean = torch.as_tensor(info.params["mean"], dtype=x.dtype, device=x.device)
        cov = torch.as_tensor(info.params["cov"], dtype=x.dtype, device=x.device)
        df = info.params["df"]
        d = mean.shape[0]
        diff = x - mean
        cov_inv = torch.inverse(cov)
        maha = diff @ cov_inv @ diff
        sign, logdet = torch.linalg.slogdet(cov)
        ll = (
            torch.lgamma(torch.tensor((df + d) / 2.0, device=x.device))
            - torch.lgamma(torch.tensor(df / 2.0, device=x.device))
            - 0.5 * d * math.log(df * math.pi)
            - 0.5 * logdet
            - ((df + d) / 2.0) * torch.log(1.0 + maha / df)
        )
        return ll / d

    @staticmethod
    def _loglik_vmf_kent(x, info):
        """Normalized log-likelihood under vMF or Kent (directional + radial)."""
        d = x.shape[0]
        mu = torch.as_tensor(info.params["mu"], dtype=x.dtype, device=x.device)
        kappa = info.params["kappa"]

        # Directional component
        x_norm = torch.norm(x)
        x_unit = x / (x_norm + 1e-12)
        dot = x_unit @ mu
        log_C = _vmf_log_norm_const(d, kappa)
        ll_dir = log_C + kappa * dot
        ll_constant = (d-1)*torch.log(x_norm + 1e-12)  # Jacobian from directional to Cartesian
        # Radial component
        norm_dist = info.params.get("norm_dist")
        ll_rad = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if norm_dist is not None:
            if norm_dist["type"] == "normal":
                mu_r = norm_dist["mean"]
                sigma_r = norm_dist["std"]
                ll_rad = -0.5 * ((x_norm - mu_r) / sigma_r) ** 2 - math.log(sigma_r) - 0.5 * math.log(2 * math.pi)
            elif norm_dist["type"] == "lognormal":
                mu_ln = norm_dist["mu"]
                sigma_ln = norm_dist["sigma"]
                log_r = torch.log(x_norm + 1e-12)
                ll_rad = -0.5 * ((log_r - mu_ln) / sigma_ln) ** 2 - log_r - math.log(sigma_ln) - 0.5 * math.log(2 * math.pi)
            elif norm_dist["type"] == "gamma":
                shape = norm_dist["shape"]
                loc = norm_dist.get("loc", 0.0)
                scale = norm_dist["scale"]
                y = x_norm - loc
                ll_rad = (shape - 1.0) * torch.log(y + 1e-12) - y / scale - shape * math.log(scale) - float(gammaln(shape))

        ll = ll_dir + ll_rad - ll_constant
        return ll / d

    @staticmethod
    def _loglik_laplace(x, info):
        """Normalized log-likelihood under product-of-independent Laplace."""
        locs = torch.as_tensor(info.params["locs"], dtype=x.dtype, device=x.device)
        scales = torch.as_tensor(info.params["scales"], dtype=x.dtype, device=x.device)
        d = locs.shape[0]
        ll = -torch.sum(torch.log(2.0 * scales) + torch.abs(x - locs) / scales)
        return ll / d

    @staticmethod
    def _loglik_uniform(x, info):
        """Normalized log-likelihood under axis-aligned uniform."""
        mins = torch.as_tensor(info.params["mins"], dtype=x.dtype, device=x.device)
        maxs = torch.as_tensor(info.params["maxs"], dtype=x.dtype, device=x.device)
        d = mins.shape[0]
        widths = maxs - mins
        widths = torch.clamp(widths, min=1e-12)
        # If x is inside the box, ll = -sum(log(widths)); outside -> penalize
        inside = (x >= mins) & (x <= maxs)
        log_vol = torch.sum(torch.log(widths))
        # Soft penalty for points outside the bounding box
        margin = torch.sum(
            torch.relu(mins - x) / widths + torch.relu(x - maxs) / widths
        )
        ll = -log_vol - margin
        return ll / d

    def _loglik(self, x, info):
        """Dispatch normalized log-likelihood by distribution type."""
        dt = info.dist_type
        if dt == "gaussian":
            return self._loglik_gaussian(x, info)
        elif dt == "tstudent":
            return self._loglik_tstudent(x, info)
        elif dt in ("vmf", "kent"):
            return self._loglik_vmf_kent(x, info)
        elif dt == "laplace":
            return self._loglik_laplace(x, info)
        elif dt == "uniform":
            return self._loglik_uniform(x, info)
        else:
            return self._loglik_gaussian(x, info)

    # ---------- Vectorised (batch) log-likelihood methods ----------------

    @staticmethod
    def _loglik_gaussian_batch(x, info):
        """Batch log-likelihood under Gaussian.  x: (B, D) -> (B,)."""
        mean = torch.as_tensor(info.params["mean"], dtype=x.dtype, device=x.device)
        cov = torch.as_tensor(info.params["cov"], dtype=x.dtype, device=x.device)
        d = mean.shape[0]
        diff = x - mean.unsqueeze(0)                    # (B, D)
        cov_inv = torch.inverse(cov)                    # (D, D)
        maha = (diff @ cov_inv * diff).sum(dim=1)       # (B,)
        _, logdet = torch.linalg.slogdet(cov)
        ll = -0.5 * (maha + logdet + d * math.log(2.0 * math.pi))
        return ll / d

    @staticmethod
    def _loglik_tstudent_batch(x, info):
        """Batch log-likelihood under multivariate t-Student.  x: (B, D) -> (B,)."""
        mean = torch.as_tensor(info.params["mean"], dtype=x.dtype, device=x.device)
        cov = torch.as_tensor(info.params["cov"], dtype=x.dtype, device=x.device)
        df = info.params["df"]
        d = mean.shape[0]
        diff = x - mean.unsqueeze(0)
        cov_inv = torch.inverse(cov)
        maha = (diff @ cov_inv * diff).sum(dim=1)       # (B,)
        _, logdet = torch.linalg.slogdet(cov)
        ll = (
            torch.lgamma(torch.tensor((df + d) / 2.0, device=x.device))
            - torch.lgamma(torch.tensor(df / 2.0, device=x.device))
            - 0.5 * d * math.log(df * math.pi)
            - 0.5 * logdet
            - ((df + d) / 2.0) * torch.log(1.0 + maha / df)
        )
        return ll / d

    @staticmethod
    def _loglik_vmf_kent_batch(x, info):
        """Batch log-likelihood under vMF/Kent.  x: (B, D) -> (B,)."""
        d = x.shape[1]
        mu = torch.as_tensor(info.params["mu"], dtype=x.dtype, device=x.device)
        kappa = info.params["kappa"]
        x_norms = torch.norm(x, dim=1, keepdim=True).clamp(min=1e-12)  # (B,1)
        
        x_unit = x / x_norms                                           # (B,D)
        dots = x_unit @ mu                                              # (B,)
        log_C = _vmf_log_norm_const(d, kappa)
        ll_dir = log_C + kappa * dots                                   # (B,)
        ll_rad = torch.zeros_like(ll_dir)
        norm_dist = info.params.get("norm_dist")
        if norm_dist is not None:
            r = x_norms.squeeze(1)                                      # (B,)
            ll_normalization = (d-1)*torch.log(r + 1e-12)  # Jacobian term for radial component
            if norm_dist["type"] == "normal":
                mu_r, sigma_r = norm_dist["mean"], norm_dist["std"]
                ll_rad = -0.5 * ((r - mu_r) / sigma_r) ** 2 - math.log(sigma_r) - 0.5 * math.log(2 * math.pi)
            elif norm_dist["type"] == "lognormal":
                mu_ln, sigma_ln = norm_dist["mu"], norm_dist["sigma"]
                log_r = torch.log(r + 1e-12)
                ll_rad = -0.5 * ((log_r - mu_ln) / sigma_ln) ** 2 - log_r - math.log(sigma_ln) - 0.5 * math.log(2 * math.pi)
            elif norm_dist["type"] == "gamma":
                shape, scale = norm_dist["shape"], norm_dist["scale"]
                loc = norm_dist.get("loc", 0.0)
                y = (r - loc).clamp(min=1e-12)   # clamp to avoid log(negative) → NaN
                ll_rad = (shape - 1.0) * torch.log(y) - y / scale - shape * math.log(scale) - float(gammaln(shape))
        return (ll_dir + ll_rad - ll_normalization) / d

    @staticmethod
    def _loglik_laplace_batch(x, info):
        """Batch log-likelihood under product-of-Laplace.  x: (B, D) -> (B,)."""
        locs = torch.as_tensor(info.params["locs"], dtype=x.dtype, device=x.device)
        scales = torch.as_tensor(info.params["scales"], dtype=x.dtype, device=x.device)
        d = locs.shape[0]
        ll = -torch.sum(torch.log(2.0 * scales).unsqueeze(0) + torch.abs(x - locs.unsqueeze(0)) / scales.unsqueeze(0), dim=1)
        return ll / d

    @staticmethod
    def _loglik_uniform_batch(x, info):
        """Batch log-likelihood under axis-aligned uniform.  x: (B, D) -> (B,)."""
        mins = torch.as_tensor(info.params["mins"], dtype=x.dtype, device=x.device)
        maxs = torch.as_tensor(info.params["maxs"], dtype=x.dtype, device=x.device)
        d = mins.shape[0]
        widths = (maxs - mins).clamp(min=1e-12)
        log_vol = torch.sum(torch.log(widths))
        margin = torch.sum(
            torch.relu(mins.unsqueeze(0) - x) / widths.unsqueeze(0)
            + torch.relu(x - maxs.unsqueeze(0)) / widths.unsqueeze(0),
            dim=1,
        )
        return (-log_vol - margin) / d

    def _loglik_batch(self, x, info):
        """Dispatch batch log-likelihood.  x: (B, D) -> (B,)."""
        dt = info.dist_type
        if dt == "gaussian":
            return self._loglik_gaussian_batch(x, info)
        elif dt == "tstudent":
            return self._loglik_tstudent_batch(x, info)
        elif dt in ("vmf", "kent"):
            return self._loglik_vmf_kent_batch(x, info)
        elif dt == "laplace":
            return self._loglik_laplace_batch(x, info)
        elif dt == "uniform":
            return self._loglik_uniform_batch(x, info)
        else:
            return self._loglik_gaussian_batch(x, info)

    # ---------- type-specific distances (for homogeneous case) -----------

    @staticmethod
    def _dist_mahalanobis(x, info):
        """Standard Mahalanobis distance for Gaussian."""
        mean = torch.as_tensor(info.params["mean"], dtype=x.dtype, device=x.device)
        cov = torch.as_tensor(info.params["cov"], dtype=x.dtype, device=x.device)
        diff = x - mean
        cov_inv = torch.inverse(cov)
        return torch.sqrt(torch.clamp(diff @ cov_inv @ diff, min=1e-12))

    @staticmethod
    def _dist_pseudo_mahalanobis_t(x, info):
        """Pseudo-Mahalanobis distance for multivariate t-Student.

        Uses sqrt((df+d)/df * (x-mu)^T Sigma^{-1} (x-mu)) which accounts
        for the heavier tails via the degrees-of-freedom scaling.
        """
        mean = torch.as_tensor(info.params["mean"], dtype=x.dtype, device=x.device)
        cov = torch.as_tensor(info.params["cov"], dtype=x.dtype, device=x.device)
        df = info.params["df"]
        d = mean.shape[0]
        diff = x - mean
        cov_inv = torch.inverse(cov)
        maha_sq = diff @ cov_inv @ diff
        # Scale by (df+d)/df so that heavier tails -> shorter effective distance
        return torch.sqrt(torch.clamp((df + d) / df * maha_sq, min=1e-12))

    @staticmethod
    def _dist_vmf_kent(x, info):
        """Distance for vMF/Kent distributions.

        1. Rotate the point so that mu aligns with the north pole (e1)
           via Householder reflection.
        2. Stereographic projection to the tangent plane R^{d-1}.
        3. Mahalanobis distance in tangent plane:
           - Kent: use stored tangent_cov
           - vMF: use identity * (1/kappa)
        4. Add the norm distance contribution:
           - normal norm_dist -> Mahalanobis (1D)
           - lognormal norm_dist -> log, then Mahalanobis
           - gamma norm_dist -> Itakura-Saito distance
        """
        d = x.shape[0]
        mu_np = info.params["mu"]
        kappa = info.params["kappa"]
        norm_dist = info.params.get("norm_dist")

        mu_t = torch.as_tensor(mu_np, dtype=x.dtype, device=x.device)

        # Normalize to unit sphere
        x_norm = torch.norm(x)
        x_unit = x / (x_norm + 1e-12)

        # Step 1: Householder rotation to align mu with e1
        e1 = torch.zeros(d, dtype=x.dtype, device=x.device)
        e1[0] = 1.0
        u = e1 - mu_t
        u_norm_val = torch.norm(u)
        if u_norm_val > 1e-12:
            u = u / u_norm_val
            x_rotated = x_unit - 2.0 * (u @ x_unit) * u
        else:
            x_rotated = x_unit

        # Step 2: Check for antipodal singularity.
        # The stereographic projection from the south pole maps to infinity,
        # so a point at or near x0 = -1 has infinite distance.  We return
        # a large finite sentinel (1e12) to keep the loss differentiable
        # while correctly signalling "as far as possible".
        x0 = x_rotated[0]

        if x0 < -1.0 + 1e-3:
            directional_dist = torch.tensor(1e12, device=x.device, dtype=x.dtype)
        else:
            # Normal path: stereographic projection
            denom = 1.0 + x0
            denom = torch.clamp(denom, min=1e-12)
            x_tangent = x_rotated[1:] / denom  # (d-1,)

            # Step 3: Mahalanobis in tangent plane
            p = d - 1
            if info.dist_type == "kent" and "tangent_cov" in info.params:
                tangent_cov = torch.as_tensor(
                    info.params["tangent_cov"], dtype=x.dtype, device=x.device
                )
                tangent_mean = torch.as_tensor(
                    info.params["tangent_mean"], dtype=x.dtype, device=x.device
                )
                diff_t = x_tangent - tangent_mean
                cov_inv_t = torch.inverse(tangent_cov)
            else:
                # vMF: isotropic -> covariance = (1/kappa) * I
                diff_t = x_tangent  # mean in tangent plane is 0 for vMF
                cov_inv_t = kappa * torch.eye(p, dtype=x.dtype, device=x.device)

            directional_dist = torch.sqrt(
                torch.clamp(diff_t @ cov_inv_t @ diff_t, min=1e-12)
            )

        # Step 4: Norm distance
        norm_contribution = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if norm_dist is not None:
            if norm_dist["type"] == "normal":
                mu_r = norm_dist["mean"]
                sigma_r = max(norm_dist["std"], 1e-12)
                norm_contribution = torch.abs(x_norm - mu_r) / sigma_r
            elif norm_dist["type"] == "lognormal":
                mu_ln = norm_dist["mu"]
                sigma_ln = max(norm_dist["sigma"], 1e-12)
                log_r = torch.log(x_norm + 1e-12)
                norm_contribution = torch.abs(log_r - mu_ln) / sigma_ln
            elif norm_dist["type"] == "gamma":
                # Itakura-Saito distance: d_IS(x, mu) = x/mu - log(x/mu) - 1
                shape = norm_dist["shape"]
                scale = norm_dist["scale"]
                mean_r = norm_dist.get("mean", shape * scale)
                ratio = x_norm / (mean_r + 1e-12)
                norm_contribution = ratio - torch.log(ratio + 1e-12) - 1.0

        return directional_dist + norm_contribution

    @staticmethod
    def _dist_manhattan_weighted(x, info):
        """Weighted Manhattan distance for Laplace distribution.

        d(x, loc) = sum_j |x_j - loc_j| / scale_j
        """
        locs = torch.as_tensor(info.params["locs"], dtype=x.dtype, device=x.device)
        scales = torch.as_tensor(info.params["scales"], dtype=x.dtype, device=x.device)
        return torch.sum(torch.abs(x - locs) / scales)

    @staticmethod
    def _dist_chebyshev(x, info):
        """Chebyshev distance for Uniform distribution.

        d(x, box) = max_j |x_j - center_j| / half_width_j
        """
        mins = torch.as_tensor(info.params["mins"], dtype=x.dtype, device=x.device)
        maxs = torch.as_tensor(info.params["maxs"], dtype=x.dtype, device=x.device)
        center = (mins + maxs) / 2.0
        half_width = (maxs - mins) / 2.0
        half_width = torch.clamp(half_width, min=1e-12)
        z = torch.abs(x - center) / half_width
        valid = z[~torch.isnan(z)]
        if valid.numel() == 0:
            return torch.tensor(float("nan"), dtype=z.dtype, device=z.device)
        return torch.max(valid)

    def _typed_distance(self, x, info):
        """Dispatch distance by distribution type (for homogeneous case)."""
        dt = info.dist_type
        if dt == "gaussian":
            return self._dist_mahalanobis(x, info)
        elif dt == "tstudent":
            return self._dist_pseudo_mahalanobis_t(x, info)
        elif dt in ("vmf", "kent"):
            return self._dist_vmf_kent(x, info)
        elif dt == "laplace":
            return self._dist_manhattan_weighted(x, info)
        elif dt == "uniform":
            return self._dist_chebyshev(x, info)
        else:
            return self._dist_mahalanobis(x, info)

    # ---------- main loss ------------------------------------------------

    def minimize_distances(self, embeddings, labels):
        """Distribution-aware forget loss.

        If all classes share the same distribution family, uses the
        canonical distance for that family. Otherwise, falls back to
        normalized log-likelihood as a universal measure.
        """
        dist_types = self._get_all_dist_types()
        homogeneous = len(dist_types) == 1
        common_type = next(iter(dist_types)) if homogeneous else None

        loss = torch.tensor(0.0, device=self.device)
        batch_size = embeddings.size(0)

        for i in range(batch_size):
            emb = embeddings[i]
            class_distances = []

            for c in range(self.config["num_classes"]):
                if c in self.config['forbidden_labels']: # NOOOOOO
                    continue

                infos = self.dist_infos[c]
                if not isinstance(infos, list):
                    infos = [(infos, 1.0)]

                for info, intra_frac in infos:
                    weight = self.percentages[c]
                    if self.config["clustering"]:
                        weight = weight * intra_frac

                    if homogeneous:
                        d = self._typed_distance(emb, info)
                        class_distances.append(d * weight)
                    else:
                        # Negative normalized log-likelihood as distance
                        nll = -self._loglik(emb, info)
                        class_distances.append(nll * weight)

            loss += self._nanmin_compat(torch.stack(class_distances))

        return loss / batch_size

    def maximize_loglikelihood(self, embeddings, labels):
        """Log-likelihood-based forget loss.

        For each forget sample, compute the normalised log-likelihood to
        every non-own-class distribution (including sub-clusters) and keep
        the **maximum** (best fit).  The returned loss is the negative of
        that maximum so that minimising it pushes the embeddings toward
        the best-fitting retain-class distribution.

        This is the natural choice when the fitted distributions are
        heterogeneous (different families per class) because
        log-likelihoods are comparable across families after the
        per-dimension normalisation applied by ``_loglik``.
        """
        loss = torch.tensor(0.0, device=self.device)
        batch_size = embeddings.size(0)

        for i in range(batch_size):
            emb = embeddings[i]
            candidate_lls = []

            for c in range(self.config["num_classes"]):
                if c in self.config['forbidden_labels']: # NOOOOOO
                    continue

                infos = self.dist_infos[c]
                if not isinstance(infos, list):
                    infos = [(infos, 1.0)]

                for info, _intra_frac in infos:
                    ll = self._loglik(emb, info)
                    candidate_lls.append(ll)

            # Best fit = highest log-likelihood among all non-own distributions
            best_ll = self._nanmax_compat(torch.stack(candidate_lls))
            loss += -best_ll  # minimise ⇒ maximise best log-likelihood

        return loss / batch_size

    # =====================================================================
    #  setup
    # =====================================================================

    def setup(
        self,
        train_X=None, train_y=None,
        retain_X=None, retain_y=None,
        forget_X=None, forget_y=None,
        transform_train=None, transform_retain=None, transform_forget=None,
        adapted_dist_infos=None,
    ):
        if retain_X is None:
            retain_X = []
        if retain_y is None:
            retain_y = []

        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = DataLoader(
            ClassificationDataset(train_X, train_y, transforms=ConditionalTransform(transform_train)),
            batch_size=self.config["train_batch_size"],
            shuffle=True,
            num_workers=4,
        )

        # Freeze
        if self.config["embunfreeze"] == "embs":
            self.model.emb_freeze()
        elif self.config["embunfreeze"] == "freeze":
            self.model.freeze()

        print("Calculating distributional stats per class/cluster", flush=True)

        if self.config["robust"]:
            using_robust = "robust"
        else:
            using_robust = "standard"
        if self.config["clustering"]:
            using_robust += "_clustering"

        # Encode covariance type in cache path so different configs don't clash
        cov_type = self.config.get("covariance_type", "full")
        if cov_type != "full":
            cov_tag = f"_cov_{cov_type}"
            n_pca = self.config.get("n_pca_components")
            if cov_type == "pca" and n_pca is not None:
                cov_tag += f"_k{n_pca}"
            using_robust += cov_tag
        ms = self.config.get("model_selection", "bic")
        if ms != "bic":
            using_robust += f"_sel_{ms}"

        cache_prefix = os.path.join(
            f"{using_robust}",
            f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}",
        )
        dist_cache = f"{cache_prefix}_dist_infos.pt"

        if os.path.exists(dist_cache):
            print(f"  Loading cached distributional stats from {dist_cache}", flush=True)
            saved = torch.load(dist_cache, map_location="cpu", weights_only=False)
            self.dist_infos = saved["dist_infos"]
            self.centroids = saved["centroids"]
            self.covariance = saved["covariance"]
            self.percentages = saved["percentages"]
            self.intra_percentage = saved["intra_percentage"]
            self.num_samples_original = saved["num_samples_original"]
            # Move tensors to device
            for i in range(self.config["num_classes"]):
                if self.config["clustering"]:
                    for j in range(len(self.centroids[i])):
                        self.centroids[i][j] = self.centroids[i][j].to(self.device)
                        self.covariance[i][j] = self.covariance[i][j].to(self.device)
                else:
                    if isinstance(self.centroids[i], torch.Tensor):
                        self.centroids[i] = self.centroids[i].to(self.device)
                        self.covariance[i] = self.covariance[i].to(self.device)
            self.percentages = self.percentages.to(self.device)
            self.intra_percentage = [p.to(self.device) for p in self.intra_percentage]
            if isinstance(self.num_samples_original, torch.Tensor):
                self.num_samples_original = self.num_samples_original.to(self.device)
        else:
            # Compute from scratch
            class_embeddings, total_samples, total_per_class = self._collect_embeddings(train_loader)
            self._build_distribution_stats(class_embeddings, total_samples, total_per_class)

            # Save cache
            os.makedirs(os.path.dirname(dist_cache) if os.path.dirname(dist_cache) else ".", exist_ok=True)
            os.makedirs(f"{using_robust}", exist_ok=True)
            torch.save(
                {
                    "dist_infos": self.dist_infos,
                    "centroids": self.centroids,
                    "covariance": self.covariance,
                    "percentages": self.percentages,
                    "intra_percentage": self.intra_percentage,
                    "num_samples_original": self.num_samples_original,
                },
                dist_cache,
            )
            print(f"  Distributional stats saved to {dist_cache}", flush=True)

        # ── ICU: override with distributions adapted during the previous step ──
        # The on-disk cache above always holds the base model's original
        # per-class fits. When carrying state across ICU steps, replace them
        # here — before the logit-profile reference and synthetic samples are
        # built below — so that this step's fixed synthetic data (and the
        # logit-profile reference) reflect the embedding drift accumulated so
        # far, instead of the original (stale) distributions.
        if adapted_dist_infos is not None:
            print("  [ICU adapter] Using distributions adapted in the previous "
                  "ICU step for this step's synthetic sample generation.", flush=True)
            self.dist_infos = deepcopy(adapted_dist_infos)

        # Print summary
        for i in range(self.config["num_classes"]):
            infos = self.dist_infos[i]
            if isinstance(infos, list):
                types = [f"{info.dist_type}({info.decided_by})" for info, _ in infos]
                print(f"  Class {i}: {len(infos)} sub-groups → {types}")
            else:
                print(f"  Class {i}: {infos}")

        # Centroids / covariance lifting for embunfreeze
        if (self.config["update_centroids"] or self.config["update_covariance"]) and (self.config["embunfreeze"] != "freeze"):
            print("Lifting data to pre-embedding space", flush=True)
            self.calculate_minimun_pre_data()
            using_robust += "_centroids_moving"
        elif (self.config["update_centroids"] or self.config["update_covariance"]) and (self.config["embunfreeze"] == "freeze"):
            raise ValueError(
                "Centroids or Covariance update is only allowed when not freezing the embeddings layer."
            )

        if (self.config["embunfreeze"] == "freeze") and self.config["loss_forget"] in ("distances", "loglikelihood"):
            raise ValueError(
                "Using embedding-based losses (distances / loglikelihood) is not allowed when freezing the embeddings layer."
            )

        # Build Gaussian samplers for the Gaussian sub-groups that need
        # to be updated during training (actualizeGaussianNoise path)
        # self.actualizeGaussianNoise()

        # Logit-profile regularizer: snapshot the reference profile from the
        # originally generated synthetic retain samples, before any
        # unlearning step touches the model.
        if self.config.get("logit_profile_lambda", 0.0) > 0.0:
            self._compute_logit_profile_reference(self.config["forbidden_labels"])

        # Synthetic samples
        if self.config["syntetic_samples"] != -1:
            if self.config["fixed_syntetic"]:
                print("Generating fixed synthetic samples", flush=True)
                syntetic_embeddings, syntetic_labels = self.generate_syntetic_embeddings(
                    self.config["syntetic_samples"],
                    self.config['forbidden_labels']
                )
                using_robust += "_syntetic"

                if self.config["update_syntetic"]:
                    syntetic_preembeddings = self.calculate_minimun_preEmbeddings(
                        syntetic_embeddings,
                        labels=syntetic_labels - self.config["num_classes"],
                    )
                    retain_X = [e for e in syntetic_preembeddings.cpu().numpy()]
                    using_robust += "_moving"
                else:
                    retain_X = [e for e in syntetic_embeddings.cpu().numpy()]
                    using_robust += "_fixed"
                retain_y = [l for l in syntetic_labels.cpu().numpy()]

                if (
                    self.config["update_centroids"]
                    or self.config["update_covariance"]
                ):
                    raise ValueError(
                        "Fixed synthetic samples cannot be used when centroids/covariance are updated."
                    )
            else:
                retain_y = [-1] * self.config["syntetic_samples"]
                retain_X = [forget_X[0]] * self.config["syntetic_samples"]

        if self.config["embunfreeze"] == "embs":
            using_robust += "_embs"
        elif self.config["embunfreeze"] == "freeze":
            using_robust += "_freeze"

        else: 
            using_robust += "_nofreeze"

        # Gradient subspace projection — computed here so retain/forget are still original
        if (self.config.get('grad_proj_forget_variance') is not None
                or self.config.get('grad_proj_retain_ortho', False)):
            self._compute_grad_subspaces(
                forget_X, forget_y, transform_forget,
                retain_X=retain_X, retain_y=retain_y, transform_retain=transform_retain,
            )

        if self.config["self_forget"]:
            print("Generating fixed forget synthetic samples", flush=True)
            syntetic_embeddings, syntetic_labels = self.generate_syntetic_embeddings(
                    self.config["forget_syntetic_samples"],
                    self.config["retain_labels"]
                )
            if self.config['update_syntetic']:
                    syntetic_embeddings = self.calculate_minimun_preEmbeddings(
                        syntetic_embeddings,
                        labels=syntetic_labels - self.config["num_classes"],
                    )
            forget_X = [e for e in syntetic_embeddings.cpu().numpy()]
            forget_y = [l for l in syntetic_labels.cpu().numpy()]

            using_robust += f"_self_forget_{self.config['forget_syntetic_samples']}"

        
        self.lambda_margin = self.config.get("lambda_margin", 0.0)

        _loss_tag = self.config['loss_forget']
        if self.config['loss_forget'] == 'notop':
            _loss_tag += f"_margin_{self.config['notop_margin']}"
        using_robust += f"_forget_{_loss_tag}"

        if self.lambda_margin > 0.0:
            using_robust += f"_lambda_{self.lambda_margin}"

        self.using_robust = using_robust

        # Classifier anchor: snapshot retain-class rows before unlearning
        if self.config.get('classifier_anchor_lambda', 0.0) > 0.0:
            retain_labels = self.config['retain_labels']
            fc = self.model.classifier[-2]
            idx = torch.tensor(retain_labels, dtype=torch.long, device=self.device)
            self._anchor_retain_indices = idx
            self._anchor_retain_weights = fc.weight.data[idx].clone().detach()
            print(
                f"  [Anchor] Snapshotted classifier rows for {len(retain_labels)} retain classes.",
                flush=True,
            )

        # Learnable temperature: activate and add to optimizer if requested
        if self.config.get('train_temperature', False):
            temp_param = self.model.classifier[-1].temperature
            temp_param.requires_grad_(True)
            self.optimizer.add_param_group({'params': [temp_param]})
            print("  [Temperature] Temperature parameter added to optimizer.", flush=True)

        # Temperature inference-only: bypass scaling while self.model.training
        # is True, apply it only in eval() mode (validation / downstream
        # inference). Independent of --train-temperature and
        # --forget-temperature-calibration — just gates *when* an already
        # active temperature applies.
        if self.config.get('temperature_inference_only', False):
            self.model.classifier[-1].inference_only.fill_(True)
            print(
                "  [Temperature] inference_only=True: temperature bypassed during "
                "training, applied only in eval() mode.", flush=True
            )
            if self.config.get('train_temperature', False):
                print(
                    "  [Temperature] WARNING: --train-temperature is also set, but "
                    "with inference_only=True the temperature is bypassed during "
                    "training, so it will never receive gradients and stays fixed.",
                    flush=True,
                )

        return train_X, train_y, retain_X, retain_y, forget_X, forget_y

    # =====================================================================
    #  unlearn  (same training loop as NOISYMethod, using adaptive sampling)
    # =====================================================================

    def unlearn(
        self,
        train_X=None, train_y=None,
        retain_X=None, retain_y=None,
        forget_X=None, forget_y=None,
        transform_train=None, transform_retain=None, transform_forget=None,
        icu_adapter=None,
    ):
        forget_train_x, forget_val_x, forget_train_y, forget_val_y = train_test_split(
            forget_X, forget_y, test_size=0.2, stratify=forget_y, random_state=42
        )

        forget_y_random = []
        if self.config["loss_forget"] == "random_class_corrected":
            candidates = [c for c in range(self.config["num_classes"]) if c not in self.config['forbidden_labels']]
            for orig_label in forget_train_y:
                # Choose a label from allowed candidates
                forget_y_random.append(pyrandom.choice(candidates))
            forget_train_y = forget_y_random

        val_dataset = ClassificationDataset(
            forget_val_x, forget_val_y, transforms=transform_forget
        )
        
        val_loader_forget = DataLoader(
            val_dataset,
            batch_size=self.config["val_batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            persistent_workers=True,
        )

        if self.config["self_forget"]:
            val_dataset = ClassificationDataset(
                forget_X, forget_y, transforms=ConditionalTransform(transform_forget)
            )

            val_loader_forget = DataLoader(
                val_dataset,
                batch_size=self.config["val_batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=False,
                persistent_workers=True,
                collate_fn=collate_fn_mixed
            )

        if self.config["loss_forget"] == "none":
            unlearning_x = retain_X
            unlearning_y = retain_y
        else:
            unlearning_x = retain_X + forget_train_x
            unlearning_y = retain_y + forget_train_y

        train_dataset_unlearn = ClassificationDataset(
            unlearning_x, unlearning_y, transforms=ConditionalTransform(transform_forget)
        )
        train_loader_unlearn = DataLoader(
            train_dataset_unlearn,
            batch_size=self.config["train_batch_size"],
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn_mixed,
        )

        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        if self.config.get("loss_forget") == "delete":
            self.ref_model = deepcopy(self.model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

        # Save hyperparameters
        os.makedirs(self.config["hyperparameters_path"], exist_ok=True)
        hyperparameters = {
            "clustering": self.config["clustering"],
            "robust": self.config["robust"],
            "update_centroids": self.config["update_centroids"],
            "update_covariance": self.config["update_covariance"],
            "embunfreeze": self.config["embunfreeze"],
            "loss_forget": self.config["loss_forget"],
            "forbidden_labels": self.config["forbidden_labels"],
            "fixed_syntetic": self.config["fixed_syntetic"],
            "update_syntetic": self.config["update_syntetic"],
            "syntetic_samples": self.config["syntetic_samples"],
            "covariance_type": self.config.get("covariance_type", "full"),
            "n_pca_components": self.config.get("n_pca_components", None),
            "model_selection": self.config.get("model_selection", "bic"),
            "logit_profile_lambda": self.config.get("logit_profile_lambda", 0.0),
            "logit_profile_mode": self.config.get("logit_profile_mode", "logits"),
            "forget_temperature_calibration": self.config.get("forget_temperature_calibration", False),
            "forget_temperature_always_apply": self.config.get("forget_temperature_always_apply", True),
            "forget_temperature_monotonic_decrease": self.config.get("forget_temperature_monotonic_decrease", False),
            "temperature_inference_only": self.config.get("temperature_inference_only", False),
            "method": "DISTRIBUTIONAL",
        }
        with open(f'{self.config["hyperparameters_path"]}/hyperparameters.yml', "w") as f:
            yaml.dump(hyperparameters, f, default_flow_style=False)
        print(f"✅ Hyperparameters saved to {self.config['hyperparameters_path']}/hyperparameters.yml")

        n_step = 0
        print("Starting unlearning process (DistributionalMethod)", flush=True)
        
        # Early stopping state
        early_stop_state = {
            'succed_saved': False,
            'best_forget_acc': float('inf'),
            'consecutive_low_count': 0,
        }

        if not self.config["self_forget"]:
            for epoch in range(self.config["max_epoch"]):
                self.model.train()

                # Pre-sample the full epoch's synthetic embeddings in one vectorised call
                # so each batch just slices from the buffer instead of calling
                # generate_syntetic_embeddings per batch. Each epoch gets fresh
                # random samples because the internal seed auto-increments.
                _epoch_synth_embs = _epoch_synth_labels = None
                _synth_ptr = 0
                if not self.config["fixed_syntetic"]:
                    _n_epoch = (len(train_loader_unlearn) + 1) * self.config["train_batch_size"]
                    _epoch_synth_embs, _epoch_synth_labels = self.generate_syntetic_embeddings(
                        _n_epoch, self.config['forbidden_labels']
                    )

                for batch_idx, (images, target) in tqdm(
                    enumerate(train_loader_unlearn), total=len(train_loader_unlearn)
                ):
                    n_step += 1
                    labels = target.to(self.model.device)
                    self.optimizer.zero_grad()
                    if not isinstance(images, list):
                        images = images.to(self.model.device)

                    loss = torch.tensor(0.0, device=self.model.device)
                    has_real_images = True
                    forget_loss = torch.tensor(0.0, device=self.model.device)
                    loss_syntetic = torch.tensor(0.0, device=self.model.device)
                    n_syntetic = 0

                    images_aux, labels_aux = [], []
                    images_syntetic, labels_syntetic = [], []

                    

                    for i in range(len(images)):
                        if 0 <= labels[i] and labels[i] < self.config["num_classes"]:
                            images_aux.append(images[i].to(self.model.device))
                            labels_aux.append(labels[i].to(self.model.device))
                        else:
                            images_syntetic.append(images[i].to(self.model.device))
                            labels_syntetic.append(labels[i].to(self.model.device))

                    if len(images_aux) > 0:
                        images = torch.stack(images_aux)
                        labels = torch.stack(labels_aux)
                    else:
                        has_real_images = False

                    if len(images_syntetic) > 0:
                        images_syntetic = torch.stack(images_syntetic)
                        labels_syntetic = torch.stack(labels_syntetic)

                        if not self.config["update_syntetic"] and not self.config["fixed_syntetic"]:
                            # Placeholders from dataloader — delegate to on-the-fly path
                            n_syntetic = images_syntetic.size(0)
                        else:
                            if self.config["update_syntetic"]:
                                images_syntetic = self.model.get_embs_layer()(images_syntetic)
                            # fixed_syntetic=True + update_syntetic=False: images_syntetic are already embeddings

                            logits_syntetic = self.model.classifier(images_syntetic)
                            # Hinge margin loss: penalise when gap between top-1
                            # and top-2 logits is below `margin` (default 5).
                            top2 = torch.topk(logits_syntetic, k=2, dim=1).values
                            gap = top2[:, 0] - top2[:, 1]          # (B,)
                            margin = 5.0
                            loss_separate_logits = F.relu(margin - gap).mean()
                            loss_syntetic = self.model.criterion(
                                logits_syntetic, labels_syntetic - self.config["num_classes"]
                            )
                            lp_lambda = self.config.get("logit_profile_lambda", 0.0)
                            logit_profile_loss = torch.tensor(0.0, device=self.model.device)
                            if lp_lambda > 0.0:
                                logit_profile_loss = self._logit_profile_loss(
                                    logits_syntetic, labels_syntetic - self.config["num_classes"]
                                )
                            print(f"  Syntetic conservative loss: {loss_syntetic.item():.4f}  margin_loss: {self.lambda_margin*loss_separate_logits.item():.4f}  logit_profile_loss: {(lp_lambda*logit_profile_loss).item():.4f}")
                            loss = loss + loss_syntetic + self.lambda_margin*loss_separate_logits + lp_lambda*logit_profile_loss


                    else:
                        n_syntetic = len(labels)
                        if self.config["syntetic_samples"] != -1:
                            images = images[labels >= 0]
                            labels = labels[labels >= 0]
                            n_syntetic = n_syntetic - images.size(0)

                    if has_real_images and images.size(0) > 0 and self.config["loss_forget"] != "none":
                        if self.config["loss_forget"] not in ("distances", "loglikelihood"):
                            logits, embeddings = self.model(images, return_feature=True)
                            
                        else:
                            embeddings = self.model(images, return_only_embs=True)

                        if self.config["loss_forget"] == "softmax":
                            uniform_target = torch.ones_like(logits) / logits.size(1)
                            forget_loss = F.kl_div(
                                F.log_softmax(logits, dim=1), uniform_target, reduction="batchmean"
                            ) 
                            
                        elif self.config["loss_forget"] == "energy":
                            energy = 1.0 * torch.logsumexp(logits / 1.0, dim=1)
                            forget_loss = torch.mean(F.softplus(energy))

                        elif self.config["loss_forget"] == "confidence_penalty":
                            probs = torch.softmax(logits, dim=1)
                            forget_loss = (probs * torch.log(probs + 1e-12)).sum(dim=1).mean() 

                        elif self.config["loss_forget"] == "corrected_confidence_penalty":
                            logits_without_forget = logits[:, [i for i in range(self.config["num_classes"]) if i not in self.config["forbidden_labels"]]]
                            probs = torch.softmax(logits_without_forget, dim=1)
                            forget_loss = (probs * torch.log(probs + 1e-12)).sum(dim=1).mean()  

                        elif self.config["loss_forget"] == "random_class":
                            random_labels = torch.randint(
                                0, self.config["num_classes"], (logits.size(0),), device=logits.device
                            )
                            forget_loss = self.model.criterion(logits, random_labels)

                        elif self.config["loss_forget"] == "random_class_corrected":
                            forget_loss = self.model.criterion(logits, labels)

                        elif self.config["loss_forget"] == "negative_entropy":
                            forget_loss = -self.model.criterion(logits, labels) 

                        elif self.config["loss_forget"] == "distances":
                            forget_loss = self.minimize_distances(embeddings, labels)

                        elif self.config["loss_forget"] == "loglikelihood":
                            forget_loss = self.maximize_loglikelihood(embeddings, labels)

                        elif self.config["loss_forget"] == "notop":
                            cols = torch.arange(self.config["num_classes"]).repeat(logits.size(0), 1)
                            cols = cols.to(logits.device)
                            mask = cols != labels.unsqueeze(1)
                            mask = mask.to(logits.device)
                            other_logits = logits.masked_select(mask).view(logits.size(0), -1)
                            best_logits = torch.topk(other_logits, k=1, dim=1).values
                            mylogit = logits[:, labels]

                            forget_loss = F.relu(mylogit - best_logits + self.config["notop_margin"]).mean()

                        elif self.config["loss_forget"] == "notop_corrected_confidence_penalty":
                            # Mix of corrected_confidence_penalty (flatten the retain-only
                            # softmax, penalising any single retain logit standing out) and
                            # notop (penalise your own class logit for being the largest
                            # one among all logits, using the same --notop-margin).
                            logits_without_forget = logits[:, [i for i in range(self.config["num_classes"]) if i not in self.config["forbidden_labels"]]]
                            probs = torch.softmax(logits_without_forget, dim=1)
                            confidence_term = (probs * torch.log(probs + 1e-12)).sum(dim=1).mean()

                            cols = torch.arange(self.config["num_classes"]).repeat(logits.size(0), 1)
                            cols = cols.to(logits.device)
                            mask = cols != labels.unsqueeze(1)
                            mask = mask.to(logits.device)
                            other_logits = logits.masked_select(mask).view(logits.size(0), -1)
                            best_logits = torch.topk(other_logits, k=1, dim=1).values
                            mylogit = logits[:, labels]
                            notop_term = F.relu(mylogit - best_logits + self.config["notop_margin"]).mean()

                            forget_loss = confidence_term + notop_term

                        elif self.config["loss_forget"] == "notop_confidence_penalty_balanced":
                            # Same mix as notop_corrected_confidence_penalty, but each term
                            # is scaled by the inverse of its own (detached) magnitude so
                            # both contribute equally regardless of scale — otherwise
                            # notop_term (a hinge, typically much larger) would dominate
                            # confidence_term (a bounded negative-entropy term).
                            logits_without_forget = logits[:, [i for i in range(self.config["num_classes"]) if i not in self.config["forbidden_labels"]]]
                            probs = torch.softmax(logits_without_forget, dim=1)
                            confidence_term = (probs * torch.log(probs + 1e-12)).sum(dim=1).mean()

                            cols = torch.arange(self.config["num_classes"]).repeat(logits.size(0), 1)
                            cols = cols.to(logits.device)
                            mask = cols != labels.unsqueeze(1)
                            mask = mask.to(logits.device)
                            other_logits = logits.masked_select(mask).view(logits.size(0), -1)
                            best_logits = torch.topk(other_logits, k=1, dim=1).values
                            mylogit = logits[:, labels]
                            notop_term = F.relu(mylogit - best_logits + self.config["notop_margin"]).mean()

                            eps = 1e-8
                            confidence_scale = 1.0 / (confidence_term.detach().abs() + eps)
                            notop_scale = 1.0 / (notop_term.detach().abs() + eps)
                            forget_loss = confidence_scale * confidence_term + notop_scale * notop_term

                        elif self.config["loss_forget"] == "notopexpand":
                            cols = torch.arange(self.config["num_classes"]).repeat(logits.size(0), 1)
                            cols = cols.to(logits.device)
                            mask = cols != labels.unsqueeze(1)
                            mask = mask.to(logits.device)
                            other_logits = logits.masked_select(mask).view(logits.size(0), -1)
                            best_logits = torch.topk(other_logits, k=1, dim=1).values
                            mylogit = logits[:, labels]

                            forbidden_labels = self.config["forbidden_labels"]
                            avg_intra_distance = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

                            if len(forbidden_labels) > 0 and embeddings.size(0) > 1:
                                forbidden_labels_t = torch.as_tensor(
                                    forbidden_labels, device=labels.device, dtype=labels.dtype
                                )

                                pairwise_dists = torch.cdist(embeddings, embeddings, p=2)  # (B, B)
                                same_label = labels[:, None] == labels[None, :]
                                not_self = ~torch.eye(
                                    embeddings.size(0), device=embeddings.device, dtype=torch.bool
                                )
                                is_forbidden_row = (labels[:, None] == forbidden_labels_t[None, :]).any(dim=1)
                                valid_pairs = same_label & not_self & is_forbidden_row[:, None]

                                row_distance_sum = (pairwise_dists * valid_pairs.to(pairwise_dists.dtype)).sum(dim=1)
                                row_pair_count = valid_pairs.sum(dim=1).to(pairwise_dists.dtype)

                                label_membership = (labels[:, None] == forbidden_labels_t[None, :]).to(
                                    pairwise_dists.dtype
                                )  # (B, F)
                                per_label_distance_sum = (label_membership * row_distance_sum[:, None]).sum(dim=0)
                                per_label_pair_count = (label_membership * row_pair_count[:, None]).sum(dim=0)

                                per_label_mean = torch.where(
                                    per_label_pair_count > 0,
                                    per_label_distance_sum / per_label_pair_count,
                                    torch.zeros_like(per_label_distance_sum),
                                )
                                existing_clusters = (per_label_pair_count > 0).sum()
                                if existing_clusters.item() > 0:
                                    avg_intra_distance = per_label_mean.sum() / existing_clusters.to(
                                        per_label_mean.dtype
                                    )

                            forget_loss = F.relu(mylogit - best_logits + 10).mean() - avg_intra_distance

                        elif self.config["loss_forget"] == "notopexpandalign":
                            cols = torch.arange(self.config["num_classes"]).repeat(logits.size(0), 1)
                            cols = cols.to(logits.device)
                            mask = cols != labels.unsqueeze(1)
                            mask = mask.to(logits.device)
                            other_logits = logits.masked_select(mask).view(logits.size(0), -1)
                            best_logits = torch.topk(other_logits, k=1, dim=1).values
                            mylogit = logits[:, labels]

                            forbidden_labels = self.config["forbidden_labels"]
                            avg_intra_distance = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
                            avg_intra_distance_l2 = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

                            if len(forbidden_labels) > 0 and embeddings.size(0) > 1:
                                forbidden_labels_t = torch.as_tensor(
                                    forbidden_labels, device=labels.device, dtype=labels.dtype
                                )

                                pairwise_dists_l2 = torch.cdist(embeddings, embeddings, p=2)  # (B, B)
                                normalized_embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
                                pairwise_dists = 1.0 - normalized_embeddings @ normalized_embeddings.T
                                pairwise_dists = pairwise_dists.clamp_min(0.0)  # (B, B)
                                same_label = labels[:, None] == labels[None, :]
                                not_self = ~torch.eye(
                                    embeddings.size(0), device=embeddings.device, dtype=torch.bool
                                )
                                is_forbidden_row = (labels[:, None] == forbidden_labels_t[None, :]).any(dim=1)
                                valid_pairs = same_label & not_self & is_forbidden_row[:, None]

                                row_distance_sum_l2 = (pairwise_dists_l2 * valid_pairs.to(pairwise_dists.dtype)).sum(dim=1)
                                row_distance_sum = (pairwise_dists * valid_pairs.to(pairwise_dists.dtype)).sum(dim=1)
                                row_pair_count = valid_pairs.sum(dim=1).to(pairwise_dists.dtype)

                                label_membership = (labels[:, None] == forbidden_labels_t[None, :]).to(
                                    pairwise_dists.dtype
                                )  # (B, F)
                                per_label_distance_sum_l2 = (label_membership * row_distance_sum_l2[:, None]).sum(dim=0)
                                per_label_distance_sum = (label_membership * row_distance_sum[:, None]).sum(dim=0)
                                per_label_pair_count = (label_membership * row_pair_count[:, None]).sum(dim=0)

                                per_label_mean = torch.where(
                                    per_label_pair_count > 0,
                                    per_label_distance_sum / per_label_pair_count,
                                    torch.zeros_like(per_label_distance_sum),
                                )
                                per_label_mean_l2 = torch.where(
                                    per_label_pair_count > 0,
                                    per_label_distance_sum_l2 / per_label_pair_count,
                                    torch.zeros_like(per_label_distance_sum_l2),
                                )
                                existing_clusters = (per_label_pair_count > 0).sum()
                                if existing_clusters.item() > 0:
                                    avg_intra_distance = per_label_mean.sum() / existing_clusters.to(
                                        per_label_mean.dtype
                                    )
                                    avg_intra_distance_l2 = per_label_mean_l2.sum() / existing_clusters.to(
                                        per_label_mean_l2.dtype
                                    )

                            forget_loss = F.relu(mylogit - best_logits + 10).mean() - 0.1*(avg_intra_distance_l2) - avg_intra_distance

                        elif self.config["loss_forget"] == "notopalign":
                            cols = torch.arange(self.config["num_classes"]).repeat(logits.size(0), 1)
                            cols = cols.to(logits.device)
                            mask = cols != labels.unsqueeze(1)
                            mask = mask.to(logits.device)
                            other_logits = logits.masked_select(mask).view(logits.size(0), -1)
                            best_logits = torch.topk(other_logits, k=1, dim=1).values
                            mylogit = logits[:, labels]

                            forbidden_labels = self.config["forbidden_labels"]
                            avg_intra_distance = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

                            if len(forbidden_labels) > 0 and embeddings.size(0) > 1:
                                forbidden_labels_t = torch.as_tensor(
                                    forbidden_labels, device=labels.device, dtype=labels.dtype
                                )

                                normalized_embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
                                pairwise_dists = 1.0 - normalized_embeddings @ normalized_embeddings.T
                                pairwise_dists = pairwise_dists.clamp_min(0.0)  # (B, B)
                                same_label = labels[:, None] == labels[None, :]
                                not_self = ~torch.eye(
                                    embeddings.size(0), device=embeddings.device, dtype=torch.bool
                                )
                                is_forbidden_row = (labels[:, None] == forbidden_labels_t[None, :]).any(dim=1)
                                valid_pairs = same_label & not_self & is_forbidden_row[:, None]

                                row_distance_sum = (pairwise_dists * valid_pairs.to(pairwise_dists.dtype)).sum(dim=1)
                                row_pair_count = valid_pairs.sum(dim=1).to(pairwise_dists.dtype)

                                label_membership = (labels[:, None] == forbidden_labels_t[None, :]).to(
                                    pairwise_dists.dtype
                                )  # (B, F)
                                per_label_distance_sum = (label_membership * row_distance_sum[:, None]).sum(dim=0)
                                per_label_pair_count = (label_membership * row_pair_count[:, None]).sum(dim=0)

                                per_label_mean = torch.where(
                                    per_label_pair_count > 0,
                                    per_label_distance_sum / per_label_pair_count,
                                    torch.zeros_like(per_label_distance_sum),
                                )
                                existing_clusters = (per_label_pair_count > 0).sum()
                                if existing_clusters.item() > 0:
                                    avg_intra_distance = per_label_mean.sum() / existing_clusters.to(
                                        per_label_mean.dtype
                                    )

                            forget_loss = F.relu(mylogit - best_logits + 10).mean() - 0.1*(avg_intra_distance)

                        elif self.config["loss_forget"] == "delete":
                            # DELETE method: KL(student || teacher) where teacher is the
                            # reference model with forgotten-class logits masked to -inf.
                            with torch.no_grad():
                                ref_logits = self.ref_model(images)
                            mask = torch.zeros_like(ref_logits)
                            if len(self.config["forbidden_labels"]) > 0:
                                forget_classes_t = torch.tensor(self.config["forbidden_labels"], device=ref_logits.device)
                                mask[:, forget_classes_t] = float('-inf')
                            logits_teacher_masked = ref_logits + mask
                            soft_targets = F.softmax(logits_teacher_masked, dim=1)
                            softmax_student = F.softmax(logits, dim=1)
                            forget_loss = F.kl_div(softmax_student, soft_targets, reduction="batchmean")

                        else:
                            raise ValueError(
                                "Invalid loss_forget! Choose 'softmax', 'energy', 'confidence_penalty', "
                                "'corrected_confidence_penalty', 'random_class', 'negative_entropy', "
                                "'none', 'notop', 'notopexpand', 'notop_corrected_confidence_penalty', "
                                "'notop_confidence_penalty_balanced', 'loglikelihood', 'distances' "
                                "or 'delete'."
                            )

                    if n_syntetic > 0 or self.config["syntetic_samples"] == -1:
                        if _epoch_synth_embs is not None:
                            _take = n_syntetic
                            syntetic_embeddings = _epoch_synth_embs[_synth_ptr:_synth_ptr + _take]
                            syntetic_labels = _epoch_synth_labels[_synth_ptr:_synth_ptr + _take]
                            _synth_ptr += _take
                        else:
                            syntetic_embeddings, syntetic_labels = self.generate_syntetic_embeddings(n_syntetic, self.config['forbidden_labels'])
                        syntetic_embeddings = syntetic_embeddings.to(self.model.device)
                        syntetic_labels = syntetic_labels.to(self.model.device)
                        syntetic_logits = self.model.classifier(syntetic_embeddings)
                        # Hinge margin loss on on-the-fly synthetic samples
                        top2 = torch.topk(syntetic_logits, k=2, dim=1).values
                        gap = top2[:, 0] - top2[:, 1]
                        margin = 5.0
                        loss_separate_logits = F.relu(margin - gap).mean()
                        noisy_loss = self.model.criterion(
                            syntetic_logits, syntetic_labels - self.config["num_classes"]
                        )
                        lp_lambda = self.config.get("logit_profile_lambda", 0.0)
                        logit_profile_loss = torch.tensor(0.0, device=self.model.device)
                        if lp_lambda > 0.0:
                            logit_profile_loss = self._logit_profile_loss(
                                syntetic_logits, syntetic_labels - self.config["num_classes"]
                            )
                        loss = loss + noisy_loss + self.lambda_margin*loss_separate_logits + lp_lambda*logit_profile_loss
                    loss = loss + forget_loss
                    if self._anchor_retain_weights is not None:
                        fc = self.model.classifier[-2]
                        cur = fc.weight[self._anchor_retain_indices]
                        anchor_loss = self.config['classifier_anchor_lambda'] * (
                            (cur - self._anchor_retain_weights) ** 2
                        ).sum()
                        loss = loss + anchor_loss
                    if self.config.get('train_temperature', False):
                        T = self.model.classifier[-1].temperature
                        lam_T = self.config.get('temperature_penalty_lambda', 1.0)
                        loss = loss + lam_T * F.relu(0.1 - T) ** 2
                    loss.backward()
                    if self._grad_proj_classifier_S is not None or self._grad_proj_emb_S is not None:
                        self._apply_grad_projection()
                    self.optimizer.step()

                    print(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Forget Loss: {forget_loss.item() if has_real_images else 'N/A'}, "
                        f"Syntetic Loss: {loss_syntetic.item() if n_syntetic > 0 else 'N/A'}"
                    )

                    # if self.config["update_centroids"] or self.config["update_covariance"]:
                    #     self.actualizeCentroidsAndCovariance()
                    #     self.actualizeGaussianNoise()

                # ----- Validation -------------------------------------------------
                print(f"Epoch {epoch} completed. Validation...", flush=True)
                forget_validation_loss = 0
                acc_forget = 0
                acc_syntetic = 0
                syntetic_validation_loss = 0
                self.model.eval()
                with torch.no_grad():
                    n_total_forget = 0
                    for batch_idx, (images, target) in tqdm(
                        enumerate(val_loader_forget), total=len(val_loader_forget)
                    ):
                        images = images.to(self.model.device)
                        labels = target.to(self.model.device)
                        logits = self.model(images)
                        val_loss = self.model.criterion(logits, labels)
                        n_total_forget += labels.size(0)
                        forget_validation_loss += val_loss.item() * images.size(0)
                        acc_forget += (logits.argmax(dim=1) == labels).sum().item()

                        # se, sl = self.generate_syntetic_embeddings(images.size(0), self.config['forbidden_labels'])
                        # se = se.to(self.model.device)
                        # sl = sl.to(self.model.device)
                        # s_logits = self.model.classifier(se)
                        # s_loss = self.model.criterion(s_logits, sl - self.config["num_classes"])
                        # syntetic_validation_loss += s_loss.item() * se.size(0)
                        # acc_syntetic += (s_logits.argmax(dim=1) == sl - self.config["num_classes"]).sum().item()

                    syntetic_validation_loss /= max(n_total_forget, 1)
                    forget_validation_loss /= max(n_total_forget, 1)
                    acc_forget /= max(n_total_forget, 1)
                    acc_syntetic /= max(n_total_forget, 1)

                print(
                    f"Epoch {epoch} validation completed.\n"
                    f" Forget Loss: {forget_validation_loss}, Syntetic Loss: {syntetic_validation_loss}\n"
                    f" Acc Forget: {acc_forget}, Acc Syntetic: {acc_syntetic}\n\n"
                )

                # Check if checkpoint should be saved / early stopping
                checkpoints_to_save, stop = should_save_checkpoint(epoch, acc_forget, self.config['num_classes'], self.config, early_stop_state)
                for ckpt_name, reason in checkpoints_to_save:
                    print(f"Saving checkpoint at epoch {epoch}: {reason}", flush=True)
                    self.save_results(epoch, n_step, filename=ckpt_name)

                if icu_adapter is not None:
                    _upd, _mtr = icu_adapter.step(self.model)
                    self.dist_infos = _upd
                    _drift_str = ""
                    if not math.isnan(_mtr['mean_angular_shift_deg']):
                        _drift_str += (
                            f"  angular_shift=({_mtr['mean_angular_shift_deg']:.3f}°"
                            f"±{_mtr['std_angular_shift_deg']:.3f}° "
                            f"max={_mtr['max_angular_shift_deg']:.3f}°)"
                            f"  kappa_ratio=({_mtr['mean_kappa_ratio']:.3f}±{_mtr['std_kappa_ratio']:.3f})"
                        )
                    if not math.isnan(_mtr['mean_centroid_shift']):
                        _drift_str += (
                            f"  shift=({_mtr['mean_centroid_shift']:.3f}"
                            f"±{_mtr['std_centroid_shift']:.3f} "
                            f"max={_mtr['max_centroid_shift']:.3f})"
                            f"  std_ratio=({_mtr['mean_std_ratio']:.3f}±{_mtr['std_std_ratio']:.3f})"
                        )
                    _cond_str = (
                        f"  σ_min={_mtr['sigma_min']:.4f}  cond={_mtr['condition_number']:.2f}"
                        if not math.isnan(_mtr['sigma_min']) else ""
                    )
                    print(
                        f"  [ICU adapt epoch {epoch}] "
                        f"‖ΔR‖_F={_mtr['R_norm_diff']:.4f}  "
                        f"‖R-I‖_F={_mtr['R_drift']:.4f}  "
                        f"‖t‖={_mtr['t_norm']:.4f}  "
                        f"disp={_mtr['mean_displacement']:.4f}±{_mtr['std_displacement']:.4f}"
                        f"{_cond_str}  "
                        f"ll_adapted={_mtr['avg_ll_adapted']:.4f}  "
                        f"ll_original={_mtr['avg_ll_original']:.4f}"
                        f"{_drift_str}",
                        flush=True,
                    )

                if stop:
                    print(f"🛑 Early stopping at epoch {epoch}", flush=True)
                    break


        else:
            for epoch in range(self.config["max_epoch"]):
                self.model.train()

                # Pre-sample full epoch's synthetic embeddings in one vectorised call.
                _epoch_synth_embs = _epoch_synth_labels = None
                _synth_ptr = 0
                if not self.config["fixed_syntetic"]:
                    _n_epoch = (len(train_loader_unlearn) + 1) * self.config["train_batch_size"]
                    _epoch_synth_embs, _epoch_synth_labels = self.generate_syntetic_embeddings(
                        _n_epoch, self.config['forbidden_labels']
                    )

                for batch_idx, (images, target) in tqdm(
                    enumerate(train_loader_unlearn), total=len(train_loader_unlearn)
                ):
                    n_step += 1
                    labels = target.to(self.model.device)
                    self.optimizer.zero_grad()
                    if not isinstance(images, list):
                        images = images.to(self.model.device)

                    labels = labels - self.config["num_classes"]  # Shift labels for synthetic samples
                    loss = torch.tensor(0.0, device=self.model.device)
                    forget_loss = torch.tensor(0.0, device=self.model.device)
                    loss_syntetic = torch.tensor(0.0, device=self.model.device)

                    retain_data = [(images[i], labels[i]) for i in range(len(labels)) if labels[i] in self.config["retain_labels"]]
                    forget_data = [(images[i], labels[i]) for i in range(len(labels)) if labels[i] not in self.config["retain_labels"]]

                    if retain_data:
                        retain_images, retain_labels = zip(*retain_data)
                        retain_images = torch.stack(retain_images)
                        retain_labels = torch.stack(retain_labels)
                    else:
                        retain_images = torch.tensor([], device=self.model.device)
                        retain_labels = torch.tensor([], device=self.model.device, dtype=torch.long)
                    
                    if forget_data:
                        forget_images, forget_labels = zip(*forget_data)
                        forget_images = torch.stack(forget_images)
                        forget_labels = torch.stack(forget_labels)
                    else:
                        forget_images = torch.tensor([], device=self.model.device)
                        forget_labels = torch.tensor([], device=self.model.device, dtype=torch.long)

                    n_syntetic = len(forget_labels) if self.config["syntetic_samples"] == -1 else len(labels) - len(retain_data) - len(forget_data)

                    if self.config["update_syntetic"] and len(retain_images) > 0:
                        retain_images = self.model.get_embs_layer()(retain_images)

                    logits_syntetic = self.model.classifier(retain_images)
                    # Hinge margin loss: penalise when gap between top-1
                    # and top-2 logits is below `margin` (default 5).
                    top2 = torch.topk(logits_syntetic, k=2, dim=1).values
                    gap = top2[:, 0] - top2[:, 1]          # (B,)
                    margin = 5.0
                    loss_separate_logits = F.relu(margin - gap).mean()
                    loss_syntetic = self.model.criterion(
                        logits_syntetic, retain_labels
                    )
                    lp_lambda = self.config.get("logit_profile_lambda", 0.0)
                    logit_profile_loss = torch.tensor(0.0, device=self.model.device)
                    if lp_lambda > 0.0:
                        logit_profile_loss = self._logit_profile_loss(logits_syntetic, retain_labels)
                    print(f"  Syntetic conservative loss: {loss_syntetic.item():.4f}  margin_loss: {self.lambda_margin*loss_separate_logits.item():.4f}  logit_profile_loss: {(lp_lambda*logit_profile_loss).item():.4f}")
                    loss = loss + loss_syntetic + self.lambda_margin*loss_separate_logits + lp_lambda*logit_profile_loss

                    forget_labels = forget_labels.to(self.model.device)
                    forget_images = forget_images.to(self.model.device)
                    forget_loss = torch.tensor(0.0, device=self.model.device) 
                    if forget_images.size(0) > 0 and self.config["loss_forget"] != "none":
                        if self.config["loss_forget"] not in ("distances", "loglikelihood"):
                            if self.config["update_syntetic"]:
                                    forget_images = self.model.get_embs_layer()(forget_images)

                            logits = self.model.classifier(forget_images)
                            
                        else:
                            if self.config["update_syntetic"]:
                                forget_images = self.model.get_embs_layer()(forget_images)

                            embeddings = forget_images

                        if self.config["loss_forget"] == "softmax":
                            uniform_target = torch.ones_like(logits) / logits.size(1)
                            forget_loss = F.kl_div(
                                F.log_softmax(logits, dim=1), uniform_target, reduction="batchmean"
                            ) 
                            
                        elif self.config["loss_forget"] == "energy":
                            energy = 1.0 * torch.logsumexp(logits / 1.0, dim=1)
                            forget_loss = torch.mean(F.softplus(energy))

                        elif self.config["loss_forget"] == "confidence_penalty":
                            probs = torch.softmax(logits, dim=1)
                            forget_loss = (probs * torch.log(probs + 1e-12)).sum(dim=1).mean() 

                        elif self.config["loss_forget"] == "corrected_confidence_penalty":
                            logits_without_forget = logits[:, [i for i in range(self.config["num_classes"]) if i not in self.config["forbidden_labels"]]]
                            probs = torch.softmax(logits_without_forget, dim=1)
                            forget_loss = (probs * torch.log(probs + 1e-12)).sum(dim=1).mean()  

                        elif self.config["loss_forget"] == "random_class":
                            random_labels = torch.randint(
                                0, self.config["num_classes"], (logits.size(0),), device=logits.device
                            )
                            forget_loss = self.model.criterion(logits, random_labels) 

                        elif self.config["loss_forget"] == "random_class_corrected":
                            forget_loss = self.model.criterion(logits, random_labels)

                        elif self.config["loss_forget"] == "negative_entropy":
                            forget_loss = -self.model.criterion(logits, forget_labels) 

                        elif self.config["loss_forget"] == "distances":
                            forget_loss = self.minimize_distances(embeddings, forget_labels)

                        elif self.config["loss_forget"] == "loglikelihood":
                            forget_loss = self.maximize_loglikelihood(embeddings, forget_labels)

                        elif self.config["loss_forget"] == "notop":
                            cols = torch.arange(self.config["num_classes"]).repeat(logits.size(0), 1)
                            cols = cols.to(logits.device)
                            mask = cols != forget_labels.unsqueeze(1)
                            mask = mask.to(logits.device)
                            other_logits = logits.masked_select(mask).view(logits.size(0), -1)
                            best_logits = torch.topk(other_logits, k=1, dim=1).values
                            mylogit = logits[:, forget_labels]

                            forget_loss = F.relu(mylogit - best_logits + self.config["notop_margin"]).mean()

                        elif self.config["loss_forget"] == "notop_corrected_confidence_penalty":
                            # Mix of corrected_confidence_penalty (flatten the retain-only
                            # softmax, penalising any single retain logit standing out) and
                            # notop (penalise your own class logit for being the largest
                            # one among all logits, using the same --notop-margin).
                            logits_without_forget = logits[:, [i for i in range(self.config["num_classes"]) if i not in self.config["forbidden_labels"]]]
                            probs = torch.softmax(logits_without_forget, dim=1)
                            confidence_term = (probs * torch.log(probs + 1e-12)).sum(dim=1).mean()

                            cols = torch.arange(self.config["num_classes"]).repeat(logits.size(0), 1)
                            cols = cols.to(logits.device)
                            mask = cols != forget_labels.unsqueeze(1)
                            mask = mask.to(logits.device)
                            other_logits = logits.masked_select(mask).view(logits.size(0), -1)
                            best_logits = torch.topk(other_logits, k=1, dim=1).values
                            mylogit = logits[:, forget_labels]
                            notop_term = F.relu(mylogit - best_logits + self.config["notop_margin"]).mean()

                            forget_loss = confidence_term + notop_term

                        elif self.config["loss_forget"] == "notop_confidence_penalty_balanced":
                            # Same mix as notop_corrected_confidence_penalty, but each term
                            # is scaled by the inverse of its own (detached) magnitude so
                            # both contribute equally regardless of scale — otherwise
                            # notop_term (a hinge, typically much larger) would dominate
                            # confidence_term (a bounded negative-entropy term).
                            logits_without_forget = logits[:, [i for i in range(self.config["num_classes"]) if i not in self.config["forbidden_labels"]]]
                            probs = torch.softmax(logits_without_forget, dim=1)
                            confidence_term = (probs * torch.log(probs + 1e-12)).sum(dim=1).mean()

                            cols = torch.arange(self.config["num_classes"]).repeat(logits.size(0), 1)
                            cols = cols.to(logits.device)
                            mask = cols != forget_labels.unsqueeze(1)
                            mask = mask.to(logits.device)
                            other_logits = logits.masked_select(mask).view(logits.size(0), -1)
                            best_logits = torch.topk(other_logits, k=1, dim=1).values
                            mylogit = logits[:, forget_labels]
                            notop_term = F.relu(mylogit - best_logits + self.config["notop_margin"]).mean()

                            eps = 1e-8
                            confidence_scale = 1.0 / (confidence_term.detach().abs() + eps)
                            notop_scale = 1.0 / (notop_term.detach().abs() + eps)
                            forget_loss = confidence_scale * confidence_term + notop_scale * notop_term

                        elif self.config["loss_forget"] == "delete":
                            # DELETE method: KL(student || teacher) where teacher is the
                            # reference model with forgotten-class logits masked to -inf.
                            with torch.no_grad():
                                ref_logits = self.ref_model.classifier(forget_images)
                            mask = torch.zeros_like(ref_logits)
                            if len(self.config["forbidden_labels"]) > 0:
                                forget_classes_t = torch.tensor(self.config["forbidden_labels"], device=ref_logits.device)
                                mask[:, forget_classes_t] = float('-inf')
                            logits_teacher_masked = ref_logits + mask
                            soft_targets = F.softmax(logits_teacher_masked, dim=1)
                            softmax_student = F.softmax(logits, dim=1)
                            forget_loss = F.kl_div(softmax_student, soft_targets, reduction="batchmean")

                        else:
                            raise ValueError(
                                "Invalid loss_forget! Choose 'softmax', 'energy', 'confidence_penalty', "
                                "'corrected_confidence_penalty', 'random_class', 'negative_entropy', "
                                "'none', 'notop', 'notop_corrected_confidence_penalty', "
                                "'notop_confidence_penalty_balanced', 'loglikelihood', 'distances' "
                                "or 'delete'."
                            )

                    if not self.config["fixed_syntetic"]:
                        if _epoch_synth_embs is not None:
                            _take = n_syntetic
                            syntetic_embeddings = _epoch_synth_embs[_synth_ptr:_synth_ptr + _take]
                            syntetic_labels = _epoch_synth_labels[_synth_ptr:_synth_ptr + _take]
                            _synth_ptr += _take
                        else:
                            syntetic_embeddings, syntetic_labels = self.generate_syntetic_embeddings(n_syntetic, self.config['forbidden_labels'])
                        syntetic_embeddings = syntetic_embeddings.to(self.model.device)
                        syntetic_labels = syntetic_labels.to(self.model.device)
                        syntetic_logits = self.model.classifier(syntetic_embeddings)
                        # Hinge margin loss on on-the-fly synthetic samples
                        top2 = torch.topk(syntetic_logits, k=2, dim=1).values
                        gap = top2[:, 0] - top2[:, 1]
                        margin = 5.0
                        loss_separate_logits = F.relu(margin - gap).mean()
                        noisy_loss = self.model.criterion(
                            syntetic_logits, syntetic_labels - self.config["num_classes"]
                        )
                        lp_lambda = self.config.get("logit_profile_lambda", 0.0)
                        logit_profile_loss = torch.tensor(0.0, device=self.model.device)
                        if lp_lambda > 0.0:
                            logit_profile_loss = self._logit_profile_loss(
                                syntetic_logits, syntetic_labels - self.config["num_classes"]
                            )
                        loss = loss + noisy_loss + self.lambda_margin*loss_separate_logits + lp_lambda*logit_profile_loss
                    loss = loss + forget_loss
                    if self._anchor_retain_weights is not None:
                        fc = self.model.classifier[-2]
                        cur = fc.weight[self._anchor_retain_indices]
                        anchor_loss = self.config['classifier_anchor_lambda'] * (
                            (cur - self._anchor_retain_weights) ** 2
                        ).sum()
                        loss = loss + anchor_loss
                    if self.config.get('train_temperature', False):
                        T = self.model.classifier[-1].temperature
                        lam_T = self.config.get('temperature_penalty_lambda', 1.0)
                        loss = loss + lam_T * F.relu(0.1 - T) ** 2
                    loss.backward()
                    if self._grad_proj_classifier_S is not None or self._grad_proj_emb_S is not None:
                        self._apply_grad_projection()
                    self.optimizer.step()

                    print(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Forget Loss: {forget_loss.item()}, "
                        f"Syntetic Loss: {loss_syntetic.item() if n_syntetic > 0 else 'N/A'}"
                    )

                    # if self.config["update_centroids"] or self.config["update_covariance"]:
                    #     self.actualizeCentroidsAndCovariance()
                    #     self.actualizeGaussianNoise()

                # ----- Validation -------------------------------------------------
                print(f"Epoch {epoch} completed. Validation...", flush=True)
                forget_validation_loss = 0
                acc_forget = 0
                acc_syntetic = 0
                syntetic_validation_loss = 0
                self.model.eval()
                with torch.no_grad():
                    n_total_forget = 0
                    for batch_idx, (images, target) in tqdm(
                        enumerate(val_loader_forget), total=len(val_loader_forget)
                    ):
                        images = images.to(self.model.device)
                        labels = target.to(self.model.device)
                        labels = labels - self.config["num_classes"]  # Shift labels for synthetic samples
                        if self.config["update_syntetic"]:
                            images = self.model.get_embs_layer()(images)

                        logits = self.model.classifier(images)
                        val_loss = self.model.criterion(logits, labels)
                        n_total_forget += labels.size(0)
                        forget_validation_loss += val_loss.item() * len(labels)
                        acc_forget += (logits.argmax(dim=1) == labels).sum().item()

                        # se, sl = self.generate_syntetic_embeddings(images.size(0), self.config['forbidden_labels'])
                        # se = se.to(self.model.device)
                        # sl = sl.to(self.model.device)
                        # s_logits = self.model.classifier(se)
                        # s_loss = self.model.criterion(s_logits, sl - self.config["num_classes"])
                        # syntetic_validation_loss += s_loss.item() * se.size(0)
                        # acc_syntetic += (s_logits.argmax(dim=1) == sl - self.config["num_classes"]).sum().item()

                    syntetic_validation_loss /= max(n_total_forget, 1)
                    forget_validation_loss /= max(n_total_forget, 1)
                    acc_forget /= max(n_total_forget, 1)
                    acc_syntetic /= max(n_total_forget, 1)


                print(
                    f"Epoch {epoch} validation completed.\n"
                    f" Forget Loss: {forget_validation_loss}, Syntetic Loss: {syntetic_validation_loss}\n"
                    f" Acc Forget: {acc_forget}, Acc Syntetic: {acc_syntetic}\n\n"
                )

                # Check if checkpoint should be saved / early stopping
                checkpoints_to_save, stop = should_save_checkpoint(epoch, acc_forget, self.config['num_classes'], self.config, early_stop_state)
                for ckpt_name, reason in checkpoints_to_save:
                    print(f"Saving checkpoint at epoch {epoch}: {reason}", flush=True)
                    self.save_results(epoch, n_step, filename=ckpt_name)

                if icu_adapter is not None:
                    _upd, _mtr = icu_adapter.step(self.model)
                    self.dist_infos = _upd
                    _drift_str = ""
                    if not math.isnan(_mtr['mean_angular_shift_deg']):
                        _drift_str += (
                            f"  angular_shift=({_mtr['mean_angular_shift_deg']:.3f}°"
                            f"±{_mtr['std_angular_shift_deg']:.3f}° "
                            f"max={_mtr['max_angular_shift_deg']:.3f}°)"
                            f"  kappa_ratio=({_mtr['mean_kappa_ratio']:.3f}±{_mtr['std_kappa_ratio']:.3f})"
                        )
                    if not math.isnan(_mtr['mean_centroid_shift']):
                        _drift_str += (
                            f"  shift=({_mtr['mean_centroid_shift']:.3f}"
                            f"±{_mtr['std_centroid_shift']:.3f} "
                            f"max={_mtr['max_centroid_shift']:.3f})"
                            f"  std_ratio=({_mtr['mean_std_ratio']:.3f}±{_mtr['std_std_ratio']:.3f})"
                        )
                    _cond_str = (
                        f"  σ_min={_mtr['sigma_min']:.4f}  cond={_mtr['condition_number']:.2f}"
                        if not math.isnan(_mtr['sigma_min']) else ""
                    )
                    print(
                        f"  [ICU adapt epoch {epoch}] "
                        f"‖ΔR‖_F={_mtr['R_norm_diff']:.4f}  "
                        f"‖R-I‖_F={_mtr['R_drift']:.4f}  "
                        f"‖t‖={_mtr['t_norm']:.4f}  "
                        f"disp={_mtr['mean_displacement']:.4f}±{_mtr['std_displacement']:.4f}"
                        f"{_cond_str}  "
                        f"ll_adapted={_mtr['avg_ll_adapted']:.4f}  "
                        f"ll_original={_mtr['avg_ll_original']:.4f}"
                        f"{_drift_str}",
                        flush=True,
                    )

                if stop:
                    print(f"🛑 Early stopping at epoch {epoch}", flush=True)
                    break

        if self.config.get("forget_temperature_calibration", False):
            if self.config["self_forget"]:
                print(
                    "  [ForgetTempCalib] Skipped: no real forgotten-class data "
                    "available under --self-forget.", flush=True
                )
            else:
                # checkpoint_best.pt (not the in-memory last-epoch weights) is
                # the authoritative artifact: downstream consumers (ICU's
                # per-step reload, standalone evaluation scripts) load it from
                # disk, discarding whatever is only in memory. Reload it here
                # before calibrating so we (a) calibrate the weights that will
                # actually be used, not stale last-epoch ones, and (b) can
                # persist the calibration back into that same file.
                best_ckpt_path = os.path.join(self.config["checkpoint_path"], "checkpoint_best.pt")
                if os.path.exists(best_ckpt_path):
                    print(f"Calibrating forget temperature on {best_ckpt_path}", flush=True)
                    best_ckpt = torch.load(best_ckpt_path, map_location=self.device)
                    self.model.load_state_dict(
                        best_ckpt["state_dict"] if "state_dict" in best_ckpt else best_ckpt,
                        strict=False,
                    )
                    self._calibrate_forget_temperature(val_loader_forget)
                    self.save_results(
                        epoch=best_ckpt.get("epoch", -1),
                        global_step=best_ckpt.get("global_step", n_step),
                        filename="checkpoint_best.pt",
                    )
                    print(f"  [ForgetTempCalib] Re-saved calibrated {best_ckpt_path}", flush=True)
                else:
                    print(
                        f"  [ForgetTempCalib] {best_ckpt_path} not found — calibrating "
                        "in-memory model only (not persisted to any checkpoint).",
                        flush=True,
                    )
                    self._calibrate_forget_temperature(val_loader_forget)

        return self.model
