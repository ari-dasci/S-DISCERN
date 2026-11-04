"""
ParametricMethod: DistributionalMethod with cross-validated family selection.

Same interface, setup()/unlearn() flow and synthetic-sample generation as
DistributionalMethod.  The only difference is HOW the per-class distribution
is chosen (and, consequently, how it is sampled):

Candidate families (one fit per class, no clustering)
-----------------------------------------------------
  gaussian            mu = mean, Sigma = Ledoit-Wolf
  tstudent            EM for mu, Sigma (LW on the weighted scatter at every
                      M-step) and nu (1-D maximisation, ECME)
  laplace_elliptical  x = mu + sqrt(w) Sigma^{1/2} eps, w ~ Exp(1).  EM with a
                      GIG E-step; Bessel functions in log scale (Debye for
                      large order).  Cov(x) = Sigma.
  vmf_norm            u = x/||x|| ~ vMF(mu, kappa)  +  norm model
  kent_tangent        v = Log_mu(u) ~ N(0, Sigma_T), Sigma_T (p-1)x(p-1) LW
                      +  norm model

DISCERN (the method of the paper) is the configuration
  pipeline="tests", selection="ll_tiebreak", tiebreak_by="ks_norm",
  cv_folds=5, cv_repeats=3, tie_margin=0.05, em_max_iter=20
(run tag ``cv5x3_tbn_m0.05_tests``; launchers: VARIANT=tiebreak_norm PIPELINE=tests).
The other options are the variants compared in the ablations.

Step 0 (once per class): norm-direction independence
-----------------------------------------------------
r = ||x||, u = x/r.  Spearman(r, cos(u, mu)), ridge LOO R^2 of log r on u
(permutation p-value) and distance correlation (permutation p-value),
Benjamini-Hochberg over the three p-values of the class.
  * not rejected:  log r ~ N(m, s2), independent of u
  * rejected:      log r | u ~ N(a.u + b, s2), a, b from ridge (alpha by LOO), s2 = mean
                   squared LOO residual
This decision fixes the norm model of vmf_norm / kent_tangent both in CV and
in sampling.  It is always computed, but only used when the chosen family is
vmf_norm or kent_tangent.

Step 1 (pipeline="tests"): DISTRIBUTIONAL's test cascade
--------------------------------------------------------
See ``distributional_tests``: Henze-Zirkler normality -> gaussian; thin shell
(Cr < 0.10) + Rayleigh + Rbar > 0.7 -> vmf_norm, or kent_tangent if the
sphericity test of the tangent-plane covariance rejects.  The first test that
accepts fixes the family and the CV of step 2 is skipped.  pipeline="cv" skips
this step.

Step 2: repeated K-fold CV (``cv_folds`` x ``cv_repeats``) in R^p
------------------------------------------------------------------
Only when no test accepts.  Every family is fitted on the training folds only
and scored with its full density in R^p on the validation fold (normalising
constants, log|Sigma|, polar and exponential-map Jacobians included); the KS
D of ||x - mean||, the KS D of ||x|| and the energy score of samples drawn
from the fit are recorded too.  ``selection`` decides (see ``select_family``):
"ll" takes the highest mean log-likelihood per point; "ll_tiebreak" (DISCERN)
treats as ties every family not significantly worse than the best
(one-sided Nadeau-Bengio corrected t-test, p > 0.05) or within ``tie_margin``
nats per dimension of it, and resolves the tie by ``tiebreak_by`` (DISCERN:
lowest KS D of ||x||; ties on that D again -> highest log-likelihood).
Classes with fewer than 2 * cv_folds points that reach this step get a
Gaussian.

Step 3: the chosen family is refitted on all the class data and sampled (if
that refit is degenerate, the next family in the CV ranking is used).
"""

import json
import math
import os
import warnings

import numpy as np
import torch
import yaml
from scipy import stats as sp_stats
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from scipy.special import gammaln, ive, kve
from sklearn.covariance import ledoit_wolf
from sklearn.model_selection import RepeatedKFold

from .DistributionalMethod import (
    DistributionalMethod,
    DistributionInfo,
    sample_from_distribution,
    _fit_kent_from_vmf,
    _fit_uniform,
    _hz_test,
    _radio_dispersion,
    _rayleigh_test,
    _uniformity_test,
)


LOG_2PI = math.log(2.0 * math.pi)
FAMILIES = ("gaussian", "tstudent", "laplace_elliptical", "vmf_norm", "kent_tangent")
# Families whose log-density is not already implemented by DistributionalMethod
# ("gaussian" / "tstudent" use the same params and parametrisation there).
NEW_TYPES = ("laplace_elliptical", "vmf_norm", "kent_tangent")

# Bessel orders >= this use the uniform asymptotic (Debye) expansion with
# corrections up to 1/nu^3; truncation error ~ u4/nu^4 < 1e-7 in log.
DEBYE_MIN_ORDER = 50.0

# Step-0 test settings
INDEP_ALPHA = 0.05
INDEP_N_PERM_DCOR = 200
INDEP_N_PERM_RIDGE = 100
INDEP_MAX_N = 2000          # subsample for the O(n^2) / permutation tests
RIDGE_ALPHAS = np.logspace(-3, 4, 15)


# =====================================================================
#  Log-Bessel functions (numerically stable)
# =====================================================================

def _debye_u(q):
    """Debye polynomials u1, u2, u3 evaluated at q = 1/sqrt(1+t^2)."""
    q2 = q * q
    u1 = q * (3.0 - 5.0 * q2) / 24.0
    u2 = q2 * (81.0 - 462.0 * q2 + 385.0 * q2 * q2) / 1152.0
    u3 = q * q2 * (30375.0 - 369603.0 * q2 + 765765.0 * q2 ** 2
                   - 425425.0 * q2 ** 3) / 414720.0
    return u1, u2, u3


def _log_kv_debye(v, z):
    """log K_v(z) by the uniform asymptotic expansion (large v)."""
    t = z / v
    s = np.sqrt(1.0 + t * t)
    eta = s + np.log(t / (1.0 + s))
    u1, u2, u3 = _debye_u(1.0 / s)
    series = 1.0 - u1 / v + u2 / v ** 2 - u3 / v ** 3
    return -v * eta + 0.5 * np.log(np.pi / (2.0 * v)) - 0.5 * np.log(s) + np.log(series)


def _log_iv_debye(v, z):
    """log I_v(z) by the uniform asymptotic expansion (large v)."""
    t = z / v
    s = np.sqrt(1.0 + t * t)
    eta = s + np.log(t / (1.0 + s))
    u1, u2, u3 = _debye_u(1.0 / s)
    series = 1.0 + u1 / v + u2 / v ** 2 + u3 / v ** 3
    return v * eta - 0.5 * np.log(2.0 * np.pi * v) - 0.5 * np.log(s) + np.log(series)


def log_kv(v, z):
    """Stable log K_v(z) for z > 0 (array-friendly).

    Large order: Debye expansion (kve overflows there: K_v(z) ~ (2v/z)^v).
    Small order: log(kve) - z, with the small-argument limit as fallback.
    """
    v = abs(float(v))
    z = np.maximum(np.asarray(z, dtype=np.float64), 1e-300)
    if v >= DEBYE_MIN_ORDER:
        return _log_kv_debye(v, z)
    with np.errstate(all="ignore"):
        out = np.log(kve(v, z)) - z
    bad = ~np.isfinite(out)
    if np.any(bad):
        zb = z[bad] if out.ndim else z
        if v > 0:
            small = gammaln(v) - math.log(2.0) + v * np.log(2.0 / zb)
        else:
            small = np.log(-np.log(zb / 2.0) - np.euler_gamma)
        if out.ndim:
            out[bad] = small
        else:
            out = small
    return out


def log_iv(v, z):
    """Stable log I_v(z) for z > 0 (array-friendly)."""
    v = abs(float(v))
    z = np.maximum(np.asarray(z, dtype=np.float64), 1e-300)
    if v >= DEBYE_MIN_ORDER:
        return _log_iv_debye(v, z)
    with np.errstate(all="ignore"):
        out = np.log(ive(v, z)) + z
    bad = ~np.isfinite(out)
    if np.any(bad):
        zb = z[bad] if out.ndim else z
        small = v * np.log(zb / 2.0) - gammaln(v + 1.0)
        if out.ndim:
            out[bad] = small
        else:
            out = small
    return out


class _TorchLogKv(torch.autograd.Function):
    """log K_v(z) for torch tensors, exact for every order: the value comes
    from ``log_kv`` (scipy / Debye, see above) and the gradient from
        d/dz log K_v(z) = -K_{v-1}(z) / K_v(z) - v / z
    (computed as a log-Bessel difference, so it never overflows)."""

    @staticmethod
    def forward(ctx, z, v):
        z_np = z.detach().cpu().double().numpy()
        out = torch.as_tensor(log_kv(v, z_np), dtype=z.dtype, device=z.device)
        ctx.save_for_backward(z)
        ctx.v = v
        return out

    @staticmethod
    def backward(ctx, grad):
        (z,) = ctx.saved_tensors
        v = abs(ctx.v)
        z_np = np.maximum(z.detach().cpu().double().numpy(), 1e-300)
        dlog = -np.exp(log_kv(v - 1.0, z_np) - log_kv(v, z_np)) - v / z_np
        return grad * torch.as_tensor(dlog, dtype=z.dtype, device=z.device), None


def _torch_log_kv(v, z):
    return _TorchLogKv.apply(z, float(v))


# =====================================================================
#  Linear-algebra helpers
# =====================================================================

def _chol(S):
    """Cholesky factor with increasing jitter if needed."""
    S = 0.5 * (S + S.T)
    d = S.shape[0]
    jitter = 0.0
    base = max(float(np.trace(S)) / d, 1e-12)
    for _ in range(8):
        try:
            return np.linalg.cholesky(S + jitter * np.eye(d))
        except np.linalg.LinAlgError:
            jitter = base * 1e-10 if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("Cholesky failed even with jitter")


def _logdet(L):
    return 2.0 * float(np.sum(np.log(np.diag(L))))


def _maha(X, mu, L):
    """Squared Mahalanobis distances of the rows of X, Sigma = L L^T."""
    Z = solve_triangular(L, (X - mu).T, lower=True, check_finite=False)
    return np.sum(Z * Z, axis=0)


def _lw(Y):
    """Ledoit-Wolf shrinkage of (1/n) Y^T Y (Y already centred / weighted)."""
    cov, _ = ledoit_wolf(Y, assume_centered=True)
    return cov


def _polar(X):
    r = np.maximum(np.linalg.norm(X, axis=1), 1e-300)
    return r, X / r[:, None]


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-300)


# =====================================================================
#  Sphere geometry: Householder frame, log / exp maps
# =====================================================================

def _householder_vec(mu):
    """h such that H = I - 2 h h^T swaps mu and e1 (None if mu ~ e1)."""
    e1 = np.zeros_like(mu)
    e1[0] = 1.0
    h = e1 - mu
    nh = np.linalg.norm(h)
    if nh < 1e-12:
        return None
    return h / nh


def _h_apply(h, V):
    """Apply the Householder reflection to the rows of V."""
    if h is None:
        return V
    return V - 2.0 * np.outer(V @ h, h)


def _log_map(mu, U):
    """Riemannian log map at mu of the unit rows U.  Returns (V, theta)."""
    c = np.clip(U @ mu, -1.0, 1.0)
    W = U - c[:, None] * mu
    s = np.linalg.norm(W, axis=1)
    theta = np.arctan2(s, c)
    scale = np.where(s > 1e-300, theta / np.maximum(s, 1e-300), 1.0)
    return W * scale[:, None], theta


def _exp_map(mu, V):
    nv = np.linalg.norm(V, axis=1)
    sinc = np.where(nv > 1e-12, np.sin(nv) / np.maximum(nv, 1e-300), 1.0)
    return np.cos(nv)[:, None] * mu + sinc[:, None] * V


def _log_theta_over_sin(theta):
    th = np.maximum(theta, 1e-8)
    return np.log(th) - np.log(np.maximum(np.sin(th), 1e-300))


def _karcher_mean(U, max_iter=50, tol=1e-10):
    """Intrinsic (Frechet) mean direction, so that E[Log_mu(u)] = 0."""
    mu = _unit(U.mean(axis=0))
    for _ in range(max_iter):
        V, _ = _log_map(mu, U)
        step = V.mean(axis=0)
        mu = _unit(_exp_map(mu, step[None, :])[0])
        if np.linalg.norm(step) < tol:
            break
    return mu


# =====================================================================
#  Norm model (log r, optionally regressed on u)
# =====================================================================

def _ridge_decomp(U):
    """SVD of the centred design, reused for every alpha and every response y.

    Ridge with an unpenalised intercept = ridge on centred U, y.  With
    U_c = A diag(s) B^T and f = s^2 / (s^2 + alpha):
        y_hat = A diag(f) A^T y_c + mean(y),   h_ii = sum_k A_ik^2 f_k + 1/n,
        LOO residual_i = (y_i - y_hat_i) / (1 - h_ii)
    — the same closed form RidgeCV (GCV mode) uses, but the SVD is computed
    once, so permutation nulls only cost a few matrix-vector products.
    """
    Um = U.mean(axis=0)
    A, s, Bt = np.linalg.svd(U - Um, full_matrices=False)
    return {"A": A, "A2": A * A, "s2": s * s, "s": s, "Bt": Bt, "Um": Um, "n": U.shape[0]}


def _ridge_loo_eval(dec, y, alphas=RIDGE_ALPHAS, with_coef=False):
    """Best alpha by LOO MSE.  Returns (alpha, loo_mse[, coef, intercept])."""
    ym = float(y.mean())
    Aty = dec["A"].T @ (y - ym)
    best_alpha, best_mse = None, np.inf
    for alpha in alphas:
        f = dec["s2"] / (dec["s2"] + alpha)
        resid = (y - (dec["A"] @ (f * Aty) + ym)) / (1.0 - (dec["A2"] @ f + 1.0 / dec["n"]))
        mse = float(np.mean(resid * resid))
        if mse < best_mse:
            best_alpha, best_mse = float(alpha), mse
    if not with_coef:
        return best_alpha, best_mse
    coef = dec["Bt"].T @ (dec["s"] / (dec["s2"] + best_alpha) * Aty)
    return best_alpha, best_mse, coef, ym - float(dec["Um"] @ coef)


def _fit_norm_model(U, r, dependent):
    log_r = np.log(r)
    if not dependent:
        s2 = float(np.var(log_r, ddof=1)) if len(log_r) > 1 else 1e-12
        return {"dependent": False, "m": float(log_r.mean()), "s2": max(s2, 1e-12)}
    alpha, loo_mse, coef, intercept = _ridge_loo_eval(_ridge_decomp(U), log_r, with_coef=True)
    return {
        "dependent": True,
        "a": np.asarray(coef, dtype=np.float64),
        "b": float(intercept),
        "s2": max(loo_mse, 1e-12),
        "alpha": alpha,
    }


def _norm_mean(nm, U):
    if nm["dependent"]:
        return U @ nm["a"] + nm["b"]
    return np.full(U.shape[0], nm["m"])


def _radial_logpdf(nm, U, r, p):
    """log N(log r; m(u), s2) - log r - (p-1) log r."""
    lr = np.log(r)
    s2 = nm["s2"]
    return -0.5 * (LOG_2PI + math.log(s2) + (lr - _norm_mean(nm, U)) ** 2 / s2) - p * lr


def _sample_norm(nm, U, rng):
    return np.exp(_norm_mean(nm, U) + math.sqrt(nm["s2"]) * rng.standard_normal(U.shape[0]))


# =====================================================================
#  Step 0: norm-direction independence test
# =====================================================================

def _double_centered(D):
    return D - D.mean(0, keepdims=True) - D.mean(1, keepdims=True) + D.mean()


def _dcor_perm_test(r, U, n_perm, rng):
    """Distance correlation between scalar r and vectors U, permutation p-value."""
    A = _double_centered(np.abs(r[:, None] - r[None, :]))
    sq = np.sum(U ** 2, 1)
    B = _double_centered(np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * U @ U.T, 0, None)))
    denom = math.sqrt((A * A).mean() * (B * B).mean()) + 1e-30
    # float32 halves the memory traffic of the O(n^2) permutation loop;
    # row-then-column take is faster than np.ix_ fancy indexing.
    A32, B32 = A.astype(np.float32), B.astype(np.float32)

    def stat(perm):
        Ap = A32.take(perm, 0).take(perm, 1)
        return float((Ap * B32).sum(dtype=np.float64)) / A.size / denom

    obs = stat(np.arange(len(r)))
    null = np.array([stat(rng.permutation(len(r))) for _ in range(n_perm)])
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return math.sqrt(max(obs, 0.0)), float(p)


def _bh_reject(pvals, alpha):
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    passed = p[order] <= alpha * np.arange(1, len(p) + 1) / len(p)
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(len(p), dtype=bool)
    rej[order[:k]] = True
    return rej


def norm_direction_test(X, rng, alpha=INDEP_ALPHA, n_perm_dcor=INDEP_N_PERM_DCOR,
                        n_perm_ridge=INDEP_N_PERM_RIDGE, max_n=INDEP_MAX_N):
    """Step 0.  Returns a dict; ``dependent`` is True when BH rejects independence."""
    r, U = _polar(X)
    mu = _unit(U.mean(axis=0))
    rho, p_rho = sp_stats.spearmanr(r, U @ mu)
    if not np.isfinite(p_rho):
        rho, p_rho = 0.0, 1.0

    if len(r) > max_n:
        idx = rng.choice(len(r), max_n, replace=False)
        r_s, U_s = r[idx], U[idx]
    else:
        r_s, U_s = r, U
    log_r = np.log(r_s)
    var_y = max(float(np.var(log_r)), 1e-300)
    dec = _ridge_decomp(U_s)
    _, loo = _ridge_loo_eval(dec, log_r)
    r2 = 1.0 - loo / var_y
    r2_null = np.array([1.0 - _ridge_loo_eval(dec, rng.permutation(log_r))[1] / var_y
                        for _ in range(n_perm_ridge)])
    p_ridge = float((1 + np.sum(r2_null >= r2)) / (n_perm_ridge + 1))

    dcor, p_dcor = _dcor_perm_test(r_s, U_s, n_perm_dcor, rng)

    pvals = [float(p_rho), p_ridge, p_dcor]
    rej = _bh_reject(pvals, alpha)
    return {
        "dependent": bool(rej.any()),
        "spearman_rho": float(rho), "p_spearman": float(p_rho),
        "ridge_loo_r2": float(r2), "p_ridge": p_ridge,
        "dcor": float(dcor), "p_dcor": p_dcor,
        "bh_reject": [bool(x) for x in rej],
        "n_tested": int(len(r_s)),
    }


# =====================================================================
#  Families: fit / logpdf (per point, in R^p) / sample
# =====================================================================

# ---- Gaussian ---------------------------------------------------------

def _fit_gaussian(X, **_):
    mu = X.mean(axis=0)
    return {"mean": mu, "cov": _lw(X - mu)}


def _logpdf_gaussian(X, prm):
    L = _chol(prm["cov"])
    p = X.shape[1]
    return -0.5 * (p * LOG_2PI + _logdet(L) + _maha(X, prm["mean"], L))


# ---- multivariate t (Sigma = scatter / dispersion matrix) -------------

def _t_logpdf_from_q(q, p, logdet, nu):
    return (gammaln(0.5 * (nu + p)) - gammaln(0.5 * nu) - 0.5 * p * math.log(nu * math.pi)
            - 0.5 * logdet - 0.5 * (nu + p) * np.log1p(q / nu))


# ---- Scale mixtures (t, Laplace): moment-matched scale + degeneracy ----
#
# Plain EM for the scatter of a scale mixture is unreliable on embeddings:
#   * the MLE does not exist when the points lie (close to) a lower-dimensional
#     subspace (Kent & Tyler 1991) — effective rank ~30 << p ~512, or n <= p
#     (CIFAR100): tr(Sigma) explodes, q_i -> 0, nu -> 0;
#   * even when it exists, Ledoit-Wolf inside the M-step shrinks the
#     eigenvalues, the in-sample q_i come out too small and the weights
#     E[1/w] ~ p/q inflate the scale (true Laplace data, p=256, n=500:
#     tr(Sigma) = 3.4 x tr(Cov) after 4 iterations, radii of the samples x1.8).
# Therefore the SCALE of Sigma is fixed by the second moment after every M-step
# and the EM only estimates mu, the SHAPE of Sigma (and nu):
#     Laplace : Cov(x) = Sigma              ->  tr(Sigma) = tr(S_LW)
#     t       : Cov(x) = nu/(nu-2) Sigma    ->  tr(Sigma) = (nu-2)/nu tr(S_LW), nu > 2
# (S_LW = Ledoit-Wolf sample covariance, consistent for Cov; LW preserves the
# trace, so this is also the scale the Gaussian uses).  The fit is the MLE of
# mu / shape / nu under the moment constraint; the CV log-likelihood is still
# the full, honest density.  A fit is flagged ``degenerate`` (excluded) when
#   * nu reaches its lower bound (t needs finite variance, nu > 2);
#   * the EM does not meet the 1e-6 relative tolerance in EM_MAX_ITER iterations;
#   * (Laplace) some q_i collapses, i.e. mu is moving onto a data point.

EM_MAX_ITER = 20   # default; --parametric-em-max-iter


def _degenerate(prm, reason):
    prm["degenerate"] = reason
    return prm


def _rescale(cov, target_trace):
    return cov * (target_trace / float(np.trace(cov)))


def _fit_tstudent(X, max_iter=EM_MAX_ITER, tol=1e-6, nu_bounds=(2.05, 1000.0), **_):
    """ECME for mu, shape(Sigma) (LW on the weighted scatter) and nu, with
    tr(Sigma) = (nu-2)/nu tr(S_LW) (see the note above)."""
    n, p = X.shape
    mu = X.mean(axis=0)
    cov = _lw(X - mu)
    tr_ref = float(np.trace(cov))
    nu = 10.0
    cov = _rescale(cov, (nu - 2.0) / nu * tr_ref)
    prev = -np.inf
    log_lo, log_hi = math.log(nu_bounds[0]), math.log(nu_bounds[1])

    def _opt_nu(q, logdet):
        res = minimize_scalar(
            lambda ln: -float(np.sum(_t_logpdf_from_q(q, p, logdet, math.exp(ln)))),
            bounds=(log_lo, log_hi), method="bounded", options={"xatol": 1e-4},
        )
        return math.exp(res.x), -float(res.fun)

    for it in range(max_iter):
        L = _chol(cov)
        logdet = _logdet(L)
        q = _maha(X, mu, L)
        nu, ll = _opt_nu(q, logdet)
        if nu <= nu_bounds[0] * 1.01:
            return _degenerate({"mean": mu, "cov": cov, "df": float(nu), "em_iter": it + 1},
                               f"nu at lower bound ({nu:.2f}, infinite variance)")
        if np.isfinite(prev) and abs(ll - prev) <= tol * abs(prev):
            break
        prev = ll
        w = (nu + p) / (nu + q)
        mu = (w @ X) / w.sum()
        cov = _rescale(_lw(np.sqrt(w)[:, None] * (X - mu)), (nu - 2.0) / nu * tr_ref)
    else:
        return _degenerate({"mean": mu, "cov": cov, "df": float(nu), "em_iter": max_iter},
                           f"no convergence in {max_iter} iterations")
    return {"mean": mu, "cov": cov, "df": float(nu), "em_iter": it + 1}


def _logpdf_tstudent(X, prm):
    L = _chol(prm["cov"])
    return _t_logpdf_from_q(_maha(X, prm["mean"], L), X.shape[1], _logdet(L), prm["df"])


# ---- symmetric multivariate (elliptical) Laplace ---------------------

_Q_FLOOR = 1e-12


def _laplace_logpdf_from_q(q, p, logdet):
    """log f = log 2 - p/2 log 2pi - 1/2 log|S| + (lam/2) log(q/2) + log K_{p/2-1}(sqrt(2q))."""
    q = np.maximum(q, _Q_FLOOR)
    lam = 1.0 - 0.5 * p
    return (math.log(2.0) - 0.5 * p * LOG_2PI - 0.5 * logdet
            + 0.5 * lam * np.log(0.5 * q) + log_kv(0.5 * p - 1.0, np.sqrt(2.0 * q)))


def _laplace_inv_w(q, p):
    """E[1/w | x] for w | x ~ GIG(1 - p/2, chi=q, psi=2), via log-Bessel differences."""
    q = np.maximum(q, _Q_FLOOR)
    z = np.sqrt(2.0 * q)
    return np.sqrt(2.0 / q) * np.exp(log_kv(0.5 * p, z) - log_kv(0.5 * p - 1.0, z))


def _fit_laplace_elliptical(X, max_iter=EM_MAX_ITER, tol=1e-6, **_):
    """EM for mu and shape(Sigma) with tr(Sigma) = tr(S_LW) (Cov(x) = Sigma);
    stops when the train log-likelihood changes < tol (relative).

    The density is unbounded at mu for p >= 2, so the likelihood can be driven
    to +inf by placing mu on a data point.  If an iteration makes some q_i
    collapse (or the log-likelihood non-finite) the fit is flagged degenerate.
    """
    n, p = X.shape
    mu = X.mean(axis=0)
    cov = _lw(X - mu)
    tr_ref = float(np.trace(cov))
    prev = -np.inf
    prev_params = (mu, cov)
    for it in range(max_iter):
        L = _chol(cov)
        logdet = _logdet(L)
        q = _maha(X, mu, L)
        ll = float(np.sum(_laplace_logpdf_from_q(q, p, logdet)))
        if not np.isfinite(ll) or np.min(q) < 1e-8 * np.median(q):
            mu, cov = prev_params
            return _degenerate({"mean": mu, "cov": cov, "em_iter": it + 1},
                               "q collapse (mu moving onto a data point)")
        if np.isfinite(prev) and abs(ll - prev) <= tol * abs(prev):
            break
        prev, prev_params = ll, (mu, cov)
        a = _laplace_inv_w(q, p)
        mu = (a @ X) / a.sum()
        cov = _rescale(_lw(np.sqrt(a)[:, None] * (X - mu)), tr_ref)
    else:
        return _degenerate({"mean": mu, "cov": cov, "em_iter": max_iter},
                           f"no convergence in {max_iter} iterations")
    return {"mean": mu, "cov": cov, "em_iter": it + 1}



def _logpdf_laplace_elliptical(X, prm):
    L = _chol(prm["cov"])
    return _laplace_logpdf_from_q(_maha(X, prm["mean"], L), X.shape[1], _logdet(L))


# ---- von Mises-Fisher direction + norm model ------------------------

def _vmf_log_c(p, kappa):
    v = 0.5 * p - 1.0
    return v * math.log(kappa) - 0.5 * p * LOG_2PI - float(log_iv(v, kappa))


def _vmf_kappa(rbar, p, n_newton=10):
    """Banerjee et al. (2005) approximation refined with Newton on A_p(kappa) = rbar."""
    rbar = float(np.clip(rbar, 1e-12, 1.0 - 1e-12))
    kappa = rbar * (p - rbar ** 2) / (1.0 - rbar ** 2)
    for _ in range(n_newton):
        A = math.exp(float(log_iv(0.5 * p, kappa) - log_iv(0.5 * p - 1.0, kappa)))
        dA = 1.0 - A * A - (p - 1.0) / kappa * A
        if not np.isfinite(dA) or dA <= 0:
            break
        new = kappa - (A - rbar) / dA
        if new <= 0:
            new = 0.5 * kappa
        if abs(new - kappa) <= 1e-10 * kappa:
            kappa = new
            break
        kappa = new
    return float(kappa)


def _fit_vmf_norm(X, dependent=False, **_):
    r, U = _polar(X)
    p = X.shape[1]
    m = U.mean(axis=0)
    rbar = float(np.linalg.norm(m))
    kappa = _vmf_kappa(rbar, p)
    return {
        "mu": _unit(m), "kappa": kappa, "log_c": _vmf_log_c(p, kappa),
        "norm": _fit_norm_model(U, r, dependent),
    }


def _logpdf_vmf_norm(X, prm):
    r, U = _polar(X)
    return prm["log_c"] + prm["kappa"] * (U @ prm["mu"]) + _radial_logpdf(prm["norm"], U, r, X.shape[1])


def _sample_vmf_dirs(mu, kappa, n, rng):
    """Wood (1994), vectorised rejection."""
    p = len(mu)
    pm1 = p - 1.0
    b = pm1 / (2.0 * kappa + math.sqrt(4.0 * kappa ** 2 + pm1 ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + pm1 * math.log(1.0 - x0 ** 2)
    w = np.empty(n)
    filled = 0
    while filled < n:
        m = max(2 * (n - filled), 16)
        z = rng.beta(0.5 * pm1, 0.5 * pm1, size=m)
        ww = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        acc = kappa * ww + pm1 * np.log(1.0 - x0 * ww) - c >= np.log(rng.random(m))
        take = ww[acc][: n - filled]
        w[filled:filled + len(take)] = take
        filled += len(take)
    xi = rng.standard_normal((n, p - 1))
    xi /= np.maximum(np.linalg.norm(xi, axis=1, keepdims=True), 1e-300)
    local = np.hstack([w[:, None], np.sqrt(np.clip(1.0 - w * w, 0.0, None))[:, None] * xi])
    return _h_apply(_householder_vec(mu), local)


# ---- "Kent": normal in the tangent plane + norm model ---------------

def _tangent_coords(mu, U):
    V, theta = _log_map(mu, U)
    return _h_apply(_householder_vec(mu), V)[:, 1:], theta


def _fit_kent_tangent(X, dependent=False, **_):
    r, U = _polar(X)
    mu = _karcher_mean(U)
    C, _ = _tangent_coords(mu, U)
    return {"mu": mu, "tangent_cov": _lw(C), "norm": _fit_norm_model(U, r, dependent)}


def _logpdf_kent_tangent(X, prm):
    r, U = _polar(X)
    p = X.shape[1]
    C, theta = _tangent_coords(prm["mu"], U)
    L = _chol(prm["tangent_cov"])
    log_n = -0.5 * ((p - 1) * LOG_2PI + _logdet(L) + _maha(C, 0.0, L))
    jac = (p - 2) * _log_theta_over_sin(theta)
    return log_n + jac + _radial_logpdf(prm["norm"], U, r, p)


_FIT = {
    "gaussian": _fit_gaussian,
    "tstudent": _fit_tstudent,
    "laplace_elliptical": _fit_laplace_elliptical,
    "vmf_norm": _fit_vmf_norm,
    "kent_tangent": _fit_kent_tangent,
}
_LOGPDF = {
    "gaussian": _logpdf_gaussian,
    "tstudent": _logpdf_tstudent,
    "laplace_elliptical": _logpdf_laplace_elliptical,
    "vmf_norm": _logpdf_vmf_norm,
    "kent_tangent": _logpdf_kent_tangent,
}


# ---- Option B: PCA subspace + isotropic residual (elliptical only) ----
#
# x = V z + e, V (p x d) orthonormal = span(mean direction, top-k centred PCs
# explaining ``var_frac`` of the variance); z = V^T x is modelled by the
# family in d dims; e lives in the (p-d)-dim complement, e ~ N(m_e, s2 I).
# Since [V, V_perp] is a rotation, the density in R^p is
#     log f(x) = log f_family(V^T x) + log N(e; m_e, s2 I_{p-d}).
#
# Only applied to the elliptical families (SUBSPACE_FAMILIES): a linear
# projection of an elliptical law is elliptical of the same family, so the
# model stays coherent (the full-space law is "family x independent Gaussian
# residual", PPCA-style: the mixing weight w only sees the subspace).  The
# projection is linear and not injective, so it does NOT preserve the
# norm/direction structure vMF/Kent are built on (||V^T x|| != ||x||, and
# the isotropic residual perturbs norm and direction independently of the
# fitted norm model): those families are always fitted in full R^p.

SUBSPACE_FAMILIES = ("gaussian", "tstudent", "laplace_elliptical")

def _fit_subspace(X, var_frac):
    n, p = X.shape
    m = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - m, full_matrices=False)
    ev = s * s / n
    k = int(np.searchsorted(np.cumsum(ev) / max(ev.sum(), 1e-300), var_frac) + 1)
    k = min(k, len(ev), p - 1)
    cols = [Vt[:k].T]
    nm = np.linalg.norm(m)
    if nm > 1e-12:
        cols.insert(0, (m / nm)[:, None])
    Q, R = np.linalg.qr(np.hstack(cols))
    V = Q[:, np.abs(np.diag(R)) > 1e-8 * max(np.abs(np.diag(R)).max(), 1e-300)]
    d = V.shape[1]
    E = X - (X @ V) @ V.T
    m_e = E.mean(axis=0)
    r = p - d
    s2 = float(np.sum((E - m_e) ** 2) / (n * r)) if r > 0 else float("nan")
    return {"V": V, "m_e": m_e, "s2": max(s2, 1e-12) if r > 0 else s2, "k": int(k), "d": int(d)}


def _subspace_residual_logpdf(X, sub):
    V = sub["V"]
    r = X.shape[1] - V.shape[1]
    if r == 0:
        return np.zeros(X.shape[0])
    E = X - (X @ V) @ V.T - sub["m_e"]
    return -0.5 * (r * (LOG_2PI + math.log(sub["s2"])) + np.sum(E * E, axis=1) / sub["s2"])


def family_logpdf(dist_type, X, params):
    """Per-point log-density in R^p of a fitted PARAMETRIC family."""
    X = np.asarray(X, dtype=np.float64)
    if "subspace" in params:
        sub = params["subspace"]
        return (_LOGPDF[dist_type](X @ sub["V"], params["inner"])
                + _subspace_residual_logpdf(X, sub))
    return _LOGPDF[dist_type](X, params)


def fit_family(dist_type, X, dependent=False, subspace=None, em_max_iter=None):
    """Fit a family in R^p, or inside a precomputed ``_fit_subspace`` basis.
    ``em_max_iter`` caps the t / Laplace EM (hitting it = degenerate fit)."""
    X = np.asarray(X, dtype=np.float64)
    kw = {"dependent": dependent}
    if em_max_iter is not None:
        kw["max_iter"] = int(em_max_iter)
    if subspace is not None:
        return {"subspace": subspace, "inner": _FIT[dist_type](X @ subspace["V"], **kw)}
    return _FIT[dist_type](X, **kw)


def sample_parametric(dist_type, params, n, rng):
    """Draw n samples (numpy).  Returns (samples, diagnostics dict)."""
    if "subspace" in params:
        sub = params["subspace"]
        V = sub["V"]
        Z, diag = sample_parametric(dist_type, params["inner"], n, rng)
        X = Z @ V.T
        if V.shape[0] > V.shape[1]:
            G = rng.standard_normal((n, V.shape[0]))
            X += sub["m_e"] + math.sqrt(sub["s2"]) * (G - (G @ V) @ V.T)
        return X, diag

    diag = {}
    if dist_type in ("gaussian", "tstudent", "laplace_elliptical"):
        mu = params["mean"]
        L = _chol(params["cov"])
        eps = rng.standard_normal((n, len(mu)))
        if dist_type == "gaussian":
            w = np.ones(n)
        elif dist_type == "tstudent":
            nu = params["df"]
            w = nu / rng.chisquare(nu, size=n)
        else:
            w = rng.exponential(1.0, size=n)
        return mu + np.sqrt(w)[:, None] * (eps @ L.T), diag

    if dist_type == "vmf_norm":
        U = _sample_vmf_dirs(params["mu"], params["kappa"], n, rng)
    elif dist_type == "kent_tangent":
        mu = params["mu"]
        L = _chol(params["tangent_cov"])
        C = rng.standard_normal((n, len(mu) - 1)) @ L.T
        diag["frac_v_gt_pi"] = float(np.mean(np.linalg.norm(C, axis=1) > math.pi))
        V = _h_apply(_householder_vec(mu), np.hstack([np.zeros((n, 1)), C]))
        U = _exp_map(mu, V)
        U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-300)
    else:
        raise ValueError(f"Unknown PARAMETRIC family: {dist_type}")
    r = _sample_norm(params["norm"], U, rng)
    return r[:, None] * U, diag


# =====================================================================
#  Step 1: CV selection
# =====================================================================

def _nadeau_bengio(a, b, n_train, n_test):
    """Corrected resampled paired t-test (Nadeau & Bengio 2003)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    k = len(d)
    if k < 2 or not np.all(np.isfinite(d)):
        return float("nan"), float("nan")
    var = (1.0 / k + n_test / n_train) * d.var(ddof=1)
    if var <= 0:
        return float("nan"), float("nan")
    t = d.mean() / math.sqrt(var)
    return float(t), float(2.0 * sp_stats.t.sf(abs(t), df=k - 1))


def _nb_tie_p(a, b, n_train, n_test, alternative):
    """p-value of the corrected resampled paired t-test (Nadeau & Bengio 2003)
    on d = a - b, used to decide ties in ``tiebreak``.
      "two-sided": H1 mean(d) != 0
      "less"     : H1 mean(d) < 0   (a significantly worse when higher is better)
    Folds with identical differences (zero variance) give t = 0 (or +-inf if
    the constant difference is not 0); non-finite differences give nan."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    k = len(d)
    if k < 2 or not np.all(np.isfinite(d)):
        return float("nan")
    var = (1.0 / k + n_test / n_train) * d.var(ddof=1)
    if var > 0:
        t = d.mean() / math.sqrt(var)
    else:
        t = 0.0 if d.mean() == 0 else math.copysign(np.inf, d.mean())
    if alternative == "less":
        return float(sp_stats.t.cdf(t, df=k - 1))
    return float(2.0 * sp_stats.t.sf(abs(t), df=k - 1))




def _ks_dist_to_mean(X_ref, X_syn, centre):
    return float(sp_stats.ks_2samp(np.linalg.norm(X_ref - centre, axis=1),
                                   np.linalg.norm(X_syn - centre, axis=1)).statistic)


def _ks_norm(X_ref, X_syn):
    """KS D between the norms ||x|| (the spherical shell): sensitive to the
    shell thickness / curvature that ||x - mean|| barely sees."""
    return float(sp_stats.ks_2samp(np.linalg.norm(X_ref, axis=1),
                                   np.linalg.norm(X_syn, axis=1)).statistic)


ENERGY_MAX_N = 1000


def _energy_score(Y, S, rng, max_n=ENERGY_MAX_N):
    """Sample estimate of the energy score of the model behind samples S on
    the observations Y (lower is better):
        ES = E||S - y|| - 1/2 E||S - S'||,   averaged over y in Y.
    Strictly proper scoring rule in R^p (minimised in expectation by the true
    distribution), density-free: it judges the samples themselves."""
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]
    if len(S) > max_n:
        S = S[rng.choice(len(S), max_n, replace=False)]
    m = len(S)
    term_ys = cdist(Y, S).mean()
    term_ss = cdist(S, S).sum() / (m * (m - 1))
    return float(term_ys - 0.5 * term_ss)


TIE_MARGIN = 0.05   # nats per dimension; default of --parametric-tie-margin


def _ks_noise(n_class):
    """5% two-sample KS critical value for the MEAN D over the CV folds.

    Each repetition of the K-fold CV puts every point of the class in exactly
    one validation fold, so the mean of the K fold-wise D values carries about
    the information of a single KS with n = n_class points per side:
    c_0.05 sqrt(2 / n_class) (~0.027 for n = 5000, vs ~0.061 for one fold of
    1000).  Averaging over repetitions only reduces it further, so this is
    still a conservative tie band.

    Only used by the "ll_ks" / "ll_ksnorm" filters; the "ll_tiebreak" KS
    tie-break uses a corrected paired test instead (see ``_ks_rule``)."""
    return 1.358 * math.sqrt(2.0 / n_class)


def tiebreak(table, ks_table, es_table, n_train, n_val, p, margin=TIE_MARGIN, alpha=0.05,
             knorm_table=None):
    """ll_tiebreak rule on the per-fold CV tables (can also be applied offline
    to the stored tables to vary the margin).

    Ties with the best-ll family: every family whose mean ll is
      * not significantly worse: one-sided corrected paired t-test
        (Nadeau-Bengio, H1: worse than the best), p > alpha, or
      * practically equivalent: worse by at most ``margin`` nats per dimension
        (i.e. margin * p nats per point).
    Resolution, all always computed:
      KS     : lowest mean D of ||x - mean||; families whose per-fold D is not
               significantly different from it (two-sided corrected paired
               t-test, p > alpha) tie again -> highest ll.
      KS norm: same rule on the KS D of ||x|| (only if knorm_table is given).
      energy : lowest mean energy score; energies not significantly worse
               (corrected t-test, p > alpha) tie again -> highest ll.
    """
    fams = list(table)
    mean = {f: float(np.mean(table[f])) for f in fams}
    es_mean = {f: float(np.mean(es_table[f])) for f in fams}
    by_ll = sorted(fams, key=lambda f: -mean[f])
    best = by_ll[0]
    tie_p, gap = {}, {}
    for f in by_ll[1:]:
        if np.isfinite(mean[f]):
            tie_p[f] = _nb_tie_p(table[f], table[best], n_train, n_val, "less")
            gap[f] = (mean[best] - mean[f]) / p
    candidates = [best] + [f for f in by_ll[1:] if f in tie_p and
                           ((np.isfinite(tie_p[f]) and tie_p[f] > alpha) or gap[f] <= margin)]
    e_best = min(candidates, key=lambda f: es_mean[f])
    fin_es = [e_best]
    for f in candidates:
        if f != e_best and np.isfinite(es_mean[f]):
            pe = _nadeau_bengio(es_table[f], es_table[e_best], n_train, n_val)[1]
            if np.isfinite(pe) and pe > alpha:
                fin_es.append(f)
    return {"tie_margin": margin, "tie_candidates": candidates, "tie_p_vs_best": tie_p,
            "tie_gap_per_dim": gap,
            "tie_finalists_energy": fin_es, "tie_winner_energy": max(fin_es, key=lambda f: mean[f]),
            **_ks_rule(ks_table, candidates, mean, n_train, n_val, alpha, "ks"),
            **_ks_rule(knorm_table, candidates, mean, n_train, n_val, alpha, "ksnorm")}


def _ks_rule(ks_table, candidates, mean, n_train, n_val, alpha, name):
    """Lowest mean D among the candidates; families whose per-fold D is not
    significantly different from it (two-sided corrected paired t-test,
    p > alpha) tie again -> highest ll."""
    if ks_table is None:
        return {}
    ks_mean = {f: float(np.mean(ks_table[f])) for f in candidates}
    f_min = min(candidates, key=lambda f: ks_mean[f])
    fin, ks_p = [f_min], {}
    for f in candidates:
        if f == f_min:
            continue
        ks_p[f] = _nb_tie_p(ks_table[f], ks_table[f_min], n_train, n_val, "two-sided")
        if np.isfinite(ks_p[f]) and ks_p[f] > alpha:
            fin.append(f)
    return {f"tie_finalists_{name}": fin, f"tie_winner_{name}": max(fin, key=lambda f: mean[f]),
            f"tie_{name}_p_vs_min": ks_p}


def select_family(X, cv_folds, seed, dependent, families=FAMILIES,
                  selection="ll", ks_max=0.1, subspace="none", pca_variance=0.99,
                  cv_repeats=1, tiebreak_by="ks", em_max_iter=EM_MAX_ITER, tie_margin=TIE_MARGIN):
    """K-fold CV per family: held-out log-likelihood per point (R^p density),
    KS D of ||x - mean_train|| and energy score of samples vs the validation fold.

    All three tables are always computed and logged; ``selection`` decides:
      "ll"     : highest mean log-likelihood.
      "ll_ksnorm" : same filter on the KS D of the norms ||x|| (the spherical
                 shell the classifier sees), then highest log-likelihood.
      "ll_ks"  : (option A) highest log-likelihood among families whose mean
                 KS D <= ks_max + _ks_noise(n) (ks_max beyond the
                 KS sampling noise); if none: smallest D.
      "ll_tiebreak" : see ``tiebreak``: families whose ll is not significantly
                 worse than the best (one-sided corrected paired t-test, p > 0.05) or
                 within ``tie_margin`` nats per dimension of it are ties.  The tie is
                 resolved by ``tiebreak_by`` and the winner under the other
                 criterion is reported too:
                   "ks"     : lowest KS D; D not significantly different
                              (two-sided corrected paired t-test on the
                              per-fold D, p > 0.05) tie again -> highest ll.
                   "ks_norm": same on the KS D of ||x||.
                   "energy" : lowest energy score; energies not significantly
                              worse (corrected t-test, p > 0.05) tie again -> highest ll.
      "energy" : lowest mean energy score (samples closest to validation in R^p).

    Degenerate scale-mixture fits (see the scale-mixture note) get ll = -inf,
    D = 1, ES = inf on that fold, so they never win.

    subspace="pca" (option B): the elliptical families are fitted in the PCA
    subspace of the training fold (see ``_fit_subspace``); vMF / Kent stay in
    full R^p.  All densities are in R^p, so the comparison remains valid.

    cv_repeats > 1: repeated K-fold (different shuffles).  The paired tests
    then use all K*r folds with the Nadeau-Bengio variance correction
    (1/(K r) + n_val/n_train), i.e. the corrected repeated k-fold t-test of
    Bouckaert & Frank (2004), df = K r - 1: more power than a single K-fold.

    Returns (ranking, report); ranking[0] is the winner, the rest are the
    fall-backs in order should the final refit on all data be degenerate.
    """
    X = np.asarray(X, dtype=np.float64)
    kf = RepeatedKFold(n_splits=cv_folds, n_repeats=max(int(cv_repeats), 1), random_state=seed)
    table = {f: [] for f in families}
    ks_table = {f: [] for f in families}
    kn_table = {f: [] for f in families}
    es_table = {f: [] for f in families}
    degenerate = {f: [] for f in families}
    n_tr, n_te, dims = [], [], []
    for fold, (tr, va) in enumerate(kf.split(X)):
        n_tr.append(len(tr))
        n_te.append(len(va))
        sub = _fit_subspace(X[tr], pca_variance) if subspace == "pca" else None
        if sub is not None:
            dims.append(sub["d"])
        centre = X[tr].mean(axis=0)
        fold_rng = np.random.default_rng(seed + 1000 + fold)
        for fam in families:
            val, ks, kn, es, why = -np.inf, 1.0, 1.0, np.inf, None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    prm = fit_family(fam, X[tr], dependent=dependent,
                                     subspace=sub if fam in SUBSPACE_FAMILIES else None,
                                     em_max_iter=em_max_iter)
                    why = prm.get("inner", prm).get("degenerate")
                    if why is None:
                        val = float(np.mean(family_logpdf(fam, X[va], prm)))
                        syn, _ = sample_parametric(fam, prm, len(va), fold_rng)
                        ks = _ks_dist_to_mean(X[va], syn, centre)
                        kn = _ks_norm(X[va], syn)
                        es = _energy_score(X[va], syn, fold_rng)
            except Exception as exc:  # a failing family just loses
                print(f"    [PARAMETRIC] {fam} failed on fold {fold}: {exc!r}", flush=True)
                why = f"error: {exc!r}"
            if why is not None:
                val, ks, kn, es = -np.inf, 1.0, 1.0, np.inf
            table[fam].append(val if np.isfinite(val) else -np.inf)
            ks_table[fam].append(ks if np.isfinite(ks) else 1.0)
            kn_table[fam].append(kn if np.isfinite(kn) else 1.0)
            es_table[fam].append(es if np.isfinite(es) else np.inf)
            degenerate[fam].append(why)

    mean = {f: float(np.mean(v)) for f, v in table.items()}
    ks_mean = {f: float(np.mean(v)) for f, v in ks_table.items()}
    kn_mean = {f: float(np.mean(v)) for f, v in kn_table.items()}
    es_mean = {f: float(np.mean(v)) for f, v in es_table.items()}
    out = {"cv_table": table, "cv_mean": mean, "selection": selection,
           "ks_table": ks_table, "ks_mean": ks_mean,
           "ksnorm_table": kn_table, "ksnorm_mean": kn_mean,
           "energy_table": es_table, "energy_mean": es_mean,
           "degenerate": {f: v for f, v in degenerate.items() if any(v)}}
    if subspace == "pca":
        out["subspace_dims"] = dims

    by_ll = sorted(families, key=lambda f: -mean[f])
    if selection in ("ll_ks", "ll_ksnorm"):
        # Filter, then likelihood: only families whose samples reproduce the
        # distribution of ||x - mean|| ("ll_ks") or of the norm ||x||, i.e. the
        # spherical shell ("ll_ksnorm"), are eligible; the highest ll among them
        # wins.  If none passes: the smallest D.
        crit = ks_mean if selection == "ll_ks" else kn_mean
        thr = ks_max + _ks_noise(np.mean(n_tr) + np.mean(n_te))
        eligible = [f for f in by_ll if crit[f] <= thr and np.isfinite(mean[f])]
        rest = sorted((f for f in families if f not in eligible), key=lambda f: crit[f])
        ranking = eligible + rest
        if not eligible:
            print(f"    [PARAMETRIC] no family passes the {selection} filter (thr={thr:.3f}); "
                  f"taking the smallest D: {ranking[0]}", flush=True)
        out.update({"ks_threshold": thr, "eligible": eligible, "ks_fallback": not eligible})
    elif selection == "ll_tiebreak":
        tb = tiebreak(table, ks_table, es_table, np.mean(n_tr), np.mean(n_te), X.shape[1], tie_margin,
                      knorm_table=kn_table)
        key = {"ks": "ks", "energy": "energy", "ks_norm": "ksnorm"}[tiebreak_by]
        winner = tb[f"tie_winner_{key}"]
        ranking = [winner] + [f for f in by_ll if f != winner]
        out.update(tb)
        out.update({"tiebreak_by": tiebreak_by, "tie_finalists": tb[f"tie_finalists_{key}"]})
    elif selection == "energy":
        ranking = sorted(families, key=lambda f: es_mean[f])
    else:
        ranking = by_ll
    out["ranking"] = ranking

    nb, nb_es = {}, {}
    if "gaussian" in table:
        for f in families:
            if f != "gaussian":
                t, p = _nadeau_bengio(table[f], table["gaussian"], np.mean(n_tr), np.mean(n_te))
                nb[f] = {"t": t, "p": p}
                t, p = _nadeau_bengio(es_table[f], es_table["gaussian"], np.mean(n_tr), np.mean(n_te))
                nb_es[f] = {"t": t, "p": p}
    out["nadeau_bengio_vs_gaussian"] = nb
    out["nadeau_bengio_energy_vs_gaussian"] = nb_es
    out["cv_repeats"] = int(cv_repeats)
    out["n_train_mean"], out["n_val_mean"] = float(np.mean(n_tr)), float(np.mean(n_te))
    return ranking, out


# =====================================================================
#  Alternative pipeline: DISTRIBUTIONAL's test cascade, CV instead of BIC
# =====================================================================

DIST_ALPHA = 0.05       # same thresholds as DistributionalMethod.fit_best_distribution
DIST_CR_THRESHOLD = 0.10
DIST_RBAR_MIN = 0.7


def distributional_tests(X, alpha=DIST_ALPHA, cr_threshold=DIST_CR_THRESHOLD, rbar_min=DIST_RBAR_MIN):
    """Steps 1-4 of DistributionalMethod.fit_best_distribution, unchanged:
      1. Henze-Zirkler multivariate normality         -> gaussian
      2. shell (Cr < 0.10) + Rayleigh (p < 0.05) + Rbar > 0.7
         -> vMF, upgraded to Kent if the tangent-plane sphericity test rejects
      3. marginal uniformity                           -> uniform
    Returns (family or None, info).  None = no test accepted: the caller then
    runs the PARAMETRIC CV selection where DISTRIBUTIONAL would run the BIC duel.
    Families accepted here are fitted with PARAMETRIC's own estimators
    (gaussian: LW; vmf_norm / kent_tangent: with the step-0 norm model)."""
    info = {}
    is_normal, p_hz = _hz_test(X, alpha=alpha)
    info["hz_p"] = float(p_hz)
    if is_normal:
        info["accepted_by"] = "Henze-Zirkler"
        return "gaussian", info
    cr = float(_radio_dispersion(X))
    info["cr"] = cr
    if cr < cr_threshold:
        p_ray = float(_rayleigh_test(X))
        _, U = _polar(X)
        m = U.mean(axis=0)
        rbar = float(np.linalg.norm(m))
        info.update({"rayleigh_p": p_ray, "rbar": rbar})
        if p_ray < alpha and rbar > rbar_min:
            kind, _ = _fit_kent_from_vmf(X, _unit(m), None, None, alpha=alpha)
            info["accepted_by"] = "shell+Rayleigh+" + ("sphericity rejected" if kind == "kent" else "sphericity kept")
            return ("kent_tangent" if kind == "kent" else "vmf_norm"), info
    # Kept from DistributionalMethod for completeness; not part of DISCERN as
    # described in the paper: no class of any reported experiment passes it.
    if _uniformity_test(X, alpha=alpha):
        info["accepted_by"] = "uniformity"
        return "uniform", info
    return None, info


def fit_parametric_class(X, cv_folds=5, seed=42, rng=None,
                         selection="ll", ks_max=0.1, subspace="none", pca_variance=0.99,
                         cv_repeats=1, tiebreak_by="ks", em_max_iter=EM_MAX_ITER, tie_margin=TIE_MARGIN,
                         pipeline="cv"):
    """Steps 0-2 for one class.  Returns a DistributionInfo with ``.selection``.

    pipeline="cv"    : every family competes in the CV selection (``selection``).
    pipeline="tests" : DISTRIBUTIONAL's logic — its statistical tests first
                       (``distributional_tests``); only if none accepts, the CV
                       selection replaces DISTRIBUTIONAL's BIC duel.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if rng is None:
        rng = np.random.default_rng(seed)

    report = {"n": int(n), "p": int(p), "cv_folds": int(cv_folds), "cv_repeats": int(cv_repeats),
              "em_max_iter": int(em_max_iter), "pipeline": pipeline,
              "subspace": subspace, "pca_variance": pca_variance if subspace == "pca" else None}
    step0 = norm_direction_test(X, rng)
    report["step0"] = step0
    dependent = step0["dependent"]

    test_family = None
    if pipeline == "tests":
        test_family, report["tests"] = distributional_tests(X)
    decided_by = "test" if test_family is not None else "cv"
    report["decided_by"] = decided_by

    if test_family is not None:
        ranking, sel = [test_family], {"cv_table": {}, "cv_mean": {}, "nadeau_bengio_vs_gaussian": {},
                                       "selection": "test"}
    elif n < 2 * cv_folds:
        print(f"    [PARAMETRIC] only {n} samples (< 2*cv_folds) — falling back to gaussian", flush=True)
        ranking, sel = ["gaussian"], {"cv_table": {}, "cv_mean": {}, "nadeau_bengio_vs_gaussian": {},
                                      "selection": selection}
    else:
        ranking, sel = select_family(X, cv_folds, seed, dependent, selection=selection,
                                     ks_max=ks_max, subspace=subspace, pca_variance=pca_variance,
                                     cv_repeats=cv_repeats, tiebreak_by=tiebreak_by,
                                     em_max_iter=em_max_iter, tie_margin=tie_margin)
    report.update(sel)

    if ranking[0] == "uniform":
        # Accepted by DISTRIBUTIONAL's uniformity test: keep its axis-aligned
        # uniform (sampled / scored by DistributionalMethod's own code).
        mins, maxs, ll, _ = _fit_uniform(X)
        params = {"mins": mins, "maxs": maxs}
        chosen, report["refit_skipped"] = "uniform", {}
        synth, diag = rng.uniform(mins, maxs, size=(n, p)), {}
    else:
        # Refit on all the data; if that fit is degenerate, take the next in the ranking.
        full_sub = _fit_subspace(X, pca_variance) if subspace == "pca" else None
        refit_skipped = {}
        for chosen in list(ranking) + ["gaussian"]:
            sub = full_sub if chosen in SUBSPACE_FAMILIES else None
            params = fit_family(chosen, X, dependent=dependent, subspace=sub, em_max_iter=em_max_iter)
            why = params.get("inner", params).get("degenerate")
            if why is None:
                break
            refit_skipped[chosen] = why
            print(f"    [PARAMETRIC] refit of {chosen} on all data is degenerate ({why}); "
                  f"trying the next family", flush=True)
        report["refit_skipped"] = refit_skipped
        if sub is not None:
            report["subspace_dim"] = sub["d"]
        ll = float(np.sum(family_logpdf(chosen, X, params)))
        synth, diag = sample_parametric(chosen, params, n, rng)
    report["chosen"] = chosen

    # ---- Post-fit checks ----
    centre = X.mean(axis=0)
    ks = sp_stats.ks_2samp(np.linalg.norm(X - centre, axis=1), np.linalg.norm(synth - centre, axis=1))
    report["ks_dist_to_mean"] = {"stat": float(ks.statistic), "p": float(ks.pvalue)}
    kn = sp_stats.ks_2samp(np.linalg.norm(X, axis=1), np.linalg.norm(synth, axis=1))
    report["ks_norm"] = {"stat": float(kn.statistic), "p": float(kn.pvalue)}
    # Kent only: fraction of tangent samples with ||v|| > pi (should be ~0)
    report["kent_frac_v_gt_pi"] = diag.get("frac_v_gt_pi")

    info = DistributionInfo(chosen, params, ll, float("nan"), decided_by=decided_by)
    info.selection = report
    return info



def format_selection_report(class_id, rep):
    head = f"  Class {class_id}: n={rep['n']} p={rep['p']}"
    if rep.get("subspace") == "pca":
        head += f" (PCA subspace d={rep.get('subspace_dim')}, folds {rep.get('subspace_dims')})"
    head += f" -> chosen = {rep['chosen']}  [selection={rep.get('selection', 'll')}"
    if rep.get("pipeline") == "tests":
        t = rep.get("tests", {})
        tinfo = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items())
        head += (f", DISTRIBUTIONAL tests -> "
                 + (f"accepted ({tinfo})" if rep.get("decided_by") == "test"
                    else f"none accepted ({tinfo}), CV instead of BIC"))
    if rep.get("selection") in ("ll_ks", "ll_ksnorm"):
        head += (f", filter on {'||x - mean||' if rep['selection'] == 'll_ks' else '||x||'}: thr D<={rep['ks_threshold']:.3f}, eligible={rep['eligible']}"
                 + (", FALLBACK min D" if rep["ks_fallback"] else ""))
    if rep.get("selection") == "ll_tiebreak":
        gaps = rep.get("tie_gap_per_dim", {})
        tp = ", ".join(f"{f}:p={p:.3g},gap={gaps.get(f, float('nan')):.3f}/dim"
                       for f, p in rep["tie_p_vs_best"].items())
        # caches written before the paired KS test store the old tolerance band
        ks_rule = (f"noise {rep['ks_noise']:.3f}" if "ks_noise" in rep else "paired NB test")
        head += (f", by {rep.get('tiebreak_by', 'ks')}; ll ties (p>0.05 or gap<={rep.get('tie_margin', 0):.3f}"
                 f" nats/dim)={rep['tie_candidates']} (vs best: {tp});"
                 f" KS -> {rep.get('tie_winner_ks')} (finalists {rep.get('tie_finalists_ks')}, {ks_rule});"
                 f" energy -> {rep.get('tie_winner_energy')} (finalists {rep.get('tie_finalists_energy')});"
                 f" KS norm -> {rep.get('tie_winner_ksnorm')} (finalists {rep.get('tie_finalists_ksnorm')})")
    if rep.get("cv_repeats", 1) > 1:
        head += f"  [CV {rep['cv_folds']}x{rep['cv_repeats']}]"
    head += "]"
    if rep.get("refit_skipped"):
        head += f"  refit skipped: {rep['refit_skipped']}"
    lines = [head]
    s0 = rep["step0"]
    lines.append(
        f"    step0 norm|dir: {'DEPENDENT (log r ~ a.u + b)' if s0['dependent'] else 'independent (log r ~ N)'}"
        f"  spearman rho={s0['spearman_rho']:.3f} p={s0['p_spearman']:.3g}"
        f"  ridgeLOO R2={s0['ridge_loo_r2']:.3f} p={s0['p_ridge']:.3g}"
        f"  dCor={s0['dcor']:.3f} p={s0['p_dcor']:.3g}  BH={s0['bh_reject']}"
    )
    nb = rep.get("nadeau_bengio_vs_gaussian", {})
    nb_es = rep.get("nadeau_bengio_energy_vs_gaussian", {})
    degen = rep.get("degenerate", {})
    lines.append(f"     {'family':<19} {'ll/pt':>10} {'KS D':>6} {'KS|x|':>6} {'energy':>9}   ll folds")
    for fam, vals in rep.get("cv_table", {}).items():
        mark = "*" if fam == rep["chosen"] else " "
        folds = " ".join(f"{v:8.2f}" for v in vals)
        line = (f"   {mark} {fam:<19} {rep['cv_mean'][fam]:10.3f} {rep.get('ks_mean', {}).get(fam, float('nan')):6.3f}"
                f" {rep.get('ksnorm_mean', {}).get(fam, float('nan')):6.3f}"
                f" {rep.get('energy_mean', {}).get(fam, float('nan')):9.4f}   {folds}")
        if fam in nb:
            line += f"  NB(ll) p={nb[fam]['p']:.3g}"
        if fam in nb_es:
            line += f"  NB(energy) p={nb_es[fam]['p']:.3g}"
        if fam in degen:
            reasons = sorted({r for r in degen[fam] if r})
            line += f"  DEGENERATE {sum(1 for r in degen[fam] if r)}/{len(degen[fam])} folds: {'; '.join(reasons)}"
        lines.append(line)
    ks = rep["ks_dist_to_mean"]
    kn = rep.get("ks_norm", {})
    lines.append(f"    final fit KS(||x-mean|| real vs synth): D={ks['stat']:.3f} p={ks['p']:.3g}"
                 + (f"  KS(||x||): D={kn['stat']:.3f} p={kn['p']:.3g}" if kn else "")
                 + (f"  kent frac(||v||>pi)={rep['kent_frac_v_gt_pi']:.2e}"
                    if rep.get("kent_frac_v_gt_pi") is not None else ""))
    return "\n".join(lines)



def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


# =====================================================================
#  Torch log-densities (used by the embedding-space losses and the
#  pre-embedding optimisation, same role as DistributionalMethod._loglik*)
# =====================================================================

def _t(x, ref):
    return torch.as_tensor(np.asarray(x), dtype=ref.dtype, device=ref.device)


def _torch_polar(x):
    r = torch.sqrt((x * x).sum(dim=1) + 1e-24)
    return r, x / r.unsqueeze(1)


def _torch_radial(nm, u, r, p):
    lr = torch.log(r)
    mean = u @ _t(nm["a"], u) + nm["b"] if nm["dependent"] else nm["m"]
    s2 = nm["s2"]
    return -0.5 * (LOG_2PI + math.log(s2) + (lr - mean) ** 2 / s2) - p * lr


def _torch_chol_maha(x, mean, cov):
    L = torch.linalg.cholesky(cov)
    sol = torch.linalg.solve_triangular(L, (x - mean).T, upper=False)
    return (sol * sol).sum(dim=0), 2.0 * torch.log(torch.diagonal(L)).sum()


def _is_parametric_specific(info):
    """True when DistributionalMethod's own log-likelihoods cannot handle info."""
    return info.dist_type in NEW_TYPES or "subspace" in info.params


def torch_parametric_loglik(x, info):
    """Unnormalised per-point log-density (B,) for NEW_TYPES families and for
    any family fitted in a PCA subspace (option B)."""
    dt, prm = info.dist_type, info.params
    if "subspace" in prm:
        sub = prm["subspace"]
        V = _t(sub["V"], x)
        z = x @ V
        d, r = V.shape[1], x.shape[1] - V.shape[1]
        inner = DistributionInfo(dt, prm["inner"], 0.0, 0.0)
        if dt in NEW_TYPES:
            ll = torch_parametric_loglik(z, inner)
        elif dt == "gaussian":
            ll = DistributionalMethod._loglik_gaussian_batch(z, inner) * d
        else:
            ll = DistributionalMethod._loglik_tstudent_batch(z, inner) * d
        if r > 0:
            E = x - z @ V.T - _t(sub["m_e"], x)
            ll = ll - 0.5 * (r * (LOG_2PI + math.log(sub["s2"])) + (E * E).sum(dim=1) / sub["s2"])
        return ll
    p = x.shape[1]
    if dt == "laplace_elliptical":
        q, logdet = _torch_chol_maha(x, _t(prm["mean"], x), _t(prm["cov"], x))
        q = q.clamp_min(_Q_FLOOR)
        lk = _torch_log_kv(0.5 * p - 1.0, torch.sqrt(2.0 * q))
        return (math.log(2.0) - 0.5 * p * LOG_2PI - 0.5 * logdet
                + 0.5 * (1.0 - 0.5 * p) * torch.log(0.5 * q) + lk)
    r, u = _torch_polar(x)
    radial = _torch_radial(prm["norm"], u, r, p)
    mu = _t(prm["mu"], x)
    if dt == "vmf_norm":
        return prm["log_c"] + prm["kappa"] * (u @ mu) + radial
    if dt == "kent_tangent":
        c = (u @ mu).clamp(-1.0, 1.0)
        w = u - c.unsqueeze(1) * mu
        s = torch.sqrt((w * w).sum(dim=1) + 1e-24)
        theta = torch.atan2(s, c)
        v = w * (theta / s).unsqueeze(1)
        h = _householder_vec(np.asarray(prm["mu"], dtype=np.float64))
        if h is not None:
            ht = _t(h, x)
            v = v - 2.0 * (v @ ht).unsqueeze(1) * ht
        C = v[:, 1:]
        q, logdet = _torch_chol_maha(C, torch.zeros(p - 1, dtype=x.dtype, device=x.device),
                                     _t(prm["tangent_cov"], x))
        th = theta.clamp_min(1e-6)
        jac = (p - 2) * (torch.log(th) - torch.log(torch.sin(th).clamp_min(1e-30)))
        return -0.5 * ((p - 1) * LOG_2PI + logdet + q) + jac + radial
    raise ValueError(f"Unknown PARAMETRIC family: {dt}")


# =====================================================================
#  ParametricMethod
# =====================================================================

class ParametricMethod(DistributionalMethod):
    """DistributionalMethod whose per-class family is chosen by K-fold CV."""

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)
        # One distribution per class: the recursive clustering path does not apply.
        self.config["clustering"] = False
        self.cv_folds = int(self.config.get("cv_folds", 5))
        self.selection = self.config.get("parametric_selection", "ll")
        self.ks_max = float(self.config.get("parametric_ks_max", 0.1))
        self.subspace = self.config.get("parametric_subspace", "none")
        self.pca_variance = float(self.config.get("parametric_pca_variance", 0.99))
        self.cv_repeats = int(self.config.get("cv_repeats", 1))
        self.tiebreak_by = self.config.get("parametric_tiebreak", "ks")
        self.em_max_iter = int(self.config.get("parametric_em_max_iter", EM_MAX_ITER))
        self.tie_margin = float(self.config.get("parametric_tie_margin", TIE_MARGIN))
        self.pipeline = self.config.get("parametric_pipeline", "cv")
        self.tag = self.config.get("parametric_tag", f"cv{self.cv_folds}")

    # ----- fitting --------------------------------------------------------

    def _fit_distributions_for_class(self, embeddings, class_id):
        info = fit_parametric_class(
            embeddings, cv_folds=self.cv_folds, seed=self.config["seed"],
            rng=np.random.default_rng(self.config["seed"] + int(class_id)),
            selection=self.selection, ks_max=self.ks_max,
            subspace=self.subspace, pca_variance=self.pca_variance,
            cv_repeats=self.cv_repeats, tiebreak_by=self.tiebreak_by,
            em_max_iter=self.em_max_iter,
            tie_margin=self.tie_margin,
            pipeline=self.pipeline,
        )
        print(format_selection_report(class_id, info.selection), flush=True)
        return [(info, np.arange(len(embeddings)))]

    def _build_distribution_stats(self, class_embeddings, total_samples, total_per_class):
        num_classes = self.config["num_classes"]
        self.dist_infos = {}
        self.centroids = {i: [] for i in range(num_classes)}
        self.covariance = {i: [] for i in range(num_classes)}
        pct_inter = [0.0] * num_classes
        pct_intra = [[] for _ in range(num_classes)]

        for i in range(num_classes):
            embs = np.asarray(class_embeddings[i], dtype=np.float64)
            if len(embs) == 0:
                self.dist_infos[i] = []
                continue
            pct_inter[i] = total_per_class[i] / total_samples
            print(f"[PARAMETRIC] Fitting class {i} ({len(embs)} samples)", flush=True)
            (info, _idx), = self._fit_distributions_for_class(embs, i)
            self.dist_infos[i] = [(info, 1.0)]
            pct_intra[i].append(1.0)
            # Moment summaries for the centroid/covariance update path
            mean = embs.mean(axis=0)
            self.centroids[i] = torch.tensor(mean, dtype=torch.float32, device=self.device)
            self.covariance[i] = torch.tensor(_lw(embs - mean), dtype=torch.float32, device=self.device)

        self.percentages = torch.tensor(pct_inter, dtype=torch.float32)
        self.intra_percentage = [torch.tensor(p, dtype=torch.float32) for p in pct_intra]
        self.num_samples_original = torch.tensor(
            [total_per_class[i] for i in range(num_classes)], dtype=torch.float32, device=self.device,
        )

    # ----- setup / unlearn: inherited flow -------------------------------

    def setup(self, *args, **kwargs):
        # DistributionalMethod.setup() builds its dist_infos cache path from
        # covariance_type; give PARAMETRIC its own cache directory
        # ("<standard|robust>_cov_parametric_{tag}") so it never loads a
        # DISTRIBUTIONAL cache (or vice versa).
        saved_cov = self.config.get("covariance_type", "full")
        self.config["covariance_type"] = f"parametric_{self.tag}"
        try:
            out = super().setup(*args, **kwargs)
        finally:
            self.config["covariance_type"] = saved_cov

        reports = {}
        for c in range(self.config["num_classes"]):
            infos = self.dist_infos.get(c, [])
            if infos and getattr(infos[0][0], "selection", None) is not None:
                reports[c] = infos[0][0].selection
        if reports:
            print("[PARAMETRIC] Per-class selection summary:", flush=True)
            for c, rep in reports.items():
                print(format_selection_report(c, rep), flush=True)
            os.makedirs(self.config["hyperparameters_path"], exist_ok=True)
            path = os.path.join(self.config["hyperparameters_path"], "parametric_selection.json")
            with open(path, "w") as f:
                json.dump(_jsonable(reports), f, indent=1)
            print(f"[PARAMETRIC] Selection report saved to {path}", flush=True)
        return out

    def unlearn(self, *args, **kwargs):
        model = super().unlearn(*args, **kwargs)
        path = f'{self.config["hyperparameters_path"]}/hyperparameters.yml'
        if os.path.exists(path):
            with open(path) as f:
                hp = yaml.safe_load(f) or {}
            hp["method"] = "PARAMETRIC"
            hp["cv_folds"] = self.cv_folds
            hp["parametric_selection"] = self.selection
            hp["parametric_ks_max"] = self.ks_max
            hp["parametric_subspace"] = self.subspace
            hp["parametric_pca_variance"] = self.pca_variance
            hp["cv_repeats"] = self.cv_repeats
            hp["parametric_tiebreak"] = self.tiebreak_by
            hp["parametric_em_max_iter"] = self.em_max_iter
            hp["parametric_tie_margin"] = self.tie_margin
            hp["parametric_pipeline"] = self.pipeline
            for k in ("covariance_type", "n_pca_components", "model_selection", "clustering", "robust"):
                hp.pop(k, None)
            with open(path, "w") as f:
                yaml.dump(hp, f, default_flow_style=False)
        return model

    # ----- sampling ---------------------------------------------------------

    @staticmethod
    def _info_dim(info):
        prm = info.params
        if "subspace" in prm:
            return prm["subspace"]["V"].shape[0]
        for k in ("mean", "mu", "locs", "mins"):
            if k in prm:
                return len(prm[k])
        raise ValueError("Cannot infer embedding dimension from DistributionInfo")

    def generate_syntetic_embeddings(self, n_samples, forbidden_labels):
        """Same label assignment as DistributionalMethod; per-class sampling
        from the CV-selected family."""
        if self.percentages is None or self.dist_infos is None:
            raise RuntimeError(
                "_build_distribution_stats must be called before generate_syntetic_embeddings()"
            )

        seed = self.config["seed"]
        probs = self.percentages.to(self.device).float()
        rng = np.random.default_rng(seed)

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

        groups = {}
        for pos, class_id in enumerate(syntetic_labels.tolist()):
            groups.setdefault(class_id, []).append(pos)

        emb_dim = None
        for cid in range(self.config["num_classes"]):
            if self.dist_infos.get(cid):
                emb_dim = self._info_dim(self.dist_infos[cid][0][0])
                break

        syntetic_embeddings = torch.empty(n_samples, emb_dim, dtype=torch.float32, device=self.device)
        for class_id, positions in groups.items():
            info, _ = self.dist_infos[class_id][0]
            if info.dist_type in FAMILIES:
                samples, _ = sample_parametric(info.dist_type, info.params, len(positions), rng)
                samples = torch.tensor(samples, dtype=torch.float32, device=self.device)
            else:  # e.g. dist_infos coming from elsewhere
                samples = sample_from_distribution(info, len(positions), device=self.device, rng=rng)
            syntetic_embeddings[positions] = samples

        return syntetic_embeddings, syntetic_labels + self.config["num_classes"]

    # ----- log-likelihoods / distances for the new families ----------------

    def _loglik_batch(self, x, info):
        if _is_parametric_specific(info):
            return torch_parametric_loglik(x, info) / x.shape[1]
        return super()._loglik_batch(x, info)

    def _loglik(self, x, info):
        if _is_parametric_specific(info):
            return self._loglik_batch(x.unsqueeze(0), info)[0]
        return super()._loglik(x, info)

    def _typed_distance(self, x, info):
        # No canonical distance for the new families: use the normalised NLL.
        if _is_parametric_specific(info):
            return -self._loglik(x, info)
        return super()._typed_distance(x, info)
