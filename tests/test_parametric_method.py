"""
Tests for src/methods/ParametricMethod.py (PARAMETRIC method).

  1. log K_v / log I_v (Debye for large order) vs mpmath, rel. error in log < 1e-6.
  2. Elliptical Laplace:
       a. log-density vs a Monte Carlo estimate of the mixture integral
          f(x) = E_w[N(x; mu, w Sigma)], w ~ Exp(1)   (p = 2, 5)
       b. P(q <= c) from samples (MC) vs the density integrated over the ball
       c. EM recovers mu and Sigma (p = 512, n = 500)
  3. vMF / tangent-normal ("Kent") directional densities integrate to 1 on the
     sphere (uniform Monte Carlo, p = 3, 5) -> checks normaliser and Jacobian.
  4. Torch log-likelihoods match the numpy ones.
  5. CV recovers the generating family on simulated data (+ step-0 decision).

Run:  python -m pytest tests/test_parametric_method.py -s -q
"""

import math
import os
import sys

import mpmath
import numpy as np
import pytest
import torch
from scipy import integrate
from scipy import stats as sp_stats
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods.DistributionalMethod import DistributionInfo
from src.methods.ParametricMethod import (
    DEBYE_MIN_ORDER,
    _householder_vec, _h_apply, _exp_map, _lw,
    _laplace_logpdf_from_q,
    _fit_laplace_elliptical,
    family_logpdf, fit_family, sample_parametric,
    fit_parametric_class, format_selection_report,
    log_iv, log_kv, torch_parametric_loglik,
)


# =====================================================================
# 1. Log-Bessel
# =====================================================================

NU_GRID = [0.0, 0.5, 1.5, 4.0, 20.0, 49.5, DEBYE_MIN_ORDER, 63.0, 127.0, 255.0, 383.0, 400.0]
Z_GRID = [1e-3, 0.1, 1.0, 5.0, 22.6, 32.0, 39.2, 60.0, 200.0, 1000.0]


@pytest.mark.parametrize("nu", NU_GRID)
def test_log_kv_vs_mpmath(nu):
    mpmath.mp.dps = 40
    for z in Z_GRID:
        ref = float(mpmath.log(mpmath.besselk(nu, z)))
        got = float(log_kv(nu, z))
        assert abs(got - ref) <= 1e-6 * max(1.0, abs(ref)), (nu, z, got, ref)
    # vectorised path gives the same values
    got_vec = log_kv(nu, np.array(Z_GRID))
    ref_vec = np.array([float(mpmath.log(mpmath.besselk(nu, z))) for z in Z_GRID])
    assert np.all(np.abs(got_vec - ref_vec) <= 1e-6 * np.maximum(1.0, np.abs(ref_vec)))


@pytest.mark.parametrize("nu", [n for n in NU_GRID if n > 0])
def test_log_iv_vs_mpmath(nu):
    mpmath.mp.dps = 40
    for z in Z_GRID:
        ref = float(mpmath.log(mpmath.besseli(nu, z)))
        got = float(log_iv(nu, z))
        assert abs(got - ref) <= 1e-6 * max(1.0, abs(ref)), (nu, z, got, ref)


# =====================================================================
# 2. Elliptical Laplace
# =====================================================================

def _random_spd(p, rng, lo=0.5, hi=2.0):
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    return (Q * rng.uniform(lo, hi, p)) @ Q.T


def _sample_laplace(mu, S, n, rng):
    L = np.linalg.cholesky(S)
    w = rng.exponential(1.0, n)
    return mu + np.sqrt(w)[:, None] * (rng.standard_normal((n, len(mu))) @ L.T)


@pytest.mark.parametrize("p", [2, 5])
def test_laplace_density_vs_mc_mixture(p):
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(p)
    S = _random_spd(p, rng)
    prm = {"mean": mu, "cov": S}
    Sinv = np.linalg.inv(S)
    _, logdet = np.linalg.slogdet(S)
    X = mu + rng.standard_normal((6, p)) @ np.linalg.cholesky(S).T * 1.5
    q = np.einsum("ij,jk,ik->i", X - mu, Sinv, X - mu)

    w = rng.exponential(1.0, 2_000_000)
    for qi, li in zip(q, family_logpdf("laplace_elliptical", X, prm)):
        mc = np.mean(np.exp(-0.5 * p * np.log(2 * np.pi * w) - 0.5 * logdet - 0.5 * qi / w))
        assert abs(math.log(mc) - li) < 0.02, (p, qi, math.log(mc), li)


@pytest.mark.parametrize("p", [2, 5])
def test_laplace_density_integrates(p):
    """Density integrated over Mahalanobis balls vs MC frequencies; total mass 1."""
    rng = np.random.default_rng(1)
    # f depends on x only through q; volume element of {q <= c}:
    # dVol = |S|^{1/2} * S_{p-1} * rho^{p-1} drho, with rho = sqrt(q)
    log_area = math.log(2.0) + 0.5 * p * math.log(math.pi) - gammaln(0.5 * p)

    def mass(c):
        g = lambda rho: math.exp(log_area + (p - 1) * math.log(rho)
                                 + float(_laplace_logpdf_from_q(np.array([rho * rho]), p, 0.0)[0]))
        return integrate.quad(g, 0.0, math.sqrt(c), limit=200)[0]

    assert abs(mass(1e4) - 1.0) < 1e-6
    X = _sample_laplace(np.zeros(p), np.eye(p), 400_000, rng)
    q = np.sum(X * X, axis=1)
    for c in np.quantile(q, [0.1, 0.5, 0.9, 0.99]):
        assert abs(mass(c) - np.mean(q <= c)) < 3e-3


def test_laplace_em_recovers_params():
    """Large p of the real order, n ~ 500.  n must exceed p: with n <= p the
    points lie in an affine subspace and the scatter MLE does not exist (Kent &
    Tyler 1991) — see test_laplace_em_flags_n_below_p."""
    p, n = 256, 500
    rng = np.random.default_rng(2)
    mu = 3.0 * rng.standard_normal(p) / math.sqrt(p)
    eig = np.geomspace(0.05, 2.0, p)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    S = (Q * eig) @ Q.T
    X = _sample_laplace(mu, S, n, rng)

    prm = _fit_laplace_elliptical(X)
    xbar = X.mean(axis=0)
    err_mu_em = np.linalg.norm(prm["mean"] - mu)
    err_mu_mean = np.linalg.norm(xbar - mu)
    rel_S_em = np.linalg.norm(prm["cov"] - S) / np.linalg.norm(S)
    rel_S_lw = np.linalg.norm(_lw(X - xbar) - S) / np.linalg.norm(S)
    tr_ratio = np.trace(prm["cov"]) / np.trace(S)
    print(f"\n  |mu_em-mu|={err_mu_em:.3f} (sample mean {err_mu_mean:.3f})  "
          f"relF(S_em)={rel_S_em:.3f} (LW sample cov {rel_S_lw:.3f})  tr ratio={tr_ratio:.3f}")
    print(f"  EM iterations={prm['em_iter']}  degenerate={prm.get('degenerate')}")
    assert not prm.get("degenerate")
    assert err_mu_em <= 1.1 * err_mu_mean
    assert rel_S_em <= 1.1 * rel_S_lw
    assert 0.9 < tr_ratio < 1.1


def test_laplace_em_flags_n_below_p():
    """p = 512, n = 500 (as first specified): the MLE does not exist, so the EM
    must stop early and flag the fit instead of drifting."""
    rng = np.random.default_rng(2)
    p, n = 512, 500
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    S = (Q * np.geomspace(0.05, 2.0, p)) @ Q.T
    prm = _fit_laplace_elliptical(_sample_laplace(np.zeros(p), S, n, rng))
    print(f"\n  n<p: degenerate={prm.get('degenerate')} em_iter={prm['em_iter']}")
    assert prm.get("degenerate")


# =====================================================================
# 3. Directional densities integrate to 1 on S^{p-1}
# =====================================================================

def _uniform_sphere(n, p, rng):
    U = rng.standard_normal((n, p))
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def _sphere_area(p):
    return math.exp(math.log(2.0) + 0.5 * p * math.log(math.pi) - gammaln(0.5 * p))


def _dir_logpdf(dist_type, prm, U):
    """Directional part only: total log-density at r = 1 minus the radial part."""
    p = U.shape[1]
    nm = prm["norm"]
    radial = -0.5 * (math.log(2 * math.pi) + math.log(nm["s2"]) + nm["m"] ** 2 / nm["s2"])
    return family_logpdf(dist_type, U, prm) - radial


@pytest.mark.parametrize("p", [3, 5])
def test_vmf_and_kent_directional_normalisation(p):
    rng = np.random.default_rng(3)
    mu = rng.standard_normal(p)
    mu /= np.linalg.norm(mu)
    norm = {"dependent": False, "m": 0.0, "s2": 0.04}
    U = _uniform_sphere(1_000_000, p, rng)
    area = _sphere_area(p)

    from src.methods.ParametricMethod import _vmf_log_c
    kappa = 4.0
    vmf = {"mu": mu, "kappa": kappa, "log_c": _vmf_log_c(p, kappa), "norm": norm}
    i_vmf = area * np.mean(np.exp(_dir_logpdf("vmf_norm", vmf, U)))
    assert abs(i_vmf - 1.0) < 0.01, i_vmf
    if p == 3:  # closed form C_3 = kappa / (4 pi sinh kappa)
        assert abs(vmf["log_c"] - math.log(kappa / (4 * math.pi * math.sinh(kappa)))) < 1e-10

    Tcov = np.diag(np.linspace(0.05, 0.3, p - 1))
    kent = {"mu": mu, "tangent_cov": Tcov, "norm": norm}
    i_kent = area * np.mean(np.exp(_dir_logpdf("kent_tangent", kent, U)))
    # mass of N(0, Tcov) beyond ||v|| = pi is lost by the exp map (negligible here)
    assert abs(i_kent - 1.0) < 0.01, i_kent


# =====================================================================
# 4. Torch log-likelihood == numpy log-density
# =====================================================================

@pytest.mark.parametrize("dist_type", ["laplace_elliptical", "vmf_norm", "kent_tangent"])
@pytest.mark.parametrize("dependent", [False, True])
def test_torch_matches_numpy(dist_type, dependent):
    if dist_type == "laplace_elliptical" and dependent:
        pytest.skip("no norm model")
    rng = np.random.default_rng(4)
    p = 64
    mu = rng.standard_normal(p)
    mu = 6.0 * mu / np.linalg.norm(mu)
    X = mu + 0.6 * rng.standard_normal((800, p))
    prm = fit_family(dist_type, X, dependent=dependent)
    ref = family_logpdf(dist_type, X[:50], prm)
    got = torch_parametric_loglik(torch.tensor(X[:50], dtype=torch.float64),
                                  DistributionInfo(dist_type, prm, 0.0, 0.0)).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


# =====================================================================
# 5. CV recovers the generating family
# =====================================================================

P_SIM, N_SIM = 10, 3000


def _rand_unit(p, rng):
    v = rng.standard_normal(p)
    return v / np.linalg.norm(v)


def _sim(family, rng):
    p, n = P_SIM, N_SIM
    if family in ("gaussian", "tstudent", "laplace_elliptical"):
        prm = {"mean": np.zeros(p), "cov": _random_spd(p, rng)}
        if family == "tstudent":
            prm["df"] = 3.0
        return sample_parametric(family, prm, n, rng)[0]
    norm = {"dependent": False, "m": math.log(5.0), "s2": 0.1 ** 2}
    mu = _rand_unit(p, rng)
    if family == "vmf_norm":
        from src.methods.ParametricMethod import _vmf_log_c
        prm = {"mu": mu, "kappa": 10.0, "log_c": _vmf_log_c(p, 10.0), "norm": norm}
    elif family == "vmf_norm_dependent":
        from src.methods.ParametricMethod import _vmf_log_c
        a = 1.5 * _rand_unit(p, rng)
        norm = {"dependent": True, "a": a, "b": math.log(5.0) - float(a @ mu), "s2": 0.05 ** 2}
        prm = {"mu": mu, "kappa": 10.0, "log_c": _vmf_log_c(p, 10.0), "norm": norm}
        family = "vmf_norm"
    else:  # kent_tangent, strongly anisotropic tangent covariance
        prm = {"mu": mu, "tangent_cov": np.diag(np.geomspace(0.01, 0.4, p - 1)), "norm": norm}
    return sample_parametric(family, prm, n, rng)[0]


@pytest.mark.parametrize("family,expect_dependent", [
    ("gaussian", None),
    ("tstudent", None),
    ("laplace_elliptical", None),
    ("vmf_norm", False),
    ("kent_tangent", False),
    ("vmf_norm_dependent", True),
])
def test_cv_recovers_family(family, expect_dependent):
    rng = np.random.default_rng(5)
    X = _sim(family, rng)
    info = fit_parametric_class(X, cv_folds=5, seed=42)
    rep = info.selection
    print("\n" + format_selection_report(family, rep))
    expected = "vmf_norm" if family == "vmf_norm_dependent" else family
    if expected == "gaussian" and info.dist_type == "tstudent":
        # t nests the Gaussian (nu -> inf): accept only a negligible tie.
        gap = rep["cv_mean"]["tstudent"] - rep["cv_mean"]["gaussian"]
        assert info.params["df"] > 50 and gap < 0.01, (info.params["df"], gap)
    else:
        assert info.dist_type == expected
    if expect_dependent is not None:
        assert rep["step0"]["dependent"] == expect_dependent
    assert rep["ks_dist_to_mean"]["p"] > 1e-3


# =====================================================================
# 6. Option A (ll_ks) and option B (PCA subspace, elliptical only)
# =====================================================================

from src.methods.ParametricMethod import SUBSPACE_FAMILIES, _fit_subspace, select_family


def _low_rank_laplace(n, p, rank, rng, noise=0.05):
    """Elliptical Laplace living (mostly) in a rank-``rank`` subspace, away from 0."""
    A = rng.standard_normal((p, rank)) * np.geomspace(1.0, 0.1, rank)
    mu = 5.0 * _rand_unit(p, rng)
    w = rng.exponential(1.0, n)
    return mu + np.sqrt(w)[:, None] * (rng.standard_normal((n, rank)) @ A.T) \
        + noise * rng.standard_normal((n, p))


@pytest.mark.parametrize("dist_type", list(SUBSPACE_FAMILIES))
def test_subspace_torch_matches_numpy(dist_type):
    rng = np.random.default_rng(6)
    X = _low_rank_laplace(1500, 64, 12, rng)
    sub = _fit_subspace(X, 0.99)
    assert sub["d"] < X.shape[1]
    prm = fit_family(dist_type, X, subspace=sub)
    ref = family_logpdf(dist_type, X[:50], prm)
    got = torch_parametric_loglik(torch.tensor(X[:50], dtype=torch.float64),
                                  DistributionInfo(dist_type, prm, 0.0, 0.0)).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-5)
    synth, _ = sample_parametric(dist_type, prm, 200, rng)
    assert synth.shape == (200, X.shape[1]) and np.all(np.isfinite(synth))


def test_subspace_never_used_for_directional():
    rng = np.random.default_rng(7)
    X = _sim("vmf_norm", rng)[:1000]
    info = fit_parametric_class(X, cv_folds=3, seed=0, subspace="pca", pca_variance=0.9)
    print("\n" + format_selection_report("vmf+pca", info.selection))
    if info.dist_type in ("vmf_norm", "kent_tangent"):
        assert "subspace" not in info.params
    else:
        assert "subspace" in info.params


def test_ll_ks_keeps_gaussian_on_gaussian_data():
    rng = np.random.default_rng(8)
    X = _sim("gaussian", rng)
    ranking, rep = select_family(X, 5, 0, dependent=False, selection="ll_ks", ks_max=0.1)
    chosen = ranking[0]
    print(f"\n  chosen={chosen} eligible={rep['eligible']} ks={ {k: round(v, 3) for k, v in rep['ks_mean'].items()} }")
    assert "gaussian" in rep["eligible"]
    assert chosen in ("gaussian", "tstudent")


def test_low_rank_laplace_scale_full_vs_subspace():
    """Reproduces the KS D ~ 1 seen on real embeddings (p >> effective rank)
    and documents how much option B changes the sampled radii (printed).  The
    Laplace radius sqrt(w) * chi_d with w ~ Exp(1) is much wider than the
    concentrated radii of this data, so the subspace cannot make it match."""
    rng = np.random.default_rng(9)
    p = 256
    X = _low_rank_laplace(3000, p, 20, rng)
    centre = X.mean(axis=0)
    d_real = np.linalg.norm(X - centre, axis=1)
    lw_tr = np.trace(_lw(X - centre))
    for label, sub in (("full", None), ("pca", _fit_subspace(X, 0.99))):
        prm = fit_family("laplace_elliptical", X, subspace=sub)
        syn, _ = sample_parametric("laplace_elliptical", prm, len(X), rng)
        D = sp_stats.ks_2samp(d_real, np.linalg.norm(syn - centre, axis=1)).statistic
        inner_cov = prm["inner"]["cov"] if sub is not None else prm["cov"]
        print(f"\n  {label:4s}: KS D={D:.3f}  tr(Sigma)/tr(LW)={np.trace(inner_cov) / lw_tr:.3f}"
              + (f"  d={sub['d']}" if sub is not None else ""))
        if sub is None:
            d_full = D
        else:
            # Option B does NOT fix the Laplace (same as on the real embeddings,
            # PCA 0.9: KS D 0.5); only require that it does not make it worse.
            assert D <= d_full


# =====================================================================
# 7. Closed-form ridge LOO (one SVD) == sklearn RidgeCV
# =====================================================================

def test_fast_ridge_loo_matches_ridgecv():
    from sklearn.linear_model import RidgeCV
    from src.methods.ParametricMethod import RIDGE_ALPHAS, _ridge_decomp, _ridge_loo_eval
    rng = np.random.default_rng(10)
    for n, p in ((300, 40), (200, 400)):          # n > p and n < p
        U = rng.standard_normal((n, p))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        y = U @ rng.standard_normal(p) * 0.3 + 0.1 * rng.standard_normal(n) + 2.0
        alpha, mse, coef, intercept = _ridge_loo_eval(_ridge_decomp(U), y, with_coef=True)
        try:
            m = RidgeCV(alphas=RIDGE_ALPHAS, store_cv_results=True).fit(U, y)
            cv = m.cv_results_
        except TypeError:
            m = RidgeCV(alphas=RIDGE_ALPHAS, store_cv_values=True).fit(U, y)
            cv = m.cv_values_
        best = int(np.argmin(np.abs(RIDGE_ALPHAS - m.alpha_)))
        assert alpha == pytest.approx(m.alpha_)
        assert mse == pytest.approx(float(cv[:, best].mean()), rel=1e-6)
        np.testing.assert_allclose(coef, m.coef_, rtol=1e-6, atol=1e-9)
        assert intercept == pytest.approx(m.intercept_, rel=1e-6, abs=1e-9)


# =====================================================================
# 8. Energy-score selection and degeneracy detection
# =====================================================================

from src.methods.ParametricMethod import _energy_score, EM_MAX_ITER


def test_energy_score_is_proper():
    """The true distribution scores lower than shifted / rescaled ones."""
    rng = np.random.default_rng(11)
    p = 32
    Y = rng.standard_normal((800, p))
    true = _energy_score(Y, rng.standard_normal((800, p)), rng)
    shifted = _energy_score(Y, rng.standard_normal((800, p)) + 0.3, rng)
    wide = _energy_score(Y, 1.3 * rng.standard_normal((800, p)), rng)
    narrow = _energy_score(Y, 0.7 * rng.standard_normal((800, p)), rng)
    print(f"\n  ES true={true:.4f} shifted={shifted:.4f} wide={wide:.4f} narrow={narrow:.4f}")
    assert true < min(shifted, wide, narrow)


@pytest.mark.parametrize("dist_type", ["tstudent", "laplace_elliptical"])
def test_scale_mixture_scale_fixed_on_low_rank_data(dist_type):
    """Gaussian data of rank 20 in p = 256 (like the real embeddings), where the
    unconstrained scale-mixture MLE does not exist: with the moment-matched
    scale the trace can no longer explode, and the EM either converges or is
    flagged within EM_MAX_ITER iterations."""
    rng = np.random.default_rng(12)
    p, r = 256, 20
    A = rng.standard_normal((p, r))
    X = 5.0 * _rand_unit(p, rng) + rng.standard_normal((3000, r)) @ A.T + 1e-3 * rng.standard_normal((3000, p))
    prm = fit_family(dist_type, X)
    tr_lw = np.trace(_lw(X - X.mean(axis=0)))
    target = tr_lw * ((prm["df"] - 2.0) / prm["df"] if dist_type == "tstudent" else 1.0)
    print(f"\n  {dist_type}: degenerate={prm.get('degenerate')} em_iter={prm['em_iter']}"
          f"  tr(Sigma)/tr(LW)={np.trace(prm['cov']) / tr_lw:.3f}"
          + (f"  nu={prm['df']:.2f}" if dist_type == "tstudent" else ""))
    assert prm["em_iter"] <= EM_MAX_ITER
    if not prm.get("degenerate"):
        assert np.trace(prm["cov"]) == pytest.approx(target, rel=1e-6)
        synth, _ = sample_parametric(dist_type, prm, len(X), rng)
        c = X.mean(axis=0)
        D = sp_stats.ks_2samp(np.linalg.norm(X - c, axis=1), np.linalg.norm(synth - c, axis=1)).statistic
        print(f"  radius KS D={D:.3f}")


def test_energy_selection_excludes_degenerate_and_logs_tables():
    rng = np.random.default_rng(13)
    p, r = 128, 10
    A = rng.standard_normal((p, r))
    X = 5.0 * _rand_unit(p, rng) + rng.standard_normal((1500, r)) @ A.T + 1e-3 * rng.standard_normal((1500, p))
    info = fit_parametric_class(X, cv_folds=3, seed=0, selection="energy")
    rep = info.selection
    print("\n" + format_selection_report("lowrank", rep))
    for key in ("cv_mean", "ks_mean", "energy_mean"):
        assert set(rep[key]) == set(rep["cv_table"])
    assert not info.params.get("degenerate")
    assert rep["ks_dist_to_mean"]["stat"] < 0.15


def test_ll_tiebreak_rule():
    """t nests the Gaussian: on Gaussian data their ll tie, so the KS decides
    (or, if the KS also ties, the higher ll).  The winner must be a candidate."""
    rng = np.random.default_rng(14)
    X = _sim("gaussian", rng)
    ranking, rep = select_family(X, 5, 0, dependent=False, selection="ll_tiebreak")
    print(f"\n  ranking={ranking}\n  candidates={rep['tie_candidates']} p={rep['tie_p_vs_best']}"
          f"\n  finalists={rep['tie_finalists']} ks={ {f: round(rep['ks_mean'][f], 3) for f in rep['tie_candidates']} }")
    assert ranking[0] in rep["tie_finalists"]
    assert set(rep["tie_finalists"]) <= set(rep["tie_candidates"])
    assert ranking[0] in ("gaussian", "tstudent")


def test_repeated_cv_and_both_tiebreaks():
    rng = np.random.default_rng(15)
    X = _sim("gaussian", rng)[:1500]
    ranking, rep = select_family(X, 5, 0, dependent=False, selection="ll_tiebreak",
                                 cv_repeats=3, tiebreak_by="energy", em_max_iter=40)
    print(f"\n  ranking={ranking}  KS->{rep['tie_winner_ks']}  energy->{rep['tie_winner_energy']}")
    assert all(len(v) == 15 for v in rep["cv_table"].values())   # 5 folds x 3 repeats
    assert ranking[0] == rep["tie_winner_energy"]
    assert rep["tie_winner_ks"] in rep["tie_candidates"]


def test_tie_margin_rule():
    """Kent significantly worse in ll (constant gap, tiny fold noise) but only
    by 0.03 nats/dim: it ties with margin 0.05, not with margin 0; with a
    clearly better KS it then wins the KS tie-break."""
    from src.methods.ParametricMethod import tiebreak
    p, k = 768, 15
    rng = np.random.default_rng(16)
    g = 600.0 + rng.normal(0, 0.5, k)
    table = {"gaussian": list(g), "kent_tangent": list(g - 0.03 * p + rng.normal(0, 0.5, k)),
             "vmf_norm": list(g - 600.0)}
    ks = {"gaussian": [0.20] * k, "kent_tangent": [0.05] * k, "vmf_norm": [0.10] * k}
    es = {"gaussian": list(1.0 + rng.normal(0, 0.01, k)), "kent_tangent": list(1.0 + rng.normal(0, 0.01, k)),
          "vmf_norm": list(3.0 + rng.normal(0, 0.01, k))}
    strict = tiebreak(table, ks, es, 4000.0, 1000.0, p, margin=0.0)
    loose = tiebreak(table, ks, es, 4000.0, 1000.0, p, margin=0.05)
    print(f"\n  margin 0   : ties={strict['tie_candidates']} KS->{strict['tie_winner_ks']}"
          f"\n  margin 0.05: ties={loose['tie_candidates']} KS->{loose['tie_winner_ks']}"
          f" energy->{loose['tie_winner_energy']}")
    assert strict["tie_candidates"] == ["gaussian"] and strict["tie_winner_ks"] == "gaussian"
    assert "kent_tangent" in loose["tie_candidates"] and "vmf_norm" not in loose["tie_candidates"]
    assert loose["tie_winner_ks"] == "kent_tangent"
    assert loose["tie_winner_energy"] == "gaussian"   # energy tie -> higher ll


@pytest.mark.parametrize("v", [0.5, 5.5, 63.0, 255.0])
def test_torch_log_kv_value_and_gradient(v):
    from src.methods.ParametricMethod import _torch_log_kv
    z = torch.tensor([0.05, 1.0, 7.0, 30.0, 150.0], dtype=torch.float64, requires_grad=True)
    np.testing.assert_allclose(_torch_log_kv(v, z).detach().numpy(), log_kv(v, z.detach().numpy()), rtol=1e-12)
    assert torch.autograd.gradcheck(lambda zz: _torch_log_kv(v, zz), (z,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_ks_norm_tiebreak_reported():
    from src.methods.ParametricMethod import tiebreak
    k, p = 15, 64
    rng = np.random.default_rng(17)
    g = 10.0 + rng.normal(0, 0.1, k)
    table = {"gaussian": list(g), "kent_tangent": list(g - 0.01 * p + rng.normal(0, 0.1, k))}
    ks = {"gaussian": [0.05] * k, "kent_tangent": [0.05] * k}
    kn = {"gaussian": [0.20] * k, "kent_tangent": [0.04] * k}   # Kent reproduces the shell
    es = {"gaussian": list(1.0 + rng.normal(0, 0.01, k)), "kent_tangent": list(1.0 + rng.normal(0, 0.01, k))}
    tb = tiebreak(table, ks, es, 4000.0, 1000.0, p, margin=0.05, knorm_table=kn)
    print(f"\n  KS->{tb['tie_winner_ks']}  energy->{tb['tie_winner_energy']}  KS norm->{tb['tie_winner_ksnorm']}")
    assert tb["tie_winner_ks"] == "gaussian"          # KS tie -> higher ll
    assert tb["tie_winner_ksnorm"] == "kent_tangent"  # clearly better shell


def test_ll_ksnorm_filter_keeps_shell_family():
    """Points on a thin spherical shell (constant norm, like ViT after
    LayerNorm): elliptical fits get the norm wrong, the ll_ksnorm filter must
    keep only families that reproduce ||x||, and every winner must pass it."""
    rng = np.random.default_rng(18)
    p, n = 32, 2000
    mu = _rand_unit(p, rng)
    prm = {"mu": mu, "tangent_cov": np.diag(np.geomspace(0.02, 0.2, p - 1)),
           "norm": {"dependent": False, "m": math.log(10.0), "s2": 0.003 ** 2}}
    X, _ = sample_parametric("kent_tangent", prm, n, rng)
    ranking, rep = select_family(X, 5, 0, dependent=False, selection="ll_ksnorm", ks_max=0.1)
    print(f"\n  ranking={ranking} eligible={rep['eligible']} "
          f"KS|x|={ {f: round(v, 3) for f, v in rep['ksnorm_mean'].items()} }")
    assert ranking[0] in rep["eligible"]
    assert rep["ksnorm_mean"][ranking[0]] <= rep["ks_threshold"]
    assert "kent_tangent" in rep["eligible"]
