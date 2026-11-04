"""
Unit tests for _propagate_params_analytically in distribution_adapter.py.

Tests:
  1. Gaussian: μ' = Rμ + t,  Σ' = RΣR^T
  2. Laplace:  locs' = R*locs + t,  scales' = |R| @ scales
  3. t-Student: μ,Σ propagate as Gaussian; ν unchanged
  4. vMF: ‖μ_dir'‖ = 1; κ' ≈ κ * σ_min(R)
  5. Radial subtype normal: mean scales with σ_mean, std scales with σ_mean
  6. Radial subtype lognormal: mu_ln shifts by log(σ_mean), sigma_ln unchanged
  7. Radial subtype gamma: shape unchanged, scale / σ_mean
  8. Chaining: T2∘T1 applied sequentially matches composing first

Run with:  python test_distribution_adapter_analytical.py
"""

import sys
import math
import numpy as np

sys.path.insert(0, ".")

from src.methods.distribution_adapter import (
    _propagate_params_analytically,
    DistributionInfo,
)

EPS = 1e-6
ATOL = 1e-5


def _make_affine(d, seed):
    """Random (but well-conditioned) affine map in R^d."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    scale = rng.uniform(0.8, 1.2, size=d)
    R = Q * scale[None, :]
    t = rng.standard_normal(d) * 0.1
    sing_vals = np.linalg.svd(R, compute_uv=False)
    sigma_min  = float(sing_vals[-1])
    sigma_mean = float(np.mean(sing_vals))
    return R, t, sigma_min, sigma_mean


def assert_close(a, b, label, atol=ATOL):
    err = np.max(np.abs(np.asarray(a) - np.asarray(b)))
    assert err < atol, f"{label}: max error {err:.2e} >= {atol:.2e}"


# ---------------------------------------------------------------------------
# 1. Gaussian
# ---------------------------------------------------------------------------
def test_gaussian():
    d = 8
    R, t, sm, smean = _make_affine(d, 0)
    mu  = np.random.default_rng(1).standard_normal(d)
    cov = np.eye(d) * 2.0 + 0.5

    info = DistributionInfo("gaussian", {"mean": mu, "cov": cov}, 0.0, 0.0, "bic")
    out  = _propagate_params_analytically(info, R, t, sm, smean)

    assert out.dist_type == "gaussian"
    assert_close(out.params["mean"], mu @ R.T + t, "gaussian mu")
    assert_close(out.params["cov"],  R @ cov @ R.T + EPS * np.eye(d), "gaussian cov")
    print("  PASS  test_gaussian")


# ---------------------------------------------------------------------------
# 2. Laplace
# ---------------------------------------------------------------------------
def test_laplace():
    d = 6
    R, t, sm, smean = _make_affine(d, 2)
    locs   = np.random.default_rng(3).standard_normal(d)
    scales = np.abs(np.random.default_rng(4).standard_normal(d)) + 0.5

    info = DistributionInfo("laplace", {"locs": locs, "scales": scales}, 0.0, 0.0, "bic")
    out  = _propagate_params_analytically(info, R, t, sm, smean)

    assert out.dist_type == "laplace"
    assert_close(out.params["locs"],   locs @ R.T + t,      "laplace locs")
    assert_close(out.params["scales"], np.abs(R) @ scales,  "laplace scales")
    print("  PASS  test_laplace")


# ---------------------------------------------------------------------------
# 3. t-Student
# ---------------------------------------------------------------------------
def test_tstudent():
    d = 5
    R, t, sm, smean = _make_affine(d, 5)
    mu  = np.random.default_rng(6).standard_normal(d)
    cov = np.eye(d) * 3.0
    nu  = 4.5

    info = DistributionInfo("tstudent", {"mean": mu, "cov": cov, "df": nu}, 0.0, 0.0, "bic")
    out  = _propagate_params_analytically(info, R, t, sm, smean)

    assert out.dist_type == "tstudent"
    assert_close(out.params["mean"], mu @ R.T + t,              "tstudent mu")
    assert_close(out.params["cov"],  R @ cov @ R.T + EPS * np.eye(d), "tstudent cov")
    assert abs(out.params["df"] - nu) < 1e-10, "tstudent df must be unchanged"
    print("  PASS  test_tstudent")


# ---------------------------------------------------------------------------
# 4. vMF: unit direction + kappa scaling
# ---------------------------------------------------------------------------
def test_vmf_direction_and_kappa():
    d = 16
    R, t, sm, smean = _make_affine(d, 7)
    mu_dir = np.random.default_rng(8).standard_normal(d)
    mu_dir /= np.linalg.norm(mu_dir)
    kappa = 500.0

    norm_dist = {"type": "normal", "mean": 10.0, "std": 1.0}
    info = DistributionInfo("vmf", {"mu": mu_dir, "kappa": kappa, "norm_dist": norm_dist}, 0.0, 0.0, "bic")
    out  = _propagate_params_analytically(info, R, t, sm, smean)

    assert out.dist_type == "vmf"

    # Direction must be unit norm
    mu_new = out.params["mu"]
    assert abs(np.linalg.norm(mu_new) - 1.0) < 1e-10, f"mu_dir not unit: {np.linalg.norm(mu_new)}"

    # Direction must be R*mu_dir (normalized)
    expected_dir = mu_dir @ R.T
    expected_dir /= np.linalg.norm(expected_dir)
    assert_close(mu_new, expected_dir, "vmf direction")

    # kappa' ≈ kappa * sigma_min
    expected_kappa = kappa * sm
    assert_close(out.params["kappa"], expected_kappa, "vmf kappa", atol=1e-4)
    print("  PASS  test_vmf_direction_and_kappa")


# ---------------------------------------------------------------------------
# 5-7. Radial subtypes
# ---------------------------------------------------------------------------
def test_radial_normal():
    d = 4
    R, t, sm, smean = _make_affine(d, 9)
    mu_r, sigma_r = 15.0, 2.0
    t_norm = float(np.linalg.norm(t))

    mu_dir = np.ones(d) / math.sqrt(d)
    norm_dist = {"type": "normal", "mean": mu_r, "std": sigma_r}
    info = DistributionInfo("vmf", {"mu": mu_dir, "kappa": 100.0, "norm_dist": norm_dist}, 0.0, 0.0, "bic")
    out  = _propagate_params_analytically(info, R, t, sm, smean)

    nd = out.params["norm_dist"]
    assert nd["type"] == "normal"
    assert_close(nd["mean"], smean * mu_r + t_norm, "radial normal mean")
    assert_close(nd["std"],  smean * sigma_r,        "radial normal std")
    print("  PASS  test_radial_normal")


def test_radial_lognormal():
    d = 4
    R, t, sm, smean = _make_affine(d, 10)
    mu_ln, sigma_ln = 2.5, 0.3

    mu_dir = np.ones(d) / math.sqrt(d)
    norm_dist = {"type": "lognormal", "mu": mu_ln, "sigma": sigma_ln}
    info = DistributionInfo("vmf", {"mu": mu_dir, "kappa": 100.0, "norm_dist": norm_dist}, 0.0, 0.0, "bic")
    out  = _propagate_params_analytically(info, R, t, sm, smean)

    nd = out.params["norm_dist"]
    assert nd["type"] == "lognormal"
    assert_close(nd["mu"],    mu_ln + math.log(max(smean, EPS)), "radial lognormal mu")
    assert_close(nd["sigma"], sigma_ln,                          "radial lognormal sigma")
    print("  PASS  test_radial_lognormal")


def test_radial_gamma():
    d = 4
    R, t, sm, smean = _make_affine(d, 11)
    shape, scale = 3.0, 5.0

    mu_dir = np.ones(d) / math.sqrt(d)
    norm_dist = {"type": "gamma", "shape": shape, "loc": 0.0, "scale": scale}
    info = DistributionInfo("vmf", {"mu": mu_dir, "kappa": 100.0, "norm_dist": norm_dist}, 0.0, 0.0, "bic")
    out  = _propagate_params_analytically(info, R, t, sm, smean)

    nd = out.params["norm_dist"]
    assert nd["type"] == "gamma"
    assert_close(nd["shape"], shape,              "radial gamma shape")
    assert_close(nd["scale"], scale / max(smean, EPS), "radial gamma scale")
    print("  PASS  test_radial_gamma")


# ---------------------------------------------------------------------------
# 8. Chaining: propagating T1 then T2 == composing T2∘T1 directly
# ---------------------------------------------------------------------------
def test_chaining_gaussian():
    d = 6
    R1, t1, sm1, smean1 = _make_affine(d, 12)
    R2, t2, sm2, smean2 = _make_affine(d, 13)

    mu  = np.random.default_rng(14).standard_normal(d)
    cov = np.eye(d) * 2.0

    info = DistributionInfo("gaussian", {"mean": mu, "cov": cov}, 0.0, 0.0, "bic")

    # Sequential
    step1 = _propagate_params_analytically(info,  R1, t1, sm1, smean1)
    step2 = _propagate_params_analytically(step1, R2, t2, sm2, smean2)

    # Direct composition: T2(T1(z)) = z @ R1.T @ R2.T + t1 @ R2.T + t2
    R_comp = R2 @ R1
    t_comp = t1 @ R2.T + t2
    sing_comp = np.linalg.svd(R_comp, compute_uv=False)
    direct = _propagate_params_analytically(
        info, R_comp, t_comp, float(sing_comp[-1]), float(np.mean(sing_comp))
    )

    assert_close(step2.params["mean"], direct.params["mean"], "chaining gaussian mu")
    assert_close(step2.params["cov"],  direct.params["cov"],  "chaining gaussian cov", atol=1e-4)
    print("  PASS  test_chaining_gaussian")


def test_chaining_vmf():
    d = 8
    R1, t1, sm1, smean1 = _make_affine(d, 15)
    R2, t2, sm2, smean2 = _make_affine(d, 16)

    mu_dir = np.random.default_rng(17).standard_normal(d)
    mu_dir /= np.linalg.norm(mu_dir)
    kappa = 200.0
    norm_dist = {"type": "lognormal", "mu": 2.3, "sigma": 0.4}

    info = DistributionInfo("vmf", {"mu": mu_dir, "kappa": kappa, "norm_dist": norm_dist}, 0.0, 0.0, "bic")

    step1 = _propagate_params_analytically(info,  R1, t1, sm1, smean1)
    step2 = _propagate_params_analytically(step1, R2, t2, sm2, smean2)

    R_comp = R2 @ R1
    t_comp = t1 @ R2.T + t2
    sing_comp = np.linalg.svd(R_comp, compute_uv=False)
    direct = _propagate_params_analytically(
        info, R_comp, t_comp, float(sing_comp[-1]), float(np.mean(sing_comp))
    )

    # Direction chains exactly: normalize(R2 @ normalize(R1 @ mu)) == normalize(R2@R1 @ mu)
    assert_close(step2.params["mu"], direct.params["mu"], "chaining vmf direction")
    # Lognormal mu_ln: sequential gives log(sm1)+log(sm2), direct gives log(sm_comp).
    # These differ because sigma_mean(R2@R1) != sigma_mean(R1)*sigma_mean(R2).
    # Check that the sequential and direct are at least in the same ballpark (< 5% relative).
    mu_ln_seq  = step2.params["norm_dist"]["mu"]
    mu_ln_dir  = direct.params["norm_dist"]["mu"]
    rel_err = abs(mu_ln_seq - mu_ln_dir) / (abs(mu_ln_dir) + 1e-12)
    assert rel_err < 0.05, f"chaining vmf lognormal mu_ln relative error {rel_err:.4f} >= 0.05"
    print("  PASS  test_chaining_vmf")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_gaussian,
        test_laplace,
        test_tstudent,
        test_vmf_direction_and_kappa,
        test_radial_normal,
        test_radial_lognormal,
        test_radial_gamma,
        test_chaining_gaussian,
        test_chaining_vmf,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed.")
    sys.exit(0 if failed == 0 else 1)
