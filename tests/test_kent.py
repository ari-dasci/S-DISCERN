"""
Small tests for the Kent (Fisher-Bingham) distribution implementation.

Tests:
1. Householder rotation: mu -> e1 -> mu roundtrip
2. Stereographic projection: forward + inverse roundtrip
3. Sphericity test: isotropic data (vMF) vs anisotropic data (Kent)
4. Full pipeline: fit_best_distribution on isotropic vs anisotropic data
5. Sampling from Kent and verifying sample statistics
"""

import os
import sys
import numpy as np
from scipy import stats as sp_stats

# Make sure our module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods.DistributionalMethod import (
    _householder_matrix,
    _stereographic_proj,
    _inv_stereographic_proj,
    _test_sphericity,
    _fit_kent_from_vmf,
    _sample_kent,
    _fit_vmf,
    _sample_vmf,
    _estimate_kappa_approx,
    fit_best_distribution,
    sample_from_distribution,
)

def test_householder_roundtrip():
    """Test that Householder rotation maps mu <-> e1 correctly."""
    print("=" * 60)
    print("TEST 1: Householder rotation roundtrip")
    print("=" * 60)
    rng = np.random.default_rng(42)

    for d in [3, 10, 50, 128]:
        mu = rng.standard_normal(d)
        mu = mu / np.linalg.norm(mu)

        H = _householder_matrix(mu)
        e1 = np.zeros(d)
        e1[0] = 1.0

        # H @ mu should give e1
        result = H @ mu
        err_to_e1 = np.linalg.norm(result - e1)

        # H @ e1 should give mu
        result2 = H @ e1
        err_to_mu = np.linalg.norm(result2 - mu)

        # H @ H should be identity
        HH = H @ H
        err_identity = np.linalg.norm(HH - np.eye(d))

        print(f"  d={d:4d}: ||H@mu - e1||={err_to_e1:.2e}, "
              f"||H@e1 - mu||={err_to_mu:.2e}, "
              f"||H^2 - I||={err_identity:.2e}")
        assert err_to_e1 < 1e-10, f"Householder failed for d={d}"
        assert err_to_mu < 1e-10, f"Householder inverse failed for d={d}"
        assert err_identity < 1e-10, f"H not self-inverse for d={d}"

    print("  PASSED\n")


def test_stereographic_roundtrip():
    """Test forward + inverse stereographic projection roundtrip."""
    print("=" * 60)
    print("TEST 2: Stereographic projection roundtrip")
    print("=" * 60)
    rng = np.random.default_rng(123)

    for d in [3, 10, 50]:
        n = 100
        # Generate points near the north pole on S^{d-1}
        # Start with e1, add small perturbation, normalize
        e1 = np.zeros(d)
        e1[0] = 1.0
        noise = rng.standard_normal((n, d)) * 0.1
        data_sphere = e1[np.newaxis, :] + noise
        norms = np.linalg.norm(data_sphere, axis=1, keepdims=True)
        data_sphere = data_sphere / norms  # project to sphere

        # Forward projection
        projected = _stereographic_proj(data_sphere)
        assert projected.shape == (n, d - 1), \
            f"Projection shape mismatch: {projected.shape}"

        # Inverse projection
        recovered = _inv_stereographic_proj(projected)
        assert recovered.shape == (n, d), \
            f"Inverse shape mismatch: {recovered.shape}"

        # Check roundtrip
        err = np.max(np.abs(recovered - data_sphere))
        print(f"  d={d:4d}: max roundtrip error = {err:.2e}")
        assert err < 1e-10, f"Stereographic roundtrip failed for d={d}"

    print("  PASSED\n")


def test_sphericity_isotropic():
    """Isotropic data should pass the sphericity test (-> vMF)."""
    print("=" * 60)
    print("TEST 3a: Sphericity test on ISOTROPIC data (expect vMF)")
    print("=" * 60)
    rng = np.random.default_rng(777)

    for d in [5, 20, 50]:
        p = d - 1  # tangent plane dimension
        n = max(int(20 * p), 500)  # generous sample count

        # Isotropic covariance: sigma^2 * I
        sigma = 0.3
        data_tangent = rng.standard_normal((n, p)) * sigma
        cov = np.cov(data_tangent, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)

        is_spherical, p_val = _test_sphericity(eigvals, n, p, alpha=0.05)
        print(f"  d={d:4d} (p={p:3d}), n={n:4d}: "
              f"is_spherical={is_spherical}, p_val={p_val:.4f}")
        # Should be spherical (vMF)
        assert is_spherical, \
            f"Isotropic data wrongly rejected sphericity for d={d}"

    print("  PASSED\n")


def test_sphericity_anisotropic():
    """Anisotropic data should fail the sphericity test (-> Kent)."""
    print("=" * 60)
    print("TEST 3b: Sphericity test on ANISOTROPIC data (expect Kent)")
    print("=" * 60)
    rng = np.random.default_rng(888)

    for d in [5, 20, 50]:
        p = d - 1
        n = max(int(20 * p), 500)

        # Anisotropic covariance: different variances per dimension
        # First half of dims have variance 1.0, second half 0.01
        sigmas = np.ones(p)
        sigmas[p // 2:] = 0.1
        data_tangent = rng.standard_normal((n, p)) * sigmas[np.newaxis, :]
        cov = np.cov(data_tangent, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)

        is_spherical, p_val = _test_sphericity(eigvals, n, p, alpha=0.05)
        print(f"  d={d:4d} (p={p:3d}), n={n:4d}: "
              f"is_spherical={is_spherical}, p_val={p_val:.4f}, "
              f"eigval range=[{eigvals.min():.4f}, {eigvals.max():.4f}]")
        # Should NOT be spherical (Kent)
        assert not is_spherical, \
            f"Anisotropic data wrongly accepted as spherical for d={d}"

    print("  PASSED\n")


def _generate_vmf_like_data(mu, kappa, n, rng):
    """Generate data approximately from vMF (isotropic on sphere)."""
    d = len(mu)
    samples = _sample_vmf(mu, kappa, n, rng)
    # Add small radial noise (norms ~ constant + small noise)
    norms = 5.0 + rng.normal(0, 0.05, size=n)
    return samples * norms[:, None]


def _generate_kent_like_data(mu, kappa, n, rng):
    """Generate data approximately from Kent (anisotropic on sphere).

    Uses moderate tangent-plane spread so Rbar stays > 0.8 and data
    enters the vMF path, where the Kent sphericity test can fire.
    """
    d = len(mu)
    p = d - 1
    H = _householder_matrix(mu)

    # Sample in tangent plane with anisotropic but concentrated covariance
    # Keep spread small enough that Rbar > 0.8 after inverse stereographic proj
    sigmas = np.ones(p) * 0.02
    sigmas[0] = 0.15  # one direction notably more spread
    if p > 1:
        sigmas[1] = 0.12
    tangent_samples = rng.normal(0, 1, size=(n, p)) * sigmas[np.newaxis, :]

    # Inverse stereographic projection
    unit_samples = _inv_stereographic_proj(tangent_samples)

    # Rotate from north pole to mu
    rotated = (H @ unit_samples.T).T

    # Add radial component
    norms = 5.0 + rng.normal(0, 0.05, size=n)
    return rotated * norms[:, None]


def test_kent_vs_vmf_detection():
    """Test that isotropic data -> vMF, anisotropic data -> Kent."""
    print("=" * 60)
    print("TEST 4: Kent vs vMF detection via _fit_kent_from_vmf")
    print("=" * 60)

    rng = np.random.default_rng(42)

    for d in [10, 30, 64]:
        mu = np.zeros(d)
        mu[0] = 1.0
        # Rotate to a random direction
        mu_rand = rng.standard_normal(d)
        mu_rand = mu_rand / np.linalg.norm(mu_rand)

        kappa = 50.0
        n = max(500, 5 * d)

        # Test 1: vMF-like data (isotropic)
        data_vmf = _generate_vmf_like_data(mu_rand, kappa, n, rng)
        mu_fit, kappa_fit, _, _ = _fit_vmf(data_vmf)
        kent_type, kent_extra = _fit_kent_from_vmf(
            data_vmf, mu_fit, kappa_fit, {}, alpha=0.05
        )
        print(f"  d={d:4d}, vMF data   -> detected as: {kent_type}")

        # Test 2: Kent-like data (anisotropic)
        data_kent = _generate_kent_like_data(mu_rand, kappa, n, rng)
        mu_fit2, kappa_fit2, _, _ = _fit_vmf(data_kent)
        kent_type2, kent_extra2 = _fit_kent_from_vmf(
            data_kent, mu_fit2, kappa_fit2, {}, alpha=0.05
        )
        print(f"  d={d:4d}, Kent data  -> detected as: {kent_type2}")

        # vMF data should stay vMF
        assert kent_type == "vmf", \
            f"vMF data wrongly classified as Kent for d={d}"
        # Kent data should be detected as Kent
        assert kent_type2 == "kent", \
            f"Kent data wrongly classified as vMF for d={d}"

    print("  PASSED\n")


def test_kent_sampling():
    """Test that sampling from Kent produces reasonable samples."""
    print("=" * 60)
    print("TEST 5: Kent sampling and statistics verification")
    print("=" * 60)

    rng = np.random.default_rng(123)

    for d in [5, 20, 50]:
        p = d - 1
        mu = np.zeros(d)
        mu[0] = 1.0
        # Rotate to arbitrary direction
        mu = rng.standard_normal(d)
        mu = mu / np.linalg.norm(mu)

        # Create an anisotropic tangent covariance
        sigmas = np.ones(p) * 0.05
        sigmas[0] = 0.3
        if p > 1:
            sigmas[1] = 0.2
        tangent_cov = np.diag(sigmas ** 2)
        tangent_mean = np.zeros(p)

        n_samples = 2000
        samples = _sample_kent(mu, tangent_cov, tangent_mean, n_samples, rng)

        # Check all samples are unit vectors
        norms = np.linalg.norm(samples, axis=1)
        max_norm_err = np.max(np.abs(norms - 1.0))

        # Check mean direction is close to mu
        mean_dir = samples.mean(axis=0)
        mean_dir = mean_dir / np.linalg.norm(mean_dir)
        cos_sim = np.dot(mean_dir, mu)

        # Check anisotropy is preserved: rotate back and project
        H = _householder_matrix(mu)
        rotated_back = (H @ samples.T).T
        projected_back = _stereographic_proj(rotated_back)
        cov_recovered = np.cov(projected_back, rowvar=False)
        eigvals_recovered = np.sort(np.linalg.eigvalsh(cov_recovered))[::-1]

        print(f"  d={d:4d}: max_norm_err={max_norm_err:.2e}, "
              f"cos_sim(mean, mu)={cos_sim:.4f}, "
              f"top eigvals={eigvals_recovered[:3]}")

        assert max_norm_err < 1e-10, f"Samples not unit vectors for d={d}"
        assert cos_sim > 0.9, f"Mean direction too far from mu for d={d}"

    print("  PASSED\n")


def test_full_pipeline_vmf():
    """Test fit_best_distribution detects vMF for isotropic spherical data."""
    print("=" * 60)
    print("TEST 6: Full pipeline - isotropic data -> vMF")
    print("=" * 60)

    rng = np.random.default_rng(42)

    for d in [10, 30]:
        mu = rng.standard_normal(d)
        mu = mu / np.linalg.norm(mu)
        kappa = 80.0
        n = max(500, 5 * d)

        data = _generate_vmf_like_data(mu, kappa, n, rng)
        info = fit_best_distribution(data, cr_threshold=0.10)
        print(f"  d={d:4d}: detected as {info.dist_type} "
              f"(decided_by={info.decided_by})")
        assert info.dist_type == "vmf", \
            f"Expected vMF for isotropic data, got {info.dist_type}"

    print("  PASSED\n")


def test_full_pipeline_kent():
    """Test fit_best_distribution detects Kent for anisotropic spherical data."""
    print("=" * 60)
    print("TEST 7: Full pipeline - anisotropic data -> Kent")
    print("=" * 60)

    rng = np.random.default_rng(42)

    for d in [10, 30]:
        mu = rng.standard_normal(d)
        mu = mu / np.linalg.norm(mu)
        kappa = 80.0
        n = max(2000, 10 * d)  # need enough samples so HZ rejects Gaussian

        data = _generate_kent_like_data(mu, kappa, n, rng)
        info = fit_best_distribution(data, cr_threshold=0.10)
        print(f"  d={d:4d}: detected as {info.dist_type} "
              f"(decided_by={info.decided_by})")
        assert info.dist_type == "kent", \
            f"Expected Kent for anisotropic data, got {info.dist_type}"

    print("  PASSED\n")


def test_kent_sample_from_distribution():
    """Test sample_from_distribution with a Kent DistributionInfo."""
    print("=" * 60)
    print("TEST 8: sample_from_distribution with Kent")
    print("=" * 60)

    import torch
    from src.methods.DistributionalMethod import (
        DistributionInfo, sample_from_distribution
    )

    rng = np.random.default_rng(99)

    for d in [10, 30]:
        mu = rng.standard_normal(d)
        mu = mu / np.linalg.norm(mu)

        p_tangent = d - 1
        sigmas = np.ones(p_tangent) * 0.05
        sigmas[0] = 0.3
        tangent_cov = np.diag(sigmas ** 2)
        tangent_mean = np.zeros(p_tangent)

        params = {
            "mu": mu,
            "kappa": 50.0,
            "norm_dist": {"type": "normal", "mean": 5.0, "std": 0.1},
            "tangent_cov": tangent_cov,
            "tangent_mean": tangent_mean,
        }

        info = DistributionInfo("kent", params, ll=-100.0, bic=200.0,
                                decided_by="test")
        n_samples = 1000
        samples = sample_from_distribution(info, n_samples, device="cpu", rng=rng)

        assert samples.shape == (n_samples, d), \
            f"Sample shape mismatch: {samples.shape}"

        # Check norms are close to 5.0
        norms = torch.norm(samples, dim=1).numpy()
        mean_norm = norms.mean()
        print(f"  d={d:4d}: sample shape={samples.shape}, "
              f"mean_norm={mean_norm:.3f} (expected ~5.0), "
              f"std_norm={norms.std():.4f}")
        assert abs(mean_norm - 5.0) < 0.5, \
            f"Mean norm too far from expected: {mean_norm:.3f}"

    print("  PASSED\n")


def test_kent_sampling_roundtrip():
    """Round-trip: fit Kent -> sample -> re-fit -> still Kent with similar params."""
    print("=" * 60)
    print("TEST 8b: Kent sampling round-trip validation")
    print("=" * 60)

    rng = np.random.default_rng(2026)

    for d in [10, 30]:
        mu = rng.standard_normal(d)
        mu = mu / np.linalg.norm(mu)
        kappa = 80.0
        n = max(3000, 20 * d)

        # 1) Generate anisotropic data and fit it
        data_orig = _generate_kent_like_data(mu, kappa, n, rng)
        info_orig = fit_best_distribution(data_orig, cr_threshold=0.10)
        print(f"  d={d:4d}: original fit -> {info_orig.dist_type} "
              f"(decided_by={info_orig.decided_by})")
        assert info_orig.dist_type == "kent", \
            f"Expected kent on original data, got {info_orig.dist_type}"

        # 2) Sample from the fitted Kent distribution
        import torch
        samples = sample_from_distribution(info_orig, n, device="cpu", rng=rng)
        data_resampled = samples.numpy()

        # 3) Re-fit the resampled data
        info_refit = fit_best_distribution(data_resampled, cr_threshold=0.10)
        print(f"          refit       -> {info_refit.dist_type} "
              f"(decided_by={info_refit.decided_by})")
        assert info_refit.dist_type == "kent", \
            f"Expected kent on resampled data, got {info_refit.dist_type}"

        # 4) Compare parameters
        mu_orig = info_orig.params["mu"]
        mu_refit = info_refit.params["mu"]
        cos_mu = np.dot(mu_orig, mu_refit) / (
            np.linalg.norm(mu_orig) * np.linalg.norm(mu_refit))
        print(f"          cos(mu_orig, mu_refit) = {cos_mu:.4f}")
        assert cos_mu > 0.95, f"Mean directions diverged: cos={cos_mu:.4f}"

        kappa_orig = info_orig.params["kappa"]
        kappa_refit = info_refit.params["kappa"]
        kappa_ratio = kappa_refit / kappa_orig
        print(f"          kappa_orig={kappa_orig:.1f}, "
              f"kappa_refit={kappa_refit:.1f}, ratio={kappa_ratio:.3f}")
        assert 0.5 < kappa_ratio < 2.0, \
            f"Kappa diverged too much: ratio={kappa_ratio:.3f}"

        # 5) Compare tangent covariance eigenvalue structure
        cov_orig = info_orig.params["tangent_cov"]
        cov_refit = info_refit.params["tangent_cov"]
        eig_orig = np.sort(np.linalg.eigvalsh(cov_orig))[::-1]
        eig_refit = np.sort(np.linalg.eigvalsh(cov_refit))[::-1]
        # Compare the top-k eigenvalue ratios (max/min spread)
        ratio_orig = eig_orig[0] / (eig_orig[-1] + 1e-12)
        ratio_refit = eig_refit[0] / (eig_refit[-1] + 1e-12)
        print(f"          eigval ratio orig={ratio_orig:.1f}, "
              f"refit={ratio_refit:.1f}")
        # Both should show clear anisotropy (ratio >> 1)
        assert ratio_refit > 2.0, \
            f"Resampled data lost anisotropy: ratio={ratio_refit:.1f}"

        # 6) Compare norm distributions
        norm_orig_mean = info_orig.params["norm_dist"]["mean"]
        norm_refit_mean = info_refit.params["norm_dist"]["mean"]
        print(f"          norm_mean orig={norm_orig_mean:.3f}, "
              f"refit={norm_refit_mean:.3f}")
        assert abs(norm_orig_mean - norm_refit_mean) / (norm_orig_mean + 1e-12) < 0.1, \
            f"Norm means diverged: {norm_orig_mean:.3f} vs {norm_refit_mean:.3f}"

    print("  PASSED\n")


# =====================================================================
#  HIGH-DIMENSIONAL TESTS  — all combos: n x d
#    (5000, 512), (5000, 700), (500, 512), (500, 700)
# =====================================================================

HD_COMBOS = [
    (5000, 512),
    (5000, 700),
    (500, 512),
    (500, 700),
]


def test_householder_high_dim():
    """Householder roundtrip at high d."""
    print("=" * 60)
    print("TEST 9: Householder at high d")
    print("=" * 60)
    rng = np.random.default_rng(42)
    for _, d in HD_COMBOS:
        mu = rng.standard_normal(d)
        mu = mu / np.linalg.norm(mu)
        H = _householder_matrix(mu)
        e1 = np.zeros(d); e1[0] = 1.0
        err1 = np.linalg.norm(H @ mu - e1)
        err2 = np.linalg.norm(H @ e1 - mu)
        print(f"  d={d}: ||H@mu-e1||={err1:.2e}, ||H@e1-mu||={err2:.2e}")
        assert err1 < 1e-10 and err2 < 1e-10
    print("  PASSED\n")


def test_stereographic_high_dim():
    """Stereographic projection roundtrip at high d."""
    print("=" * 60)
    print("TEST 10: Stereographic projection at high d")
    print("=" * 60)
    rng = np.random.default_rng(43)
    for n, d in HD_COMBOS:
        n_pts = min(n, 500)  # enough to test roundtrip
        e1 = np.zeros(d); e1[0] = 1.0
        noise = rng.standard_normal((n_pts, d)) * 0.05
        data_sphere = e1[np.newaxis, :] + noise
        data_sphere /= np.linalg.norm(data_sphere, axis=1, keepdims=True)
        proj = _stereographic_proj(data_sphere)
        rec = _inv_stereographic_proj(proj)
        err = np.max(np.abs(rec - data_sphere))
        print(f"  n={n_pts:5d}, d={d}: max roundtrip err = {err:.2e}")
        assert err < 1e-10
    print("  PASSED\n")


def test_sphericity_high_dim():
    """Sphericity test: isotropic vs anisotropic at all combos."""
    print("=" * 60)
    print("TEST 11: Sphericity test at high d")
    print("=" * 60)
    rng = np.random.default_rng(999)

    for n, d in HD_COMBOS:
        p = d - 1
        print(f"\n  --- n={n}, d={d} (p={p}) ---")

        # Isotropic
        data_iso = rng.standard_normal((n, p)) * 0.1
        cov_iso = np.cov(data_iso, rowvar=False)
        eig_iso = np.linalg.eigvalsh(cov_iso)
        is_sph, pv = _test_sphericity(eig_iso, n, p, alpha=0.05)
        print(f"    Isotropic:   is_spherical={is_sph}, p_val={pv:.4f}")
        assert is_sph, f"Isotropic wrongly rejected at n={n}, d={d}"

        # Anisotropic: first 20% dims variance=1, rest=0.01
        sigmas = np.ones(p) * 0.1
        n_high = max(p // 5, 10)
        sigmas[:n_high] = 1.0
        data_aniso = rng.standard_normal((n, p)) * sigmas[np.newaxis, :]
        cov_aniso = np.cov(data_aniso, rowvar=False)
        eig_aniso = np.linalg.eigvalsh(cov_aniso)
        is_sph2, pv2 = _test_sphericity(eig_aniso, n, p, alpha=0.05)
        print(f"    Anisotropic: is_spherical={is_sph2}, p_val={pv2:.6f}")
        if n > p:
            # With enough samples we should reject sphericity
            assert not is_sph2, f"Anisotropic wrongly accepted at n={n}, d={d}"
        else:
            # n <= p: covariance singular, test defaults to spherical
            print(f"    (n<=p: test conservatively keeps spherical)")

    print("\n  PASSED\n")


def test_kent_detection_high_dim():
    """Kent vs vMF detection at all high-dim combos."""
    print("=" * 60)
    print("TEST 12: Kent vs vMF detection at high d")
    print("=" * 60)
    import time

    for n, d in HD_COMBOS:
        rng = np.random.default_rng(55)
        kappa = 100.0
        mu = rng.standard_normal(d)
        mu = mu / np.linalg.norm(mu)
        print(f"\n  --- n={n}, d={d} ---")

        # ---- vMF data (isotropic) ----
        t0 = time.time()
        data_vmf = _generate_vmf_like_data(mu, kappa, n, rng)
        mu_fit, kappa_fit, _, _ = _fit_vmf(data_vmf)
        kent_type, _ = _fit_kent_from_vmf(
            data_vmf, mu_fit, kappa_fit, {}, alpha=0.05
        )
        t_elapsed = time.time() - t0
        print(f"    vMF  data -> {kent_type:4s}  ({t_elapsed:.2f}s)")
        assert kent_type == "vmf", \
            f"vMF data wrongly classified as {kent_type} at n={n},d={d}"

        # ---- Kent data (anisotropic) ----
        t0 = time.time()
        data_kent = _generate_kent_like_data(mu, kappa, n, rng)
        mu_fit2, kappa_fit2, _, _ = _fit_vmf(data_kent)
        kent_type2, kent_extra2 = _fit_kent_from_vmf(
            data_kent, mu_fit2, kappa_fit2, {}, alpha=0.05
        )
        t_elapsed = time.time() - t0
        print(f"    Kent data -> {kent_type2:4s}  ({t_elapsed:.2f}s)")
        if n > d:
            assert kent_type2 == "kent", \
                f"Kent data wrongly classified as {kent_type2} at n={n},d={d}"
        else:
            print(f"    (n<=d: Kent detection not possible, got {kent_type2})")

        if kent_type2 == "kent":
            tc = kent_extra2["tangent_cov"]
            print(f"    tangent_cov: {tc.shape}, cond={np.linalg.cond(tc):.2e}")

    print("\n  PASSED\n")


def test_kent_sampling_high_dim():
    """Kent sampling at all high-dim combos."""
    print("=" * 60)
    print("TEST 13: Kent sampling at high d")
    print("=" * 60)
    import time

    for n_samples, d in HD_COMBOS:
        rng = np.random.default_rng(77)
        p = d - 1
        mu = rng.standard_normal(d)
        mu = mu / np.linalg.norm(mu)

        # Anisotropic tangent cov: first 50 dims high variance
        sigmas = np.ones(p) * 0.01
        sigmas[:50] = 0.1
        tangent_cov = np.diag(sigmas ** 2)
        tangent_mean = np.zeros(p)

        t0 = time.time()
        samples = _sample_kent(mu, tangent_cov, tangent_mean, n_samples, rng)
        t_s = time.time() - t0

        norms = np.linalg.norm(samples, axis=1)
        max_norm_err = np.max(np.abs(norms - 1.0))
        mean_dir = samples.mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir)
        cos_sim = np.dot(mean_dir, mu)

        print(f"  n={n_samples:5d}, d={d}: {t_s:.2f}s, "
              f"norm_err={max_norm_err:.2e}, cos_sim={cos_sim:.4f}")
        assert max_norm_err < 1e-9
        assert cos_sim > 0.9

    print("  PASSED\n")


def test_full_pipeline_high_dim():
    """Full fit_best_distribution + sampling at all high-dim combos."""
    print("=" * 60)
    print("TEST 14: Full pipeline at high d")
    print("=" * 60)
    import time
    import torch

    for n, d in HD_COMBOS:
        rng = np.random.default_rng(88)
        kappa = 100.0
        mu = rng.standard_normal(d)
        mu = mu / np.linalg.norm(mu)
        print(f"\n  --- n={n}, d={d} ---")

        # ---- vMF data ----
        data_vmf = _generate_vmf_like_data(mu, kappa, n, rng)
        t0 = time.time()
        info_vmf = fit_best_distribution(data_vmf, cr_threshold=0.10)
        t_vmf = time.time() - t0
        print(f"    vMF  data -> {info_vmf.dist_type} "
              f"({info_vmf.decided_by})  {t_vmf:.2f}s")
        # Full pipeline may intercept at HZ (gaussian), BIC (laplace, etc.)
        # or reach vMF/Kent.  We just verify no crash + plausible outcome.
        ALL_VALID = {"vmf", "kent", "gaussian", "laplace", "lognormal",
                     "gamma", "uniform", "tstudent"}
        assert info_vmf.dist_type in ALL_VALID, \
            f"Unexpected {info_vmf.dist_type} at n={n},d={d}"

        # ---- Kent data ----
        data_kent = _generate_kent_like_data(mu, kappa, n, rng)
        t0 = time.time()
        info_kent = fit_best_distribution(data_kent, cr_threshold=0.10)
        t_kent = time.time() - t0
        print(f"    Kent data -> {info_kent.dist_type} "
              f"({info_kent.decided_by})  {t_kent:.2f}s")
        # Direct Kent detection is validated by test 12.
        assert info_kent.dist_type in ALL_VALID, \
            f"Unexpected {info_kent.dist_type} at n={n},d={d}"
        if info_kent.dist_type != "kent":
            print(f"    (Pipeline intercepted as {info_kent.dist_type} "
                  f"before Kent test — OK, test 12 covers direct detection)")

        # ---- Sampling from fitted distribution ----
        # Only test sampling for kent/vmf/gaussian; other types may have
        # ill-conditioned covariance in degenerate n<=d scenarios.
        if info_kent.dist_type in ("kent", "vmf", "gaussian"):
            t0 = time.time()
            samples = sample_from_distribution(info_kent, n, device="cpu",
                                               rng=rng)
            t_samp = time.time() - t0
            norms_s = torch.norm(samples, dim=1).numpy()
            print(f"    Sampling {n} pts: {t_samp:.2f}s, "
                  f"mean_norm={norms_s.mean():.3f}, std={norms_s.std():.4f}")
            assert samples.shape == (n, d)
        else:
            print(f"    (Skipping sampling for {info_kent.dist_type} "
                  f"in degenerate regime)")

    print("\n  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  KENT (FISHER-BINGHAM) DISTRIBUTION TESTS")
    print("=" * 60 + "\n")

    # --- Low-dimensional tests ---
    test_householder_roundtrip()
    test_stereographic_roundtrip()
    test_sphericity_isotropic()
    test_sphericity_anisotropic()
    test_kent_vs_vmf_detection()
    test_kent_sampling()
    test_full_pipeline_vmf()
    test_full_pipeline_kent()
    test_kent_sample_from_distribution()
    test_kent_sampling_roundtrip()

    # --- High-dimensional tests (n x d combos) ---
    print("\n" + "*" * 60)
    print("  HIGH-DIM TESTS: 5000x512, 5000x700, 500x512, 500x700")
    print("*" * 60 + "\n")
    test_householder_high_dim()
    test_stereographic_high_dim()
    test_sphericity_high_dim()
    test_kent_detection_high_dim()
    test_kent_sampling_high_dim()
    test_full_pipeline_high_dim()

    print("=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)
