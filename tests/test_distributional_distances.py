"""
Tests for the distribution-aware distance / log-likelihood functions
in DistributionalMethod.minimize_mahalanobis.

Covers:
  1. Gaussian  -> Mahalanobis distance  &  log-likelihood
  2. t-Student -> pseudo-Mahalanobis-t   &  log-likelihood
  3. vMF       -> rotation + stereo + Mahalanobis (isotropic tangent) + norm
  4. Kent      -> rotation + stereo + Mahalanobis (anisotropic tangent) + norm
  5. Laplace   -> weighted Manhattan     &  log-likelihood
  6. Uniform   -> Chebyshev distance     &  log-likelihood
  7. Heterogeneous (mixed types) -> falls back to log-likelihood
  8. End-to-end minimize_mahalanobis with homogeneous & heterogeneous setups
"""

import math
import sys
import os
import numpy as np
import torch
import pytest

# ---------------------------------------------------------------------------
# Make sure the project root is on the path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods.DistributionalMethod import (
    DistributionInfo,
    DistributionalMethod,
    _vmf_log_norm_const,
)


# ===========================================================================
# Helpers to build fake DistributionInfo objects
# ===========================================================================

def _make_gaussian_info(mean, cov):
    return DistributionInfo(
        "gaussian",
        {"mean": np.array(mean, dtype=np.float64),
         "cov": np.array(cov, dtype=np.float64)},
        ll=0.0, bic=0.0, decided_by="test",
    )


def _make_tstudent_info(mean, cov, df):
    return DistributionInfo(
        "tstudent",
        {"mean": np.array(mean, dtype=np.float64),
         "cov": np.array(cov, dtype=np.float64),
         "df": float(df)},
        ll=0.0, bic=0.0, decided_by="test",
    )


def _make_vmf_info(mu, kappa, norm_dist=None):
    params = {
        "mu": np.array(mu, dtype=np.float64),
        "kappa": float(kappa),
    }
    if norm_dist is not None:
        params["norm_dist"] = norm_dist
    else:
        params["norm_dist"] = {"type": "normal", "mean": 1.0, "std": 0.1}
    return DistributionInfo("vmf", params, ll=0.0, bic=0.0, decided_by="test")


def _make_kent_info(mu, kappa, tangent_cov, tangent_mean, norm_dist=None):
    params = {
        "mu": np.array(mu, dtype=np.float64),
        "kappa": float(kappa),
        "tangent_cov": np.array(tangent_cov, dtype=np.float64),
        "tangent_mean": np.array(tangent_mean, dtype=np.float64),
    }
    if norm_dist is not None:
        params["norm_dist"] = norm_dist
    else:
        params["norm_dist"] = {"type": "normal", "mean": 1.0, "std": 0.1}
    return DistributionInfo("kent", params, ll=0.0, bic=0.0, decided_by="test")


def _make_laplace_info(locs, scales):
    return DistributionInfo(
        "laplace",
        {"locs": np.array(locs, dtype=np.float64),
         "scales": np.array(scales, dtype=np.float64)},
        ll=0.0, bic=0.0, decided_by="test",
    )


def _make_uniform_info(mins, maxs):
    return DistributionInfo(
        "uniform",
        {"mins": np.array(mins, dtype=np.float64),
         "maxs": np.array(maxs, dtype=np.float64)},
        ll=0.0, bic=0.0, decided_by="test",
    )


# ===========================================================================
# A thin mock of DistributionalMethod (just the static/instance methods)
# ===========================================================================

class _FakeMethod:
    """Mimics the relevant parts of DistributionalMethod for unit testing."""

    def __init__(self, dist_infos, num_classes, clustering=False):
        self.config = {"num_classes": num_classes, "clustering": clustering}
        self.dist_infos = dist_infos
        n_per_class = 1.0 / num_classes
        self.percentages = torch.tensor(
            [n_per_class] * num_classes, dtype=torch.float32
        )
        # intra_percentage: list of tensors, one per class
        self.intra_percentage = []
        for c in range(num_classes):
            infos = dist_infos[c]
            if not isinstance(infos, list):
                infos = [(infos, 1.0)]
            n_sub = len(infos)
            self.intra_percentage.append(
                torch.tensor([1.0 / n_sub] * n_sub, dtype=torch.float32)
            )

    @property
    def device(self):
        return torch.device("cpu")

    # Bind the actual methods from DistributionalMethod
    _get_all_dist_types = DistributionalMethod._get_all_dist_types
    _loglik_gaussian = staticmethod(DistributionalMethod._loglik_gaussian)
    _loglik_tstudent = staticmethod(DistributionalMethod._loglik_tstudent)
    _loglik_vmf_kent = staticmethod(DistributionalMethod._loglik_vmf_kent)
    _loglik_laplace = staticmethod(DistributionalMethod._loglik_laplace)
    _loglik_uniform = staticmethod(DistributionalMethod._loglik_uniform)
    _loglik = DistributionalMethod._loglik
    _dist_mahalanobis = staticmethod(DistributionalMethod._dist_mahalanobis)
    _dist_pseudo_mahalanobis_t = staticmethod(DistributionalMethod._dist_pseudo_mahalanobis_t)
    _dist_vmf_kent = staticmethod(DistributionalMethod._dist_vmf_kent)
    _dist_manhattan_weighted = staticmethod(DistributionalMethod._dist_manhattan_weighted)
    _dist_chebyshev = staticmethod(DistributionalMethod._dist_chebyshev)
    _typed_distance = DistributionalMethod._typed_distance
    minimize_mahalanobis = DistributionalMethod.minimize_mahalanobis


# ===========================================================================
# 1. GAUSSIAN
# ===========================================================================

class TestGaussian:
    def test_mahalanobis_at_mean_is_zero(self):
        info = _make_gaussian_info([1.0, 2.0, 3.0], np.eye(3))
        x = torch.tensor([1.0, 2.0, 3.0])
        d = DistributionalMethod._dist_mahalanobis(x, info)
        assert d.item() < 1e-4

    def test_mahalanobis_positive(self):
        info = _make_gaussian_info([0.0, 0.0], np.eye(2))
        x = torch.tensor([3.0, 4.0])
        d = DistributionalMethod._dist_mahalanobis(x, info)
        assert abs(d.item() - 5.0) < 1e-4  # sqrt(9+16) = 5

    def test_mahalanobis_with_scaled_cov(self):
        # With cov = 4*I, Mahalanobis of (2,0) should be 1
        info = _make_gaussian_info([0.0, 0.0], 4.0 * np.eye(2))
        x = torch.tensor([2.0, 0.0])
        d = DistributionalMethod._dist_mahalanobis(x, info)
        assert abs(d.item() - 1.0) < 1e-4

    def test_loglik_peaks_at_mean(self):
        info = _make_gaussian_info([1.0, 2.0], np.eye(2))
        x_at_mean = torch.tensor([1.0, 2.0])
        x_away = torch.tensor([5.0, 5.0])
        ll_mean = DistributionalMethod._loglik_gaussian(x_at_mean, info)
        ll_away = DistributionalMethod._loglik_gaussian(x_away, info)
        assert ll_mean > ll_away

    def test_loglik_gradient_flows(self):
        info = _make_gaussian_info([0.0, 0.0], np.eye(2))
        x = torch.tensor([1.0, 1.0], requires_grad=True)
        ll = DistributionalMethod._loglik_gaussian(x, info)
        ll.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


# ===========================================================================
# 2. t-STUDENT
# ===========================================================================

class TestTStudent:
    def test_pseudo_mahalanobis_at_mean_is_zero(self):
        info = _make_tstudent_info([0.0, 0.0, 0.0], np.eye(3), df=5.0)
        x = torch.tensor([0.0, 0.0, 0.0])
        d = DistributionalMethod._dist_pseudo_mahalanobis_t(x, info)
        assert d.item() < 1e-4

    def test_pseudo_mahalanobis_scales_with_df(self):
        """Lower df -> heavier tails -> larger scaling factor -> larger distance."""
        info_low = _make_tstudent_info([0.0, 0.0], np.eye(2), df=2.0)
        info_high = _make_tstudent_info([0.0, 0.0], np.eye(2), df=100.0)
        x = torch.tensor([1.0, 1.0])
        d_low = DistributionalMethod._dist_pseudo_mahalanobis_t(x, info_low)
        d_high = DistributionalMethod._dist_pseudo_mahalanobis_t(x, info_high)
        assert d_low > d_high  # (df+d)/df larger for smaller df

    def test_loglik_peaks_at_mean(self):
        info = _make_tstudent_info([0.0, 0.0], np.eye(2), df=5.0)
        ll_mean = DistributionalMethod._loglik_tstudent(
            torch.tensor([0.0, 0.0]), info)
        ll_away = DistributionalMethod._loglik_tstudent(
            torch.tensor([3.0, 3.0]), info)
        assert ll_mean > ll_away

    def test_loglik_gradient_flows(self):
        info = _make_tstudent_info([0.0, 0.0], np.eye(2), df=5.0)
        x = torch.tensor([1.0, 1.0], requires_grad=True)
        ll = DistributionalMethod._loglik_tstudent(x, info)
        ll.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


# ===========================================================================
# 3. vMF
# ===========================================================================

class TestVMF:
    def test_dist_vmf_at_mu_is_small(self):
        """A point in the direction of mu with correct norm should have small distance."""
        mu = np.array([1.0, 0.0, 0.0])
        info = _make_vmf_info(mu, kappa=50.0,
                              norm_dist={"type": "normal", "mean": 1.0, "std": 0.1})
        x = torch.tensor([1.0, 0.0, 0.0])  # exactly mu * norm
        d = DistributionalMethod._dist_vmf_kent(x, info)
        assert d.item() < 1.0  # should be very close

    def test_dist_vmf_opposite_is_infinite(self):
        """A point at the antipode of mu should have effectively infinite distance
        (the stereographic projection of the south pole is infinity)."""
        mu = np.array([1.0, 0.0, 0.0])
        info = _make_vmf_info(mu, kappa=50.0,
                              norm_dist={"type": "normal", "mean": 1.0, "std": 0.1})
        x_far = torch.tensor([-1.0, 0.0, 0.0])
        d_far = DistributionalMethod._dist_vmf_kent(x_far, info)
        assert d_far.item() > 1e10  # effectively infinite

    def test_dist_vmf_norm_normal(self):
        """Norm contribution: normal distribution."""
        mu = np.array([1.0, 0.0, 0.0])
        info = _make_vmf_info(mu, kappa=50.0,
                              norm_dist={"type": "normal", "mean": 5.0, "std": 0.5})
        # Point in mu direction but wrong norm
        x = torch.tensor([1.0, 0.0, 0.0])  # norm=1 but expected 5
        d = DistributionalMethod._dist_vmf_kent(x, info)
        # Norm contribution should be significant: |1-5|/0.5 = 8
        assert d.item() > 5.0

    def test_dist_vmf_norm_lognormal(self):
        """Norm contribution: lognormal distribution."""
        mu = np.array([1.0, 0.0, 0.0])
        info = _make_vmf_info(mu, kappa=50.0,
                              norm_dist={"type": "lognormal", "mu": 0.0, "sigma": 0.5})
        x = torch.tensor([1.0, 0.0, 0.0])
        d = DistributionalMethod._dist_vmf_kent(x, info)
        # log(1) = 0 should be close to mu_ln=0
        assert torch.isfinite(d)

    def test_dist_vmf_norm_gamma(self):
        """Norm contribution: gamma distribution with Itakura-Saito."""
        mu = np.array([1.0, 0.0, 0.0])
        info = _make_vmf_info(mu, kappa=50.0,
                              norm_dist={"type": "gamma", "shape": 2.0,
                                         "scale": 1.0, "mean": 2.0})
        x = torch.tensor([2.0, 0.0, 0.0])  # norm=2, mean=2 -> IS=0
        d = DistributionalMethod._dist_vmf_kent(x, info)
        assert torch.isfinite(d)

    def test_vmf_loglik_gradient_flows(self):
        mu = np.array([1.0, 0.0, 0.0])
        info = _make_vmf_info(mu, kappa=10.0)
        x = torch.tensor([0.8, 0.5, 0.3], requires_grad=True)
        ll = DistributionalMethod._loglik_vmf_kent(x, info)
        ll.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_vmf_higher_kappa_sharper(self):
        """Higher kappa should give larger distance for off-axis points."""
        mu = np.array([1.0, 0.0, 0.0])
        x = torch.tensor([0.7, 0.7, 0.1])
        info_low_k = _make_vmf_info(mu, kappa=5.0)
        info_high_k = _make_vmf_info(mu, kappa=100.0)
        d_low = DistributionalMethod._dist_vmf_kent(x, info_low_k)
        d_high = DistributionalMethod._dist_vmf_kent(x, info_high_k)
        assert d_high > d_low


# ===========================================================================
# 4. KENT
# ===========================================================================

class TestKent:
    def test_dist_kent_at_mu_is_small(self):
        mu = np.array([1.0, 0.0, 0.0])
        tangent_cov = np.eye(2) * 0.01  # small variance in tangent plane
        tangent_mean = np.zeros(2)
        info = _make_kent_info(mu, 50.0, tangent_cov, tangent_mean)
        x = torch.tensor([1.0, 0.0, 0.0])
        d = DistributionalMethod._dist_vmf_kent(x, info)
        assert d.item() < 2.0

    def test_kent_uses_anisotropic_cov(self):
        """Kent with anisotropic tangent_cov should give different distances
        for perturbations in different tangent directions."""
        mu = np.array([1.0, 0.0, 0.0])
        # Very elongated covariance in tangent plane
        tangent_cov = np.diag([0.01, 1.0])
        tangent_mean = np.zeros(2)
        info = _make_kent_info(mu, 50.0, tangent_cov, tangent_mean)

        # Two points with same angular displacement but in different tangent directions
        x1 = torch.tensor([0.99, 0.1, 0.0])  # perturbation in 1st tangent dir
        x2 = torch.tensor([0.99, 0.0, 0.1])  # perturbation in 2nd tangent dir
        d1 = DistributionalMethod._dist_vmf_kent(x1, info)
        d2 = DistributionalMethod._dist_vmf_kent(x2, info)
        # d1 should be larger because cov is smaller in that direction (0.01)
        assert d1 > d2

    def test_kent_loglik_gradient_flows(self):
        mu = np.array([1.0, 0.0, 0.0])
        tangent_cov = np.eye(2) * 0.1
        tangent_mean = np.zeros(2)
        info = _make_kent_info(mu, 10.0, tangent_cov, tangent_mean)
        x = torch.tensor([0.8, 0.5, 0.3], requires_grad=True)
        ll = DistributionalMethod._loglik_vmf_kent(x, info)
        ll.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_kent_with_lognormal_norm(self):
        mu = np.array([1.0, 0.0, 0.0])
        tangent_cov = np.eye(2) * 0.1
        tangent_mean = np.zeros(2)
        norm_dist = {"type": "lognormal", "mu": 1.0, "sigma": 0.3}
        info = _make_kent_info(mu, 50.0, tangent_cov, tangent_mean, norm_dist)
        x = torch.tensor([2.5, 0.1, 0.0])
        d = DistributionalMethod._dist_vmf_kent(x, info)
        assert torch.isfinite(d)
        assert d.item() > 0


# ===========================================================================
# 5. LAPLACE
# ===========================================================================

class TestLaplace:
    def test_manhattan_at_loc_is_zero(self):
        info = _make_laplace_info([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
        x = torch.tensor([1.0, 2.0, 3.0])
        d = DistributionalMethod._dist_manhattan_weighted(x, info)
        assert d.item() < 1e-6

    def test_manhattan_positive(self):
        info = _make_laplace_info([0.0, 0.0], [1.0, 1.0])
        x = torch.tensor([3.0, 4.0])
        d = DistributionalMethod._dist_manhattan_weighted(x, info)
        assert abs(d.item() - 7.0) < 1e-4  # |3|/1 + |4|/1 = 7

    def test_manhattan_weighted_by_scale(self):
        info = _make_laplace_info([0.0, 0.0], [0.5, 2.0])
        x = torch.tensor([1.0, 1.0])
        d = DistributionalMethod._dist_manhattan_weighted(x, info)
        # 1.0/0.5 + 1.0/2.0 = 2.0 + 0.5 = 2.5
        assert abs(d.item() - 2.5) < 1e-4

    def test_loglik_peaks_at_loc(self):
        info = _make_laplace_info([0.0, 0.0], [1.0, 1.0])
        ll_loc = DistributionalMethod._loglik_laplace(
            torch.tensor([0.0, 0.0]), info)
        ll_away = DistributionalMethod._loglik_laplace(
            torch.tensor([5.0, 5.0]), info)
        assert ll_loc > ll_away

    def test_loglik_gradient_flows(self):
        info = _make_laplace_info([0.0, 0.0], [1.0, 1.0])
        x = torch.tensor([1.0, 1.0], requires_grad=True)
        ll = DistributionalMethod._loglik_laplace(x, info)
        ll.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


# ===========================================================================
# 6. UNIFORM
# ===========================================================================

class TestUniform:
    def test_chebyshev_at_center_is_zero(self):
        info = _make_uniform_info([0.0, 0.0], [2.0, 2.0])
        x = torch.tensor([1.0, 1.0])  # center
        d = DistributionalMethod._dist_chebyshev(x, info)
        assert d.item() < 1e-6

    def test_chebyshev_on_boundary_is_one(self):
        info = _make_uniform_info([0.0, 0.0], [2.0, 2.0])
        x = torch.tensor([2.0, 1.0])  # on boundary
        d = DistributionalMethod._dist_chebyshev(x, info)
        assert abs(d.item() - 1.0) < 1e-4

    def test_chebyshev_outside_greater_than_one(self):
        info = _make_uniform_info([0.0, 0.0], [2.0, 2.0])
        x = torch.tensor([3.0, 1.0])  # outside
        d = DistributionalMethod._dist_chebyshev(x, info)
        assert d.item() > 1.0

    def test_chebyshev_scales_with_width(self):
        info = _make_uniform_info([0.0, 0.0], [10.0, 2.0])
        x = torch.tensor([6.0, 1.5])  # center=(5,1), hw=(5,1)
        d = DistributionalMethod._dist_chebyshev(x, info)
        # max(|6-5|/5, |1.5-1|/1) = max(0.2, 0.5) = 0.5
        assert abs(d.item() - 0.5) < 1e-4

    def test_loglik_gradient_flows(self):
        info = _make_uniform_info([0.0, 0.0], [2.0, 2.0])
        x = torch.tensor([1.0, 1.0], requires_grad=True)
        ll = DistributionalMethod._loglik_uniform(x, info)
        ll.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


# ===========================================================================
# 7. HETEROGENEOUS (mixed types) -> log-likelihood fallback
# ===========================================================================

class TestHeterogeneous:
    def test_mixed_types_detected(self):
        dist_infos = {
            0: [(_make_gaussian_info([0, 0], np.eye(2)), 1.0)],
            1: [(_make_laplace_info([1, 1], [0.5, 0.5]), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2)
        types = fake._get_all_dist_types()
        assert len(types) == 2
        assert "gaussian" in types
        assert "laplace" in types

    def test_vmf_kent_grouped(self):
        mu = [1.0, 0.0, 0.0]
        dist_infos = {
            0: [(_make_vmf_info(mu, 10.0), 1.0)],
            1: [(_make_kent_info(mu, 10.0, np.eye(2)*0.1, np.zeros(2)), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2)
        types = fake._get_all_dist_types()
        assert len(types) == 1
        assert "vmf_kent" in types

    def test_heterogeneous_uses_loglik(self):
        """With mixed types, minimize_mahalanobis should still work
        (falls back to log-likelihood)."""
        dist_infos = {
            0: [(_make_gaussian_info([0, 0], np.eye(2)), 1.0)],
            1: [(_make_laplace_info([3, 3], [0.5, 0.5]), 1.0)],
            2: [(_make_uniform_info([-1, -1], [1, 1]), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=3)

        embs = torch.tensor([[0.0, 0.0], [3.0, 3.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        loss = fake.minimize_mahalanobis(embs, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert embs.grad is not None
        assert torch.all(torch.isfinite(embs.grad))


# ===========================================================================
# 8. END-TO-END minimize_mahalanobis
# ===========================================================================

class TestMinimizeMahalanobis:
    def test_homogeneous_gaussian(self):
        """All classes Gaussian -> uses Mahalanobis distance."""
        dist_infos = {
            0: [(_make_gaussian_info([0, 0], np.eye(2)), 1.0)],
            1: [(_make_gaussian_info([5, 5], np.eye(2)), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2)
        embs = torch.tensor([[0.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0])
        loss = fake.minimize_mahalanobis(embs, labels)
        # Should minimize distance to class 1 centroid at (5,5)
        assert torch.isfinite(loss)
        assert loss.item() > 0
        loss.backward()
        assert embs.grad is not None

    def test_homogeneous_tstudent(self):
        dist_infos = {
            0: [(_make_tstudent_info([0, 0], np.eye(2), 5.0), 1.0)],
            1: [(_make_tstudent_info([5, 5], np.eye(2), 5.0), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2)
        embs = torch.tensor([[0.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0])
        loss = fake.minimize_mahalanobis(embs, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert embs.grad is not None

    def test_homogeneous_vmf(self):
        mu1 = [1.0, 0.0, 0.0]
        mu2 = [0.0, 1.0, 0.0]
        dist_infos = {
            0: [(_make_vmf_info(mu1, 50.0), 1.0)],
            1: [(_make_vmf_info(mu2, 50.0), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2)
        embs = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0])
        loss = fake.minimize_mahalanobis(embs, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert embs.grad is not None

    def test_homogeneous_kent(self):
        mu1 = [1.0, 0.0, 0.0]
        mu2 = [0.0, 1.0, 0.0]
        dist_infos = {
            0: [(_make_kent_info(mu1, 50.0, np.eye(2)*0.01, np.zeros(2)), 1.0)],
            1: [(_make_kent_info(mu2, 50.0, np.eye(2)*0.01, np.zeros(2)), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2)
        embs = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0])
        loss = fake.minimize_mahalanobis(embs, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert embs.grad is not None

    def test_homogeneous_laplace(self):
        dist_infos = {
            0: [(_make_laplace_info([0, 0], [1, 1]), 1.0)],
            1: [(_make_laplace_info([5, 5], [1, 1]), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2)
        embs = torch.tensor([[0.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0])
        loss = fake.minimize_mahalanobis(embs, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert embs.grad is not None

    def test_homogeneous_uniform(self):
        dist_infos = {
            0: [(_make_uniform_info([0, 0], [2, 2]), 1.0)],
            1: [(_make_uniform_info([5, 5], [7, 7]), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2)
        embs = torch.tensor([[1.0, 1.0]], requires_grad=True)
        labels = torch.tensor([0])
        loss = fake.minimize_mahalanobis(embs, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert embs.grad is not None

    def test_mixed_vmf_kent_is_homogeneous(self):
        """vMF and Kent should be grouped together as homogeneous."""
        mu1 = [1.0, 0.0, 0.0]
        mu2 = [0.0, 1.0, 0.0]
        dist_infos = {
            0: [(_make_vmf_info(mu1, 50.0), 1.0)],
            1: [(_make_kent_info(mu2, 50.0, np.eye(2)*0.01, np.zeros(2)), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2)
        types = fake._get_all_dist_types()
        assert len(types) == 1
        assert "vmf_kent" in types

        embs = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0])
        loss = fake.minimize_mahalanobis(embs, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert embs.grad is not None

    def test_clustering_multiple_subgroups(self):
        """With clustering, multiple sub-groups per class."""
        dist_infos = {
            0: [
                (_make_gaussian_info([0, 0], np.eye(2)), 0.6),
                (_make_gaussian_info([2, 0], np.eye(2)), 0.4),
            ],
            1: [(_make_gaussian_info([5, 5], np.eye(2)), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2, clustering=True)
        embs = torch.tensor([[1.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0])
        loss = fake.minimize_mahalanobis(embs, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert embs.grad is not None

    def test_batch_processing(self):
        """Multiple embeddings in a batch."""
        dist_infos = {
            0: [(_make_gaussian_info([0, 0], np.eye(2)), 1.0)],
            1: [(_make_gaussian_info([5, 5], np.eye(2)), 1.0)],
            2: [(_make_gaussian_info([10, 0], np.eye(2)), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=3)
        embs = torch.tensor(
            [[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]], requires_grad=True
        )
        labels = torch.tensor([0, 1, 2])
        loss = fake.minimize_mahalanobis(embs, labels)
        assert torch.isfinite(loss)
        assert loss.item() > 0
        loss.backward()
        assert embs.grad is not None

    def test_loss_decreases_toward_target(self):
        """Loss should be smaller when embedding is closer to non-own class."""
        dist_infos = {
            0: [(_make_gaussian_info([0, 0], np.eye(2)), 1.0)],
            1: [(_make_gaussian_info([10, 0], np.eye(2)), 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2)

        # Point far from class 1
        embs_far = torch.tensor([[0.0, 0.0]])
        labels = torch.tensor([0])
        loss_far = fake.minimize_mahalanobis(embs_far, labels)

        # Point close to class 1
        embs_close = torch.tensor([[9.0, 0.0]])
        loss_close = fake.minimize_mahalanobis(embs_close, labels)

        assert loss_close < loss_far


# ===========================================================================
# 9. NORM DISTANCE EDGE CASES
# ===========================================================================

class TestNormDistanceEdgeCases:
    def test_gamma_itakura_saito_at_mean(self):
        """Itakura-Saito distance is 0 when x/mu = 1."""
        mu = np.array([1.0, 0.0, 0.0])
        norm_dist = {"type": "gamma", "shape": 2.0, "scale": 1.0, "mean": 2.0}
        info = _make_vmf_info(mu, 50.0, norm_dist=norm_dist)
        # x with norm=2 (matching gamma mean)
        x = torch.tensor([2.0, 0.0, 0.0])
        d = DistributionalMethod._dist_vmf_kent(x, info)
        # Directional part should be small (aligned with mu),
        # Itakura-Saito part should be ~0 since ratio=1 -> IS = 1-0-1 = 0
        assert torch.isfinite(d)

    def test_no_norm_dist(self):
        """vMF without norm_dist should still work."""
        mu = np.array([1.0, 0.0, 0.0])
        info = DistributionInfo(
            "vmf",
            {"mu": mu, "kappa": 10.0, "norm_dist": None},
            ll=0.0, bic=0.0,
        )
        x = torch.tensor([1.0, 0.0, 0.0])
        d = DistributionalMethod._dist_vmf_kent(x, info)
        assert torch.isfinite(d)


# ===========================================================================
# 10. DISPATCH TESTS
# ===========================================================================

class TestDispatch:
    def test_loglik_dispatch(self):
        """_loglik dispatches correctly for each type."""
        fake = _FakeMethod(
            {0: [(_make_gaussian_info([0, 0], np.eye(2)), 1.0)]},
            num_classes=1,
        )
        x = torch.tensor([0.0, 0.0])
        g = _make_gaussian_info([0, 0], np.eye(2))
        t = _make_tstudent_info([0, 0], np.eye(2), 5.0)
        l = _make_laplace_info([0, 0], [1, 1])
        u = _make_uniform_info([-1, -1], [1, 1])

        for info in [g, t, l, u]:
            ll = fake._loglik(x, info)
            assert torch.isfinite(ll)

    def test_typed_distance_dispatch(self):
        """_typed_distance dispatches correctly for each type."""
        fake = _FakeMethod(
            {0: [(_make_gaussian_info([0, 0], np.eye(2)), 1.0)]},
            num_classes=1,
        )
        x = torch.tensor([1.0, 1.0])
        g = _make_gaussian_info([0, 0], np.eye(2))
        t = _make_tstudent_info([0, 0], np.eye(2), 5.0)
        l = _make_laplace_info([0, 0], [1, 1])
        u = _make_uniform_info([-1, -1], [3, 3])

        for info in [g, t, l, u]:
            d = fake._typed_distance(x, info)
            assert torch.isfinite(d)
            assert d.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
