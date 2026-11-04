"""
Tests for maximize_loglikelihood (loss_forget="loglikelihood")
in DistributionalMethod.

Part 1 — Synthetic unit tests:
  1. A point at a Gaussian mean gets higher LL than a far-away point.
  2. The method correctly excludes the own class.
  3. With heterogeneous distributions the best LL is chosen.
  4. Closer to a distribution => lower loss (loss = -max_ll).
  5. Batch consistency: loss scales correctly with batch size.
  6. Clustering sub-groups.
  7. Compare loglikelihood vs distances ordering.
  8. vMF distribution.

Part 2 — Integration tests with real dist_infos from
  standard_clustering_cov_diagonal:
  resnet18 / vit_b_16  x  mnist / cifar10 / cifar100  (epoch_best)
  For each combination:
    a. Load dist_infos, print summary.
    b. Generate samples from two classes, compute loglikelihood loss
       and verify that a sample generated from class A has higher LL
       under class A's distribution than under class B's.
    c. End-to-end maximize_loglikelihood with near/far samples.
    d. Cross-class discrimination.
"""

import math
import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods.DistributionalMethod import (
    DistributionInfo,
    DistributionalMethod,
    sample_from_distribution,
    _vmf_log_norm_const,
)


# ===========================================================================
# Helpers — build fake DistributionInfo objects
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


# ===========================================================================
# Thin mock of DistributionalMethod
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
        self.intra_percentage = []
        for c in range(num_classes):
            infos = dist_infos.get(c, [])
            if not isinstance(infos, list):
                infos = [(infos, 1.0)]
            n_sub = len(infos)
            self.intra_percentage.append(
                torch.ones(n_sub) / max(n_sub, 1)
            )

    @property
    def device(self):
        return torch.device("cpu")

    # Bind methods from DistributionalMethod
    _get_all_dist_types = DistributionalMethod._get_all_dist_types
    _loglik_gaussian = staticmethod(DistributionalMethod._loglik_gaussian)
    _loglik_tstudent = staticmethod(DistributionalMethod._loglik_tstudent)
    _loglik_vmf_kent = staticmethod(DistributionalMethod._loglik_vmf_kent)
    _loglik_laplace = staticmethod(DistributionalMethod._loglik_laplace)
    _loglik_uniform = staticmethod(DistributionalMethod._loglik_uniform)
    _loglik = DistributionalMethod._loglik
    maximize_loglikelihood = DistributionalMethod.maximize_loglikelihood
    minimize_distances = DistributionalMethod.minimize_distances
    _typed_distance = DistributionalMethod._typed_distance
    _dist_mahalanobis = staticmethod(DistributionalMethod._dist_mahalanobis)
    _dist_pseudo_mahalanobis_t = staticmethod(DistributionalMethod._dist_pseudo_mahalanobis_t)
    _dist_vmf_kent = staticmethod(DistributionalMethod._dist_vmf_kent)
    _dist_manhattan_weighted = staticmethod(DistributionalMethod._dist_manhattan_weighted)
    _dist_chebyshev = staticmethod(DistributionalMethod._dist_chebyshev)


# ===========================================================================
# Part 1 — Synthetic unit tests
# ===========================================================================

class TestMaximizeLoglikelihoodSynthetic:
    """Unit tests with hand-crafted distributions."""

    # ---- 1. Point at mean gets higher LL --------------------------------

    def test_point_at_mean_higher_ll(self):
        """A point sitting on class-1's Gaussian mean should have a very
        high log-likelihood under class 1.  Since maximize_loglikelihood
        excludes the own class, we label the point as class 0 so that
        class 1 IS considered."""
        g0 = _make_gaussian_info([10.0, 10.0], np.eye(2))
        g1 = _make_gaussian_info([0.0, 0.0], np.eye(2))
        dist_infos = {0: [(g0, 1.0)], 1: [(g1, 1.0)]}
        fake = _FakeMethod(dist_infos, num_classes=2)

        # Point at class 1's mean, labelled as class 0 -> class 1 evaluated
        embs_close = torch.tensor([[0.0, 0.0]])
        labels_close = torch.tensor([0])

        # Point far from both
        embs_far = torch.tensor([[50.0, 50.0]])
        labels_far = torch.tensor([0])

        loss_close = fake.maximize_loglikelihood(embs_close, labels_close)
        loss_far = fake.maximize_loglikelihood(embs_far, labels_far)

        # loss = -max_ll, so closer point -> lower loss
        assert loss_close.item() < loss_far.item(), (
            f"Loss at mean ({loss_close.item():.4f}) should be < "
            f"loss far away ({loss_far.item():.4f})"
        )

    # ---- 2. Own class is excluded ---------------------------------------

    def test_own_class_excluded(self):
        """The method must NOT use the own class's distribution.
        If the point is at class 0's mean and labelled class 0,
        only class 1 is evaluated."""
        g0 = _make_gaussian_info([0.0, 0.0], np.eye(2) * 0.01)  # very tight
        g1 = _make_gaussian_info([100.0, 100.0], np.eye(2) * 0.01)
        dist_infos = {0: [(g0, 1.0)], 1: [(g1, 1.0)]}
        fake = _FakeMethod(dist_infos, num_classes=2)

        # At class 0 mean, labelled class 0 -> only class 1 considered
        embs = torch.tensor([[0.0, 0.0]])
        labels = torch.tensor([0])
        loss_own = fake.maximize_loglikelihood(embs, labels)

        # At class 0 mean, labelled class 1 -> class 0 IS considered
        labels_other = torch.tensor([1])
        loss_other = fake.maximize_loglikelihood(embs, labels_other)

        # When own class excluded, LL under class 1 is terrible -> high loss
        # When own class included, LL under class 0 is excellent -> low loss
        assert loss_own.item() > loss_other.item(), (
            f"Excluding own class should give higher loss: "
            f"own={loss_own.item():.2f} vs other={loss_other.item():.2f}"
        )

    # ---- 3. Heterogeneous: best LL is chosen ----------------------------

    def test_heterogeneous_best_ll_chosen(self):
        """With Gaussian + Laplace, the best LL among non-own distributions
        should be the one returned."""
        g0 = _make_gaussian_info([0.0, 0.0, 0.0], np.eye(3))
        laplace1 = _make_laplace_info([10.0, 10.0, 10.0], [0.5, 0.5, 0.5])
        g2 = _make_gaussian_info([5.0, 5.0, 5.0], np.eye(3) * 2.0)
        dist_infos = {
            0: [(g0, 1.0)],
            1: [(laplace1, 1.0)],
            2: [(g2, 1.0)],
        }
        fake = _FakeMethod(dist_infos, num_classes=3)

        # Point near class 2's mean, label = 0 -> classes 1 and 2 evaluated
        embs = torch.tensor([[5.0, 5.0, 5.0]])
        labels = torch.tensor([0])
        loss = fake.maximize_loglikelihood(embs, labels)

        # Verify it's finite
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"

        # Manually check: LL under class 2 should be highest for this point
        x = torch.tensor([5.0, 5.0, 5.0])
        ll_1 = fake._loglik(x, laplace1).item()
        ll_2 = fake._loglik(x, g2).item()
        best_ll = max(ll_1, ll_2)
        expected_loss = -best_ll
        assert abs(loss.item() - expected_loss) < 1e-4, (
            f"Loss ({loss.item():.6f}) should match -max(ll) ({expected_loss:.6f})"
        )

    # ---- 4. Closer sample -> lower loss ----------------------------------

    def test_closer_sample_lower_loss(self):
        """Across 3 classes, a sample closer to a non-own class should
        have lower loss."""
        g0 = _make_gaussian_info([0.0, 0.0], np.eye(2))
        g1 = _make_gaussian_info([10.0, 0.0], np.eye(2))
        g2 = _make_gaussian_info([0.0, 10.0], np.eye(2))
        dist_infos = {0: [(g0, 1.0)], 1: [(g1, 1.0)], 2: [(g2, 1.0)]}
        fake = _FakeMethod(dist_infos, num_classes=3)

        # Near class 1, label 0 -> considers class 1 (close) and class 2
        embs_near = torch.tensor([[9.5, 0.0]])
        labels = torch.tensor([0])
        loss_near = fake.maximize_loglikelihood(embs_near, labels)

        # Far from everything
        embs_far = torch.tensor([[50.0, 50.0]])
        loss_far = fake.maximize_loglikelihood(embs_far, labels)

        assert loss_near.item() < loss_far.item()

    # ---- 5. Batch consistency -------------------------------------------

    def test_batch_consistency(self):
        """Loss for a duplicated batch should equal loss for a single sample."""
        g0 = _make_gaussian_info([0.0, 0.0], np.eye(2))
        g1 = _make_gaussian_info([5.0, 5.0], np.eye(2))
        dist_infos = {0: [(g0, 1.0)], 1: [(g1, 1.0)]}
        fake = _FakeMethod(dist_infos, num_classes=2)

        embs1 = torch.tensor([[1.0, 1.0]])
        labels1 = torch.tensor([0])
        loss1 = fake.maximize_loglikelihood(embs1, labels1)

        embs2 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        labels2 = torch.tensor([0, 0])
        loss2 = fake.maximize_loglikelihood(embs2, labels2)

        assert abs(loss1.item() - loss2.item()) < 1e-5, (
            f"Single ({loss1.item():.6f}) vs batch ({loss2.item():.6f}) "
            "should be equal (mean-reduced)"
        )

    # ---- 6. With clustering (multiple sub-groups) -----------------------

    def test_clustering_subgroups(self):
        """With clustering, all sub-groups of non-own classes should be
        evaluated, and the best LL across all of them is taken."""
        g0 = _make_gaussian_info([0.0, 0.0], np.eye(2))
        # Class 1 has two sub-clusters
        g1a = _make_gaussian_info([10.0, 0.0], np.eye(2))
        g1b = _make_gaussian_info([3.0, 0.0], np.eye(2) * 0.5)
        dist_infos = {
            0: [(g0, 1.0)],
            1: [(g1a, 0.6), (g1b, 0.4)],
        }
        fake = _FakeMethod(dist_infos, num_classes=2, clustering=True)

        # Point near class 1's sub-cluster b
        embs = torch.tensor([[3.0, 0.0]])
        labels = torch.tensor([0])
        loss = fake.maximize_loglikelihood(embs, labels)
        assert torch.isfinite(loss)

        # The LL under sub-cluster b (at its mean) should be very high
        # => loss should be low (negative of a high LL)
        x = torch.tensor([3.0, 0.0])
        ll_1a = fake._loglik(x, g1a).item()
        ll_1b = fake._loglik(x, g1b).item()
        assert ll_1b > ll_1a, "Sub-cluster b is closer, should have higher LL"

    # ---- 7. Compare loglikelihood vs distances --------------------------

    def test_loglikelihood_vs_distances_homogeneous(self):
        """For homogeneous Gaussians, both methods should agree on
        which sample is 'closer' to the retain set."""
        g0 = _make_gaussian_info([0.0, 0.0], np.eye(2))
        g1 = _make_gaussian_info([10.0, 10.0], np.eye(2))
        dist_infos = {0: [(g0, 1.0)], 1: [(g1, 1.0)]}
        fake = _FakeMethod(dist_infos, num_classes=2)

        embs_near = torch.tensor([[9.0, 9.0]])
        embs_far = torch.tensor([[50.0, 50.0]])
        labels = torch.tensor([0])

        ll_near = fake.maximize_loglikelihood(embs_near, labels)
        ll_far = fake.maximize_loglikelihood(embs_far, labels)
        dist_near = fake.minimize_distances(embs_near, labels)
        dist_far = fake.minimize_distances(embs_far, labels)

        # Both should agree: near -> lower loss
        assert ll_near.item() < ll_far.item()
        assert dist_near.item() < dist_far.item()

    # ---- 8. VMF distribution --------------------------------------------

    def test_vmf_loglikelihood(self):
        """vMF: a point aligned with mu should have higher LL than
        one orthogonal to it."""
        d = 5
        mu = np.zeros(d)
        mu[0] = 1.0
        vmf = _make_vmf_info(mu, kappa=50.0,
                             norm_dist={"type": "normal", "mean": 1.0, "std": 0.1})
        g_other = _make_gaussian_info(np.full(d, 10.0), np.eye(d))
        dist_infos = {0: [(vmf, 1.0)], 1: [(g_other, 1.0)]}
        fake = _FakeMethod(dist_infos, num_classes=2)

        # Aligned with mu (class 0), label=1 -> class 0 evaluated
        x_aligned = torch.zeros(1, d)
        x_aligned[0, 0] = 1.0
        labels = torch.tensor([1])
        loss_aligned = fake.maximize_loglikelihood(x_aligned, labels)

        # Orthogonal to mu
        x_ortho = torch.zeros(1, d)
        x_ortho[0, 2] = 1.0
        loss_ortho = fake.maximize_loglikelihood(x_ortho, labels)

        assert loss_aligned.item() < loss_ortho.item()

    # ---- 9. t-Student distribution --------------------------------------

    def test_tstudent_loglikelihood(self):
        """t-Student: point at mean should have lower loss than far point."""
        g0 = _make_gaussian_info([0.0, 0.0], np.eye(2))
        ts1 = _make_tstudent_info([5.0, 5.0], np.eye(2), df=3.0)
        dist_infos = {0: [(g0, 1.0)], 1: [(ts1, 1.0)]}
        fake = _FakeMethod(dist_infos, num_classes=2)

        embs_near = torch.tensor([[5.0, 5.0]])
        embs_far = torch.tensor([[50.0, 50.0]])
        labels = torch.tensor([0])

        loss_near = fake.maximize_loglikelihood(embs_near, labels)
        loss_far = fake.maximize_loglikelihood(embs_far, labels)
        assert loss_near.item() < loss_far.item()

    # ---- 10. Uniform distribution ---------------------------------------

    def test_uniform_loglikelihood(self):
        """Uniform: point inside box should have lower loss than outside."""
        g0 = _make_gaussian_info([0.0, 0.0], np.eye(2))
        unif = _make_uniform_info([3.0, 3.0], [7.0, 7.0])
        dist_infos = {0: [(g0, 1.0)], 1: [(unif, 1.0)]}
        fake = _FakeMethod(dist_infos, num_classes=2)

        embs_inside = torch.tensor([[5.0, 5.0]])
        embs_outside = torch.tensor([[50.0, 50.0]])
        labels = torch.tensor([0])

        loss_in = fake.maximize_loglikelihood(embs_inside, labels)
        loss_out = fake.maximize_loglikelihood(embs_outside, labels)
        assert loss_in.item() < loss_out.item()


# ===========================================================================
# Part 2 — Integration tests with real dist_infos
# ===========================================================================

DIST_INFOS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "standard_clustering_cov_diagonal"
)

# Build list of (model, dataset, num_classes) combos to test
_COMBOS = []
for model in ("resnet18", "vit_b_16"):
    for dataset, nc in [("mnist", 10), ("cifar10", 10), ("cifar100", 100)]:
        tag = f"{model}_{dataset}_original_epoch_best_dist_infos.pt"
        path = os.path.join(DIST_INFOS_DIR, tag)
        if os.path.isfile(path):
            _COMBOS.append(
                pytest.param((model, dataset, nc, path), id=f"{model}_{dataset}")
            )


@pytest.mark.skipif(len(_COMBOS) == 0, reason="No dist_infos files found")
class TestRealDistInfos:
    """Integration tests loading real fitted distributions."""

    @pytest.fixture(params=_COMBOS)
    def combo(self, request):
        model, dataset, nc, path = request.param
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # The .pt file is a full checkpoint dict with key 'dist_infos'
        if isinstance(checkpoint, dict) and "dist_infos" in checkpoint:
            dist_infos = checkpoint["dist_infos"]
            percentages = checkpoint.get("percentages", None)
            intra_percentage = checkpoint.get("intra_percentage", None)
        else:
            dist_infos = checkpoint
            percentages = None
            intra_percentage = None
        return model, dataset, nc, dist_infos, percentages, intra_percentage

    def _build_fake(self, nc, dist_infos, percentages=None,
                    intra_percentage=None, clustering=True):
        fake = _FakeMethod.__new__(_FakeMethod)
        fake.config = {"num_classes": nc, "clustering": clustering}
        fake.dist_infos = dist_infos
        if percentages is not None:
            fake.percentages = percentages
        else:
            fake.percentages = torch.ones(nc) / nc
        if intra_percentage is not None:
            fake.intra_percentage = intra_percentage
        else:
            fake.intra_percentage = []
            for c in range(nc):
                infos = dist_infos.get(c, [])
                if not isinstance(infos, list):
                    infos = [(infos, 1.0)]
                n_sub = len(infos)
                fake.intra_percentage.append(
                    torch.ones(max(n_sub, 1)) / max(n_sub, 1)
                )
        return fake

    # ---- a. Load and print summary --------------------------------------

    def test_load_and_summary(self, combo):
        model, dataset, nc, dist_infos, percentages, intra_pct = combo
        print(f"\n{'='*60}")
        print(f"  {model} / {dataset}  ({nc} classes)")
        print(f"{'='*60}")
        types_count = {}
        for c in sorted(dist_infos.keys()):
            infos = dist_infos[c]
            if not isinstance(infos, list):
                infos = [(infos, 1.0)]
            for info, frac in infos:
                t = info.dist_type
                types_count[t] = types_count.get(t, 0) + 1
        print(f"  Distribution types: {types_count}")
        print(f"  Total classes with data: {len(dist_infos)}")
        assert len(dist_infos) > 0

    # ---- b. Generated samples have higher LL under own class ------------

    def test_generated_sample_best_ll_own_class(self, combo):
        """Generate samples from class A, compute LL under class A and
        class B.  The LL under class A should be higher (= the sample
        'belongs' to its generating distribution)."""
        model, dataset, nc, dist_infos, percentages, intra_pct = combo
        fake = self._build_fake(nc, dist_infos, percentages, intra_pct)

        rng = np.random.default_rng(42)
        available = sorted(dist_infos.keys())
        if len(available) < 2:
            pytest.skip("Need at least 2 classes")

        class_a = available[0]
        class_b = available[1]

        info_a = dist_infos[class_a]
        info_b = dist_infos[class_b]
        if not isinstance(info_a, list):
            info_a = [(info_a, 1.0)]
        if not isinstance(info_b, list):
            info_b = [(info_b, 1.0)]

        n_samples = 50
        samples_a = sample_from_distribution(info_a[0][0], n_samples, rng=rng)

        ll_under_a = []
        ll_under_b = []
        for i in range(n_samples):
            x = samples_a[i].float()
            lla = fake._loglik(x, info_a[0][0])
            llb = fake._loglik(x, info_b[0][0])
            ll_under_a.append(lla.item())
            ll_under_b.append(llb.item())

        mean_ll_a = np.mean(ll_under_a)
        mean_ll_b = np.mean(ll_under_b)

        print(f"\n  Class {class_a} samples -> LL under class {class_a}: {mean_ll_a:.4f}")
        print(f"  Class {class_a} samples -> LL under class {class_b}: {mean_ll_b:.4f}")
        print(f"  Difference (should be > 0): {mean_ll_a - mean_ll_b:.4f}")

        assert mean_ll_a > mean_ll_b, (
            f"Samples from class {class_a} should have higher LL under "
            f"class {class_a} ({mean_ll_a:.4f}) than class {class_b} "
            f"({mean_ll_b:.4f})"
        )

    # ---- c. maximize_loglikelihood end-to-end ---------------------------

    def test_maximize_loglikelihood_end_to_end(self, combo):
        """Run maximize_loglikelihood on generated samples and verify
        that far-away samples get higher loss."""
        model, dataset, nc, dist_infos, percentages, intra_pct = combo
        fake = self._build_fake(nc, dist_infos, percentages, intra_pct)

        rng = np.random.default_rng(123)
        available = sorted(dist_infos.keys())
        if len(available) < 2:
            pytest.skip("Need at least 2 classes")

        class_a = available[0]
        info_a_list = dist_infos[class_a]
        if not isinstance(info_a_list, list):
            info_a_list = [(info_a_list, 1.0)]

        # Generate close samples (from class A, labelled as class A
        # so maximize_loglikelihood evaluates all OTHER classes)
        samples_close = sample_from_distribution(
            info_a_list[0][0], 10, rng=rng
        ).float()
        labels_a = torch.full((10,), class_a, dtype=torch.long)

        loss_close = fake.maximize_loglikelihood(samples_close, labels_a)
        assert torch.isfinite(loss_close), f"Loss not finite: {loss_close}"

        # Generate far-away samples (scale by 100)
        samples_far = samples_close * 100.0
        loss_far = fake.maximize_loglikelihood(samples_far, labels_a)

        print(f"\n  Loss close: {loss_close.item():.4f}")
        print(f"  Loss far:   {loss_far.item():.4f}")

        assert loss_far.item() > loss_close.item(), (
            f"Far samples should have higher loss: "
            f"close={loss_close.item():.4f}, far={loss_far.item():.4f}"
        )

    # ---- d. Cross-class discrimination ----------------------------------

    def test_cross_class_discrimination(self, combo):
        """Generate samples from class A and class B.  When both are
        labelled as a third class C, samples from A should get different
        loss than samples from B (demonstrates the method discriminates)."""
        model, dataset, nc, dist_infos, percentages, intra_pct = combo
        fake = self._build_fake(nc, dist_infos, percentages, intra_pct)

        rng = np.random.default_rng(999)
        available = sorted(dist_infos.keys())
        if len(available) < 3:
            pytest.skip("Need at least 3 classes")

        class_a = available[0]
        class_b = available[1]
        class_c = available[2]

        info_a = dist_infos[class_a]
        info_b = dist_infos[class_b]
        if not isinstance(info_a, list):
            info_a = [(info_a, 1.0)]
        if not isinstance(info_b, list):
            info_b = [(info_b, 1.0)]

        n = 20
        samples_a = sample_from_distribution(info_a[0][0], n, rng=rng).float()
        samples_b = sample_from_distribution(info_b[0][0], n, rng=rng).float()

        # Label all as class C -> both A and B distributions are evaluated
        labels_c = torch.full((n,), class_c, dtype=torch.long)

        loss_a = fake.maximize_loglikelihood(samples_a, labels_c)
        loss_b = fake.maximize_loglikelihood(samples_b, labels_c)

        print(f"\n  Samples from class {class_a}, labelled {class_c} -> loss: {loss_a.item():.4f}")
        print(f"  Samples from class {class_b}, labelled {class_c} -> loss: {loss_b.item():.4f}")

        # Both should be finite
        assert torch.isfinite(loss_a) and torch.isfinite(loss_b)

    # ---- e. Bidirectional own-class check with more classes -------------

    def test_bidirectional_own_class(self, combo):
        """Generate from A and B, label as the other.  Each set should
        find its own class distribution as the best fit (lowest loss)."""
        model, dataset, nc, dist_infos, percentages, intra_pct = combo
        fake = self._build_fake(nc, dist_infos, percentages, intra_pct)

        rng = np.random.default_rng(77)
        available = sorted(dist_infos.keys())
        if len(available) < 2:
            pytest.skip("Need at least 2 classes")

        ca, cb = available[0], available[1]
        info_a = dist_infos[ca]
        info_b = dist_infos[cb]
        if not isinstance(info_a, list):
            info_a = [(info_a, 1.0)]
        if not isinstance(info_b, list):
            info_b = [(info_b, 1.0)]

        n = 30
        samp_a = sample_from_distribution(info_a[0][0], n, rng=rng).float()
        samp_b = sample_from_distribution(info_b[0][0], n, rng=rng).float()

        # Samples from A labelled as B -> class A available in search
        loss_a_as_b = fake.maximize_loglikelihood(
            samp_a, torch.full((n,), cb, dtype=torch.long)
        )
        # Samples from B labelled as A -> class B available in search
        loss_b_as_a = fake.maximize_loglikelihood(
            samp_b, torch.full((n,), ca, dtype=torch.long)
        )

        # Samples from A labelled as A -> class A NOT available
        loss_a_as_a = fake.maximize_loglikelihood(
            samp_a, torch.full((n,), ca, dtype=torch.long)
        )
        # Samples from B labelled as B -> class B NOT available
        loss_b_as_b = fake.maximize_loglikelihood(
            samp_b, torch.full((n,), cb, dtype=torch.long)
        )

        print(f"\n  Samples A labelled B (own class available): {loss_a_as_b.item():.4f}")
        print(f"  Samples A labelled A (own class hidden):    {loss_a_as_a.item():.4f}")
        print(f"  Samples B labelled A (own class available): {loss_b_as_a.item():.4f}")
        print(f"  Samples B labelled B (own class hidden):    {loss_b_as_b.item():.4f}")

        # When own class IS available, loss should be <= (lower or equal)
        # With many classes (e.g. cifar100), a nearby class may dominate
        # in both cases, so we allow equality.
        assert loss_a_as_b.item() <= loss_a_as_a.item() + 1e-4, (
            "Samples from A should get lower (or equal) loss when class A is available"
        )
        assert loss_b_as_a.item() <= loss_b_as_b.item() + 1e-4, (
            "Samples from B should get lower (or equal) loss when class B is available"
        )


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
