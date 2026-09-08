"""Variance inflation x_i <- mean + lam(x_i - mean): the properties it must have.

Tested because the whole point of trying lambda BEFORE a learned head is that
it is interpretable -- and that rests on two invariants which are easy to break
in an implementation:
  1. the ensemble MEAN is exactly unchanged, so ens-mean L2 cannot move and any
     CRPS change is pure calibration rather than accuracy;
  2. spread scales exactly linearly in lambda.
"""
import unittest
import numpy as np
import torch

from eval.metrics import crps_ensemble


def inflate(members, lam):
    m = torch.stack(members).mean(0)
    return [m + lam * (x - m) for x in members]


class InflationTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        truth = torch.randn(4, 1, 16, 16, generator=g)
        self.truth = truth
        # UNDERdispersed, matching the measured ensembles (spread/err 0.62-0.65).
        # The structure matters: members must share a COMMON systematic error
        # that the spread does not see, and differ only by sampling noise. A
        # first attempt used truth + 0.3*noise_i, whose mean error is
        # 0.3/sqrt(8) = 0.106 against a spread of 0.3 -- spread/err 2.8, i.e.
        # heavily OVERdispersed, so inflation correctly made it worse.
        common = 1.0 * torch.randn(4, 1, 16, 16, generator=g)
        self.members = [truth + common + 0.25 * torch.randn(4, 1, 16, 16, generator=g)
                        for _ in range(8)]

    def test_mean_is_exactly_preserved(self):
        base = torch.stack(self.members).mean(0)
        for lam in (0.5, 1.0, 1.7, 3.0):
            got = torch.stack(inflate(self.members, lam)).mean(0)
            self.assertLess((got - base).abs().max().item(), 1e-5,
                            f"lambda={lam} moved the ensemble mean")

    def test_spread_scales_linearly(self):
        s1 = torch.stack(self.members).std(0).mean().item()
        for lam in (0.5, 1.5, 2.0):
            s = torch.stack(inflate(self.members, lam)).std(0).mean().item()
            self.assertAlmostEqual(s, lam * s1, places=4)

    def test_lambda_one_is_the_identity(self):
        for a, b in zip(inflate(self.members, 1.0), self.members):
            self.assertLess((a - b).abs().max().item(), 1e-6)

    def test_fixture_really_is_underdispersed(self):
        """Guard the guard: if the fixture is not underdispersed, the next test
        proves nothing."""
        st = torch.stack(self.members)
        spread = st.std(0).mean().item()
        err = (st.mean(0) - self.truth).pow(2).mean().sqrt().item()
        self.assertLess(spread / err, 0.7, "fixture is not underdispersed")

    def test_inflation_helps_an_underdispersed_ensemble(self):
        """The claim being tested on real data: CRPS should improve for some
        lambda > 1 when the ensemble is too narrow."""
        base = crps_ensemble(self.members, self.truth)
        best = min(crps_ensemble(inflate(self.members, l), self.truth)
                   for l in (1.2, 1.4, 1.6, 1.8, 2.0))
        self.assertLess(best, base)

    def test_inflation_hurts_an_already_wide_ensemble(self):
        """Guards against 'bigger lambda is always better' — it must have an
        optimum, or the sweep is meaningless."""
        wide = [self.truth + 3.0 * torch.randn_like(self.truth) for _ in range(8)]
        base = crps_ensemble(wide, self.truth)
        self.assertGreater(crps_ensemble(inflate(wide, 2.0), self.truth), base)


if __name__ == "__main__":
    unittest.main()
