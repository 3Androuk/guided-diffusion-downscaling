"""Multi-channel metric behaviour, with the real 20-var scale spread.

The 20-variable config mixes channels whose standard deviations span ~10^6
(z500 1453.7 vs q500 0.001). Averaging PHYSICAL per-channel RMSE across them
lets the largest-magnitude variable define the score: measured on the real
patches, mean-sea-level pressure plus the three geopotentials supply ~90% of
the pooled number while 2m temperature contributes under 1%.
"""

import unittest

import numpy as np
import torch

from eval.metrics import (l2_norm, l2_per_channel, spectrum_log_l1,
                          spectrum_log_l1_per_channel)

# Rough per-channel stds from the wb220 training patches.
STDS = np.array([8.47, 5.12, 4.26, 723.21, 15.97, 1453.73, 874.40, 613.93,
                 6.98, 6.84, 7.61, 11.65, 8.77, 7.68, 7.35, 5.63, 5.24,
                 0.001, 0.003, 0.004])


class PerChannelL2Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n, c, s = 8, len(STDS), 32
        base = rng.standard_normal((n, c, s, s))
        self.truth = torch.tensor(base * STDS[None, :, None, None])
        # every channel wrong by the SAME relative amount (5% of its own std)
        err = rng.standard_normal((n, c, s, s)) * (0.05 * STDS)[None, :, None, None]
        self.pred = torch.tensor(base * STDS[None, :, None, None] + err)

    def test_per_channel_returns_one_number_per_channel(self):
        v = l2_per_channel(self.pred, self.truth)
        self.assertEqual(len(v), len(STDS))

    def test_equal_relative_error_looks_wildly_unequal_in_physical_units(self):
        """Every channel is 5% of its std wrong, yet physical RMSE spans ~10^6."""
        v = np.array(l2_per_channel(self.pred, self.truth))
        rel = v / STDS
        np.testing.assert_allclose(rel, rel[0], rtol=0.15)   # equally wrong
        self.assertGreater(v.max() / v.min(), 1e5)           # but not in physical units

    def test_pooled_physical_mean_is_dominated_by_a_few_channels(self):
        """The failure mode, measured as SHARE of the pooled sum.

        On the real patches: msl 42%, the three geopotentials 48% between them,
        t2m 0.9%. Here, with every channel equally wrong in relative terms, the
        big-magnitude channels still take almost the whole score.
        """
        v = np.array(l2_per_channel(self.pred, self.truth))
        share = v / v.sum()
        self.assertGreater(share[5], 0.25, "z500 alone should take a large share")
        self.assertGreater(share[[3, 5, 6, 7]].sum(), 0.85,
                           "msl + geopotentials should dominate")
        self.assertLess(share[0], 0.01, "t2m should be negligible")
        self.assertLess(share[17:].sum(), 1e-5, "humidity should vanish entirely")

    def test_normalized_pooling_is_fair(self):
        """In normalized units every channel contributes comparably."""
        t = self.truth / torch.tensor(STDS)[None, :, None, None]
        p = self.pred / torch.tensor(STDS)[None, :, None, None]
        per_ch = np.array(l2_per_channel(p, t))
        self.assertLess(per_ch.max() / per_ch.min(), 1.5)
        self.assertAlmostEqual(l2_norm(p, t), per_ch.mean(), places=6)

    def test_spectrum_metric_is_scale_invariant(self):
        """Why pooling channels IS legitimate for the log-spectrum error."""
        a = self.truth[:, :1]
        b = self.pred[:, :1]
        base = spectrum_log_l1(b, a)
        for k in (1e-3, 1e3):
            self.assertAlmostEqual(spectrum_log_l1(b * k, a * k), base, places=6)

    def test_per_channel_spectrum_returns_one_number_per_channel(self):
        v = spectrum_log_l1_per_channel(self.pred, self.truth)
        self.assertEqual(len(v), len(STDS))

    def test_per_channel_spectrum_is_scale_invariant_channel_by_channel(self):
        """The property that makes the POOLED spectrum fair must hold per channel.

        Rescaling one channel by 10^3 must not move its own score or any other's
        — otherwise the per-channel split would inherit the magnitude bias that
        test_pooled_physical_mean_is_dominated_by_a_few_channels documents for L2.
        """
        base = np.array(spectrum_log_l1_per_channel(self.pred, self.truth))
        p, t = self.pred.clone(), self.truth.clone()
        p[:, 0] *= 1e3
        t[:, 0] *= 1e3
        np.testing.assert_allclose(
            np.array(spectrum_log_l1_per_channel(p, t)), base, rtol=1e-6)

    def test_per_channel_spectrum_agrees_with_the_scalar_on_one_channel(self):
        one = self.pred[:, 3:4], self.truth[:, 3:4]
        self.assertAlmostEqual(spectrum_log_l1_per_channel(*one)[0],
                               spectrum_log_l1(*one), places=9)

    def test_per_channel_spectrum_separates_a_single_bad_channel(self):
        """A channel blurred to a different spectrum must stand out alone."""
        p = self.pred.clone()
        p[:, 7] = self.truth[:, 7].mean(dim=(-2, -1), keepdim=True).expand_as(p[:, 7])
        v = np.array(spectrum_log_l1_per_channel(p, self.truth))
        others = np.delete(v, 7)
        self.assertGreater(v[7], 10 * others.max(),
                           "the flattened channel should dominate its own score")

    def test_single_channel_behaviour_unchanged(self):
        """t2m/z500 configs are 1-channel; they must be unaffected."""
        a = self.truth[:, :1]
        b = self.pred[:, :1]
        self.assertAlmostEqual(l2_per_channel(b, a)[0], l2_norm(b, a), places=6)


if __name__ == "__main__":
    unittest.main()
