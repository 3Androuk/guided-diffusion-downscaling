"""SpikeGuard, tested against the gradient traces of the runs that motivated it.

The 20-var no-geo baseline collapsed twice from different RNG states — job
6169725 at epoch 92 and job 6189791 at epoch 82 — while both geo arms ran 200
epochs untouched. The numbers below are the observed 50-step mean gradient
norms from those four runs, not invented ones.

Two things must both hold or the guard is useless:
  - it fires on the no-geo excursions (otherwise it does not prevent the collapse)
  - it is silent on the geo arms (otherwise it perturbs runs that were fine, and
    the study loses its shared recipe)
"""

import unittest

from utils import SpikeGuard, build_spike_guard

# Observed 50-step mean grad norms. Healthy late-training steps sit at ~0.09-0.13
# on every arm; these are the extremes each run actually reached.
HEALTHY_LATE = [0.090, 0.084, 0.101, 0.106, 0.088, 0.110, 0.094, 0.104, 0.096,
                0.094, 0.108, 0.128, 0.126, 0.106]
HPX_MAX = 0.374        # job 6160937, max over all 200 epochs
STATIC_MAX = 1.578     # job 6169727, max over all 200 epochs (early)
NOGEO_P999 = 0.548     # job 6189791, p99.9 — routine, must NOT fire
# The collapse windows. These are 50-step MEANS, so the individual step that
# caused them was far larger: a mean of 3.167 over 50 steps whose others are
# ~0.1 implies one step near 150.
COLLAPSE_WINDOW_MEANS = [3.167, 4.716]


class SpikeGuardTests(unittest.TestCase):
    def _warm(self, guard, values=None, n=400):
        """Fill the history with healthy steps so the guard is armed."""
        vals = values or HEALTHY_LATE
        for i in range(n):
            guard.check(vals[i % len(vals)])
        return guard

    def test_silent_on_healthy_training(self):
        g = self._warm(SpikeGuard())
        self.assertEqual(g.skipped, 0)

    def test_does_not_fire_during_warmup(self):
        """Before enough history exists there is no median to compare against."""
        g = SpikeGuard(warmup=200)
        for _ in range(199):
            self.assertFalse(g.check(1e6))
        self.assertEqual(g.skipped, 0)

    def test_the_window_mean_alone_is_below_threshold(self):
        """Documents what the guard does NOT do, so the next test is not misread.

        3.167 is a mean over 50 steps and is itself only ~30x the median, below
        the 50x threshold. The guard never sees this number — it sees each step.
        If it had to fire on the smoothed value the threshold would have to drop
        to ~30x, which is inside the range a legitimate step can reach.
        """
        for m in COLLAPSE_WINDOW_MEANS:
            g = self._warm(SpikeGuard())
            self.assertFalse(g.check(m))

    def test_fires_on_the_collapse_replayed_step_by_step(self):
        """The real scenario: 49 healthy steps and one pathological one.

        A 50-step window averaging 3.167 whose other members are healthy implies
        a single step near 150: even assuming the other 49 all sat at the no-geo
        p99.9 of 0.548 (sum 26.9), the remaining step is >= 130. The same window
        showed the loss mean jump from ~0.014 to 0.408, i.e. one batch at ~20 —
        consistent with one bad batch rather than 50 mildly bad ones.

        130 is >1300x the running median, against a 50x threshold: the gap
        between the largest healthy gradient ever observed (1.578) and the
        smallest possible spike (130) spans ~two orders of magnitude, so the
        threshold sits in an empty region rather than on a contested boundary.
        """
        for mean in COLLAPSE_WINDOW_MEANS:
            g = self._warm(SpikeGuard())
            implied = 50 * mean - 49 * NOGEO_P999      # conservative lower bound
            self.assertGreater(implied, 100)
            for _ in range(49):
                self.assertFalse(g.check(0.548))
            self.assertTrue(g.check(implied),
                            f"missed the spike implied by window mean {mean}")
            self.assertEqual(g.skipped, 1, "exactly one step should be skipped")

    def test_silent_on_both_geo_arms_extremes(self):
        """The completed arms must be unaffected, or the recipe is not shared.

        hpx never exceeded 0.374 and static never exceeded 1.578 across 200
        epochs. Neither may trip the guard, otherwise enabling it retroactively
        would mean those runs were trained under a different recipe.
        """
        for v, arm in ((HPX_MAX, "hpx"), (STATIC_MAX, "static"),
                       (NOGEO_P999, "no-geo p99.9")):
            g = self._warm(SpikeGuard())
            self.assertFalse(g.check(v), f"false positive on {arm} ({v})")

    def test_static_early_max_has_real_margin(self):
        """1.578 is the closest any healthy run came — check it is not marginal.

        Early epochs run a higher gradient scale than late ones, so warm the
        history with early-epoch values rather than late ones.
        """
        g = self._warm(SpikeGuard(), values=[0.9, 1.1, 1.0, 1.2, 0.95])
        self.assertFalse(g.check(STATIC_MAX))
        med = 1.0
        self.assertGreater(50 * med / STATIC_MAX, 10,
                           "want an order of magnitude of headroom, not a hair")

    def test_non_finite_is_skipped_even_during_warmup(self):
        g = SpikeGuard(warmup=200)
        self.assertTrue(g.check(float("nan")))
        self.assertTrue(g.check(float("inf")))
        self.assertEqual(g.skipped, 2)

    def test_spikes_do_not_enter_the_history(self):
        """A diverging run must not raise its own threshold to accommodate itself."""
        g = self._warm(SpikeGuard())
        for _ in range(50):
            g.check(100.0)
        self.assertEqual(g.skipped, 50, "should still be skipping, not adapting")

    def test_exhausted_after_sustained_skipping(self):
        """Persistent skipping is a stall, not a rescue — hand over to the other guard."""
        g = self._warm(SpikeGuard(max_consecutive=20))
        self.assertFalse(g.exhausted())
        for _ in range(20):
            g.check(100.0)
        self.assertTrue(g.exhausted())

    def test_one_healthy_step_resets_the_consecutive_count(self):
        g = self._warm(SpikeGuard(max_consecutive=5))
        for _ in range(4):
            g.check(100.0)
        g.check(0.1)
        self.assertEqual(g.consecutive, 0)
        self.assertFalse(g.exhausted())

    def test_zero_median_does_not_skip_everything(self):
        """A fully converged run can have a ~0 median; the guard must not jam."""
        g = SpikeGuard(warmup=10)
        for _ in range(20):
            g.check(0.0)
        self.assertFalse(g.check(0.5))

    def test_history_is_bounded_by_the_window(self):
        g = SpikeGuard(window=200, warmup=50)
        self._warm(g, n=1000)
        self.assertEqual(len(g.history), 200)

    def test_build_from_config(self):
        self.assertIsNone(build_spike_guard({}), "must be opt-in, not on by default")
        self.assertIsNone(build_spike_guard({"spike": {"enabled": False}}))
        g = build_spike_guard({"spike": {"enabled": True, "factor": 25.0,
                                         "window": 100, "warmup": 10}})
        self.assertEqual((g.factor, g.window, g.warmup), (25.0, 100, 10))


if __name__ == "__main__":
    unittest.main()
