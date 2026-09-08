"""Read the TensorBoard scalars around the two no-geo collapses.

Login-node CPU only. All wb220 arms share results_wb220/tb, one event file per
process start, so files are mapped to jobs by wall time. Windows of interest:

  job 6169725 (collapse ~epoch 92): steps 18000-19200, plus the recovered
      epoch-10 spike at steps 1800-2400
  job 6189791 (collapse epoch 82):  steps 16000-17250

Per-step gradient norms are NOT recorded anywhere — train/grad_norm is a
50-step mean — so single-step magnitudes stay inferences.
"""

import datetime as dt
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB = Path("results_wb220/tb")
TAGS = ["train/grad_norm", "train/loss",
        "train/loss_t_q1", "train/loss_t_q2", "train/loss_t_q3", "train/loss_t_q4"]
WINDOWS = [("run1-epoch10-spike", 1800, 2400),
           ("run1-collapse", 18000, 19200),
           ("run2-collapse", 16000, 17250)]


def load(f):
    acc = EventAccumulator(str(f), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


files = sorted(TB.glob("events.out.tfevents.*"))
print(f"{len(files)} event files\n")
accs = []
for f in files:
    acc = load(f)
    tags = acc.Tags().get("scalars", [])
    if "train/grad_norm" not in tags:
        print(f"  {f.name}: no train/grad_norm ({len(tags)} tags) — skipped")
        continue
    ev = acc.Scalars("train/grad_norm")
    t0 = dt.datetime.fromtimestamp(ev[0].wall_time)
    t1 = dt.datetime.fromtimestamp(ev[-1].wall_time)
    print(f"  {f.name}\n    {t0} .. {t1} | steps {ev[0].step}..{ev[-1].step} "
          f"| {len(ev)} points")
    accs.append((f.name, t0, acc, ev))

# ── long-run drift: per-2000-step median and max of grad_norm ────────────────
print("\n=== grad_norm drift (per 2000-step bucket: median / max) ===")
for name, t0, acc, ev in accs:
    if len(ev) < 50:
        continue
    print(f"\n{name}  (start {t0})")
    buckets = {}
    for e in ev:
        buckets.setdefault(e.step // 2000, []).append(e.value)
    for b in sorted(buckets):
        v = sorted(buckets[b])
        print(f"  steps {b*2000:>6}-{(b+1)*2000:>6}: med {v[len(v)//2]:.4f}  "
              f"max {v[-1]:.4f}")

# ── collapse windows, all tags side by side ──────────────────────────────────
for label, lo, hi in WINDOWS:
    print(f"\n=== window {label} (steps {lo}-{hi}) ===")
    for name, t0, acc, ev in accs:
        if not any(lo <= e.step <= hi for e in ev):
            continue
        series = {}
        for tag in TAGS:
            try:
                series[tag] = {e.step: e.value for e in acc.Scalars(tag)
                               if lo <= e.step <= hi}
            except KeyError:
                pass
        try:
            series["val/loss"] = {e.step: e.value for e in acc.Scalars("val/loss")
                                  if lo - 400 <= e.step <= hi + 400}
        except KeyError:
            pass
        steps = sorted({s for d in series.values() for s in d})
        print(f"\n  file {name} (start {t0})")
        hdr = "  step   " + "".join(f"{t.split('/')[-1]:>10}" for t in series)
        print(hdr)
        for s in steps:
            row = f"  {s:>6} "
            for t in series:
                v = series[t].get(s)
                row += f"{v:>10.4f}" if v is not None else f"{'':>10}"
            print(row)
