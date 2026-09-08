"""Where did the collapse move the weights? Epoch 80 (best) vs epoch 83 (mid-
collapse), SAME run (job 6189791) — checkpoints_wb220/diffusion_best.pt vs
diffusion.pt. Three epochs apart, the collapse in between, so per-layer drift
localizes the damage.

Login node: 4 GiB cgroup, two ~3.9 GB checkpoints — mmap keeps the tensors
file-backed (clean, evictable pages) and we touch them one tensor at a time.
"""

import torch

A = "checkpoints_wb220/diffusion_best.pt"   # epoch 80, val 0.01282
B = "checkpoints_wb220/diffusion.pt"        # epoch 83, val 0.99977

a = torch.load(A, map_location="cpu", mmap=True, weights_only=False)
b = torch.load(B, map_location="cpu", mmap=True, weights_only=False)
print(f"A: epoch {a['epoch']} step {a['step']}   B: epoch {b['epoch']} step {b['step']}")

sa, sb = a["model"], b["model"]
assert sa.keys() == sb.keys()

rows, tot_d2, tot_w2 = [], 0.0, 0.0
groups = {}
for k in sa:
    wa = sa[k].float()
    d = (sb[k].float() - wa)
    dn, wn = d.norm().item(), wa.norm().item()
    tot_d2 += dn * dn
    tot_w2 += wn * wn
    rows.append((k, dn, wn, dn / (wn + 1e-12)))
    g = ".".join(k.split(".")[:2])
    gg = groups.setdefault(g, [0.0, 0.0])
    gg[0] += dn * dn
    gg[1] += wn * wn
    del wa, d

print(f"\nGLOBAL ||delta|| / ||w_ep80|| = {tot_d2**0.5:.4f} / {tot_w2**0.5:.4f} "
      f"= {tot_d2**0.5 / tot_w2**0.5:.4f}")

print("\n=== drift by module group (top 15 by relative drift) ===")
for g, (d2, w2) in sorted(groups.items(), key=lambda kv: -kv[1][0] / (kv[1][1] + 1e-12))[:15]:
    print(f"  {g:35s} rel {d2**0.5 / (w2**0.5 + 1e-12):8.4f}   "
          f"||d|| {d2**0.5:9.4f}  ||w|| {w2**0.5:9.4f}")

print("\n=== top 15 tensors by RELATIVE drift ===")
for k, dn, wn, r in sorted(rows, key=lambda t: -t[3])[:15]:
    print(f"  {k:60s} rel {r:8.4f}  ||d|| {dn:8.4f}  ||w|| {wn:8.4f}")

print("\n=== top 15 tensors by ABSOLUTE drift ===")
for k, dn, wn, r in sorted(rows, key=lambda t: -t[1])[:15]:
    print(f"  {k:60s} ||d|| {dn:8.4f}  ||w|| {wn:8.4f}  rel {r:8.4f}")

# The trivial solution is eps_hat = 0: did the final conv actually go to zero,
# or does the zero arise upstream (dead features feeding a healthy head)?
print("\n=== output head ===")
for k in sa:
    if "out" in k.lower() or k.startswith(tuple(sorted({kk.rsplit('.', 2)[0] for kk in sa})[-1:])):
        pass
last_keys = list(sa.keys())[-6:]
for k in last_keys:
    print(f"  {k:60s} ||w80|| {sa[k].float().norm().item():9.5f}  "
          f"||w83|| {sb[k].float().norm().item():9.5f}")

# Weight-norm growth question (weight_decay = 0): total norm at 80 vs 83 is in
# the numbers above; also print a few GroupNorm gain norms since normalization
# layers set the effective gain.
print("\n=== groupnorm gains (first 8) ===")
n = 0
for k in sa:
    if k.endswith(".weight") and sa[k].dim() == 1:
        print(f"  {k:60s} ||w80|| {sa[k].float().norm().item():9.4f}  "
              f"||w83|| {sb[k].float().norm().item():9.4f}")
        n += 1
        if n >= 8:
            break
