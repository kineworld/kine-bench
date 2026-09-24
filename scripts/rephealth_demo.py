"""Real-number sanity run for KINE-REP-1.

Question: does the probe actually separate a healthy feature space from a
collapsed one *on KINE-Bench's own synthetic distribution* (not just on
hand-built tensors)? A diagnostic that only works on toy inputs would be useless.

This is a development check, not published evidence.
"""
import sys

import torch

sys.path.insert(0, ".")
from kinebench.rephealth import representational_health
from kinebench.synth import synthetic_clips


class Enc:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, v):
        B, C, T, H, W = v.shape
        if self.mode == "collapse":
            return torch.ones(B, 8, 16)                 # constant, no information
        g = torch.Generator().manual_seed(0)
        base = torch.randn(B, 8, 16, generator=g)
        if self.mode == "lowrank":
            base[:, :, 4:] = 0.0                        # variance in 4 of 16 dims
        return base


class M:
    def __init__(self, e):
        self.target = e


clips = synthetic_clips(8, num_frames=4, size=32, seed=0)
hdr = f"{'mode':10}{'eff_rank':>10}{'rank_frac':>11}{'top1':>8}{'top4':>8}{'disp_acc':>10}{'disp_lift':>11}"
print(hdr)
print("-" * len(hdr))
for mode in ("healthy", "lowrank", "collapse"):
    r = representational_health(M(Enc(mode)), clips, device="cpu")
    d = r["displacement"]
    print(f"{mode:10}{r['effective_rank']:>10.3f}{r['rank_fraction']:>11.3f}"
          f"{r['dim_utilisation']['top1']:>8.3f}{r['dim_utilisation']['top4']:>8.3f}"
          f"{str(d['accuracy']):>10}{str(d['lift']):>11}")
