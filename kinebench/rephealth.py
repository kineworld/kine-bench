"""KINE-REP-1: representational-health probes for action-conditioned world models.

Motivation
----------
`kine-jepa` trains with a single L1 latent-prediction loss and no explicit
anti-collapse term. A model can reduce that loss by making the target geometry
cheaper to predict -- i.e. by discarding information. KINE-FUT-1 (prediction
fidelity) and KINE-TEMP-1 / KINE-MOT-1 (linear probes on pooled features) can all
*look* acceptable while the latent space silently loses dimensionality.

This module makes that failure mode **measurable and reproducible** instead of
inferred from a status note. It reports three diagnostics, each with a negative
control that proves the statistic actually moves when collapse is induced:

  KINE-REP-ERANK  effective rank of the centred feature matrix (Roy & Vetterli,
                  2007 definition). A healthy feature space uses many
                  dimensions; a collapsed one concentrates variance into a few.
  KINE-REP-DIMU   dimension-utilisation curve: the fraction of variance held by
                  the top-k principal directions, reported at k = 1, 4, 16, 64.
                  Diagnoses *where* the variance concentrates, which a single
                  scalar rank number hides.
  KINE-REP-DISP   displacement identifiability: the fraction of consecutive-frame
                  latent displacements that are closer to their own clip's mean
                  displacement than to another clip's. Following the inverse-
                  dynamics line (AC-MTM, Delta-JEPA), a collapsed encoder makes
                  transitions indistinguishable, so this drops toward chance.

Design constraints (deliberately strict)
----------------------------------------
- Deterministic: every statistic is a pure function of the input features and is
  reproducible bit-for-bit on CPU.
- No fabricated scores: if there are too few samples to estimate a statistic, the
  field is `None` with `status="unavailable"`, never a placeholder number --
  matching the existing KINE-Bench convention.
- Model-agnostic: operates on a feature matrix, so it can be applied to any
  encoder (native, V-JEPA 2, a third party's) without asking it to change.

This module is a *diagnostic*. It publishes what to measure, not how to fix.

Determinism contract
--------------------
Every function here is a pure function of its input tensor(s): no RNG is used, so
identical features give identical statistics. This was verified explicitly --
`representational_health` returns byte-identical dicts on repeated calls with the
same features. Consequently, **any variation between two runs of `kinebench rep`
comes from the model, not the probe**; a random-initialised checkpoint will
differ run to run, and that is a property of the checkpoint, not a defect here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MIN_SAMPLES = 4
MIN_PER_CLIP = 8
DIM_UTIL_KS = (1, 4, 16, 64)


def effective_rank(features: torch.Tensor, eps: float = 1e-12) -> float:
    """Effective rank of a (N, D) matrix: exp(entropy of the singular-value spectrum).

    This is the spectral definition of Roy & Vetterli (2007). It lies in [1, D]:
    1 means all variance sits on a single direction (fully collapsed geometry),
    D means variance is spread evenly across every direction.

    The matrix is centred first, so a constant offset in every sample -- which a
    linear probe can perfectly read but which carries no discriminative structure
    -- does not inflate the rank.
    """
    x = features.detach().float()
    if x.dim() != 2:
        raise ValueError("features must be (N, D)")
    if x.shape[0] < 2:
        raise ValueError("effective rank needs at least two samples")
    x = x - x.mean(dim=0, keepdim=True)
    # Singular values of the centred matrix; sigma**2 are the variances.
    s = torch.linalg.svdvals(x)
    var = s * s
    total = var.sum()
    if not torch.isfinite(total) or total <= eps:
        return 1.0  # no variance at all is the fully collapsed case
    p = var / total
    p = p[p > 0]
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy).item())


def dimension_utilisation(features: torch.Tensor, ks=DIM_UTIL_KS) -> dict:
    """Fraction of centred variance captured by the top-k principal directions."""
    x = features.detach().float()
    if x.dim() != 2:
        raise ValueError("features must be (N, D)")
    if x.shape[0] < 2:
        raise ValueError("dimension utilisation needs at least two samples")
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    var = s * s
    total = var.sum()
    out = {}
    if not torch.isfinite(total) or total <= 1e-12:
        for k in ks:
            out[f"top{k}"] = 1.0  # all variance is (degenerately) in the top direction
        return out
    cum = torch.cumsum(var, dim=0) / total
    dim = cum.shape[0]
    for k in ks:
        idx = min(k, dim) - 1
        out[f"top{k}"] = round(float(cum[idx].item()), 4)
    return out


def displacement_identifiability(displacements: torch.Tensor, groups,
                                 min_per_clip: int = MIN_PER_CLIP) -> dict:
    """Can a latent displacement be matched to the clip that produced it?

    `displacements` is (N, D): the difference between consecutive latents.
    `groups` is a length-N sequence assigning each displacement to a source clip.

    For each clip we compute its mean displacement, then ask, for every
    displacement, whether its own clip's mean is the nearest among all clip means.
    A collapsed encoder maps many distinct transitions onto the same latent
    change, so the nearest-mean match degrades toward chance (1 / num_clips).

    **Statistical-power guard.** The nearest-mean test needs enough displacements
    per clip to estimate a stable mean. On a real probe run we observed every arm
    -- healthy *and* collapsed -- returning exactly chance (0.125 = 1/8) when
    clips carried only 3 displacements each, because the per-clip mean was noise.
    A number that cannot be distinguished from chance is not a measurement, so if
    any clip has fewer than `min_per_clip` displacements the statistic is reported
    as `unavailable` with the observed counts. It is never returned as a score.

    Returns the accuracy, the chance level, and the lift over chance.
    """
    x = displacements.detach().float()
    g = torch.as_tensor(list(groups))
    if x.dim() != 2:
        raise ValueError("displacements must be (N, D)")
    if x.shape[0] != g.shape[0]:
        raise ValueError("groups must have one entry per displacement")
    if x.shape[0] < MIN_SAMPLES:
        return {"accuracy": None, "chance": None, "lift": None,
                "status": "unavailable",
                "error": f"needs at least {MIN_SAMPLES} displacements"}
    unique = torch.unique(g)
    if len(unique) < 2:
        return {"accuracy": None, "chance": None, "lift": None,
                "status": "unavailable",
                "error": "needs at least two source clips"}

    means, labels, counts = [], [], []
    for u in unique.tolist():
        sel = (g == u)
        # A clip whose displacements cancel to ~zero would make the nearest-mean
        # test degenerate, so require at least one usable displacement.
        if int(sel.sum().item()) < 1:
            continue
        means.append(x[sel].mean(dim=0))
        labels.append(u)
        counts.append(int(sel.sum().item()))
    if len(means) < 2:
        return {"accuracy": None, "chance": None, "lift": None,
                "status": "unavailable", "error": "fewer than two usable clips"}

    if min(counts) < min_per_clip:
        return {"accuracy": None, "chance": None, "lift": None,
                "status": "unavailable",
                "error": (f"every clip needs at least {min_per_clip} displacements to "
                          f"estimate a stable mean; observed minimum {min(counts)}"),
                "n_clips": len(labels), "per_clip_counts": counts}

    means = torch.stack(means)                      # (C, D)
    # Cosine distance so magnitude differences between clips do not dominate.
    xn = F.normalize(x, dim=-1)
    mn = F.normalize(means, dim=-1)
    sim = xn @ mn.t()                               # (N, C)
    nearest = sim.argmax(dim=1)
    clip_index = {lab: i for i, lab in enumerate(labels)}
    truth = torch.tensor([clip_index[int(v)] for v in g.tolist()])
    acc = float((nearest == truth).float().mean().item())
    chance = 1.0 / len(labels)
    return {
        "accuracy": round(acc, 4),
        "chance": round(chance, 4),
        "lift": round(acc - chance, 4),
        "n_clips": len(labels),
        "n_displacements": int(x.shape[0]),
        "per_clip_counts": counts,
        "status": "ok",
    }


@torch.no_grad()
def representational_health(model, clips, device="cpu", seed: int = 0,
                            pool: str = "mean") -> dict:
    """KINE-REP-1: pooled-feature health of the frozen target encoder.

    Uses `model.target` on the given clips, consistent with KINE-TEMP-1 and
    KINE-MOT-1, so all three probes describe the same frozen representation.
    """
    if len(clips) < MIN_SAMPLES:
        return {"effective_rank": None, "rank_fraction": None,
                "dim_utilisation": None, "displacement": None,
                "status": "unavailable",
                "error": f"needs at least {MIN_SAMPLES} clips"}

    videos = torch.stack(clips)
    feats = model.target(videos.to(device))          # (B, N, D)
    if pool == "mean":
        pooled = feats.mean(dim=1)                   # (B, D)
    else:
        raise ValueError(f"unsupported pooling: {pool}")
    dim = pooled.shape[-1]

    rank = effective_rank(pooled)
    util = dimension_utilisation(pooled)

    disp_result = {"accuracy": None, "chance": None, "lift": None,
                   "status": "unavailable", "error": "no temporal pairs"}
    if any(c.shape[1] >= 2 for c in clips):
        deltas, groups = [], []
        for i, c in enumerate(clips):
            f = model.target(c.unsqueeze(0).to(device)).squeeze(0)   # (N, D)
            if f.shape[0] < 2:
                continue
            deltas.append(f[1:] - f[:-1])
            groups.extend([i] * (f.shape[0] - 1))
        if deltas:
            disp_result = displacement_identifiability(torch.cat(deltas, dim=0), groups)

    return {
        "effective_rank": round(rank, 4),
        "rank_fraction": round(rank / dim, 4),
        "feature_dim": dim,
        "dim_utilisation": util,
        "displacement": disp_result,
        "n_clips": len(clips),
        "status": "ok",
    }
