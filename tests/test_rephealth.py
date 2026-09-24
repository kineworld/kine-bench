"""KINE-REP-1 must move in the right direction on constructed inputs.

Every test here uses a hand-built feature matrix with a *known* geometry, so the
expected statistic is derivable by hand. The negative controls are the point of
the file: a diagnostic that always returns "healthy" is worse than no diagnostic
at all, so each statistic is proven to separate a full-rank space from a collapsed
one.
"""

import math

import torch

from kinebench.rephealth import (
    dimension_utilisation,
    displacement_identifiability,
    effective_rank,
    representational_health,
)


class _IdentityEncoder:
    """Pooled features = per-clip mean over (T, H, W), keeping channels.

    Lets a test construct an exact (B, D) feature matrix by choosing clip
    statistics, without a real ViT.
    """

    def __call__(self, videos):
        # videos: (B, C, T, H, W) -> (B, T, C)
        return videos.mean(dim=(3, 4)).transpose(1, 2)


class _Model:
    def __init__(self, encoder):
        self.target = encoder


def _orthogonal_features(n, d):
    """(n, d) with independent standard-normal columns -> rank ~ d."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, d, generator=g)


def _collapsed_features(n, d, dims=1):
    """Variance confined to the first `dims` directions -> effective rank ~ dims."""
    x = torch.zeros(n, d)
    g = torch.Generator().manual_seed(0)
    x[:, :dims] = torch.randn(n, dims, generator=g)
    return x


# ---------------------------------------------------------------- effective rank

def test_full_rank_features_use_many_dimensions():
    x = _orthogonal_features(256, 64)
    rank = effective_rank(x)
    # Random Gaussian in 64-d: entropy near maximum, so exp(entropy) close to 64.
    assert rank > 40, rank


def test_collapsed_features_report_near_one_dimension():
    x = _collapsed_features(256, 64, dims=1)
    assert 0.99 <= effective_rank(x) <= 1.01


def test_partial_collapse_lands_between():
    full = effective_rank(_orthogonal_features(256, 64))
    part = effective_rank(_collapsed_features(256, 64, dims=4))
    assert 3.5 <= part <= 4.5
    assert part < full


def test_constant_offset_does_not_inflate_rank():
    """A shared mean carries no structure; centring must remove it."""
    base = _collapsed_features(256, 64, dims=1)
    shifted = base + 100.0
    assert abs(effective_rank(shifted) - effective_rank(base)) < 1e-4


def test_zero_variance_is_the_collapsed_case_not_nan():
    rank = effective_rank(torch.ones(32, 8))
    assert rank == 1.0 and math.isfinite(rank)


# ------------------------------------------------------------- dimension usage

def test_dimension_utilisation_reaches_one_by_full_width():
    util = dimension_utilisation(_orthogonal_features(256, 64))
    assert util["top64"] > 0.95


def test_collapsed_space_saturates_the_first_bucket():
    util = dimension_utilisation(_collapsed_features(256, 64, dims=1))
    assert util["top1"] > 0.99
    assert util["top16"] > 0.99


def test_utilisation_is_monotone_in_k():
    util = dimension_utilisation(_orthogonal_features(256, 64))
    ks = [1, 4, 16, 64]
    vals = [util[f"top{k}"] for k in ks]
    assert vals == sorted(vals)


# --------------------------------------------------- displacement identifiability

def test_distinct_clips_are_identifiable():
    """Four clips with well-separated mean displacements -> perfect matching.

    Each clip gets 8 displacements so the per-clip mean is estimable and the
    statistical-power guard does not fire.
    """
    disp, groups = [], []
    seeds = {0: [1.0, 0.0], 1: [0.0, 1.0], 2: [-1.0, 0.0], 3: [0.0, -1.0]}
    g = torch.Generator().manual_seed(0)
    for clip, centre in seeds.items():
        for _ in range(8):
            # small jitter around a well-separated centre
            disp.append(torch.tensor(centre) + 0.02 * torch.randn(2, generator=g))
            groups.append(clip)
    result = displacement_identifiability(torch.stack(disp), groups)
    assert result["status"] == "ok"
    assert result["accuracy"] == 1.0
    assert result["chance"] == 0.25


def test_collapsed_displacements_fall_to_chance():
    """All transitions identical -> the statistic cannot do better than chance.

    This is the negative control that proves the probe detects collapse.
    """
    n = 40
    disp = torch.ones(n, 2) * 5.0
    groups = [i % 4 for i in range(n)]
    result = displacement_identifiability(disp, groups)
    assert result["status"] == "ok"
    assert result["accuracy"] == 0.25
    assert result["lift"] == 0.0


def test_underpowered_run_is_unavailable_not_a_chance_score():
    """Few displacements per clip must NOT be reported as a numeric accuracy.

    Observed on a real probe run: healthy and collapsed arms both returned
    exactly 1/8 when clips had only 3 displacements, because the per-clip mean
    was noise. Reporting 0.125 as a score would be a fabricated measurement.
    """
    disp = torch.randn(12, 6)
    groups = [i % 4 for i in range(12)]          # 3 displacements per clip
    result = displacement_identifiability(disp, groups)
    assert result["accuracy"] is None
    assert result["status"] == "unavailable"
    assert result["per_clip_counts"] == [3, 3, 3, 3]


def test_single_clip_is_unavailable_not_perfect():
    disp = torch.randn(8, 4)
    result = displacement_identifiability(disp, [0] * 8)
    assert result["accuracy"] is None
    assert result["status"] == "unavailable"


def test_too_few_displacements_is_unavailable():
    result = displacement_identifiability(torch.randn(3, 4), [0, 0, 1])
    assert result["accuracy"] is None
    assert result["status"] == "unavailable"


# ------------------------------------------------------------- end-to-end probe

def _clips(n=8, t=4, size=2):
    """Clips whose temporal profile distinguishes them by index."""
    clips = []
    for i in range(n):
        clip = torch.zeros(1, t, size, size)
        clip[0, :, :, :] = torch.arange(t).view(t, 1, 1) + i * 10.0
        clips.append(clip)
    return clips


def _separated_clips(n=6, t=12, size=2):
    """Each clip's consecutive-frame latent *displacement points in its own
    direction*, which is what the probe actually tests.

    Note: the probe uses cosine similarity, so it measures whether displacement
    *directions* are separable, not whether their magnitudes differ. Clips that
    move at different speeds along the same axis are legitimately not
    identifiable by this statistic -- that is a property of the metric, and it
    is why the fixture varies direction rather than magnitude.
    """
    clips = []
    for i in range(n):
        # alternate the channel that receives the temporal ramp, and flip sign
        ch = i % size
        sign = 1.0 if (i // size) % 2 == 0 else -1.0
        clip = torch.zeros(1, t, size, size)
        ramp = torch.arange(t).float() * sign        # (t,)
        clip[0, :, ch, :] = ramp.view(t, 1)
        clips.append(clip)
    return clips


def test_health_probe_reports_all_three_diagnostics():
    """rank and dimension-utilisation are always available; displacement needs
    enough frames per clip, and in this small fixture it correctly declines."""
    result = representational_health(_Model(_IdentityEncoder()), _clips())
    assert result["status"] == "ok"
    assert result["effective_rank"] is not None
    assert result["dim_utilisation"] is not None
    # 4 frames -> 3 displacements per clip: below the power guard, so the
    # displacement probe must decline rather than emit a chance-level number.
    assert result["displacement"]["status"] == "unavailable"


def test_health_probe_is_unavailable_below_min_samples():
    result = representational_health(_Model(_IdentityEncoder()), _clips(n=3))
    assert result["status"] == "unavailable"
    assert result["effective_rank"] is None


def test_health_probe_displacement_works_when_adequately_powered():
    """With enough frames per clip the displacement probe activates and, on
    an encoder whose clips are well separated, identifies them far above chance.
    This proves the power guard declines only when it must."""
    clips = _separated_clips(n=6, t=12)
    result = representational_health(_Model(_IdentityEncoder()), clips)
    assert result["status"] == "ok"
    assert result["displacement"]["status"] == "ok"
    assert result["displacement"]["accuracy"] > result["displacement"]["chance"]


def test_health_probe_rejects_non_2d_features():
    try:
        effective_rank(torch.randn(4, 4, 4))
    except ValueError:
        return
    raise AssertionError("expected ValueError on a 3-D feature tensor")
