"""Pearson motion scores need enough distinct held-out observations."""

from unittest.mock import patch

import torch

from kinebench.metrics import _split_indices, motion_magnitude


class _MeanEncoder:
    def __call__(self, videos):
        return videos.mean(dim=(1, 2, 3, 4)).view(len(videos), 1, 1)


class _Model:
    target = _MeanEncoder()


def _clips(n, constant_motion=False):
    clips = []
    for i in range(n):
        clip = torch.zeros(3, 2, 2, 2)
        clip[:, 1] = 1.0 if constant_motion else i / max(n - 1, 1)
        clips.append(clip)
    return clips


def test_motion_split_has_three_test_and_three_train_clips():
    train, test = _split_indices(8, seed=0, min_test=3)
    assert len(train) == 5 and len(test) == 3
    assert set(train.tolist()).isdisjoint(test.tolist())


def test_nonconstant_motion_reports_finite_three_clip_correlation():
    result = motion_magnitude(_Model(), _clips(8), "cpu")
    assert result["n_test"] == 3
    assert -1.0 <= result["pearson_r"] <= 1.0


def test_too_few_clips_do_not_report_spurious_perfect_correlation():
    result = motion_magnitude(_Model(), _clips(5), "cpu")
    assert result["pearson_r"] is None
    assert result["status"] == "unavailable"


def test_constant_heldout_motion_does_not_report_nan():
    result = motion_magnitude(_Model(), _clips(8, constant_motion=True), "cpu")
    assert result["pearson_r"] is None
    assert result["status"] == "unavailable"


def test_constant_prediction_does_not_report_nan():
    class ConstantProbe:
        def __call__(self, x):
            return torch.zeros(len(x), 1)

    with patch("kinebench.metrics._train_probe", return_value=(ConstantProbe(), torch.tensor([0, 1, 2]))):
        result = motion_magnitude(_Model(), _clips(8), "cpu")
    assert result["pearson_r"] is None
    assert result["status"] == "unavailable"
