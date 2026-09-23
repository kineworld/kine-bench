"""The temporal probe must split by video before comparing its two views."""

import torch

from kinebench.metrics import _split_indices, temporal_order


def test_augmented_views_never_cross_train_test_boundary():
    n_videos = 12
    groups = torch.arange(n_videos).repeat(2)
    train, test = _split_indices(2 * n_videos, seed=17, groups=groups)
    assert set(groups[train].tolist()).isdisjoint(groups[test].tolist())
    assert len(train) + len(test) == 2 * n_videos
    assert len(test) == 2 * 3
    labels = torch.tensor([1] * n_videos + [0] * n_videos)
    assert set(labels[train].tolist()) == {0, 1}
    assert set(labels[test].tolist()) == {0, 1}
    same_train, same_test = _split_indices(2 * n_videos, seed=17, groups=groups)
    torch.testing.assert_close(train, same_train)
    torch.testing.assert_close(test, same_test)


def test_temporal_probe_rejects_single_frame_clip():
    clip = torch.zeros(3, 1, 4, 4)
    try:
        temporal_order(object(), [clip, clip], "cpu")
    except ValueError as exc:
        assert "at least two frames" in str(exc)
    else:
        raise AssertionError("single-frame clip would hang in shuffle loop")
