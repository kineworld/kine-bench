"""Check the offline fallback keeps every frame and color channel in place."""

import numpy as np
import torch

from kinebench.adapters.vjepa2 import _VJEPA2Facade


def _facade(size=3):
    return _VJEPA2Facade(None, None, num_frames=2, img_size=size,
                         patch=1, tubelet=1, batch_size=1)


def test_manual_preserves_frame_and_channel_order():
    clips = []
    for batch in range(2):
        clip = np.zeros((2, 3, 4, 3), dtype=np.uint8)
        for frame in range(2):
            clip[frame] = [20 + batch * 40 + frame * 10,
                           80 + batch * 30 + frame * 10,
                           140 + batch * 20 + frame * 10]
        clips.append(clip)

    result = _facade()._manual(clips)["pixel_values_videos"]
    assert result.shape == (2, 3, 2, 3, 3)
    for batch in range(2):
        for frame in range(2):
            for channel in range(3):
                expected = (clips[batch][frame, 0, 0, channel] / 255.0
                            - _facade()._mean[channel]) / _facade()._std[channel]
                torch.testing.assert_close(
                    result[batch, channel, frame],
                    torch.full((3, 3), expected), rtol=0, atol=1e-6,
                )


def test_manual_resizes_each_frame_independently():
    clip = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    facade = _facade(size=5)
    actual = facade._manual([clip])["pixel_values_videos"]
    frames = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0
    resized = torch.nn.functional.interpolate(
        frames, size=(5, 5), mode="bilinear", align_corners=False,
    )
    expected = resized.permute(1, 0, 2, 3).unsqueeze(0)
    mean = torch.tensor(facade._mean).view(1, 3, 1, 1, 1)
    std = torch.tensor(facade._std).view(1, 3, 1, 1, 1)
    torch.testing.assert_close(actual, (expected - mean) / std)
