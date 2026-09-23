"""Wiring test for the V-JEPA 2 adapter: proves the facade correctly bridges
the probe modules to a real transformers VJEPA2Model API -- WITHOUT downloading
the 1.2GB weights. We monkeypatch AutoModel/AutoVideoProcessor with stubs that
mirror the real interface (pixel_values_videos -> last_hidden_state).

Uses only the standard library (unittest.mock) so the repo stays lean:
    torch  is the only runtime requirement.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinebench.adapters import build_adapter  # noqa: E402
from kinebench.metrics import temporal_order, motion_magnitude  # noqa: E402
from kinebench.cau import intervention_auc  # noqa: E402


class _StubProcessor:
    def __call__(self, videos, return_tensors="pt", do_sample_frames=False):
        # `videos` is a list of B clips, each a list of T (H,W,C) frames.
        t = torch.stack([
            torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in clip])
            for clip in videos
        ])  # (B, T, C, H, W)
        return {"pixel_values_videos": t.permute(0, 2, 1, 3, 4).contiguous()}


class _StubModel(SimpleNamespace):
    config = SimpleNamespace(image_size=256, patch_size=16, tubelet_size=2, frames_per_clip=16)
    _params = [torch.zeros(1)]  # so next(self.model.parameters()) works

    def parameters(self):
        return iter(self._params)

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, pixel_values_videos):
        B, C, T, H, W = pixel_values_videos.shape
        N = (T // 2) * (H // 16) * (W // 16)
        h = (torch.arange(N, dtype=torch.float32).view(1, N, 1) / N).expand(B, N, 64).contiguous()
        return SimpleNamespace(last_hidden_state=h)


def test_vjepa2_facade_wiring():
    import kinebench.adapters.vjepa2 as m

    with patch.object(m, "AutoModel", SimpleNamespace(from_pretrained=lambda *a, **k: _StubModel())), \
         patch.object(m, "AutoVideoProcessor", SimpleNamespace(from_pretrained=lambda *a, **k: _StubProcessor())):
        adapter = build_adapter("vjepa2-vitl-256")
        model = adapter.build("cpu")
        assert model.grid == (8, 16, 16), model.grid
        assert model.predictor is None
        assert model.intervention_head is None

        clips = [torch.randn(3, 16, 64, 64) for _ in range(4)]
        r1 = temporal_order(model, clips, "cpu")
        assert r1["accuracy"] >= 0 and r1["accuracy"] <= 1.0, r1
        r2 = motion_magnitude(model, clips, "cpu")
        assert "pearson_r" in r2
        # encoder-only: cau auc_do must be None, encoder auc still computed
        r3 = intervention_auc(model, "cpu", n_pairs=4, frames=16, size=64)
        assert r3["auc_do"] is None
        assert "auc" in r3
        print("PASS test_vjepa2_facade_wiring  grid=%s auc=%.3f mot=%s" % (
            model.grid, r3["auc"], r2["pearson_r"]))


if __name__ == "__main__":
    test_vjepa2_facade_wiring()
    print("all wiring tests passed")
