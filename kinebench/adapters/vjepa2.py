"""Adapter for Meta's V-JEPA 2 / V-JEPA 2.1 (MIT code, MIT or Apache-2.0 weights).

Why this adapter exists
-----------------------
V-JEPA 2 is the strongest *public* model on the same axis KINE-Bench measures
(physical understanding from video, no labels). Benchmarking KineOne-WM against
it is the fastest way to tell whether a KINE-Bench number is actually good or
just better than random.

Licensing (verified 2026-09-01)
-------------------------------
- code:   MIT  (facebookresearch/vjepa2)
- vitl:   MIT          (facebook/vjepa2-vitl-fpc64-256)
- vitg:   Apache-2.0   (facebook/vjepa2-vitg-fpc64-256 / -384)

Both are commercially usable. Note that V-JEPA **1** (facebookresearch/jepa)
is CC-BY-NC -- do not use that one in a commercial product.

Known upstream gotcha: `src/hub/backbones.py` on main still hardcodes
VJEPA_BASE_URL = "http://localhost:8300", so `torch.hub.load` cannot fetch
weights. We therefore load through HuggingFace `transformers` instead.

Offline use
-----------
`build()` downloads from the HF hub by default (the normal path for users).
In locked-down / air-gapped CI you can pass `local_dir="/path/to/weights"`
(or set env `KINE_VJEPA2_LOCAL`) and the adapter will load directly from the
local `config.json` + `model.safetensors`, bypassing the hub entirely.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

# Guarded at module level so the adapter is importable without `transformers`
# installed (it is only needed when you actually build a V-JEPA 2 facade), and
# so tests can monkeypatch `kinebench.adapters.vjepa2.AutoModel`.
try:  # pragma: no cover - depends on environment
    from transformers import AutoModel
except ImportError:  # pragma: no cover
    AutoModel = None
try:  # pragma: no cover - AutoVideoProcessor may be a lazy stub that raises on access
    from transformers import AutoVideoProcessor
except Exception:  # pragma: no cover
    AutoVideoProcessor = None

from .base import AdapterInfo, WorldModelAdapter, as_uint8_numpy

# bench alias -> (hf repo, weight license, size label)
MODELS: Dict[str, tuple] = {
    "vjepa2-vitl-256": ("facebook/vjepa2-vitl-fpc64-256", "MIT", "ViT-L/16 (300M)"),
    "vjepa2-vitg-256": ("facebook/vjepa2-vitg-fpc64-256", "Apache-2.0", "ViT-g/16 (1B)"),
    "vjepa2-vitg-384": ("facebook/vjepa2-vitg-fpc64-384", "Apache-2.0", "ViT-g/16 (1B)"),
}

DEFAULT = "vjepa2-vitl-256"


class _VJEPA2Facade:
    """KineJEPA-shaped facade over an encoder-only V-JEPA 2 model.

    `predictor` and `intervention_head` are intentionally absent: the public
    V-JEPA 2 release does not ship a usable masked-token predictor for
    arbitrary clips, so FUT-1 / EMB-1 / CAU-1 are reported as n/a rather than
    approximated.
    """

    predictor = None
    intervention_head = None

    def __init__(self, model, processor, num_frames: int, img_size: int,
                 patch: int, tubelet: int, batch_size: int):
        self.model = model
        self.processor = processor
        # `num_frames` = the model's own frames_per_clip (e.g. 64). Incoming
        # clips are resampled to this length so the token grid is always valid.
        self.num_frames = num_frames
        # `img_size` is the model's own resolution, NOT the incoming clip size:
        # the preprocessor resizes, so the token grid is set by the model.
        self.img_size = img_size
        self.patch = patch
        self.tubelet = tubelet
        self.batch_size = batch_size
        self._grid = (max(1, num_frames // tubelet), img_size // patch, img_size // patch)

    @property
    def grid(self):
        return self._grid

    # -- internal ----------------------------------------------------------
    @staticmethod
    def _resample_time(clip: np.ndarray, t_target: int) -> np.ndarray:
        """(T,H,W,C) -> (t_target,H,W,C) by nearest-time index selection."""
        T = clip.shape[0]
        if T == t_target:
            return clip
        idx = np.linspace(0, T - 1, t_target).round().astype(int)
        return clip[idx]

    def _preprocess(self, videos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(B,C,T,H,W) float -> processor inputs on the model's device."""
        clips: List[np.ndarray] = [as_uint8_numpy(v) for v in videos]
        clips = [self._resample_time(c, self.num_frames) for c in clips]
        if self.processor is not None:
            try:
                return self.processor(
                    videos=[list(c) for c in clips],
                    return_tensors="pt",
                    do_sample_frames=False,
                )
            except TypeError:
                # older processors without do_sample_frames
                return self.processor(videos=[list(c) for c in clips], return_tensors="pt")
        return self._manual(clips)

    def _manual(self, clips: Sequence) -> Dict[str, torch.Tensor]:
        """Fallback: resize + center-crop-free normalization, no processor."""
        import torch.nn.functional as F

        arr = torch.stack([
            torch.from_numpy(c).permute(3, 0, 1, 2).float() / 255.0 for c in clips
        ])
        batch, channels, frames, height, width = arr.shape
        # Resize each video frame as an image. A direct reshape of (B,C,T,H,W)
        # mixes channels and time, and can even change the frame count.
        images = arr.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        images = F.interpolate(
            images,
            size=(self.img_size, self.img_size), mode="bilinear", align_corners=False,
        )
        arr = images.reshape(batch, frames, channels, self.img_size, self.img_size)
        arr = arr.permute(0, 2, 1, 3, 4)
        mean = torch.tensor(self._mean, device=arr.device).view(1, 3, 1, 1, 1)
        std = torch.tensor(self._std, device=arr.device).view(1, 3, 1, 1, 1)
        return {"pixel_values_videos": (arr - mean) / std}

    _mean = (0.485, 0.456, 0.406)
    _std = (0.229, 0.224, 0.225)

    @torch.no_grad()
    def _forward(self, videos: torch.Tensor) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        outs = []
        for i in range(0, len(videos), self.batch_size):
            batch = self._preprocess(videos[i:i + self.batch_size])
            pv = batch["pixel_values_videos"].to(dev)
            out = self.model(pixel_values_videos=pv)
            h = getattr(out, "last_hidden_state", None)
            if h is None:
                h = out[0]
            outs.append(h.float().cpu())
        return torch.cat(outs, dim=0)

    # -- probe-facing API --------------------------------------------------
    @torch.no_grad()
    def target(self, videos: torch.Tensor) -> torch.Tensor:
        feats = self._forward(videos)
        return feats.to(videos.device)

    @torch.no_grad()
    def encoder(self, videos: torch.Tensor, visible_idx=None) -> torch.Tensor:
        feats = self._forward(videos).to(videos.device)
        if visible_idx is None:
            return feats
        idx = visible_idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])
        return torch.gather(feats, 1, idx)

    def __repr__(self):  # pragma: no cover
        return f"<VJEPA2Facade grid={self._grid}>"


def _load_offline(local_dir: str, device: str):
    """Load complete V-JEPA 2 weights from a local directory.

    Bypasses the HuggingFace hub -- needed when the hub is unreachable (proxy
    pollution, air-gap, offline CI). The weights themselves are unmodified
    Meta checkpoints; we only skip the download step.
    """
    from transformers import AutoModel

    local = Path(local_dir)
    if not local.is_dir() or not any(local.glob("*.safetensors")):
        raise FileNotFoundError(f"no .safetensors in {local_dir}")
    # from_pretrained reads model.safetensors.index.json and all referenced shards.
    # The old glob-first + strict=False path could score randomly initialized layers
    # when a shard or parameter was missing, while still returning finite embeddings.
    model, info = AutoModel.from_pretrained(
        local_dir, local_files_only=True, use_safetensors=True,
        output_loading_info=True,
    )
    problems = {key: info.get(key) for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if info.get(key)}
    if problems:
        raise RuntimeError(f"incomplete or incompatible local V-JEPA 2 weights: {problems}")
    return model.to(device).eval()


class VJEPA2Adapter(WorldModelAdapter):
    def __init__(self, alias: str = DEFAULT, num_frames: int = 16,
                 img_size: int | None = None, batch_size: int = 2,
                 local_dir: str | None = None):
        if alias not in MODELS:
            raise KeyError(f"unknown vjepa2 alias {alias!r}; pick from {sorted(MODELS)}")
        self.alias = alias
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.local_dir = local_dir or os.environ.get("KINE_VJEPA2_LOCAL")
        repo, lic, size = MODELS[alias]
        self.repo = repo
        # resolution comes from the checkpoint's own config (crop_size); the
        # caller's img_size only affects the input clips, which get resized.
        self.img_size = int(repo.rsplit("-", 1)[-1])
        self.info = AdapterInfo(
            name=alias,
            source=f"hf:{repo}",
            license=lic,
            capabilities=frozenset({"encode"}),
            params=size,
            notes="Meta V-JEPA 2. Encoder only: FUT-1/EMB-1/CAU-1 report n/a.",
        )

    def build(self, device: str = "cpu") -> Any:
        if AutoModel is None:
            raise RuntimeError(
                "transformers is required for the V-JEPA 2 adapter: pip install transformers"
            )
        if self.local_dir is not None:
            model = _load_offline(self.local_dir, device)
            processor = None
            if AutoVideoProcessor is not None:
                try:
                    processor = AutoVideoProcessor.from_pretrained(self.local_dir, local_files_only=True)
                except Exception:
                    processor = None
            cfg = model.config
        else:
            model = AutoModel.from_pretrained(self.repo).to(device).eval()
            try:
                processor = AutoVideoProcessor.from_pretrained(self.repo)
            except Exception:
                processor = None
            cfg = model.config

        patch = int(getattr(cfg, "patch_size", 16))
        tubelet = int(getattr(cfg, "tubelet_size", 2))
        res = int(getattr(cfg, "image_size", None)
                  or getattr(cfg, "crop_size", None) or self.img_size)
        self.img_size = res
        # The encoder always emits `frames_per_clip // tubelet_size` time steps.
        nf = int(getattr(cfg, "frames_per_clip", None) or (self.num_frames * tubelet))
        return _VJEPA2Facade(
            model=model, processor=processor,
            num_frames=nf, img_size=res,
            patch=patch, tubelet=tubelet, batch_size=self.batch_size,
        )
