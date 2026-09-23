"""Locate kineworld_jepa and load KINE-JEPA checkpoints."""

import os
import sys
from pathlib import Path

import torch


def ensure_jepa_path() -> None:
    try:
        import kineworld_jepa  # noqa: F401
        return
    except ImportError:
        pass
    env = os.environ.get("KINE_JEPA_ROOT")
    here = Path(__file__).resolve().parent
    candidates = ([env] if env else []) + [
        str(here.parents[1] / "kine-jepa"),
        str(here.parents[1] / "kine-exp001"),
    ]
    for c in candidates:
        if c and (Path(c) / "kineworld_jepa").is_dir():
            sys.path.insert(0, c)
            return
    raise ImportError(
        "kineworld_jepa not found. Clone https://github.com/zoahdev/kine-jepa next to "
        "kine-bench, or set KINE_JEPA_ROOT to its directory."
    )


def unwrap_state(state):
    if not state:
        return state, {}
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}
    head = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
    if any(k.startswith("base.") for k in state):
        base = {k[len("base."):]: v for k, v in state.items() if k.startswith("base.")}
        return base, head
    return {k: v for k, v in state.items() if not k.startswith("head.")}, head


def attach_head(model, head_state):
    if not head_state:
        model.intervention_head = None
        return model
    from kineworld_jepa.causal import InterventionHead
    dim = model.predictor.mask_token.shape[-1]
    head = InterventionHead(dim)
    head.load_state_dict(head_state, strict=True)
    model.intervention_head = head
    return model


def load_model(ckpt_path, device, img_size=224, num_frames=16):
    ensure_jepa_path()
    from kineworld_jepa.jepa import KineJEPA

    cfg, state, head_state = {}, None, {}
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw = ckpt.get("model", ckpt)
        state, head_state = unwrap_state(raw)
        cfg = ckpt.get("config") or {}
    kw = dict(
        img_size=int(cfg.get("img_size", img_size)),
        num_frames=int(cfg.get("num_frames", num_frames)),
        enc_depth=int(cfg.get("enc_depth", 12)),
        pred_depth=int(cfg.get("pred_depth", 6)),
    )
    if cfg.get("tiny") or kw["img_size"] <= 64:
        kw.update(tubelet_t=2, patch_size=16, enc_dim=64, enc_heads=4, pred_dim=64, pred_heads=4)
        if int(cfg.get("enc_depth", 12)) <= 2:
            kw["enc_depth"] = int(cfg.get("enc_depth", 1))
            kw["pred_depth"] = int(cfg.get("pred_depth", 2))
    model = KineJEPA(**kw)
    if state is not None:
        # An incomplete checkpoint must not silently benchmark random layers.
        model.load_state_dict(state, strict=True)
    attach_head(model, head_state)
    if getattr(model, "intervention_head", None) is not None:
        model.intervention_head.to(device).eval()
    model.to(device).eval()
    return model
