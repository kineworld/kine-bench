"""Incomplete native checkpoints must never be scored with random layers."""

import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch

from kinebench import load


class _TinyJEPA(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.predictor = torch.nn.Linear(2, 2)
        self.predictor.mask_token = torch.nn.Parameter(torch.zeros(1, 1, 2))


class _TinyHead(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = torch.nn.Linear(dim, dim)


def _load_checkpoint(state):
    jepa = ModuleType("kineworld_jepa")
    jepa.__path__ = []
    jepa_model = ModuleType("kineworld_jepa.jepa")
    jepa_model.KineJEPA = _TinyJEPA
    causal = ModuleType("kineworld_jepa.causal")
    causal.InterventionHead = _TinyHead
    modules = {"kineworld_jepa": jepa, "kineworld_jepa.jepa": jepa_model,
               "kineworld_jepa.causal": causal}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save({"model": state}, path)
        with patch.dict(sys.modules, modules), patch.object(load, "ensure_jepa_path"):
            return load.load_model(str(path), "cpu", img_size=64, num_frames=4)


def test_complete_native_checkpoint_loads_all_tensors():
    model = _TinyJEPA()
    state = model.state_dict()
    loaded = _load_checkpoint(state)
    for key, value in state.items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_missing_native_tensor_is_rejected():
    state = _TinyJEPA().state_dict()
    state.pop("encoder.bias")
    try:
        _load_checkpoint(state)
    except RuntimeError as exc:
        assert "encoder.bias" in str(exc)
    else:
        raise AssertionError("incomplete checkpoint was accepted")


def test_wrapped_base_and_head_load_completely():
    base = _TinyJEPA().state_dict()
    head = _TinyHead(2).state_dict()
    wrapped = {**{"base." + k: v for k, v in base.items()},
               **{"head." + k: v for k, v in head.items()}}
    loaded = _load_checkpoint(wrapped)
    for key, value in base.items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    for key, value in head.items():
        torch.testing.assert_close(loaded.intervention_head.state_dict()[key], value)


def test_missing_intervention_head_tensor_is_rejected():
    base = _TinyJEPA().state_dict()
    head = _TinyHead(2).state_dict()
    head.pop("projection.bias")
    wrapped = {**{"base." + k: v for k, v in base.items()},
               **{"head." + k: v for k, v in head.items()}}
    try:
        _load_checkpoint(wrapped)
    except RuntimeError as exc:
        assert "projection.bias" in str(exc)
    else:
        raise AssertionError("incomplete intervention head was accepted")
