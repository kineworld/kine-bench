"""Small local checkpoints test the real offline loader without downloading weights."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from kinebench.adapters.vjepa2 import _load_offline


@unittest.skipUnless(
    importlib.util.find_spec("transformers") and importlib.util.find_spec("safetensors"),
    "offline weight tests need requirements-vjepa2.txt",
)
class OfflineVJEPAWeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import BertConfig, BertModel

        cls.model = BertModel(BertConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32,
        ))

    def test_complete_sharded_checkpoint_loads_every_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp, safe_serialization=True,
                                       max_shard_size="10KB")
            self.assertGreater(len(list(Path(tmp).glob("*.safetensors"))), 1)
            restored = _load_offline(tmp, "cpu")
            for key, expected in self.model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], expected)

    def test_missing_tensor_is_rejected_instead_of_randomly_initialized(self):
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp, safe_serialization=True)
            weight_file = Path(tmp) / "model.safetensors"
            weights = load_file(weight_file)
            del weights[sorted(weights)[0]]
            save_file(weights, weight_file)
            with self.assertRaisesRegex(RuntimeError, "missing_keys"):
                _load_offline(tmp, "cpu")

    def test_missing_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp, safe_serialization=True,
                                       max_shard_size="10KB")
            shards = sorted(Path(tmp).glob("*.safetensors"))
            self.assertGreater(len(shards), 1)
            shards[0].unlink()
            with self.assertRaises((OSError, FileNotFoundError, RuntimeError)):
                _load_offline(tmp, "cpu")


if __name__ == "__main__":
    unittest.main()
