from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from maildigest.model_store import MODEL_METADATA_FILE, ensure_local_model


def _make_complete_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"test")


class ModelIdentityTests(unittest.TestCase):
    def test_recorded_model_repo_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            _make_complete_model(model_dir)
            (model_dir / MODEL_METADATA_FILE).write_text(
                json.dumps({"repo_id": "org/model-a"}),
                encoding="utf-8",
            )

            with self.assertRaises(RuntimeError):
                ensure_local_model(
                    repo_id="org/model-b",
                    local_dir=model_dir,
                    offline=True,
                )

    def test_untracked_local_model_works_without_repo_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            _make_complete_model(model_dir)

            resolved, repo = ensure_local_model(
                repo_id=None,
                local_dir=model_dir,
                offline=True,
            )

            self.assertEqual(resolved, model_dir.resolve())
            self.assertIsNone(repo)

    def test_untracked_local_model_rejects_unverifiable_repo_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            _make_complete_model(model_dir)

            with self.assertRaises(RuntimeError):
                ensure_local_model(
                    repo_id="org/model-a",
                    local_dir=model_dir,
                    offline=True,
                )


if __name__ == "__main__":
    unittest.main()
