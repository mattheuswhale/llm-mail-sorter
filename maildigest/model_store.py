
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_REPO = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_DIR = PROJECT_ROOT / ".models" / "Qwen3-0.6B"


def _has_model_weights(path: Path) -> bool:
    return (
        (path / "model.safetensors").exists()
        or (path / "pytorch_model.bin").exists()
        or bool(list(path.glob("model-*.safetensors")))
        or bool(list(path.glob("pytorch_model-*.bin")))
    )


def model_is_complete(path: Path) -> bool:
    return (
        path.exists()
        and (path / "config.json").exists()
        and (path / "tokenizer_config.json").exists()
        and _has_model_weights(path)
    )


def ensure_local_model(
    repo_id: str = DEFAULT_MODEL_REPO,
    local_dir: Path = DEFAULT_MODEL_DIR,
    *,
    offline: bool = False,
) -> Path:
    """
    Ensure the model is stored inside this program directory.

    Once the local snapshot is complete, Hugging Face Hub is put in offline
    mode for the rest of the process. Email text is never sent to Hugging Face.
    """
    local_dir = local_dir.resolve()

    # Disable Hub telemetry for this process.
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    if model_is_complete(local_dir):
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(f"Using project-local model: {local_dir}", flush=True)
        return local_dir

    if offline:
        raise RuntimeError(
            f"Offline mode requested, but the model is incomplete or missing at: {local_dir}"
        )

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} into the program directory:", flush=True)
    print(f"  {local_dir}", flush=True)
    print("This network access is only for model files; email contents are not uploaded.", flush=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
    )

    if not model_is_complete(local_dir):
        raise RuntimeError(
            f"Model download completed but required files are missing from {local_dir}"
        )

    # All subsequent Transformers loads are strictly local.
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("Model download complete. Hugging Face Hub is now offline for this process.", flush=True)
    return local_dir
