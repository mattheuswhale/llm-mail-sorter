from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_REPO = "Qwen/Qwen3-0.6B"
MODELS_ROOT = PROJECT_ROOT / ".models"
LEGACY_DEFAULT_MODEL_DIR = MODELS_ROOT / "Qwen3-0.6B"
MODEL_METADATA_FILE = ".mailsort_model.json"


def model_dir_for_repo(repo_id: str) -> Path:
    """Return a deterministic project-local directory for a Hugging Face repo."""
    # Preserve the original default path so existing installs keep working.
    if repo_id == DEFAULT_MODEL_REPO:
        return LEGACY_DEFAULT_MODEL_DIR

    safe_name = repo_id.replace("/", "__").replace("\\", "__")
    return MODELS_ROOT / safe_name


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


def _read_recorded_repo(path: Path) -> str | None:
    metadata_path = path / MODEL_METADATA_FILE
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    repo_id = data.get("repo_id")
    return str(repo_id) if repo_id else None


def _write_recorded_repo(path: Path, repo_id: str) -> None:
    metadata_path = path / MODEL_METADATA_FILE
    temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps({"repo_id": repo_id}, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(metadata_path)


def ensure_local_model(
    repo_id: str | None,
    local_dir: Path,
    *,
    offline: bool = False,
) -> tuple[Path, str | None]:
    """
    Ensure a model is available locally.

    If local_dir already contains a complete model, repo_id may be None.
    Directories downloaded by MailSort contain .mailsort_model.json so an
    explicitly supplied --model can be checked against the local directory.
    Older/manual model directories remain usable with --model-dir alone and
    are reported as local/unverified rather than being assigned a guessed repo.
    """
    local_dir = local_dir.resolve()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    if model_is_complete(local_dir):
        recorded_repo = _read_recorded_repo(local_dir)

        if repo_id and recorded_repo and repo_id != recorded_repo:
            raise RuntimeError(
                f"Model directory {local_dir} contains {recorded_repo}, "
                f"but --model requested {repo_id}. Use a different --model-dir."
            )

        if repo_id and not recorded_repo:
            raise RuntimeError(
                f"Model directory {local_dir} contains a complete but untracked model, "
                f"so MailSort cannot verify that it is {repo_id}. "
                "Either omit --model and use this directory as a local model, or "
                "download the requested repository into a new/empty --model-dir."
            )

        os.environ["HF_HUB_OFFLINE"] = "1"
        print(f"Using project-local model: {local_dir}", flush=True)
        return local_dir, recorded_repo

    if offline:
        raise RuntimeError(
            f"Offline mode requested, but the model is incomplete or missing at: {local_dir}"
        )

    if not repo_id:
        raise RuntimeError(
            f"No complete model exists at {local_dir}. "
            "Specify --model <huggingface/repo> so it can be downloaded."
        )

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} into:", flush=True)
    print(f"  {local_dir}", flush=True)
    print(
        "This network access is only for model files; email contents are not uploaded.",
        flush=True,
    )

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
    )

    if not model_is_complete(local_dir):
        raise RuntimeError(
            f"Model download completed but required files are missing from {local_dir}"
        )

    _write_recorded_repo(local_dir, repo_id)

    os.environ["HF_HUB_OFFLINE"] = "1"
    print(
        "Model download complete. Hugging Face Hub is now offline for this process.",
        flush=True,
    )
    return local_dir, repo_id
