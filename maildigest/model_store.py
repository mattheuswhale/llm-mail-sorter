from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_REPO = "Qwen/Qwen3-0.6B"
DEFAULT_LAYA_REPO = "convaiinnovations/laya"
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
    """Return True for a complete Transformers causal-LM directory."""
    return (
        path.exists()
        and (path / "config.json").exists()
        and (path / "tokenizer_config.json").exists()
        and _has_model_weights(path)
    )


def laya_model_is_complete(path: Path) -> bool:
    """Return True for a complete project-local Laya checkpoint."""
    return (
        path.exists()
        and (path / "rl_agent_config.json").exists()
        and (path / "model.safetensors").exists()
        and (path / "tokenizer" / "tokenizer_config.json").exists()
        and (path / "encoder" / "config.json").exists()
    )


def _read_recorded_repo(path: Path) -> str | None:
    metadata_path = path / MODEL_METADATA_FILE
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    repo_id = data.get("repo_id")
    return str(repo_id) if repo_id else None


def _write_recorded_repo(path: Path, repo_id: str, *, kind: str) -> None:
    metadata_path = path / MODEL_METADATA_FILE
    temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps({"repo_id": repo_id, "kind": kind}, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(metadata_path)


def _set_hub_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"


def _ensure_project_local_model(
    repo_id: str | None,
    local_dir: Path,
    *,
    offline: bool,
    lock_hub_offline: bool,
    kind: str,
    is_complete: Callable[[Path], bool],
    allow_patterns: list[str] | None = None,
) -> tuple[Path, str | None]:
    local_dir = local_dir.resolve()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    # Explicit offline mode must take effect before any Hugging Face import/load.
    if offline:
        _set_hub_offline()

    if is_complete(local_dir):
        recorded_repo = _read_recorded_repo(local_dir)

        if repo_id and recorded_repo and repo_id != recorded_repo:
            raise RuntimeError(
                f"Model directory {local_dir} contains {recorded_repo}, "
                f"but the requested model is {repo_id}. Use a different model directory."
            )

        if repo_id and not recorded_repo:
            raise RuntimeError(
                f"Model directory {local_dir} contains a complete but untracked model, "
                f"so MailSort cannot verify that it is {repo_id}. "
                "Either omit the repository ID and use this directory as a local model, "
                "or download the requested repository into a new/empty directory."
            )

        if lock_hub_offline:
            _set_hub_offline()
        print(f"Using project-local {kind} model: {local_dir}", flush=True)
        return local_dir, recorded_repo

    if offline:
        raise RuntimeError(
            f"Offline mode requested, but the {kind} model is incomplete or missing at: "
            f"{local_dir}"
        )

    if not repo_id:
        raise RuntimeError(
            f"No complete {kind} model exists at {local_dir}. "
            "Specify a Hugging Face repository so it can be downloaded."
        )

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} into:", flush=True)
    print(f"  {local_dir}", flush=True)
    print(
        "This network access is only for model files; email contents are not uploaded.",
        flush=True,
    )

    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
    }
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns

    snapshot_download(**kwargs)

    if not is_complete(local_dir):
        raise RuntimeError(
            f"{kind.capitalize()} model download completed but required files are missing "
            f"from {local_dir}"
        )

    _write_recorded_repo(local_dir, repo_id, kind=kind)

    if lock_hub_offline:
        _set_hub_offline()
        print(
            "Model download complete. Hugging Face Hub is now offline for this process.",
            flush=True,
        )
    else:
        print("Model download complete.", flush=True)

    return local_dir, repo_id


def ensure_local_model(
    repo_id: str | None,
    local_dir: Path,
    *,
    offline: bool = False,
    lock_hub_offline: bool = True,
) -> tuple[Path, str | None]:
    """
    Ensure a causal language model is available locally.

    If local_dir already contains a complete model, repo_id may be None.
    Directories downloaded by MailSort contain .mailsort_model.json so an
    explicitly supplied --model can be checked against the local directory.
    Older/manual model directories remain usable with --model-dir alone and
    are reported as local/unverified rather than being assigned a guessed repo.

    lock_hub_offline=False is used only during multi-model startup, before Gmail
    is accessed, so a second project-local model can still be downloaded.
    """
    return _ensure_project_local_model(
        repo_id,
        local_dir,
        offline=offline,
        lock_hub_offline=lock_hub_offline,
        kind="LLM",
        is_complete=model_is_complete,
    )


def ensure_local_laya_model(
    repo_id: str | None,
    local_dir: Path,
    *,
    offline: bool = False,
    lock_hub_offline: bool = True,
) -> tuple[Path, str | None]:
    """Ensure a Laya checkpoint is available in a project-local directory."""
    return _ensure_project_local_model(
        repo_id,
        local_dir,
        offline=offline,
        lock_hub_offline=lock_hub_offline,
        kind="Laya",
        is_complete=laya_model_is_complete,
        allow_patterns=[
            "rl_agent_config.json",
            "model.safetensors",
            "tokenizer/*",
            "encoder/*",
        ],
    )
