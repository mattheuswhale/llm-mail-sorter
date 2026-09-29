from __future__ import annotations

import argparse
import os
from pathlib import Path

from maildigest.model_store import (
    DEFAULT_MODEL_REPO,
    LEGACY_DEFAULT_MODEL_DIR,
    ensure_local_model,
    model_dir_for_repo,
    model_is_complete,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly download/check a local Hugging Face model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_REPO,
        help="Hugging Face repository ID.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to a project-local path derived from --model.",
    )
    parser.add_argument(
        "--disable-xet",
        action="store_true",
        help="Disable Hugging Face Xet transfers and use the HTTP fallback.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose Hugging Face Hub diagnostics.",
    )
    args = parser.parse_args()

    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if args.debug:
        os.environ["HF_HUB_VERBOSITY"] = "debug"
        os.environ["HF_DEBUG"] = "1"

    model_dir = args.model_dir or model_dir_for_repo(args.model)

    # Preserve compatibility with the original untracked default model folder.
    repo_id = args.model
    if (
        args.model == DEFAULT_MODEL_REPO
        and model_dir.resolve() == LEGACY_DEFAULT_MODEL_DIR.resolve()
        and model_is_complete(model_dir)
    ):
        repo_id = None

    ensure_local_model(
        repo_id=repo_id,
        local_dir=model_dir,
        offline=False,
    )


if __name__ == "__main__":
    main()
