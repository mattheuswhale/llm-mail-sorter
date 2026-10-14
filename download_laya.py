from __future__ import annotations

import argparse
import os
from pathlib import Path

from maildigest.model_store import (
    DEFAULT_LAYA_REPO,
    ensure_local_laya_model,
    laya_model_is_complete,
    model_dir_for_repo,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly download/check the optional project-local Laya classifier."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LAYA_REPO,
        help="Hugging Face Laya repository ID.",
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

    # A manually provisioned local checkpoint can be used without claiming a
    # repository identity. The normal default download remains tracked.
    repo_id: str | None = args.model
    if laya_model_is_complete(model_dir) and not (
        model_dir / ".mailsort_model.json"
    ).exists():
        repo_id = None

    ensure_local_laya_model(
        repo_id=repo_id,
        local_dir=model_dir,
        offline=False,
    )


if __name__ == "__main__":
    main()
