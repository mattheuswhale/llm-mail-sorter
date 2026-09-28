from __future__ import annotations

import argparse
from pathlib import Path

from maildigest.model_store import ensure_model_downloaded


def main() -> None:
    p = argparse.ArgumentParser(description="Explicitly download/check the local Qwen model.")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--model-dir", type=Path, default=Path(".models/Qwen3-0.6B"))
    p.add_argument("--disable-xet", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--timeout", type=int, default=30)
    args = p.parse_args()

    ensure_model_downloaded(
        args.model,
        args.model_dir,
        timeout=args.timeout,
        disable_xet=args.disable_xet,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
