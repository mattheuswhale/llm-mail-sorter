
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .digest import json_document, make_item, markdown_digest, sort_items
from .gmail_readonly import ReadOnlyGmail, authenticate_readonly
from .model_store import (
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_REPO,
    ensure_local_model,
)
from .priority import classify_priority


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read unread Gmail messages without modifying the mailbox, summarize locally, and prioritize deterministically."
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=Path("credentials.json"),
        help="Google OAuth desktop-app credentials JSON.",
    )
    parser.add_argument(
        "--token",
        type=Path,
        default=Path(".secrets/token_readonly.json"),
        help="Dedicated read-only OAuth token cache.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=30,
        help="Maximum unread INBOX messages to retrieve (1-500).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_REPO,
        help="Hugging Face repository used only if the local model is missing.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Project-local model directory.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device. auto prefers CUDA when available.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid model download; require the project-local model to already exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for digest.md and digest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure model is local before touching Gmail. Once present, Transformers
    # uses local_files_only and HF Hub is placed in offline mode.
    model_dir = ensure_local_model(
        repo_id=args.model,
        local_dir=args.model_dir,
        offline=args.offline,
    )

    if not args.credentials.exists():
        raise SystemExit(
            f"Missing {args.credentials}. See README.md for Google OAuth setup."
        )

    service = authenticate_readonly(args.credentials, args.token)
    gmail = ReadOnlyGmail(service)

    print(f"Retrieving up to {args.max_messages} unread INBOX messages (read-only)...")
    messages = gmail.unread_inbox(max_messages=args.max_messages)
    print(f"Retrieved {len(messages)} unread message(s).")

    items = []

    if messages:
        from .llm import LocalLLMExtractor

        extractor = LocalLLMExtractor(model_dir, device=args.device)

        for index, email in enumerate(messages, start=1):
            print(f"[{index}/{len(messages)}] Analyzing: {email.subject[:70]}")
            signals = extractor.extract(email)
            classification = classify_priority(email, signals)
            print(
                f"    -> {classification.priority.upper()} "
                f"({classification.rule})"
            )
            items.append(make_item(email, classification))

        items = sort_items(items)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    digest_json = json_document(items)
    (args.output_dir / "digest.json").write_text(
        json.dumps(digest_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "digest.md").write_text(
        markdown_digest(items),
        encoding="utf-8",
    )

    print(f"Wrote: {args.output_dir / 'digest.md'}")
    print(f"Wrote: {args.output_dir / 'digest.json'}")
    print("Mailbox mutation operations are not implemented and OAuth is gmail.readonly only.")


if __name__ == "__main__":
    main()
