from __future__ import annotations

import argparse
from pathlib import Path

from .archive import save_run_archive, update_manifest
from .digest import make_item, sort_items
from .gmail_client import GmailClient, authenticate_gmail
from .model_store import (
    DEFAULT_MODEL_REPO,
    ensure_local_model,
    model_dir_for_repo,
    model_is_complete,
)
from .priority import classify_priority


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read unread Gmail messages, summarize locally, prioritize "
            "deterministically, and optionally mark processed messages as read."
        )
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
        help="OAuth token used for normal read-only runs.",
    )
    parser.add_argument(
        "--modify-token",
        type=Path,
        default=Path(".secrets/token_modify.json"),
        help="Separate OAuth token used only with --mark-read.",
    )
    parser.add_argument(
        "--mark-read",
        action="store_true",
        help=(
            "After the local timestamped archive is written successfully, "
            "remove the UNREAD label from every processed message. This "
            "requires Gmail's broader gmail.modify OAuth scope."
        ),
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=30,
        help="Maximum unread INBOX messages to retrieve (1-500).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Hugging Face repository to download when the local model is missing. "
            f"If both --model and --model-dir are omitted, defaults to {DEFAULT_MODEL_REPO}."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Local model directory. If omitted, a directory is derived from --model. "
            "If this directory already contains a complete model, --model is optional."
        ),
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
        "--save-bodies",
        action="store_true",
        help=(
            "Store full extracted email bodies in processed_messages.json. "
            "Without this flag, only metadata/snippets/results are archived."
        ),
    )
    parser.add_argument(
        "--show-subjects",
        action="store_true",
        help="Print email subjects to the terminal while processing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help=(
            "Root output directory. Each run is saved in a timestamped "
            "subdirectory containing digest.md, digest.json, "
            "processed_messages.json, and manifest.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve model repo/path without mixing models in the same directory.
    if args.model_dir is None:
        if args.model is not None:
            model_repo = args.model
            requested_model_dir = model_dir_for_repo(model_repo)
        else:
            # Backward compatibility: older MailSort versions stored the
            # default model in .models/Qwen3-0.6B without identity metadata.
            requested_model_dir = model_dir_for_repo(DEFAULT_MODEL_REPO)
            model_repo = (
                None
                if model_is_complete(requested_model_dir)
                else DEFAULT_MODEL_REPO
            )
    else:
        requested_model_dir = args.model_dir
        model_repo = args.model

        # A custom existing local directory does not need --model. A missing or
        # incomplete custom directory does, because we need to know what to download.
        if not model_is_complete(requested_model_dir) and model_repo is None:
            raise SystemExit(
                f"No complete model found at {requested_model_dir}. "
                "Specify --model <huggingface/repo> to download it."
            )

    model_dir, recorded_model_repo = ensure_local_model(
        repo_id=model_repo,
        local_dir=requested_model_dir,
        offline=args.offline,
    )

    if not args.credentials.exists():
        raise SystemExit(
            f"Missing {args.credentials}. Configure Google OAuth credentials first."
        )

    token_path = args.modify_token if args.mark_read else args.token
    service = authenticate_gmail(
        args.credentials,
        token_path,
        allow_modify=args.mark_read,
    )
    gmail = GmailClient(service, allow_modify=args.mark_read)

    mode = "mark-read enabled" if args.mark_read else "read-only"
    print(
        f"Retrieving up to {args.max_messages} unread INBOX messages ({mode})..."
    )
    messages = gmail.unread_inbox(max_messages=args.max_messages)
    print(f"Retrieved {len(messages)} unread message(s).")

    items = []

    if messages:
        from .llm import LocalLLMExtractor

        extractor = LocalLLMExtractor(model_dir, device=args.device)

        for index, email in enumerate(messages, start=1):
            if args.show_subjects:
                print(f"[{index}/{len(messages)}] Analyzing: {email.subject[:70]}")
            else:
                print(f"[{index}/{len(messages)}] Analyzing message...")
            signals = extractor.extract(email)
            classification = classify_priority(email, signals)
            print(
                f"    -> {classification.priority.upper()} "
                f"({classification.rule})"
            )
            items.append(make_item(email, classification))

        items = sort_items(items)

    # IMPORTANT: persist the source messages and analysis before any mailbox
    # modification. If saving fails, the messages remain unread.
    run_dir, manifest = save_run_archive(
        output_dir=args.output_dir,
        messages=messages,
        items=items,
        mark_read_requested=args.mark_read,
        model_repo=recorded_model_repo or model_repo or "local/unknown",
        model_dir=model_dir,
        save_bodies=args.save_bodies,
        device=args.device,
        max_messages=args.max_messages,
    )

    print(f"Saved timestamped run archive: {run_dir}")
    print(f"  - {run_dir / 'digest.md'}")
    print(f"  - {run_dir / 'digest.json'}")
    print(f"  - {run_dir / 'processed_messages.json'}")
    print(f"  - {run_dir / 'manifest.json'}")

    if args.mark_read and messages:
        # Persist an "attempting" state before touching Gmail. If the process
        # crashes after Gmail changes but before the final manifest update, the
        # archive will not misleadingly claim the operation is still pending.
        manifest["mark_read_status"] = "attempting"
        update_manifest(run_dir, manifest)

        print("Marking processed messages as read...")
        try:
            marked_count = gmail.mark_as_read([message.id for message in messages])
        except Exception as exc:
            manifest["mark_read_status"] = "failed"
            manifest["mark_read_error"] = f"{type(exc).__name__}: {exc}"
            update_manifest(run_dir, manifest)
            raise
        else:
            manifest["mark_read_status"] = "completed"
            manifest["marked_read_count"] = marked_count
            update_manifest(run_dir, manifest)
            print(f"Marked {marked_count} processed message(s) as read.")
    elif args.mark_read:
        manifest["mark_read_status"] = "completed"
        manifest["marked_read_count"] = 0
        update_manifest(run_dir, manifest)
        print("No messages were retrieved, so nothing was marked as read.")
    else:
        print("Read-only mode: Gmail was not modified.")


if __name__ == "__main__":
    main()
