from __future__ import annotations

import argparse
from pathlib import Path

from .archive import save_run_archive, update_manifest
from .digest import make_item, sort_items
from .gmail_client import GmailClient, authenticate_gmail
from .model_store import (
    DEFAULT_LAYA_REPO,
    DEFAULT_MODEL_REPO,
    ensure_local_laya_model,
    ensure_local_model,
    laya_model_is_complete,
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
        "--classifier",
        choices=("llm", "laya"),
        default="llm",
        help=(
            "Structured-signal classifier. 'llm' keeps the existing single-model path. "
            "'laya' experimentally refines booleans/category with a local Laya checkpoint "
            "while the generative model still provides summary/deadline text."
        ),
    )
    parser.add_argument(
        "--laya-model",
        default=None,
        help=(
            "Hugging Face Laya repository to download when --classifier laya is used "
            f"and the local checkpoint is missing. Defaults to {DEFAULT_LAYA_REPO}."
        ),
    )
    parser.add_argument(
        "--laya-model-dir",
        type=Path,
        default=None,
        help=(
            "Project-local Laya checkpoint directory. If omitted, a directory is derived "
            "from --laya-model (or the default Laya repository)."
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


def _resolve_optional_laya(args: argparse.Namespace) -> tuple[Path, str | None, str | None]:
    """Resolve/download Laya before Gmail access, then lock the Hub offline."""
    if args.laya_model_dir is None:
        laya_repo = args.laya_model or DEFAULT_LAYA_REPO
        requested_laya_dir = model_dir_for_repo(laya_repo)

        # Preserve compatibility with a manually provisioned untracked directory.
        repo_for_check = (
            None
            if args.laya_model is None and laya_model_is_complete(requested_laya_dir)
            else laya_repo
        )
    else:
        requested_laya_dir = args.laya_model_dir
        repo_for_check = args.laya_model
        if not laya_model_is_complete(requested_laya_dir) and repo_for_check is None:
            raise SystemExit(
                f"No complete Laya model found at {requested_laya_dir}. "
                "Specify --laya-model <huggingface/repo> to download it."
            )

    laya_dir, recorded_laya_repo = ensure_local_laya_model(
        repo_id=repo_for_check,
        local_dir=requested_laya_dir,
        offline=args.offline,
        lock_hub_offline=True,
    )
    return laya_dir, recorded_laya_repo, repo_for_check


def _release_cuda_memory() -> None:
    """Best-effort cleanup between the LLM phase and optional Laya phase."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    # Resolve/download every model before Gmail is authenticated or email content is
    # retrieved. In hybrid mode the Hub stays online only until both model downloads
    # are complete, then HF_HUB_OFFLINE is enabled for the rest of the process.
    if args.model_dir is None:
        if args.model is not None:
            model_repo = args.model
            requested_model_dir = model_dir_for_repo(model_repo)
        else:
            requested_model_dir = model_dir_for_repo(DEFAULT_MODEL_REPO)
            model_repo = (
                None
                if model_is_complete(requested_model_dir)
                else DEFAULT_MODEL_REPO
            )
    else:
        requested_model_dir = args.model_dir
        model_repo = args.model

        if not model_is_complete(requested_model_dir) and model_repo is None:
            raise SystemExit(
                f"No complete model found at {requested_model_dir}. "
                "Specify --model <huggingface/repo> to download it."
            )

    hybrid_mode = args.classifier == "laya"
    model_dir, recorded_model_repo = ensure_local_model(
        repo_id=model_repo,
        local_dir=requested_model_dir,
        offline=args.offline,
        lock_hub_offline=not hybrid_mode,
    )

    laya_dir: Path | None = None
    recorded_laya_repo: str | None = None
    requested_laya_repo: str | None = None
    if hybrid_mode:
        laya_dir, recorded_laya_repo, requested_laya_repo = _resolve_optional_laya(args)
        try:
            import laya  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "Laya classifier requested but the optional package is not installed. "
                "Run: pip install -r requirements-laya.txt"
            ) from exc

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
    classifier_fallback_count = 0

    if messages:
        from .llm import LocalLLMExtractor

        if not hybrid_mode:
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
        else:
            # Phase 1: preserve the existing generative summary/deadline extraction.
            # Laya will replace the booleans/category used by priority rules in phase 2.
            print("Phase 1/2: extracting summaries and deadline text with the local LLM...")
            extractor = LocalLLMExtractor(model_dir, device=args.device)
            base_signals = []
            for index, email in enumerate(messages, start=1):
                if args.show_subjects:
                    print(f"[{index}/{len(messages)}] Extracting: {email.subject[:70]}")
                else:
                    print(f"[{index}/{len(messages)}] Extracting message...")
                base_signals.append(extractor.extract(email))

            # Avoid requiring enough VRAM to keep both models resident simultaneously.
            del extractor
            _release_cuda_memory()

            from .laya_classifier import LocalLayaClassifier

            assert laya_dir is not None
            print("Phase 2/2: refining structured signals with local Laya...")
            classifier = LocalLayaClassifier(laya_dir, device=args.device)

            for index, (email, base) in enumerate(
                zip(messages, base_signals), start=1
            ):
                if args.show_subjects:
                    print(f"[{index}/{len(messages)}] Classifying: {email.subject[:70]}")
                else:
                    print(f"[{index}/{len(messages)}] Classifying message...")

                try:
                    signals = classifier.refine(email, base)
                except Exception as exc:
                    classifier_fallback_count += 1
                    signals = base
                    print(
                        "    WARNING: Laya classification failed for this message; "
                        f"using LLM-extracted signals ({type(exc).__name__})"
                    )

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
        classifier_backend=args.classifier,
        classifier_model_repo=(
            recorded_laya_repo or requested_laya_repo or "local/unknown"
            if hybrid_mode
            else None
        ),
        classifier_model_dir=laya_dir if hybrid_mode else None,
        classifier_fallback_count=classifier_fallback_count,
    )

    print(f"Saved timestamped run archive: {run_dir}")
    print(f"  - {run_dir / 'digest.md'}")
    print(f"  - {run_dir / 'digest.json'}")
    print(f"  - {run_dir / 'processed_messages.json'}")
    print(f"  - {run_dir / 'manifest.json'}")

    if hybrid_mode and classifier_fallback_count:
        print(
            f"Laya fallback count: {classifier_fallback_count} message(s) used the "
            "original LLM-extracted structured signals."
        )

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
