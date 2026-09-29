from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .digest import DigestItem, json_document, markdown_digest
from .types import EmailMessage


def _unique_run_dir(output_dir: Path, local_time: datetime) -> Path:
    stamp = local_time.strftime("%Y-%m-%d_%H-%M-%S")
    candidate = output_dir / stamp
    suffix = 1

    while candidate.exists():
        candidate = output_dir / f"{stamp}_{suffix:02d}"
        suffix += 1

    return candidate


def _write_json(path: Path, value: Any) -> None:
    """Write JSON atomically so a crash does not leave a half-written file."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def save_run_archive(
    *,
    output_dir: Path,
    messages: list[EmailMessage],
    items: list[DigestItem],
    mark_read_requested: bool,
    model_repo: str,
    model_dir: Path,
    save_bodies: bool,
    device: str,
    max_messages: int,
) -> tuple[Path, dict[str, Any]]:
    """
    Save a timestamped local archive before any Gmail mutation occurs.

    processed_messages.json always contains message metadata and results. Full
    extracted bodies are included only when save_bodies=True.
    """
    local_time = datetime.now().astimezone()
    utc_time = local_time.astimezone(timezone.utc)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _unique_run_dir(output_dir, local_time)
    run_dir.mkdir(parents=False, exist_ok=False)

    result_by_id = {item.message_id: item.to_dict() for item in items}
    processed_records = []

    for message in messages:
        email_record = asdict(message)

        if not save_bodies:
            email_record.pop("body", None)
            # List-Unsubscribe can contain unique per-recipient tokens and is not
            # needed for the run history.
            email_record.pop("list_unsubscribe", None)

        processed_records.append(
            {
                "email": email_record,
                "result": result_by_id.get(message.id),
            }
        )

    manifest: dict[str, Any] = {
        "generated_at_local": local_time.isoformat(),
        "generated_at_utc": utc_time.isoformat(),
        "message_count": len(messages),
        "mark_read_requested": mark_read_requested,
        "mark_read_status": "pending" if mark_read_requested else "not_requested",
        "marked_read_count": 0,
        "model_repo": model_repo,
        "model_dir": str(model_dir),
        "save_bodies": save_bodies,
        "device": device,
        "max_messages": max_messages,
        "files": {
            "digest_markdown": "digest.md",
            "digest_json": "digest.json",
            "processed_messages": "processed_messages.json",
            "manifest": "manifest.json",
        },
    }

    digest_json = json_document(items)
    digest_json["run"] = {
        "generated_at_local": local_time.isoformat(),
        "mark_read_requested": mark_read_requested,
    }
    digest_json["safety"]["raw_bodies_saved"] = save_bodies
    digest_json["safety"]["mailbox_mutations"] = mark_read_requested
    digest_json["safety"]["gmail_access"] = (
        "gmail.modify (mark-read requested)"
        if mark_read_requested
        else "gmail.readonly"
    )

    (run_dir / "digest.md").write_text(
        markdown_digest(items, mark_read_requested=mark_read_requested),
        encoding="utf-8",
    )
    _write_json(run_dir / "digest.json", digest_json)
    _write_json(
        run_dir / "processed_messages.json",
        {
            "generated_at_local": local_time.isoformat(),
            "generated_at_utc": utc_time.isoformat(),
            "contains_raw_email_bodies": save_bodies,
            "messages": processed_records,
        },
    )
    _write_json(run_dir / "manifest.json", manifest)

    return run_dir, manifest


def update_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    _write_json(run_dir / "manifest.json", manifest)
