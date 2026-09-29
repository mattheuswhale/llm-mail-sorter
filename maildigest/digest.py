
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Iterable, Any

from .types import Classification, EmailMessage


PRIORITY_RANK = {
    "urgent": 0,
    "high": 1,
    "normal": 2,
    "low": 3,
}


@dataclass
class DigestItem:
    message_id: str
    sender: str
    subject: str
    date: str
    priority: str
    category: str
    action_required: bool
    deadline: str | None
    summary: str
    reason: str
    rule: str
    signals: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_item(email: EmailMessage, classification: Classification) -> DigestItem:
    return DigestItem(
        message_id=email.id,
        sender=email.sender,
        subject=email.subject,
        date=email.date,
        priority=classification.priority,
        category=classification.category,
        action_required=classification.action_required,
        deadline=classification.deadline,
        summary=classification.summary,
        reason=classification.reason,
        rule=classification.rule,
        signals=classification.signals.to_dict(),
    )


def sort_items(items: Iterable[DigestItem]) -> list[DigestItem]:
    return sorted(
        items,
        key=lambda x: (
            PRIORITY_RANK.get(x.priority, 2),
            0 if x.action_required else 1,
        ),
    )


def json_document(items: list[DigestItem]) -> dict[str, Any]:
    counts = {p: 0 for p in PRIORITY_RANK}
    for item in items:
        counts[item.priority] = counts.get(item.priority, 0) + 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety": {
            "gmail_access": "read-only",
            "mailbox_mutations": False,
            "raw_bodies_saved": False,
            "llm_inference": "local",
            "priority_decision": "deterministic_python_rules",
        },
        "counts": counts,
        "messages": [item.to_dict() for item in items],
    }


def markdown_digest(
    items: list[DigestItem],
    *,
    mark_read_requested: bool = False,
) -> str:
    groups = {p: [] for p in ("urgent", "high", "normal", "low")}
    for item in items:
        groups.setdefault(item.priority, []).append(item)

    if mark_read_requested:
        status_line = (
            "> This run was configured to mark processed messages as read after "
            "the local archive was written. No other mailbox changes are performed."
        )
    else:
        status_line = (
            "> Read-only report. No message was marked read, labeled, archived, "
            "deleted, replied to, or otherwise modified."
        )

    lines = [
        "# Unread Gmail Digest",
        "",
        status_line,
        "",
    ]

    headings = {
        "urgent": "🔴 Urgent",
        "high": "🟠 High",
        "normal": "⚪ Normal",
        "low": "🔵 Low",
    }

    for priority in ("urgent", "high", "normal", "low"):
        group = groups.get(priority, [])
        lines.append(f"## {headings[priority]} ({len(group)})")
        lines.append("")
        if not group:
            lines.append("_None._")
            lines.append("")
            continue

        for item in group:
            action = " — **Action likely required**" if item.action_required else ""
            lines.append(f"### {item.subject}{action}")
            lines.append(f"- **From:** {item.sender or '(unknown)'}")
            lines.append(f"- **Category:** {item.category}")
            if item.deadline:
                lines.append(f"- **Possible deadline:** {item.deadline}")
            lines.append(f"- **Summary:** {item.summary}")
            lines.append(f"- **Why this priority:** {item.reason}")
            lines.append(f"- **Rule:** `{item.rule}`")
            lines.append("")

    lines.extend([
        "---",
        "The LLM extracts facts and writes summaries locally; priority is assigned by deterministic Python rules. Review important messages manually.",
        "",
    ])
    return "\n".join(lines)
