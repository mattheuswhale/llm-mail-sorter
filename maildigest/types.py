
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class EmailMessage:
    id: str
    thread_id: str
    sender: str
    subject: str
    date: str
    snippet: str
    body: str
    list_unsubscribe: str = ""
    precedence: str = ""
    auto_submitted: str = ""


@dataclass
class ExtractedSignals:
    # Content type / provenance
    is_promotion: bool = False
    is_newsletter: bool = False
    is_automated: bool = False
    from_person: bool = False

    # User-action signals
    asks_for_reply: bool = False
    asks_for_action: bool = False
    has_deadline: bool = False
    deadline_is_immediate: bool = False

    # Critical problem signals
    security_issue: bool = False
    payment_problem: bool = False
    account_problem: bool = False
    service_outage: bool = False

    # Descriptive fields
    category: str = "other"
    deadline: str | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Classification:
    priority: str
    category: str
    action_required: bool
    deadline: str | None
    summary: str
    reason: str
    rule: str
    signals: ExtractedSignals

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
