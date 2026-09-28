
from __future__ import annotations

from dataclasses import dataclass

from .types import Classification, EmailMessage, ExtractedSignals


PROMOTION_MARKERS = (
    "% off",
    "sale",
    "discount",
    "coupon",
    "promo code",
    "shop now",
    "limited time",
    "last chance",
    "clearance",
    "special offer",
    "new arrivals",
    "free shipping",
    "save up to",
)

NEWSLETTER_MARKERS = (
    "unsubscribe",
    "view in browser",
    "weekly newsletter",
    "daily newsletter",
    "weekly digest",
)

SECURITY_MARKERS = (
    "security alert",
    "unauthorized",
    "suspicious sign-in",
    "suspicious login",
    "unusual activity",
    "account locked",
)

PAYMENT_PROBLEM_MARKERS = (
    "payment failed",
    "card declined",
    "billing failed",
    "past due",
    "overdue invoice",
    "unable to process payment",
)

ACCOUNT_PROBLEM_MARKERS = (
    "account suspended",
    "account locked",
    "verification required",
    "service suspended",
    "account disabled",
)

OUTAGE_MARKERS = (
    "service outage",
    "production outage",
    "major incident",
    "service unavailable",
)


@dataclass(frozen=True)
class HardSignals:
    promotion: bool
    newsletter: bool
    automated: bool
    security_issue: bool
    payment_problem: bool
    account_problem: bool
    service_outage: bool


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def detect_hard_signals(email: EmailMessage) -> HardSignals:
    text = (
        f"{email.subject}\n{email.snippet}\n{email.body[:4000]}"
    ).lower()

    precedence = email.precedence.lower()
    auto_submitted = email.auto_submitted.lower()

    return HardSignals(
        promotion=_contains_any(text, PROMOTION_MARKERS),
        newsletter=(
            bool(email.list_unsubscribe)
            or precedence in {"bulk", "list"}
            or _contains_any(text, NEWSLETTER_MARKERS)
        ),
        automated=(
            auto_submitted not in {"", "no"}
            or precedence in {"bulk", "list", "junk"}
        ),
        security_issue=_contains_any(text, SECURITY_MARKERS),
        payment_problem=_contains_any(text, PAYMENT_PROBLEM_MARKERS),
        account_problem=_contains_any(text, ACCOUNT_PROBLEM_MARKERS),
        service_outage=_contains_any(text, OUTAGE_MARKERS),
    )


def classify_priority(
    email: EmailMessage,
    signals: ExtractedSignals,
) -> Classification:
    """
    Deterministic ordered rules.

    The LLM extracts facts and writes the summary; it does not decide priority.
    Obvious text/header evidence can override an inconsistent tiny-model signal.
    """
    hard = detect_hard_signals(email)

    promotion = hard.promotion or signals.is_promotion
    newsletter = hard.newsletter or signals.is_newsletter
    automated = hard.automated or signals.is_automated

    security_issue = hard.security_issue or signals.security_issue
    payment_problem = hard.payment_problem or signals.payment_problem
    account_problem = hard.account_problem or signals.account_problem
    service_outage = hard.service_outage or signals.service_outage

    critical = security_issue or payment_problem or account_problem or service_outage

    # Critical operational/account problems override marketing/newsletter evidence.
    if critical:
        if security_issue or account_problem:
            category = "account_security"
        elif payment_problem:
            category = "finance"
        else:
            category = signals.category

        return Classification(
            priority="urgent",
            category=category,
            action_required=True,
            deadline=signals.deadline,
            summary=signals.summary,
            reason="Security, account, payment, or service problem detected.",
            rule="critical_problem",
            signals=signals,
        )

    # Marketing and newsletters are always low unless caught by critical rules above.
    # This deliberately runs BEFORE deadline logic so "sale ends tonight" cannot
    # become urgent just because a tiny model mistakes commercial urgency for a deadline.
    if promotion or newsletter:
        category = "promotion" if promotion else "newsletter"
        return Classification(
            priority="low",
            category=category,
            action_required=False,
            deadline=None,
            summary=signals.summary,
            reason="Promotional or newsletter content is low priority by policy.",
            rule="promotion_or_newsletter",
            signals=signals,
        )

    # A real immediate deadline is urgent only when the mail also expects action/reply.
    if signals.deadline_is_immediate and (
        signals.asks_for_action or signals.asks_for_reply
    ):
        return Classification(
            priority="urgent",
            category=signals.category,
            action_required=True,
            deadline=signals.deadline,
            summary=signals.summary,
            reason="A non-promotional action or reply is explicitly due immediately.",
            rule="immediate_action_deadline",
            signals=signals,
        )

    # Explicit non-promotional action/reply/deadline => high.
    if signals.asks_for_action or signals.asks_for_reply or signals.has_deadline:
        return Classification(
            priority="high",
            category=signals.category,
            action_required=True,
            deadline=signals.deadline,
            summary=signals.summary,
            reason="The message requests a non-promotional action/reply or has an action deadline.",
            rule="action_required",
            signals=signals,
        )

    # Pure automation with no user action is low.
    if automated:
        return Classification(
            priority="low",
            category=signals.category if signals.category != "other" else "notification",
            action_required=False,
            deadline=None,
            summary=signals.summary,
            reason="Automated message with no required action.",
            rule="automated_no_action",
            signals=signals,
        )

    # Everything else is routine informational mail.
    return Classification(
        priority="normal",
        category=signals.category,
        action_required=False,
        deadline=signals.deadline,
        summary=signals.summary,
        reason="No critical issue or required action was detected.",
        rule="routine_information",
        signals=signals,
    )
