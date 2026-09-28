from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .types import EmailMessage


READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
SCOPES = [READONLY_SCOPE]

# Explicitly reject credentials that contain mailbox-changing Gmail scopes.
FORBIDDEN_GMAIL_SCOPES = {
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.insert",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/gmail.settings.basic",
    "https://www.googleapis.com/auth/gmail.settings.sharing",
}


def _credential_scope_set(creds: Credentials) -> set[str]:
    scopes = getattr(creds, "granted_scopes", None) or getattr(creds, "scopes", None) or []
    return set(scopes)


def _assert_readonly_credentials(creds: Credentials) -> None:
    scopes = _credential_scope_set(creds)
    forbidden = scopes & FORBIDDEN_GMAIL_SCOPES
    if forbidden:
        raise RuntimeError(
            "Refusing to run: OAuth credentials contain Gmail write-capable scope(s): "
            + ", ".join(sorted(forbidden))
        )
    if scopes and READONLY_SCOPE not in scopes:
        raise RuntimeError(
            "Refusing to run: OAuth credentials do not contain the required gmail.readonly scope."
        )


def authenticate_readonly(
    credentials_path: Path,
    token_path: Path,
):
    """Authenticate using a dedicated token that requests ONLY gmail.readonly."""
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        _assert_readonly_credentials(creds)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _assert_readonly_credentials(creds)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path),
                SCOPES,
            )
            creds = flow.run_local_server(port=0)
            _assert_readonly_credentials(creds)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        try:
            token_path.chmod(0o600)
        except OSError:
            pass

    # Discovery client only; the credential itself makes mutation calls unauthorized.
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _decode_b64url(data: str | None) -> str:
    if not data:
        return ""
    try:
        raw = base64.urlsafe_b64decode(data.encode("ascii"))
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_text(payload: dict[str, Any]) -> str:
    """
    Prefer text/plain. Fall back to text/html as plain source text.
    Recurses through nested multipart messages.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")

        if mime == "text/plain" and data:
            plain_parts.append(_decode_b64url(data))
        elif mime == "text/html" and data:
            html_parts.append(_decode_b64url(data))

        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload)

    text = "\n".join(x for x in plain_parts if x.strip())
    if text.strip():
        return text

    # We intentionally do not fetch attachment bodies in this MVP.
    # HTML is kept as source text; the model can still summarize most mail.
    return "\n".join(x for x in html_parts if x.strip())


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in payload.get("headers", []) or []:
        name = str(item.get("name", "")).lower()
        value = str(item.get("value", ""))
        if name:
            result[name] = value
    return result


class ReadOnlyGmail:
    """
    Deliberately tiny Gmail wrapper.

    It exposes only:
      - users.messages.list
      - users.messages.get

    There are no modify, trash, delete, send, label, archive, or mark-read methods.
    """

    def __init__(self, service):
        self._service = service

    def unread_inbox(self, max_messages: int = 30) -> list[EmailMessage]:
        if max_messages < 1 or max_messages > 500:
            raise ValueError("max_messages must be between 1 and 500")

        result = (
            self._service.users()
            .messages()
            .list(
                userId="me",
                labelIds=["INBOX"],
                q="is:unread",
                maxResults=max_messages,
                includeSpamTrash=False,
            )
            .execute()
        )

        messages: list[EmailMessage] = []
        for ref in result.get("messages", []) or []:
            raw = (
                self._service.users()
                .messages()
                .get(
                    userId="me",
                    id=ref["id"],
                    format="full",
                )
                .execute()
            )
            payload = raw.get("payload", {})
            headers = _headers(payload)
            messages.append(
                EmailMessage(
                    id=str(raw["id"]),
                    thread_id=str(raw.get("threadId", "")),
                    sender=headers.get("from", ""),
                    subject=headers.get("subject", "(no subject)"),
                    date=headers.get("date", ""),
                    snippet=str(raw.get("snippet", "")),
                    body=_extract_text(payload),
                    list_unsubscribe=headers.get("list-unsubscribe", ""),
                    precedence=headers.get("precedence", ""),
                    auto_submitted=headers.get("auto-submitted", ""),
                )
            )

        return messages
