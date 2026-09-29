from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .types import EmailMessage


READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"
FULL_MAIL_SCOPE = "https://mail.google.com/"


# Scopes that this project never needs. gmail.modify is allowed only for the
# explicit --mark-read mode.
NEVER_ALLOWED_SCOPES = {
    FULL_MAIL_SCOPE,
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


def _token_file_scopes(token_path: Path) -> set[str]:
    """Read stored scopes before google-auth can override them with requested scopes."""
    try:
        info = json.loads(token_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    scopes = info.get("scopes", [])
    if isinstance(scopes, str):
        return set(scopes.split())
    if isinstance(scopes, list):
        return {str(scope) for scope in scopes}
    return set()


def _assert_scope_set(scopes: set[str], *, allow_modify: bool) -> None:
    forbidden = scopes & NEVER_ALLOWED_SCOPES
    if forbidden:
        raise RuntimeError(
            "Refusing to run: OAuth credentials contain scope(s) this project does not use: "
            + ", ".join(sorted(forbidden))
        )

    if allow_modify:
        if scopes and MODIFY_SCOPE not in scopes:
            raise RuntimeError(
                "Refusing to run: --mark-read requires a token with gmail.modify scope. "
                "Delete the modify token and authorize again if necessary."
            )
    else:
        if MODIFY_SCOPE in scopes:
            raise RuntimeError(
                "Refusing to use a gmail.modify token for a read-only run. "
                "Use the dedicated read-only token instead."
            )
        if scopes and READONLY_SCOPE not in scopes:
            raise RuntimeError(
                "Refusing to run: OAuth credentials do not contain gmail.readonly scope."
            )


def _assert_credentials(creds: Credentials, *, allow_modify: bool) -> None:
    _assert_scope_set(_credential_scope_set(creds), allow_modify=allow_modify)


def authenticate_gmail(
    credentials_path: Path,
    token_path: Path,
    *,
    allow_modify: bool = False,
):
    """
    Authenticate with the narrowest scope needed for this run.

    Normal run: gmail.readonly
    --mark-read run: gmail.modify

    A separate token file should be used for each mode so a normal run never
    silently inherits the broader permission.
    """
    scopes = [MODIFY_SCOPE] if allow_modify else [READONLY_SCOPE]
    creds = None

    if token_path.exists():
        _assert_scope_set(
            _token_file_scopes(token_path),
            allow_modify=allow_modify,
        )
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
        _assert_credentials(creds, allow_modify=allow_modify)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _assert_credentials(creds, allow_modify=allow_modify)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path),
                scopes,
            )
            creds = flow.run_local_server(port=0)
            _assert_credentials(creds, allow_modify=allow_modify)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        try:
            token_path.chmod(0o600)
        except OSError:
            pass

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def authenticate_readonly(credentials_path: Path, token_path: Path):
    """Backward-compatible helper for callers that only need read access."""
    return authenticate_gmail(
        credentials_path,
        token_path,
        allow_modify=False,
    )


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

    # Attachments are intentionally not fetched in this MVP.
    return "\n".join(x for x in html_parts if x.strip())


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in payload.get("headers", []) or []:
        name = str(item.get("name", "")).lower()
        value = str(item.get("value", ""))
        if name:
            result[name] = value
    return result


class GmailClient:
    """
    Small Gmail wrapper.

    Every run can list/get unread messages. Label mutation is available only
    when the client was created with allow_modify=True.
    """

    def __init__(self, service, *, allow_modify: bool = False):
        self._service = service
        self._allow_modify = allow_modify

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

    def mark_as_read(self, message_ids: list[str]) -> int:
        """Remove the Gmail UNREAD label from the supplied processed messages."""
        if not self._allow_modify:
            raise RuntimeError(
                "Mailbox modification is disabled for this Gmail client. "
                "Run with --mark-read to enable it."
            )

        ids = [message_id for message_id in message_ids if message_id]
        if not ids:
            return 0

        # The CLI caps retrieval at 500; Gmail batchModify supports up to 1000
        # IDs per request, so one request is enough for a complete run.
        (
            self._service.users()
            .messages()
            .batchModify(
                userId="me",
                body={
                    "ids": ids,
                    "removeLabelIds": ["UNREAD"],
                },
            )
            .execute()
        )
        return len(ids)


# Backward compatibility for code importing the old class name.
ReadOnlyGmail = GmailClient
