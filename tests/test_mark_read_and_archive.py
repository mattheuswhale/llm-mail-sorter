from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

# The build/test environment may not have Google's client libraries installed.
# Stub only the imports needed to load the module; Gmail behavior itself is
# exercised with fake service objects below.
google = types.ModuleType("google")
google_auth = types.ModuleType("google.auth")
google_auth_transport = types.ModuleType("google.auth.transport")
google_auth_requests = types.ModuleType("google.auth.transport.requests")
google_oauth2 = types.ModuleType("google.oauth2")
google_oauth2_credentials = types.ModuleType("google.oauth2.credentials")
google_auth_oauthlib = types.ModuleType("google_auth_oauthlib")
google_auth_oauthlib_flow = types.ModuleType("google_auth_oauthlib.flow")
googleapiclient = types.ModuleType("googleapiclient")
googleapiclient_discovery = types.ModuleType("googleapiclient.discovery")


class _DummyRequest:
    pass


class _DummyCredentials:
    pass


class _DummyFlow:
    pass


google_auth_requests.Request = _DummyRequest
google_oauth2_credentials.Credentials = _DummyCredentials
google_auth_oauthlib_flow.InstalledAppFlow = _DummyFlow
googleapiclient_discovery.build = lambda *args, **kwargs: None

sys.modules.setdefault("google", google)
sys.modules.setdefault("google.auth", google_auth)
sys.modules.setdefault("google.auth.transport", google_auth_transport)
sys.modules.setdefault("google.auth.transport.requests", google_auth_requests)
sys.modules.setdefault("google.oauth2", google_oauth2)
sys.modules.setdefault("google.oauth2.credentials", google_oauth2_credentials)
sys.modules.setdefault("google_auth_oauthlib", google_auth_oauthlib)
sys.modules.setdefault("google_auth_oauthlib.flow", google_auth_oauthlib_flow)
sys.modules.setdefault("googleapiclient", googleapiclient)
sys.modules.setdefault("googleapiclient.discovery", googleapiclient_discovery)

from maildigest.archive import save_run_archive
from maildigest.digest import DigestItem
from maildigest.gmail_client import (
    GmailClient,
    MODIFY_SCOPE,
    READONLY_SCOPE,
    _assert_scope_set,
)
from maildigest.types import EmailMessage


class _Request:
    def __init__(self, result=None):
        self.result = result if result is not None else {}

    def execute(self):
        return self.result


class _Messages:
    def __init__(self):
        self.batch_body = None

    def batchModify(self, *, userId, body):
        self.batch_body = (userId, body)
        return _Request({})


class _Users:
    def __init__(self, messages):
        self._messages = messages

    def messages(self):
        return self._messages


class _Service:
    def __init__(self):
        self.messages_resource = _Messages()
        self.users_resource = _Users(self.messages_resource)

    def users(self):
        return self.users_resource


def _message() -> EmailMessage:
    return EmailMessage(
        id="m1",
        thread_id="t1",
        sender="a@example.com",
        subject="Hello",
        date="date",
        snippet="snippet",
        body="private body",
        list_unsubscribe="https://example.com/u/secret-token",
    )


def _item() -> DigestItem:
    return DigestItem(
        message_id="m1",
        sender="a@example.com",
        subject="Hello",
        date="date",
        priority="normal",
        category="personal",
        action_required=False,
        deadline=None,
        summary="A hello message.",
        reason="No action.",
        rule="routine_information",
        signals={},
    )


class ScopeTests(unittest.TestCase):
    def test_readonly_mode_rejects_modify_scope(self):
        with self.assertRaises(RuntimeError):
            _assert_scope_set({MODIFY_SCOPE}, allow_modify=False)

    def test_mark_read_mode_accepts_modify_scope(self):
        _assert_scope_set({MODIFY_SCOPE}, allow_modify=True)

    def test_mark_read_mode_rejects_readonly_only_scope(self):
        with self.assertRaises(RuntimeError):
            _assert_scope_set({READONLY_SCOPE}, allow_modify=True)


class GmailMutationTests(unittest.TestCase):
    def test_mark_as_read_removes_only_unread_label(self):
        service = _Service()
        gmail = GmailClient(service, allow_modify=True)
        count = gmail.mark_as_read(["m1", "m2"])

        self.assertEqual(count, 2)
        self.assertEqual(
            service.messages_resource.batch_body,
            (
                "me",
                {
                    "ids": ["m1", "m2"],
                    "removeLabelIds": ["UNREAD"],
                },
            ),
        )

    def test_readonly_client_refuses_mark_as_read(self):
        gmail = GmailClient(_Service(), allow_modify=False)
        with self.assertRaises(RuntimeError):
            gmail.mark_as_read(["m1"])


class ArchiveTests(unittest.TestCase):
    def test_default_archive_excludes_full_body_and_unsubscribe_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, manifest = save_run_archive(
                output_dir=Path(tmp),
                messages=[_message()],
                items=[_item()],
                mark_read_requested=True,
                model_repo="test/model",
                model_dir=Path("models/test"),
                save_bodies=False,
                device="cpu",
                max_messages=30,
            )

            self.assertTrue((run_dir / "digest.md").exists())
            self.assertTrue((run_dir / "digest.json").exists())
            self.assertTrue((run_dir / "processed_messages.json").exists())
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertEqual(manifest["mark_read_status"], "pending")
            self.assertFalse(manifest["save_bodies"])

            processed = json.loads(
                (run_dir / "processed_messages.json").read_text(encoding="utf-8")
            )
            email = processed["messages"][0]["email"]
            self.assertNotIn("body", email)
            self.assertNotIn("list_unsubscribe", email)
            self.assertEqual(
                processed["messages"][0]["result"]["priority"],
                "normal",
            )

    def test_save_bodies_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, manifest = save_run_archive(
                output_dir=Path(tmp),
                messages=[_message()],
                items=[_item()],
                mark_read_requested=False,
                model_repo="test/model",
                model_dir=Path("models/test"),
                save_bodies=True,
                device="cpu",
                max_messages=30,
            )

            self.assertTrue(manifest["save_bodies"])
            processed = json.loads(
                (run_dir / "processed_messages.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                processed["messages"][0]["email"]["body"],
                "private body",
            )


if __name__ == "__main__":
    unittest.main()
