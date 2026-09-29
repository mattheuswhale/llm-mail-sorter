"""Backward-compatible import shim.

New code should import from maildigest.gmail_client. The client is read-only by
default and exposes mark-as-read only when explicitly authorized.
"""

from .gmail_client import *  # noqa: F401,F403
