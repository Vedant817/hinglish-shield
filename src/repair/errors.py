"""Fail-closed errors for the repair workflow.

Messages stay free of prompt text.
"""
from __future__ import annotations


class RepairError(Exception):
    """Raised when a repair gate refuses to continue."""

    def __init__(self, message: str, *, stage: str = "protocol") -> None:
        super().__init__(message)
        self.stage = stage
