"""Tests are deterministic and offline by default (Groq disabled). Run with DISHA_LIVE_TESTS=1 to hit Groq."""
import os

import pytest

from app import config, groq_client


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    if not os.getenv("DISHA_LIVE_TESTS"):
        monkeypatch.setattr(config, "GROQ_API_KEY", "")
        monkeypatch.setattr(groq_client, "_client", None)
