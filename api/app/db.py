"""Optional Supabase persistence. Always writes to an in-process store too, so every app feature
works when Supabase is down/unconfigured. /triage never touches this module."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from . import config

log = logging.getLogger("disha.db")
_mem: dict[str, dict[str, dict]] = {}
_client = None
_tried = False


def client():
    global _client, _tried
    if not _tried:
        _tried = True
        if config.SUPABASE_URL and config.SUPABASE_SERVICE_ROLE_KEY:
            try:
                from supabase import create_client
                _client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
            except Exception as e:
                log.warning("supabase init failed: %s", e)
    return _client


def configured() -> bool:
    return client() is not None


def healthy() -> bool:
    c = client()
    if c is None:
        return False
    try:
        c.table("families").select("id").limit(1).execute()
        return True
    except Exception:
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert(table: str, row: dict) -> dict:
    row = {"id": str(uuid.uuid4()), "created_at": _now(), **row}
    _mem.setdefault(table, {})[row["id"]] = row
    c = client()
    if c:
        try:
            c.table(table).insert(row).execute()
        except Exception as e:
            log.warning("supabase insert %s failed: %s", table, str(e)[:160])
    return row


def update(table: str, id: str, patch: dict) -> Optional[dict]:
    row = get(table, id)
    if row is None:
        return None
    row = {**row, **patch}
    _mem.setdefault(table, {})[id] = row
    c = client()
    if c:
        try:
            c.table(table).update(patch).eq("id", id).execute()
        except Exception as e:
            log.warning("supabase update %s failed: %s", table, str(e)[:160])
    return row


def get(table: str, id: str) -> Optional[dict]:
    if id in _mem.get(table, {}):
        return _mem[table][id]
    c = client()
    if c:
        try:
            r = c.table(table).select("*").eq("id", id).limit(1).execute().data
            if r:
                _mem.setdefault(table, {})[id] = r[0]
                return r[0]
        except Exception:
            pass
    return None


def select(table: str, **eq) -> list[dict]:
    rows = [r for r in _mem.get(table, {}).values() if all(r.get(k) == v for k, v in eq.items())]
    return sorted(rows, key=lambda r: r.get("created_at", ""))
