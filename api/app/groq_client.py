"""The only place that talks to an AI provider (Groq). chat_json / transcribe never raise: they return None."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Optional, Type

from pydantic import BaseModel, ValidationError

from . import config

log = logging.getLogger("disha.groq")
TIMEOUT_S = 8
RETRIES = 2
_client = None
# If the configured model is not available to the account (404), fall through these in order. Remembered for the process lifetime.
MODEL_FALLBACKS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
_model = {"name": None}
_cache: "OrderedDict[str, dict]" = OrderedDict()
CACHE_SIZE = 512
stats: list[tuple[str, float]] = []        # (call name, seconds) for latency reports

# Opt-in pacing for batch scripts (free tier = 8000 tokens/min). Interactive API leaves this at 0 and falls back instead of waiting.
PACE_TPM = 0
_window: list[tuple[float, int]] = []
_pace_lock = threading.Lock()


def _pace(est_tokens: int):
    if not PACE_TPM:
        return
    while True:
        with _pace_lock:
            now = time.time()
            _window[:] = [(t, n) for t, n in _window if now - t < 60]
            if sum(n for _, n in _window) + est_tokens <= PACE_TPM:
                _window.append((now, est_tokens))
                return
            wait = 60 - (now - _window[0][0]) + 0.2
        time.sleep(max(wait, 0.5))


def _get_client():
    global _client
    if _client is None and config.GROQ_API_KEY:
        from groq import Groq
        _client = Groq(api_key=config.GROQ_API_KEY, timeout=TIMEOUT_S, max_retries=0)
    return _client


def current_model() -> str:
    return _model["name"] or config.GROQ_LLM_MODEL


def _next_model() -> bool:
    """Switch to the next fallback model after a 'model not found' error. False when none is left."""
    tried = {current_model(), config.GROQ_LLM_MODEL}
    for m in MODEL_FALLBACKS:
        if m not in tried:
            log.warning("groq model %s unavailable -> falling back to %s", current_model(), m)
            _model["name"] = m
            return True
    return False


def available() -> bool:
    return bool(config.GROQ_API_KEY)


def chat_json(system: str, user: str, model_cls: Optional[Type[BaseModel]] = None, name: str = "chat",
              temperature: float = 0.0) -> Optional[dict]:
    """JSON-mode chat completion. Validates with `model_cls` if given. Returns None on any failure."""
    client = _get_client()
    if client is None:
        return None
    key = hashlib.sha1(f"{system}\x00{user}\x00{temperature}".encode()).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    retries = RETRIES + (4 if PACE_TPM else 0)
    for attempt in range(retries + 1):
        _pace((len(system) + len(user)) // 3 + 450)
        t0 = time.perf_counter()
        try:
            extra = {"reasoning_effort": "low"} if "gpt-oss" in current_model() else {}   # reasoning models: keep latency down
            r = client.chat.completions.create(
                model=current_model(), temperature=temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                seed=7,   # best-effort reproducibility (reasoning models are not fully deterministic)
                **extra,
            )
            data = json.loads(r.choices[0].message.content)
            if model_cls is not None:
                data = model_cls.model_validate(data).model_dump()
            dt = time.perf_counter() - t0
            stats.append((name, dt))
            u = getattr(r, "usage", None)
            log.info("groq %s ok %.2fs tokens=%s/%s", name, dt, getattr(u, "prompt_tokens", "?"), getattr(u, "completion_tokens", "?"))
            _cache[key] = data
            if len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
            return data
        except ValidationError as e:
            log.warning("groq %s: invalid JSON shape: %s", name, str(e)[:200])
            return None
        except Exception as e:   # network, 429, 5xx, bad JSON
            status = getattr(e, "status_code", None)
            if status in (400, 404) and "model" in str(e).lower() and ("does not exist" in str(e) or "decommission" in str(e) or "not found" in str(e).lower()) and _next_model():
                continue                      # retry at once with the next model
            retryable = status is None or status == 429 or status >= 500
            log.warning("groq %s failed (attempt %d, status=%s): %s", name, attempt + 1, status, str(e)[:160])
            if not retryable or attempt == retries:
                return None
            time.sleep(12 if (PACE_TPM and status == 429) else 0.4 * (2 ** min(attempt, 2)))
    return None


def transcribe(file_bytes: bytes, filename: str) -> Optional[tuple[str, str]]:
    """Groq Whisper -> (text, language) or None."""
    client = _get_client()
    if client is None:
        return None
    t0 = time.perf_counter()
    try:
        r = client.audio.transcriptions.create(
            file=(filename, file_bytes), model=config.GROQ_STT_MODEL, response_format="verbose_json", temperature=0)
        stats.append(("transcribe", time.perf_counter() - t0))
        lang = {"hindi": "hi", "english": "en"}.get(str(getattr(r, "language", "")).lower(), str(getattr(r, "language", "") or "hi"))
        return r.text.strip(), lang
    except Exception as e:
        log.warning("transcribe failed: %s", str(e)[:200])
        return None


_health = {"t": 0.0, "ok": False}


def ping() -> bool:
    """Cheap liveness check (cached 60 s). False if no key or API unreachable."""
    if not available():
        return False
    if time.time() - _health["t"] < 60:
        return _health["ok"]
    try:
        _get_client().models.list()
        _health.update(t=time.time(), ok=True)
    except Exception:
        _health.update(t=time.time(), ok=False)
    return _health["ok"]
