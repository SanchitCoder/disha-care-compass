# Disha — working notes for Claude

Hackathon prototype: routes home-healthcare requests (EN/HI/Hinglish, text/voice) to care type + urgency + ranked caregivers.
See README.md for architecture and commands. Hard rules:
- Route only. Never output disease names as conclusions, medicine names or doses (guardrail: `api/app/modules/explain.py::is_safe`).
- All AI via Groq (`api/app/groq_client.py` is the only caller). TTS is the browser's speechSynthesis.
- Red flags are rules, never the LLM (`api/app/modules/red_flags.py`, `api/config/red_flags.yaml`). Under-triage is the worst failure; when unsure choose the more urgent option.
- `POST /triage` must work with Supabase down and returns exactly the contract fields (`extra="forbid"`).
- Organiser data lives in `data/` (Sanocare synthetic dataset v1: JSONL requests, caregivers.csv). Format specifics are only in `api/app/data_loader.py`; label definitions and matching rules in `api/config/taxonomy.yaml`.
- No Docker. Secrets only in `.env` (commit `.env.example` only).
- Tests: `cd api && .venv/bin/python -m pytest`. Frontend check: `cd web && npx tsc --noEmit && npx vite build`.
