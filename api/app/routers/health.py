from fastapi import APIRouter

from .. import db, groq_client

router = APIRouter()


@router.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "groq": "ok" if groq_client.ping() else "down",
            "supabase": ("ok" if db.healthy() else "down") if db.configured() else "off"}
