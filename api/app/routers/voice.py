from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import groq_client
from ..schemas import TranscribeResponse

router = APIRouter()


@router.post("/v1/transcribe", response_model=TranscribeResponse, tags=["app"], summary="Speech to text (Groq Whisper)")
async def transcribe(audio: UploadFile = File(..., description="webm / ogg / wav / m4a")):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio")
    r = groq_client.transcribe(data, audio.filename or "audio.webm")
    if r is None:
        raise HTTPException(503, "transcription unavailable (Groq down or not configured) - please type instead")
    return TranscribeResponse(text=r[0], language=r[1])
