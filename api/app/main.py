import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import data_loader
from .modules import matching, red_flags
from .modules.retrieval import get_retriever
from .routers import booking, health, nurse, safety_net, triage, voice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(
    title="Disha - home-healthcare routing API",
    version="1.0.0",
    description="Routes messy Hindi/English/Hinglish requests to a care type, urgency and best caregivers. "
                "Routes only: never diagnoses or prescribes. In an emergency call 112 / 108.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def warm():
    """Load data + fit retrieval into memory so POST /triage needs no database."""
    data_loader.load_taxonomy()
    data_loader.load_caregivers()
    red_flags.load_flags()
    get_retriever()
    matching.role_map()


for r in (triage.router, voice.router, booking.router, safety_net.router, nurse.router, health.router):
    app.include_router(r)
