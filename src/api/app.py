"""FastAPI application."""

import os
import sys
from pathlib import Path

# Windows consoles default to cp1252 and crash printing non-Latin chars
# (common in PDFs). Force UTF-8 with a safe fallback for log output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import BASE_DIR
from .routes import router

UI_DIST = BASE_DIR / "ui" / "dist"

app = FastAPI(title="SimpleRAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"service": "SimpleRAG", "docs": "/docs", "health": "/api/health"}


# Serve the built React frontend if present (production mode).
if UI_DIST.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIST), html=True), name="ui")


def run():
    import uvicorn

    from ..config import API_HOST, API_PORT

    uvicorn.run("src.api.app:app", host=API_HOST, port=API_PORT, reload=False)