"""
config.py — Centralised configuration for the Investment Analyst Agent.
Loads environment variables and exposes typed settings.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (same directory as this file)
load_dotenv(Path(__file__).parent / ".env")


class Settings:
    # ── Ollama (local LLM — free, no API key) ───────────────────────────────
    # Install : https://ollama.com/download
    # Pull    : ollama pull granite3.3:8b
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "granite3.3:8b")

    # ── LLM inference parameters ─────────────────────────────────────────────
    LLM_MAX_NEW_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.1      # Low temp → factual, deterministic output

    # ── News API (optional) ──────────────────────────────────────────────────
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "")

    # ── RAG / Vector store ───────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.environ.get(
        "CHROMA_PERSIST_DIR", "./data/chroma_db"
    )
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # runs locally, no API key
    RAG_CHUNK_SIZE: int = 512
    RAG_CHUNK_OVERLAP: int = 64
    RAG_TOP_K: int = 5

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        """Check Ollama is reachable (best-effort, non-blocking)."""
        import urllib.request
        try:
            urllib.request.urlopen(self.OLLAMA_BASE_URL, timeout=3)
        except Exception:
            raise EnvironmentError(
                f"Ollama does not appear to be running at {self.OLLAMA_BASE_URL}.\n"
                "Start it with: ollama serve\n"
                f"Then pull the model: ollama pull {self.OLLAMA_MODEL}"
            )


settings = Settings()
