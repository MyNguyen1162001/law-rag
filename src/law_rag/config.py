"""Loads environment configuration and exposes paths/model names as constants."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _path(env_key: str, default: str) -> Path:
    p = Path(os.getenv(env_key, default))
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# LLM provider selection: "gemini" | "openrouter" | "ollama" | "" (auto)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()

# Gemini API (if using Google)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# OpenRouter API (if using OpenRouter)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

# Ollama API (local or remote Ollama-compatible server)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://exporter-unopened-remover.ngrok-free.dev").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4-e2b-32k").strip()

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3").strip()

QDRANT_DIR = _path("QDRANT_DIR", "./qdrant_db")
JSON_DIR = _path("JSON_DIR", "./data/json")
BM25_DIR = _path("BM25_DIR", "./data/bm25")
INBOX_DIR = _path("INBOX_DIR", "./inbox")
PROCESSED_DIR = _path("PROCESSED_DIR", "./processed")

# Collection names
COLL_CLAUSES = "clauses"
COLL_ARTICLES = "articles"
COLL_DOCUMENTS = "documents"
COLL_PROTOTYPES = "prototypes"
