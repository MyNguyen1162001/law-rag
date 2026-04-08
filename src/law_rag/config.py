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


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3").strip()

CHROMA_DIR = _path("CHROMA_DIR", "./chroma_db")
JSON_DIR = _path("JSON_DIR", "./data/json")
BM25_DIR = _path("BM25_DIR", "./data/bm25")
INBOX_DIR = _path("INBOX_DIR", "./inbox")
PROCESSED_DIR = _path("PROCESSED_DIR", "./processed")

# Collection names
COLL_CLAUSES = "clauses"
COLL_ARTICLES = "articles"
COLL_DOCUMENTS = "documents"
COLL_PROTOTYPES = "prototypes"
