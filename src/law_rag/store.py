"""ChromaDB multi-collection helpers.

Collections:
    clauses     — one entry per Khoản
    articles    — one entry per Điều (concatenated Khoản summaries)
    documents   — one entry per Văn bản (TOC + summary)
    prototypes  — one entry per (nhom) — mean embedding of all clauses with that nhom
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from . import config
from .schema import ClauseRecord

_client: Optional[chromadb.api.ClientAPI] = None


def client() -> chromadb.api.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def collection(name: str):
    return client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


# --- Metadata serialization ------------------------------------------------

# Chroma metadata only accepts str/int/float/bool. Lists and dicts are JSON-encoded.

_SCALAR_TYPES = (str, int, float, bool)


def _flatten_meta(rec: ClauseRecord) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": rec.id,
        "doc_id": rec.doc_id,
        "summary": rec.summary or "",
        "input_text": rec.input_text or "",
        "reasoning": rec.reasoning or "",
        "nhom": rec.nhom or "",
        "nhom_confidence": float(rec.nhom_confidence or 0.0),
        "nhom_source": rec.nhom_source or "none",
        # path
        "chuong": rec.path.chuong or "",
        "dieu": rec.path.dieu or 0,
        "dieu_title": rec.path.dieu_title or "",
        "khoan": rec.path.khoan or 0,
        # doc meta
        "so_hieu": rec.doc_meta.so_hieu or "",
        "co_quan_ban_hanh": rec.doc_meta.co_quan_ban_hanh or "",
        "ngay_ban_hanh": rec.doc_meta.ngay_ban_hanh or "",
        # normative
        "clause_type": rec.normative.clause_type,
        "modal": ", ".join(rec.normative.modal),
        # JSON-encoded list fields (so we can round-trip them)
        "doi_tuong_json": json.dumps(rec.doi_tuong, ensure_ascii=False),
        "references_json": json.dumps(rec.references, ensure_ascii=False),
        "keywords_json": json.dumps(rec.keywords, ensure_ascii=False),
        "tags_json": json.dumps(rec.tags, ensure_ascii=False),
        "triggers_json": json.dumps(rec.normative.triggers, ensure_ascii=False),
        "actions_json": json.dumps(rec.normative.actions, ensure_ascii=False),
        # Searchable flat copies (BM25 fed)
        "tags_flat": ", ".join(rec.tags),
        "keywords_flat": ", ".join(rec.keywords),
    }
    return {k: v for k, v in out.items() if isinstance(v, _SCALAR_TYPES)}


def text_for_embedding(rec: ClauseRecord) -> str:
    """Concatenate the fields BGE-M3 should see."""
    parts = [
        rec.path.dieu_title or "",
        rec.summary or "",
        rec.input_text or "",
    ]
    return "\n".join(p for p in parts if p)


# --- Upserts ---------------------------------------------------------------


def upsert_clauses(records: List[ClauseRecord], embeddings: np.ndarray) -> None:
    if not records:
        return
    coll = collection(config.COLL_CLAUSES)
    coll.upsert(
        ids=[r.id for r in records],
        embeddings=embeddings.tolist(),
        documents=[text_for_embedding(r) for r in records],
        metadatas=[_flatten_meta(r) for r in records],
    )


def upsert_articles(records: List[ClauseRecord], embeddings: np.ndarray) -> None:
    """Aggregate clauses up to one row per (doc_id, dieu)."""
    if not records:
        return
    by_dieu: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        key = f"{r.doc_id}__D{r.path.dieu}"
        by_dieu.setdefault(key, []).append(i)

    ids, docs, metas, vecs = [], [], [], []
    for key, idxs in by_dieu.items():
        first = records[idxs[0]]
        joined_summary = " ".join((records[i].summary or records[i].input_text)[:300] for i in idxs)
        text = f"{first.path.dieu_title or ''}\n{joined_summary}"
        vec = embeddings[idxs].mean(axis=0)
        ids.append(key)
        docs.append(text)
        vecs.append(vec.tolist())
        metas.append(
            {
                "id": key,
                "doc_id": first.doc_id,
                "dieu": first.path.dieu or 0,
                "dieu_title": first.path.dieu_title or "",
                "chuong": first.path.chuong or "",
                "so_hieu": first.doc_meta.so_hieu or "",
                "n_khoan": len(idxs),
                "summary": joined_summary[:1000],
            }
        )
    coll = collection(config.COLL_ARTICLES)
    coll.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)


def upsert_document(records: List[ClauseRecord], embeddings: np.ndarray) -> None:
    if not records:
        return
    first = records[0]
    doc_id = first.doc_id
    toc = " | ".join(
        sorted({f"Đ.{r.path.dieu}: {r.path.dieu_title or ''}" for r in records if r.path.dieu})
    )
    text = f"{first.doc_meta.so_hieu}\n{toc}"
    vec = embeddings.mean(axis=0).tolist()
    coll = collection(config.COLL_DOCUMENTS)
    coll.upsert(
        ids=[doc_id],
        embeddings=[vec],
        documents=[text],
        metadatas=[
            {
                "id": doc_id,
                "doc_id": doc_id,
                "so_hieu": first.doc_meta.so_hieu or "",
                "co_quan_ban_hanh": first.doc_meta.co_quan_ban_hanh or "",
                "ngay_ban_hanh": first.doc_meta.ngay_ban_hanh or "",
                "n_clauses": len(records),
                "toc": toc[:2000],
            }
        ],
    )


def rebuild_prototypes() -> int:
    """Recompute one prototype per distinct `nhom` from the clauses collection.
    Returns number of prototypes written."""
    coll_clauses = collection(config.COLL_CLAUSES)
    coll_proto = collection(config.COLL_PROTOTYPES)

    # Wipe prototypes
    try:
        existing = coll_proto.get()
        if existing["ids"]:
            coll_proto.delete(ids=existing["ids"])
    except Exception:
        pass

    data = coll_clauses.get(include=["embeddings", "metadatas"])
    if not data["ids"]:
        return 0

    by_nhom: Dict[str, List[int]] = {}
    for i, meta in enumerate(data["metadatas"]):
        nhom = (meta or {}).get("nhom") or ""
        if nhom:
            by_nhom.setdefault(nhom, []).append(i)

    if not by_nhom:
        return 0

    embs = np.asarray(data["embeddings"], dtype=np.float32)
    ids, vecs, metas, docs = [], [], [], []
    for nhom, idxs in by_nhom.items():
        vec = embs[idxs].mean(axis=0)
        ids.append(f"proto::{nhom}")
        vecs.append(vec.tolist())
        docs.append(nhom)
        metas.append({"nhom": nhom, "n_examples": len(idxs)})
    coll_proto.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)
    return len(ids)


def set_nhom(clause_id: str, nhom: str, source: str = "manual", confidence: float = 1.0) -> None:
    coll = collection(config.COLL_CLAUSES)
    rec = coll.get(ids=[clause_id], include=["metadatas"])
    if not rec["ids"]:
        raise KeyError(clause_id)
    meta = rec["metadatas"][0] or {}
    meta["nhom"] = nhom
    meta["nhom_source"] = source
    meta["nhom_confidence"] = float(confidence)
    coll.update(ids=[clause_id], metadatas=[meta])
