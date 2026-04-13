"""Qdrant multi-collection helpers.

Collections:
    clauses     — one entry per Khoản
    articles    — one entry per Điều (concatenated Khoản summaries)
    documents   — one entry per Văn bản (TOC + summary)
    prototypes  — one entry per (nhom) — mean embedding of all clauses with that nhom
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient, models

from . import config
from .schema import ClauseRecord

_VECTOR_DIM = 1024  # BGE-M3 dense dim
_NS = uuid.NAMESPACE_DNS  # namespace for deterministic UUIDs

_client: Optional[QdrantClient] = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(config.QDRANT_DIR))
    return _client


def _uuid(string_id: str) -> str:
    """Deterministic UUID from a string ID."""
    return str(uuid.uuid5(_NS, string_id))


def _ensure_collection(name: str) -> None:
    """Create collection if it doesn't exist."""
    c = client()
    if not c.collection_exists(name):
        c.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=_VECTOR_DIM,
                distance=models.Distance.COSINE,
            ),
        )


# --- Metadata serialization ------------------------------------------------

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
        # JSON-encoded list fields
        "doi_tuong_json": json.dumps(rec.doi_tuong, ensure_ascii=False),
        "references_json": json.dumps(rec.references, ensure_ascii=False),
        "keywords_json": json.dumps(rec.keywords, ensure_ascii=False),
        "tags_json": json.dumps(rec.tags, ensure_ascii=False),
        "triggers_json": json.dumps(rec.normative.triggers, ensure_ascii=False),
        "actions_json": json.dumps(rec.normative.actions, ensure_ascii=False),
        # Searchable flat copies
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


def insert_clauses(records: List[ClauseRecord], embeddings: np.ndarray) -> None:
    if not records:
        return
    _ensure_collection(config.COLL_CLAUSES)
    points = [
        models.PointStruct(
            id=_uuid(r.id),
            vector=embeddings[i].tolist(),
            payload={**_flatten_meta(r), "_document": text_for_embedding(r)},
        )
        for i, r in enumerate(records)
    ]
    client().upload_points(collection_name=config.COLL_CLAUSES, points=points)


def insert_articles(records: List[ClauseRecord], embeddings: np.ndarray) -> None:
    """Aggregate clauses up to one row per (doc_id, dieu)."""
    if not records:
        return
    _ensure_collection(config.COLL_ARTICLES)
    by_dieu: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        key = f"{r.doc_id}__D{r.path.dieu}"
        by_dieu.setdefault(key, []).append(i)

    points = []
    for key, idxs in by_dieu.items():
        first = records[idxs[0]]
        joined_summary = " ".join((records[i].summary or records[i].input_text)[:300] for i in idxs)
        text = f"{first.path.dieu_title or ''}\n{joined_summary}"
        vec = embeddings[idxs].mean(axis=0)
        points.append(
            models.PointStruct(
                id=_uuid(key),
                vector=vec.tolist(),
                payload={
                    "id": key,
                    "doc_id": first.doc_id,
                    "dieu": first.path.dieu or 0,
                    "dieu_title": first.path.dieu_title or "",
                    "chuong": first.path.chuong or "",
                    "so_hieu": first.doc_meta.so_hieu or "",
                    "n_khoan": len(idxs),
                    "summary": joined_summary[:1000],
                    "_document": text,
                },
            )
        )
    client().upload_points(collection_name=config.COLL_ARTICLES, points=points)


def insert_document(records: List[ClauseRecord], embeddings: np.ndarray) -> None:
    if not records:
        return
    _ensure_collection(config.COLL_DOCUMENTS)
    first = records[0]
    doc_id = first.doc_id
    toc = " | ".join(
        sorted({f"Đ.{r.path.dieu}: {r.path.dieu_title or ''}" for r in records if r.path.dieu})
    )
    text = f"{first.doc_meta.so_hieu}\n{toc}"
    vec = embeddings.mean(axis=0).tolist()
    point = models.PointStruct(
        id=_uuid(doc_id),
        vector=vec,
        payload={
            "id": doc_id,
            "doc_id": doc_id,
            "so_hieu": first.doc_meta.so_hieu or "",
            "co_quan_ban_hanh": first.doc_meta.co_quan_ban_hanh or "",
            "ngay_ban_hanh": first.doc_meta.ngay_ban_hanh or "",
            "n_clauses": len(records),
            "toc": toc[:2000],
            "_document": text,
        },
    )
    client().upload_points(collection_name=config.COLL_DOCUMENTS, points=[point])


# --- Query helpers (replace direct collection access) ----------------------


def get_all(
    coll_name: str,
    include: Optional[List[str]] = None,
    where: Optional[Dict] = None,
) -> Dict[str, List]:
    """Get all points from a collection. Returns dict with keys: ids, metadatas, documents, embeddings."""
    _ensure_collection(coll_name)
    c = client()
    with_vectors = include and "embeddings" in include

    scroll_filter = None
    if where:
        conditions = []
        for k, v in where.items():
            if isinstance(v, dict) and "$ne" in v:
                conditions.append(
                    models.FieldCondition(key=k, match=models.MatchExcept(**{"except": [v["$ne"]]}))
                )
            else:
                conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        scroll_filter = models.Filter(must=conditions)

    all_points = []
    offset = None
    while True:
        results, offset = c.scroll(
            collection_name=coll_name,
            scroll_filter=scroll_filter,
            with_vectors=with_vectors,
            with_payload=True,
            limit=256,
            offset=offset,
        )
        all_points.extend(results)
        if offset is None:
            break

    ids = [p.payload.get("id", str(p.id)) for p in all_points]
    metadatas = [p.payload for p in all_points]
    documents = [p.payload.get("_document", "") for p in all_points]
    embeddings = [p.vector for p in all_points] if with_vectors else []

    return {"ids": ids, "metadatas": metadatas, "documents": documents, "embeddings": embeddings}


def get_by_ids(coll_name: str, ids: List[str], include: Optional[List[str]] = None) -> Dict[str, List]:
    """Get points by their string IDs."""
    _ensure_collection(coll_name)
    c = client()
    with_vectors = include and "embeddings" in include
    uuid_ids = [_uuid(sid) for sid in ids]
    points = c.retrieve(collection_name=coll_name, ids=uuid_ids, with_vectors=with_vectors, with_payload=True)

    out_ids = [p.payload.get("id", str(p.id)) for p in points]
    metadatas = [p.payload for p in points]
    documents = [p.payload.get("_document", "") for p in points]
    embeddings = [p.vector for p in points] if with_vectors else []

    return {"ids": out_ids, "metadatas": metadatas, "documents": documents, "embeddings": embeddings}


def query(
    coll_name: str,
    query_embedding: List[float],
    k: int = 10,
    where: Optional[Dict] = None,
) -> Dict[str, List]:
    """Similarity search. Returns dict with ids, metadatas, documents, distances (lists of lists like Chroma)."""
    _ensure_collection(coll_name)
    c = client()

    query_filter = None
    if where:
        conditions = []
        for key, v in where.items():
            if isinstance(v, dict) and "$ne" in v:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchExcept(**{"except": [v["$ne"]]}))
                )
            else:
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=v)))
        query_filter = models.Filter(must=conditions)

    results = c.query_points(
        collection_name=coll_name,
        query=query_embedding,
        limit=k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    ids = [[p.payload.get("id", str(p.id)) for p in results]]
    metadatas = [[p.payload for p in results]]
    documents = [[p.payload.get("_document", "") for p in results]]
    # Qdrant returns similarity score (higher = more similar) for cosine
    # Convert to distance (1 - score) for compatibility
    distances = [[1.0 - p.score for p in results]]

    return {"ids": ids, "metadatas": metadatas, "documents": documents, "distances": distances}


# --- Prototypes ------------------------------------------------------------


def rebuild_prototypes() -> int:
    """Recompute one prototype per distinct `nhom` from the clauses collection.
    Returns number of prototypes written."""
    data = get_all(config.COLL_CLAUSES, include=["embeddings"])
    if not data["ids"]:
        return 0

    by_nhom: Dict[str, List[int]] = {}
    for i, meta in enumerate(data["metadatas"]):
        nhom = (meta or {}).get("nhom") or ""
        if nhom:
            by_nhom.setdefault(nhom, []).append(i)

    if not by_nhom:
        return 0

    # Recreate prototypes collection
    _ensure_collection(config.COLL_PROTOTYPES)
    c = client()
    # Delete all existing prototypes
    existing = get_all(config.COLL_PROTOTYPES)
    if existing["ids"]:
        c.delete(
            collection_name=config.COLL_PROTOTYPES,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[])
            ),
        )

    embs = np.asarray(data["embeddings"], dtype=np.float32)
    points = []
    for nhom, idxs in by_nhom.items():
        vec = embs[idxs].mean(axis=0)
        pid = f"proto::{nhom}"
        points.append(
            models.PointStruct(
                id=_uuid(pid),
                vector=vec.tolist(),
                payload={"id": pid, "nhom": nhom, "n_examples": len(idxs), "_document": nhom},
            )
        )
    c.upload_points(collection_name=config.COLL_PROTOTYPES, points=points)
    return len(points)


def set_nhom(clause_id: str, nhom: str, source: str = "manual", confidence: float = 1.0) -> None:
    _ensure_collection(config.COLL_CLAUSES)
    c = client()
    point_id = _uuid(clause_id)
    points = c.retrieve(collection_name=config.COLL_CLAUSES, ids=[point_id], with_payload=True)
    if not points:
        raise KeyError(clause_id)
    payload = points[0].payload or {}
    payload["nhom"] = nhom
    payload["nhom_source"] = source
    payload["nhom_confidence"] = float(confidence)
    c.set_payload(
        collection_name=config.COLL_CLAUSES,
        payload={"nhom": nhom, "nhom_source": source, "nhom_confidence": float(confidence)},
        points=[point_id],
    )
