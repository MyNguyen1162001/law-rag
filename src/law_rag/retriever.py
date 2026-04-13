"""Hybrid dense + BM25 retrieval with Reciprocal Rank Fusion."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from . import config, embed, store
from .segment import normalize_vi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tok(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# --- BM25 sidecar ----------------------------------------------------------


def _bm25_path(coll_name: str) -> Path:
    return config.BM25_DIR / f"{coll_name}.pkl"


def rebuild_bm25(coll_name: str = config.COLL_CLAUSES) -> int:
    """Rebuild the BM25 index for a collection. Returns doc count."""
    data = store.get_all(coll_name)
    if not data["ids"]:
        return 0
    corpus_tokens = []
    ids = []
    for i, doc_id in enumerate(data["ids"]):
        meta = data["metadatas"][i] or {}
        # Normalize on the fly (don't store the normalized text in JSON/metadata)
        normalized = normalize_vi(meta.get("input_text", "") or "")
        text = " ".join(
            [
                normalized,
                meta.get("keywords_flat", "") or "",
                meta.get("tags_flat", "") or "",
            ]
        )
        corpus_tokens.append(_tok(text))
        ids.append(doc_id)
    bm25 = BM25Okapi(corpus_tokens)
    with _bm25_path(coll_name).open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    return len(ids)


def _load_bm25(coll_name: str) -> Optional[Dict]:
    p = _bm25_path(coll_name)
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)


# --- Retrieval -------------------------------------------------------------


def dense_search(query: str, coll_name: str, k: int, where: Optional[Dict] = None) -> List[Tuple[str, float]]:
    qvec = embed.encode([query])[0].tolist()
    res = store.query(coll_name, qvec, k=k, where=where)
    out: List[Tuple[str, float]] = []
    for cid, dist in zip(res["ids"][0], res["distances"][0]):
        out.append((cid, 1.0 - float(dist)))  # cosine distance → similarity
    return out


def bm25_search(query: str, coll_name: str, k: int) -> List[Tuple[str, float]]:
    bundle = _load_bm25(coll_name)
    if not bundle:
        return []
    bm25: BM25Okapi = bundle["bm25"]
    ids: List[str] = bundle["ids"]
    scores = bm25.get_scores(_tok(normalize_vi(query)))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]


def rrf_fuse(*ranklists: List[Tuple[str, float]], k: int = 60) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion. Returns merged (id, score) sorted desc."""
    fused: Dict[str, float] = {}
    for ranklist in ranklists:
        for rank, (doc_id, _score) in enumerate(ranklist):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    coll_name: str = config.COLL_CLAUSES,
    k: int = 8,
    where: Optional[Dict] = None,
) -> List[Tuple[str, float]]:
    dense = dense_search(query, coll_name, k * 2, where=where)
    sparse = bm25_search(query, coll_name, k * 2)
    return rrf_fuse(dense, sparse)[:k]
