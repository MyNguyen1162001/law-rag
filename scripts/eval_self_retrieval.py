"""Test 3a: Self-retrieval sanity check.

For every clause in the `clauses` collection, query Chroma with its own stored
embedding and verify that the top-1 result is the clause itself.

What this catches:
    - ID collisions / overwrites during ingest
    - Embedding ↔ metadata row misalignment
    - Duplicate clause text (boilerplate)
    - Re-ingest creating phantom rows
    - Wrong collection writes

What this does NOT test:
    - Embedding model quality (see eval_sibling.py)
    - Chunking strategy
    - BM25 / hybrid retriever

Usage:
    python -m scripts.eval_self_retrieval
    python -m scripts.eval_self_retrieval --reembed   # also re-embed from text
    python -m scripts.eval_self_retrieval --top-k 5   # check top-5 instead of top-1
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from law_rag import config, store


def run_self_retrieval(top_k: int = 1, reembed: bool = False) -> None:
    data = store.get_all(config.COLL_CLAUSES, include=["embeddings"])

    ids = data["ids"]
    docs = data["documents"]
    embs = data["embeddings"]

    total = len(ids)
    if total == 0:
        print("clauses collection is empty — ingest something first")
        return

    print(f"Loaded {total} clauses from collection '{config.COLL_CLAUSES}'")

    # Optionally re-embed from text to also test the embed-time pipeline
    if reembed:
        from law_rag.embed import embed_texts  # lazy import (loads BGE-M3)

        print("Re-embedding from stored text (this loads BGE-M3)...")
        query_embs = embed_texts(docs)
    else:
        query_embs = np.asarray(embs, dtype=np.float32)

    hits_at_1 = 0
    hits_at_k = 0
    misses: list[tuple[str, str, str]] = []
    duplicate_text_ids: list[str] = []

    # Detect duplicate documents up front (a known cause of self-miss)
    doc_counter = Counter(docs)
    dup_docs = {d for d, c in doc_counter.items() if c > 1}
    if dup_docs:
        for cid, doc in zip(ids, docs):
            if doc in dup_docs:
                duplicate_text_ids.append(cid)

    # Query each clause one at a time (clear semantics; fine for ~thousands)
    for i, (cid, emb) in enumerate(zip(ids, query_embs)):
        result = store.query(
            config.COLL_CLAUSES,
            emb.tolist() if hasattr(emb, "tolist") else list(emb),
            k=top_k,
        )
        returned_ids = result["ids"][0]
        if returned_ids and returned_ids[0] == cid:
            hits_at_1 += 1
            hits_at_k += 1
        elif cid in returned_ids:
            hits_at_k += 1
            misses.append((cid, returned_ids[0], docs[i][:100]))
        else:
            misses.append((cid, returned_ids[0] if returned_ids else "<none>", docs[i][:100]))

        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{total}")

    rate1 = hits_at_1 / total
    ratek = hits_at_k / total
    print()
    print("=" * 60)
    print(f"Self-hit @1:       {hits_at_1}/{total} = {rate1:.2%}")
    if top_k > 1:
        print(f"Self-hit @{top_k}: {hits_at_k}/{total} = {ratek:.2%}")
    print(f"Duplicate-text rows: {len(duplicate_text_ids)} "
          f"(across {len(dup_docs)} distinct duplicated texts)")
    print("=" * 60)

    if misses:
        print(f"\nFirst {min(10, len(misses))} misses:")
        for cid, top_id, snippet in misses[:10]:
            print(f"  - {cid}")
            print(f"      top-1 returned: {top_id}")
            print(f"      text:           {snippet!r}")

    if duplicate_text_ids:
        print(f"\nFirst {min(5, len(duplicate_text_ids))} clauses with duplicate text:")
        for cid in duplicate_text_ids[:5]:
            print(f"  - {cid}")

    print()
    if rate1 >= 0.99:
        verdict = "OK — store/embeddings are coherent. Move on to eval_sibling.py."
    elif rate1 >= 0.95:
        verdict = "Mostly OK — a few duplicate texts (likely boilerplate). Inspect misses above."
    elif rate1 >= 0.70:
        verdict = "Suspicious — significant duplicates or re-ingest. Check pipeline upsert logic."
    else:
        verdict = ("BROKEN — embedding↔metadata likely misaligned, or wrong collection. "
                   "Stop other evals and debug src/law_rag/store.py.")
    print(f"Verdict: {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=1,
                    help="Also report self-hit within top-k (default 1)")
    ap.add_argument("--reembed", action="store_true",
                    help="Re-embed from stored text instead of using stored embeddings "
                         "(tests the embed pipeline too; loads BGE-M3)")
    args = ap.parse_args()
    run_self_retrieval(top_k=args.top_k, reembed=args.reembed)


if __name__ == "__main__":
    main()
