"""Test 3b: Sibling coherence (label-free embedding quality test).

For each clause, fetch its top-k nearest neighbors and measure how often those
neighbors share the same structural parent (same Điều, same Chương, same doc).

Intuition:
    Clauses in the same Điều are usually about the same topic. A good embedding
    space should put them close together. If neighbors are random w.r.t.
    structure, the embedding model is not capturing legal-domain semantics.

What this catches:
    - Embedding model that ignores Vietnamese legal vocabulary
    - Chunking that strips too much context (e.g. "Khoản này quy định..." with
      no parent reference)
    - Pathological hub clauses (one clause is everyone's neighbor)

What this does NOT test:
    - Whether retrieval answers real user questions (need eval_retrieval.py)
    - Whether the embedding distinguishes amended vs original text

Usage:
    python -m scripts.eval_sibling
    python -m scripts.eval_sibling --top-k 10
    python -m scripts.eval_sibling --exclude-self        # default behavior
    python -m scripts.eval_sibling --show-worst 20       # list weakest clauses
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from law_rag import config, store


def _expected_random_rate(group_sizes: list[int], k: int) -> float:
    """Approximate baseline: if neighbors were random, what fraction would share group?

    For each clause in a group of size g, the chance a random other clause is in
    the same group is (g - 1) / (N - 1). Average across all clauses.
    """
    n = sum(group_sizes)
    if n <= 1:
        return 0.0
    weighted = sum(g * (g - 1) for g in group_sizes) / (n * (n - 1))
    return weighted  # independent of k under random sampling


def run_sibling_eval(top_k: int = 10, show_worst: int = 0) -> None:
    data = store.get_all(config.COLL_CLAUSES, include=["embeddings"])

    ids = data["ids"]
    embs = data["embeddings"]
    metas = data["metadatas"]
    total = len(ids)
    if total == 0:
        print("clauses collection is empty")
        return

    print(f"Loaded {total} clauses")

    # Build structural group keys per clause
    dieu_key: dict[str, str] = {}
    chuong_key: dict[str, str] = {}
    doc_key: dict[str, str] = {}
    for cid, m in zip(ids, metas):
        m = m or {}
        doc = m.get("doc_id") or ""
        dieu_key[cid] = f"{doc}::D{m.get('dieu') or 0}"
        chuong_key[cid] = f"{doc}::C{m.get('chuong') or ''}"
        doc_key[cid] = doc

    # Group sizes for random baseline
    dieu_groups = Counter(dieu_key.values())
    chuong_groups = Counter(chuong_key.values())
    doc_groups = Counter(doc_key.values())

    base_dieu = _expected_random_rate(list(dieu_groups.values()), top_k)
    base_chuong = _expected_random_rate(list(chuong_groups.values()), top_k)
    base_doc = _expected_random_rate(list(doc_groups.values()), top_k)

    # We need k+1 results because the clause itself is its own nearest neighbor
    n_query = top_k + 1

    sibling_dieu = 0
    sibling_chuong = 0
    sibling_doc = 0
    n_neighbors = 0
    per_clause_dieu_rate: list[tuple[str, float]] = []
    hub_counter: Counter[str] = Counter()

    for i, (cid, emb) in enumerate(zip(ids, embs)):
        result = store.query(
            config.COLL_CLAUSES,
            list(emb),
            k=n_query,
        )
        neighbors = [n for n in result["ids"][0] if n != cid][:top_k]
        if not neighbors:
            continue

        same_dieu = sum(1 for n in neighbors if dieu_key.get(n) == dieu_key[cid])
        same_chuong = sum(1 for n in neighbors if chuong_key.get(n) == chuong_key[cid])
        same_doc = sum(1 for n in neighbors if doc_key.get(n) == doc_key[cid])

        sibling_dieu += same_dieu
        sibling_chuong += same_chuong
        sibling_doc += same_doc
        n_neighbors += len(neighbors)

        per_clause_dieu_rate.append((cid, same_dieu / len(neighbors)))
        for n in neighbors:
            hub_counter[n] += 1

        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{total}")

    rate_dieu = sibling_dieu / n_neighbors if n_neighbors else 0.0
    rate_chuong = sibling_chuong / n_neighbors if n_neighbors else 0.0
    rate_doc = sibling_doc / n_neighbors if n_neighbors else 0.0

    def lift(observed: float, baseline: float) -> str:
        if baseline <= 0:
            return "n/a"
        return f"{observed / baseline:.1f}x baseline"

    print()
    print("=" * 70)
    print(f"Sibling rate @top-{top_k}     observed   random baseline   lift")
    print("-" * 70)
    print(f"  same Điều     {rate_dieu:>10.2%}   {base_dieu:>13.2%}    {lift(rate_dieu, base_dieu)}")
    print(f"  same Chương   {rate_chuong:>10.2%}   {base_chuong:>13.2%}    {lift(rate_chuong, base_chuong)}")
    print(f"  same văn bản  {rate_doc:>10.2%}   {base_doc:>13.2%}    {lift(rate_doc, base_doc)}")
    print("=" * 70)

    # Hub detection: clauses that appear as neighbor for unusually many others
    if hub_counter:
        avg_appearances = sum(hub_counter.values()) / len(hub_counter)
        top_hubs = hub_counter.most_common(5)
        print(f"\nAvg times a clause appears as someone's neighbor: {avg_appearances:.2f}")
        print(f"Top-5 hubs (high = pathological if much greater than {top_k * 2}):")
        for hub_id, count in top_hubs:
            print(f"  {count:>5}x  {hub_id}")

    if show_worst > 0:
        per_clause_dieu_rate.sort(key=lambda x: x[1])
        print(f"\nWeakest {show_worst} clauses by same-Điều sibling rate:")
        for cid, rate in per_clause_dieu_rate[:show_worst]:
            print(f"  {rate:.0%}   {cid}")

    print()
    # Verdict — based on lift over random baseline for same-Điều
    lift_dieu = rate_dieu / base_dieu if base_dieu > 0 else 0.0
    if lift_dieu >= 5.0 and rate_dieu >= 0.30:
        verdict = ("STRONG — embeddings clearly capture document structure. "
                   "Move on to building the synthetic gold set (eval_retrieval.py).")
    elif lift_dieu >= 2.0:
        verdict = ("OK — embeddings beat random by a meaningful margin but not dramatically. "
                   "Acceptable; proceed to eval_retrieval.py.")
    elif lift_dieu >= 1.2:
        verdict = ("WEAK — embeddings barely beat random. Possible causes: "
                   "chunks too short, missing parent context, or model mismatch. "
                   "Inspect text_for_embedding() in src/law_rag/store.py.")
    else:
        verdict = ("BROKEN — embeddings are essentially random w.r.t. structure. "
                   "Stop and debug embed pipeline before any retrieval eval.")
    print(f"Verdict: {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of neighbors to inspect per clause (default 10)")
    ap.add_argument("--show-worst", type=int, default=0,
                    help="Print N clauses with the lowest same-Điều sibling rate")
    args = ap.parse_args()
    run_sibling_eval(top_k=args.top_k, show_worst=args.show_worst)


if __name__ == "__main__":
    main()
