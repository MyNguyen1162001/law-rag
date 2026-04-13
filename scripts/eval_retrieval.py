"""Test #1 + #2: Retrieval quality with synthetic gold set + ablation modes.

Builds a synthetic gold set from the JSON audit files in data/json/. For each
clause, generates 1-3 query variants from its own metadata (keywords, actions,
triggers, doi_tuong, summary). The expected answer is the clause's own ID.

Then runs each query through the retriever in one of three modes and reports
Recall@k, MRR, and per-document breakdown.

Modes (test #2 ablation):
    dense   — Chroma vector search only
    bm25    — BM25 sidecar only
    hybrid  — RRF fusion of both (production retriever)

Caveat:
    Synthetic queries are easier than real user questions because they are
    drawn from fields that already match the clause text. Treat numbers as an
    UPPER BOUND on real-world quality. If even synthetic Recall@10 is < 0.7,
    retrieval is broken.

Usage:
    python -m scripts.eval_retrieval                           # synthetic, hybrid
    python -m scripts.eval_retrieval --mode all                # synthetic, all 3 + table
    python -m scripts.eval_retrieval --gold eval/gold.csv      # MANUAL gold set
    python -m scripts.eval_retrieval --gold eval/gold.csv --mode all
    python -m scripts.eval_retrieval --top-k 5 10 20
    python -m scripts.eval_retrieval --sample 100
    python -m scripts.eval_retrieval --show-misses 10

Gold CSV format (header required):
    query,expected_ids
    "Phí bảo lãnh thu bằng ngoại tệ thì sao?","11_2022_TT-NHNN__D19__K4"
    "Hồ sơ vay vốn cần những gì?","39_2016_TT-NHNN__D9__K1;39_2016_TT-NHNN__D9__K2"

Multiple expected IDs (separated by ;) — query is a hit if ANY expected ID
appears in top-k. Use this for queries that legitimately match several clauses.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from law_rag import config
from law_rag.retriever import bm25_search, dense_search, hybrid_search


# --- Gold set construction --------------------------------------------------


def _first_n(items: list[str] | None, n: int) -> list[str]:
    if not items:
        return []
    return [s.strip() for s in items[:n] if s and s.strip()]


def build_queries_for_clause(rec: dict) -> list[str]:
    """Synthesize 1-3 query strings for a single clause record.

    Strategy: combine high-signal short fields. Skip clauses with too little
    structured info (we can't make a fair query out of them).
    """
    title = (rec.get("path") or {}).get("dieu_title") or ""
    keywords = _first_n(rec.get("keywords"), 4)
    tags = _first_n(rec.get("tags"), 3)
    doi_tuong = _first_n(rec.get("doi_tuong"), 2)
    actions = _first_n((rec.get("normative") or {}).get("actions"), 2)
    triggers = _first_n((rec.get("normative") or {}).get("triggers"), 2)

    queries: list[str] = []

    # Query A: keywords + first action — "what is this clause about"
    if keywords and actions:
        queries.append(f"{' '.join(keywords[:3])} {actions[0]}")
    elif keywords and title:
        queries.append(f"{title} {' '.join(keywords[:3])}")

    # Query B: doi_tuong + trigger — "who must do what when"
    if doi_tuong and triggers:
        queries.append(f"{doi_tuong[0]} {triggers[0]}")
    elif doi_tuong and actions:
        queries.append(f"{doi_tuong[0]} {actions[0]}")

    # Query C: title + tags — topical phrasing
    if title and tags:
        queries.append(f"{title} {' '.join(tags[:2])}")

    # Dedupe and trim
    seen = set()
    out = []
    for q in queries:
        q = " ".join(q.split())
        if q and q not in seen and len(q) > 8:
            seen.add(q)
            out.append(q)
    return out


def load_synthetic_gold(json_dir: Path) -> list[tuple[str, frozenset[str], str]]:
    """Synthesize (query, {expected_ids}, doc_id) triples from JSON metadata."""
    pairs: list[tuple[str, frozenset[str], str]] = []
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files in {json_dir}")
    for f in files:
        with f.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            continue
        for rec in data:
            cid = rec.get("id")
            if not cid:
                continue
            doc_id = rec.get("doc_id") or ""
            for q in build_queries_for_clause(rec):
                pairs.append((q, frozenset({cid}), doc_id))
    return pairs


def load_manual_gold(csv_path: Path) -> list[tuple[str, frozenset[str], str]]:
    """Read a hand-written gold CSV with columns: query, expected_ids."""
    if not csv_path.exists():
        raise SystemExit(f"Gold file not found: {csv_path}")
    pairs: list[tuple[str, frozenset[str], str]] = []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "query" not in reader.fieldnames \
                or "expected_ids" not in reader.fieldnames:
            raise SystemExit(
                f"Gold CSV must have columns: query, expected_ids "
                f"(got: {reader.fieldnames})"
            )
        for row in reader:
            q = (row.get("query") or "").strip()
            ids_raw = (row.get("expected_ids") or "").strip()
            if not q or not ids_raw:
                continue
            ids = frozenset(s.strip() for s in ids_raw.split(";") if s.strip())
            if not ids:
                continue
            # Infer doc_id from first expected id (for per-doc breakdown)
            first = next(iter(ids))
            doc_id = first.split("__")[0] if "__" in first else ""
            pairs.append((q, ids, doc_id))
    return pairs


# --- Search wrappers --------------------------------------------------------

SearchFn = Callable[[str, int], list[tuple[str, float]]]


def _make_search_fn(mode: str) -> SearchFn:
    coll = config.COLL_CLAUSES
    if mode == "dense":
        return lambda q, k: dense_search(q, coll, k)
    if mode == "bm25":
        return lambda q, k: bm25_search(q, coll, k)
    if mode == "hybrid":
        return lambda q, k: hybrid_search(q, coll, k)
    raise ValueError(f"unknown mode: {mode}")


# --- Metrics ----------------------------------------------------------------


def _best_rank(expected: frozenset[str], results: list[tuple[str, float]]) -> int | None:
    """Rank of the FIRST expected id that appears in results (1-indexed)."""
    for i, (cid, _) in enumerate(results):
        if cid in expected:
            return i + 1
    return None


def evaluate(
    pairs: list[tuple[str, frozenset[str], str]],
    search_fn: SearchFn,
    ks: list[int],
    show_misses: int = 0,
) -> dict:
    max_k = max(ks)
    recall_hits = {k: 0 for k in ks}
    mrr_total = 0.0
    n = 0
    per_doc_total: dict[str, int] = defaultdict(int)
    per_doc_hits: dict[str, int] = defaultdict(int)
    misses: list[tuple[str, str, str]] = []

    for query, expected, doc_id in pairs:
        results = search_fn(query, max_k)
        rank = _best_rank(expected, results)
        n += 1
        per_doc_total[doc_id] += 1
        if rank is not None:
            mrr_total += 1.0 / rank
            for k in ks:
                if rank <= k:
                    recall_hits[k] += 1
            if rank <= max_k:
                per_doc_hits[doc_id] += 1
        else:
            if len(misses) < show_misses:
                top1 = results[0][0] if results else "<none>"
                misses.append((query, ";".join(sorted(expected)), top1))

        if n % 200 == 0:
            print(f"  ...{n}/{len(pairs)}")

    return {
        "n": n,
        "recall": {k: recall_hits[k] / n if n else 0.0 for k in ks},
        "mrr": mrr_total / n if n else 0.0,
        "per_doc": {
            d: (per_doc_hits[d] / per_doc_total[d] if per_doc_total[d] else 0.0)
            for d in per_doc_total
        },
        "per_doc_n": dict(per_doc_total),
        "misses": misses,
    }


# --- Reporting --------------------------------------------------------------


def _print_report(mode: str, result: dict, ks: list[int]) -> None:
    print()
    print("=" * 60)
    print(f"Mode: {mode}    queries: {result['n']}")
    print("-" * 60)
    for k in ks:
        print(f"  Recall@{k:<3}  {result['recall'][k]:.2%}")
    print(f"  MRR        {result['mrr']:.4f}")
    print("=" * 60)

    if result["per_doc"]:
        print("\nPer-document Recall@max-k:")
        rows = sorted(result["per_doc"].items(), key=lambda x: x[1])
        for doc, rate in rows:
            n = result["per_doc_n"][doc]
            print(f"  {rate:.2%}   ({n:>4} queries)   {doc}")

    if result["misses"]:
        print(f"\nFirst {len(result['misses'])} misses:")
        for q, expected, top1 in result["misses"]:
            print(f"  query:    {q[:90]}")
            print(f"  expected: {expected}")
            print(f"  got top1: {top1}")
            print()


def _print_comparison(results: dict[str, dict], ks: list[int]) -> None:
    print()
    print("=" * 70)
    print("ABLATION COMPARISON")
    print("-" * 70)
    header = f"{'mode':<10}" + "".join(f"  R@{k:<5}" for k in ks) + f"   MRR"
    print(header)
    print("-" * 70)
    for mode, r in results.items():
        row = f"{mode:<10}"
        for k in ks:
            row += f"  {r['recall'][k]:>6.2%}"
        row += f"   {r['mrr']:.4f}"
        print(row)
    print("=" * 70)


# --- Main -------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid", "all"], default="hybrid")
    ap.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10],
                    help="One or more k values for Recall@k (default: 1 5 10)")
    ap.add_argument("--sample", type=int, default=0,
                    help="Random subsample of N queries (0 = use all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show-misses", type=int, default=0,
                    help="Print first N missed queries per mode")
    ap.add_argument("--json-dir", type=Path, default=config.JSON_DIR)
    ap.add_argument("--gold", type=Path, default=None,
                    help="Path to manual gold CSV (query,expected_ids). "
                         "If omitted, synthesizes from JSON metadata.")
    args = ap.parse_args()

    ks = sorted(set(args.top_k))

    if args.gold:
        print(f"Loading manual gold set from {args.gold}...")
        pairs = load_manual_gold(args.gold)
        print(f"Loaded {len(pairs)} hand-written queries")
    else:
        print(f"Building synthetic gold set from {args.json_dir}...")
        pairs = load_synthetic_gold(args.json_dir)
        print(f"Built {len(pairs)} synthetic (query, expected_id) pairs from "
              f"{len(set(p[2] for p in pairs))} documents")

    if args.sample and args.sample < len(pairs):
        random.seed(args.seed)
        pairs = random.sample(pairs, args.sample)
        print(f"Sampled {len(pairs)} pairs (seed={args.seed})")

    modes = ["dense", "bm25", "hybrid"] if args.mode == "all" else [args.mode]
    results: dict[str, dict] = {}
    for mode in modes:
        print(f"\nRunning mode: {mode}")
        fn = _make_search_fn(mode)
        results[mode] = evaluate(pairs, fn, ks, show_misses=args.show_misses)
        _print_report(mode, results[mode], ks)

    if len(modes) > 1:
        _print_comparison(results, ks)

        # Quick verdict on hybrid lift
        if "hybrid" in results and "dense" in results and "bm25" in results:
            h = results["hybrid"]["recall"][max(ks)]
            d = results["dense"]["recall"][max(ks)]
            b = results["bm25"]["recall"][max(ks)]
            best_single = max(d, b)
            print()
            if h >= best_single + 0.02:
                print("Verdict: hybrid beats best single retriever — RRF is earning its keep.")
            elif h >= best_single - 0.01:
                print("Verdict: hybrid ≈ best single. RRF is neutral; consider tuning weights.")
            else:
                print("Verdict: hybrid underperforms. RRF k-constant or weights likely wrong.")


if __name__ == "__main__":
    main()
