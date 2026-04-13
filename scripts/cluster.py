"""Use case 1 — Clustering. Pull all clause embeddings, run HDBSCAN/KMeans,
print clusters with their dominant tags/đối tượng, optionally write `nhom` back."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np

from law_rag import config, store


def main() -> int:
    parser = argparse.ArgumentParser(description="Cluster clauses to discover groups")
    parser.add_argument("--algo", choices=["hdbscan", "kmeans"], default="hdbscan")
    parser.add_argument("--k", type=int, default=8, help="K for kmeans")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--write", action="store_true", help="Persist nhom = cluster_<i>")
    args = parser.parse_args()

    data = store.get_all(config.COLL_CLAUSES, include=["embeddings"])
    if not data["ids"]:
        print("No clauses in DB. Run `python -m scripts.ingest <file>` first.", file=sys.stderr)
        return 1

    X = np.asarray(data["embeddings"], dtype=np.float32)
    print(f"Loaded {len(X)} clauses, dim={X.shape[1]}")

    if args.algo == "hdbscan":
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, metric="euclidean").fit_predict(X)
    else:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=args.k, n_init="auto", random_state=42).fit_predict(X)

    n_clusters = len({l for l in labels if l != -1})
    n_noise = int((labels == -1).sum())
    print(f"→ {n_clusters} clusters ({n_noise} noise points)\n")

    by_label: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(int(lab), []).append(i)

    for lab in sorted(by_label):
        if lab == -1:
            continue
        idxs = by_label[lab]
        tag_counter: Counter[str] = Counter()
        dt_counter: Counter[str] = Counter()
        types: Counter[str] = Counter()
        for i in idxs:
            meta = data["metadatas"][i] or {}
            try:
                tag_counter.update(json.loads(meta.get("tags_json", "[]")))
                dt_counter.update(json.loads(meta.get("doi_tuong_json", "[]")))
            except Exception:
                pass
            types[meta.get("clause_type", "unknown")] += 1

        print(f"Cluster {lab}  ({len(idxs)} clauses)")
        print(f"  top tags:      {[t for t, _ in tag_counter.most_common(5)]}")
        print(f"  top đối tượng: {[t for t, _ in dt_counter.most_common(5)]}")
        print(f"  clause_types:  {dict(types)}")
        print("  examples:")
        for i in idxs[:3]:
            meta = data["metadatas"][i] or {}
            snippet = (meta.get('summary') or meta.get('input_text') or '')[:80]
            print(f"    - {meta.get('id')}: {snippet}")
        print()

        if args.write:
            for i in idxs:
                store.set_nhom(data["ids"][i], f"cluster_{lab}", source="cluster", confidence=0.5)

    if args.write:
        n = store.rebuild_prototypes()
        print(f"Wrote nhom labels and rebuilt {n} prototypes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
