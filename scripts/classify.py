"""Use case 2 — Classification via prototype kNN. No prompt loops, no LLM calls."""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from law_rag import config, embed, store


def classify(query: str, k: int = 5) -> dict:
    qvec = embed.encode([query])[0].tolist()

    # 1) Try prototypes (fast, O(num_classes))
    proto_data = store.get_all(config.COLL_PROTOTYPES)
    if proto_data["ids"]:
        res = store.query(config.COLL_PROTOTYPES, qvec, k=1)
        nhom = res["metadatas"][0][0]["nhom"]
        sim = 1.0 - float(res["distances"][0][0])
        if sim >= 0.45:
            return {"nhom": nhom, "confidence": sim, "method": "prototype"}

    # 2) Fallback: kNN over labelled clauses, majority vote
    res = store.query(
        config.COLL_CLAUSES,
        qvec,
        k=k,
        where={"nhom": {"$ne": ""}},
    )
    if not res["ids"][0]:
        return {"nhom": None, "confidence": 0.0, "method": "none"}
    votes: Counter[str] = Counter()
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        votes[meta.get("nhom", "")] += 1.0 - float(dist)
    nhom, score = votes.most_common(1)[0]
    return {"nhom": nhom, "confidence": score / k, "method": "knn"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    result = classify(" ".join(args.query), k=args.k)
    print(f"nhom:       {result['nhom']}")
    print(f"confidence: {result['confidence']:.3f}")
    print(f"method:     {result['method']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
