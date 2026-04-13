"""Resume ingest from saved JSON — re-embed then insert into Qdrant + BM25.

Usage:
    python -m scripts.resume_insert data/json/54_2026_ND-CP.json
    python -m scripts.resume_insert data/json/*.json          # all files
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Insert from saved JSON(s) into Qdrant")
    parser.add_argument("json_paths", nargs="+", type=Path, help="Path(s) to saved JSON files")
    args = parser.parse_args()

    from law_rag import config, embed, retriever, store
    from law_rag.schema import ClauseRecord

    for json_path in args.json_paths:
        json_path = json_path.resolve()
        if not json_path.exists():
            log.error("File not found: %s", json_path)
            continue

        log.info("=== %s ===", json_path.name)

        # 1. Load records from JSON
        log.info("[1/3] Loading records")
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        records = [ClauseRecord(**r) for r in raw]
        log.info("       %d records loaded", len(records))

        # 2. Re-embed
        log.info("[2/3] Embed (BGE-M3, %d texts)", len(records))
        texts = [store.text_for_embedding(r) for r in records]
        embeddings = embed.encode(texts)
        log.info("       %d vectors of dim %d", embeddings.shape[0], embeddings.shape[1])

        # 3. Insert into Qdrant
        log.info("[3/3] Insert into Qdrant")
        store.insert_clauses(records, embeddings)
        log.info("       clauses done")
        store.insert_articles(records, embeddings)
        log.info("       articles done")
        store.insert_document(records, embeddings)
        log.info("       document done")

    # Rebuild BM25 sidecar (once, after all files)
    log.info("Rebuilding BM25 sidecar...")
    retriever.rebuild_bm25(config.COLL_CLAUSES)

    log.info("Done!")


if __name__ == "__main__":
    main()
