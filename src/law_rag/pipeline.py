"""End-to-end ingest pipeline: file → JSON → embeddings → Chroma + BM25."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List

from . import config, embed, extract_llm, parse_doc, retriever, rules, store
from .schema import ClauseRecord
from .segment import (
    extract_doc_meta,
    segment_appendices,
    segment_paragraphs,
    slug_doc_id,
    to_clause_records,
)

log = logging.getLogger(__name__)


def _save_json(doc_id: str, records: List[ClauseRecord]) -> Path:
    out = config.JSON_DIR / f"{doc_id}.json"
    payload = [r.model_dump() for r in records]
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def ingest_file(path: Path, *, move_to_processed: bool = True, skip_llm: bool = False) -> dict:
    """Run the full ingest pipeline for a single .doc/.docx file."""
    path = Path(path).resolve()
    total = 9 if not skip_llm else 8
    log.info("Ingesting %s", path.name)

    # 1. Parse
    log.info("[1/%d] Parse .doc → paragraphs", total)
    paragraphs = parse_doc.file_to_paragraphs(path)
    if not paragraphs:
        raise ValueError(f"No paragraphs extracted from {path}")
    log.info("       %d paragraphs", len(paragraphs))

    # 2. Doc meta + doc_id
    log.info("[2/%d] Extract doc meta", total)
    doc_id = slug_doc_id(path.stem)
    doc_meta = extract_doc_meta(paragraphs, path.stem)

    # 3. Separate appendices, then segment main body → Khoản
    log.info("[3/%d] Segment Khoản + Phụ lục", total)
    main_paragraphs, appendix_records = segment_appendices(paragraphs, doc_id, doc_meta)
    khoans = segment_paragraphs(main_paragraphs)
    records = to_clause_records(khoans, doc_id, doc_meta)
    records.extend(appendix_records)
    n_main = len(records) - len(appendix_records)
    n_full = sum(1 for r in appendix_records if r.record_type == "full_text")
    n_clause = sum(1 for r in appendix_records if r.record_type == "clause")
    log.info("       %d Khoản (main) + %d Phụ lục (%d full + %d clause)", n_main, len(appendix_records), n_full, n_clause)
    if not records:
        raise ValueError(f"No Khoản found in {path}")

    # 4. Rule-based pre-fill
    log.info("[4/%d] Rule-based prefill", total)
    for r in records:
        rules.prefill(r)

    step = 5
    # 5. LLM enrichment
    if not skip_llm:
        log.info("[%d/%d] LLM enrichment (~5 RPM, batch 8)", step, total)
        try:
            extract_llm.enrich_batch(records, batch_size=8)
        except Exception as e:  # noqa: BLE001
            log.error("LLM enrichment failed (continuing with rule output only): %s", e)
        step += 1

    # 6. Persist JSON
    log.info("[%d/%d] Write JSON", step, total)
    json_path = _save_json(doc_id, records)
    log.info("       %s", json_path)
    step += 1

    # 7. Embed
    log.info("[%d/%d] Embed (BGE-M3, %d texts)", step, total, len(records))
    texts = [store.text_for_embedding(r) for r in records]
    embeddings = embed.encode(texts)
    log.info("       %d vectors of dim %d", embeddings.shape[0], embeddings.shape[1])
    step += 1

    # 8. Insert into all collections
    log.info("[%d/%d] Insert into Qdrant (clauses + articles + document)", step, total)
    store.insert_clauses(records, embeddings)
    store.insert_articles(records, embeddings)
    store.insert_document(records, embeddings)
    step += 1

    # 9. Rebuild BM25 sidecar
    log.info("[%d/%d] Rebuild BM25 sidecar", step, total)
    retriever.rebuild_bm25(config.COLL_CLAUSES)

    # 10. Move file
    if move_to_processed and path.parent.resolve() == config.INBOX_DIR.resolve():
        dest = config.PROCESSED_DIR / path.name
        shutil.move(str(path), str(dest))
        log.info("  moved → %s", dest)

    return {
        "doc_id": doc_id,
        "n_clauses": len(records),
        "json_path": str(json_path),
    }
