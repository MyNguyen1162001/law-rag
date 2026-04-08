"""CLI: ingest one or more .doc/.docx files into the RAG database.

Usage:
    python -m scripts.ingest path/to/file1.doc path/to/file2.docx
    python -m scripts.ingest --skip-llm file.doc      # rule-only ingest (offline)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from law_rag import pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest legal .doc/.docx files into the RAG DB")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--skip-llm", action="store_true", help="Skip Gemini enrichment")
    parser.add_argument("--no-move", action="store_true", help="Don't move file to processed/")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    rc = 0
    for f in args.files:
        if not f.exists():
            print(f"ERROR: not found: {f}", file=sys.stderr)
            rc = 1
            continue
        try:
            result = pipeline.ingest_file(f, move_to_processed=not args.no_move, skip_llm=args.skip_llm)
            print(f"OK  {f.name}: {result['n_clauses']} Khoản → {result['json_path']}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {f.name}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
