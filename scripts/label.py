"""Manage labels: assign nhom to a clause, list clauses, rebuild prototypes."""
from __future__ import annotations

import argparse
import sys

from law_rag import config, store


def cmd_set(args) -> int:
    store.set_nhom(args.id, args.nhom, source="manual", confidence=1.0)
    print(f"Set {args.id}.nhom = {args.nhom}")
    return 0


def cmd_list(args) -> int:
    coll = store.collection(config.COLL_CLAUSES)
    where = {"nhom": {"$ne": ""}} if args.labelled else None
    data = coll.get(where=where)
    for i, cid in enumerate(data["ids"]):
        meta = data["metadatas"][i] or {}
        nhom = meta.get("nhom") or "-"
        snippet = (meta.get("summary") or meta.get("input_text") or "")[:80]
        print(f"{cid:50}  [{nhom:20}]  {snippet}")
    print(f"\n{len(data['ids'])} clauses")
    return 0


def cmd_rebuild(_args) -> int:
    n = store.rebuild_prototypes()
    print(f"Rebuilt {n} prototypes from labelled clauses.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser("set", help="Set nhom on one clause")
    p_set.add_argument("id")
    p_set.add_argument("nhom")
    p_set.set_defaults(func=cmd_set)

    p_list = sub.add_parser("list", help="List clauses")
    p_list.add_argument("--labelled", action="store_true", help="Only labelled")
    p_list.set_defaults(func=cmd_list)

    p_rb = sub.add_parser("rebuild-prototypes", help="Recompute prototype vectors")
    p_rb.set_defaults(func=cmd_rebuild)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
