"""Use case 3 — Chatbot. Hybrid retrieve + answer with citations."""
from __future__ import annotations

import argparse
import json
import sys

from law_rag import config, llm, retriever, store

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý ngân hàng Việt Nam. Trả lời câu hỏi của người dùng
DỰA HOÀN TOÀN trên các trích dẫn bên dưới. Mỗi luận điểm phải kèm citation theo dạng
[<so_hieu> Đ.<dieu>.<khoan>]. Nếu trích dẫn không đủ thông tin, nói rõ."""


def _format_context(ids_scores) -> str:
    blocks = []
    if not ids_scores:
        return "(không có trích dẫn)"
    data = store.get_by_ids(config.COLL_CLAUSES, [cid for cid, _ in ids_scores])
    by_id = {cid: i for i, cid in enumerate(data["ids"])}
    for cid, score in ids_scores:
        if cid not in by_id:
            continue
        m = data["metadatas"][by_id[cid]] or {}
        cite = f"[{m.get('so_hieu','?')} Đ.{m.get('dieu','?')}.{m.get('khoan','?')}]"
        text = m.get("input_text", "")
        blocks.append(f"{cite} (score={score:.3f})\n{text}")
    return "\n\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    query = " ".join(args.question)
    hits = retriever.hybrid_search(query, k=args.k)
    context = _format_context(hits)

    if args.show_context:
        print("=== CONTEXT ===")
        print(context)
        print("=== END ===\n")

    if not config.GEMINI_API_KEY:
        print("GEMINI_API_KEY not set; printing context only.")
        return 0

    prompt = f"{SYSTEM_PROMPT}\n\n=== TRÍCH DẪN ===\n{context}\n\n=== CÂU HỎI ===\n{query}"
    answer = llm.generate(prompt)
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
