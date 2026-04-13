"""LLM-based enrichment of ClauseRecord. Only fills fields rules left blank."""
from __future__ import annotations

import json
import logging
import time
from typing import List

from tqdm import tqdm

from . import config, llm as llm_module
from .schema import ClauseRecord, LLMEnrichment

log = logging.getLogger(__name__)

_PROMPT_HEADER = """Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam (ngân hàng).
Với MỖI Khoản dưới đây, hãy trả về một JSON object có CHÍNH XÁC các trường:

- summary: tóm tắt 1-2 câu, giữ nguyên ngữ nghĩa pháp lý
- triggers: danh sách điều kiện kích hoạt (khi nào áp dụng), mỗi mục là một cụm từ; [] nếu không có
- actions: danh sách hành động/đối tượng được quy định; [] nếu không có
- doi_tuong: danh sách chủ thể (TCTD, NHNN, chi nhánh, khách hàng, ...)
- keywords: 3-8 từ khóa thực thể (entity-like, không phải stopword)
- tags: 2-5 tag ngắn snake_case để gom nhóm (vd: quan_tri, cap_phep, chi_nhanh)
- sanctions: object (dict) ánh xạ từng chủ thể (đối tượng) đến danh sách chế tài/xử phạt áp dụng cho chủ thể đó; thêm khóa đặc biệt "references" là danh sách các Điều/Khoản liên quan. Ví dụ: {"tổ chức tín dụng": ["phạt tiền từ 50-200 triệu đồng"], "chi nhánh ngân hàng nước ngoài": ["đình chỉ hoạt động"], "references": ["Điều 5"]}. Dùng {} nếu Khoản không quy định chế tài.
- reasoning: 1 câu giải thích vì sao Khoản này thuộc nhóm chủ đề đó
- clause_type_override: chỉ điền nếu bạn KHÔNG đồng ý với clause_type rule-based đã suy ra; chọn 1 trong: definition|obligation|permission|prohibition|procedure|sanction|unknown. Nếu đồng ý, để null.

QUAN TRỌNG: với các trường list, dùng [] thay vì null khi không có giá trị.
Trả về DUY NHẤT một JSON array, mỗi phần tử ứng với một Khoản theo đúng thứ tự đầu vào.
KHÔNG thêm markdown, KHÔNG thêm văn bản ngoài JSON.
"""


def _build_user_message(records: List[ClauseRecord]) -> str:
    items = []
    for r in records:
        items.append(
            {
                "id": r.id,
                "dieu": r.path.dieu,
                "khoan": r.path.khoan,
                "dieu_title": r.path.dieu_title,
                "input_text": r.input_text,
                "rule_clause_type": r.normative.clause_type,
                "rule_modal": r.normative.modal,
                "rule_references": r.references,
            }
        )
    return _PROMPT_HEADER + "\n\nĐầu vào:\n" + json.dumps(items, ensure_ascii=False, indent=2)


def _parse_response(text: str, n: int) -> List[LLMEnrichment]:
    text = text.strip()
    # Some models still wrap JSON in fences despite response_mime_type
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("LLM did not return a JSON array")
    if len(data) != n:
        log.warning("LLM returned %d items, expected %d", len(data), n)
    out: List[LLMEnrichment] = []
    for item in data[:n]:
        try:
            out.append(LLMEnrichment.model_validate(item))
        except Exception as e:  # noqa: BLE001
            log.warning("Skipping invalid LLM item: %s", e)
            out.append(LLMEnrichment())
    while len(out) < n:
        out.append(LLMEnrichment())
    return out


def enrich_batch(
    records: List[ClauseRecord],
    batch_size: int = 8,
    rpm: int = 5,
    max_retries: int = 2,
) -> List[ClauseRecord]:
    """Call Gemini in batches and merge results into the records (in place).

    Skips records that already have a summary (resume-friendly).
    Throttles to `rpm` requests/minute and retries 429s with exponential backoff.
    """
    todo = [r for r in records if not r.summary]
    if not todo:
        log.info("All %d records already enriched, skipping LLM.", len(records))
        return records
    if len(todo) < len(records):
        log.info("Resuming: %d/%d records still need enrichment.", len(todo), len(records))

    min_interval = 60.0 / rpm
    last_call = 0.0

    n_batches = (len(todo) + batch_size - 1) // batch_size
    pbar = tqdm(
        range(0, len(todo), batch_size),
        total=n_batches,
        desc=f"LLM enrich ({len(todo)} Khoản)",
        unit="batch",
    )
    for i in pbar:
        batch = todo[i : i + batch_size]
        prompt = _build_user_message(batch)
        enrichments: List[LLMEnrichment] = [LLMEnrichment() for _ in batch]

        for attempt in range(max_retries + 1):
            wait = min_interval - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                resp_text = llm_module.generate(prompt)
                last_call = time.time()
                enrichments = _parse_response(resp_text, len(batch))
                break
            except Exception as e:  # noqa: BLE001
                last_call = time.time()
                msg = str(e)
                if "429" in msg or "ResourceExhausted" in msg or "quota" in msg.lower():
                    backoff = 60.0 * (attempt + 1)
                    log.warning(
                        "Batch %d hit 429 (attempt %d/%d), sleeping %.0fs...",
                        i // batch_size, attempt + 1, max_retries + 1, backoff,
                    )
                    time.sleep(backoff)
                    continue
                log.error("LLM call failed for batch %d: %s", i // batch_size, e)
                break

        for rec, enr in zip(batch, enrichments):
            _merge(rec, enr)
        done = min(i + batch_size, len(todo))
        pbar.set_postfix_str(f"{done}/{len(todo)} Khoản")
    pbar.close()
    return records


def _merge(rec: ClauseRecord, enr: LLMEnrichment) -> None:
    """Apply LLM enrichment over the rule pre-fill, never overwriting non-empty rule output
    except for clause_type when explicitly overridden."""
    if enr.summary:
        rec.summary = enr.summary
    if enr.triggers:
        rec.normative.triggers = enr.triggers
    if enr.actions:
        rec.normative.actions = enr.actions
    if enr.doi_tuong:
        rec.doi_tuong = enr.doi_tuong
    if enr.keywords:
        rec.keywords = enr.keywords
    if enr.tags:
        rec.tags = enr.tags
    if enr.sanctions:
        rec.sanctions = enr.sanctions
    if enr.reasoning:
        rec.reasoning = enr.reasoning
    if enr.clause_type_override and enr.clause_type_override != rec.normative.clause_type:
        log.info(
            "clause_type override on %s: rule=%s llm=%s",
            rec.id,
            rec.normative.clause_type,
            enr.clause_type_override,
        )
        rec.normative.clause_type = enr.clause_type_override
