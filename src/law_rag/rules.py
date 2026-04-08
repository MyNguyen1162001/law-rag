"""Cheap rule-based pre-fill for ClauseRecord fields, run BEFORE the LLM."""
from __future__ import annotations

import re
from typing import List

from .schema import ClauseRecord, ClauseType

# --- Modal verbs (order matters: prohibition > obligation > permission) ----

RE_MODAL_PROHIBIT = re.compile(r"\b(không được|cấm|nghiêm cấm)\b", re.IGNORECASE)
RE_MODAL_OBLIGE = re.compile(r"\b(phải|có trách nhiệm|có nghĩa vụ)\b", re.IGNORECASE)
RE_MODAL_PERMIT = re.compile(r"\b(được|có quyền|được phép)\b", re.IGNORECASE)

# Definitions: "X là Y", "X được hiểu là Y", "X nghĩa là Y"
RE_DEFINITION = re.compile(r"\b(là|được hiểu là|nghĩa là|có nghĩa)\b", re.IGNORECASE)

# References to other legal artifacts
RE_REF_DIEU = re.compile(r"Điều\s+\d+(?:\s+(?:của|tại)\s+[^,;.]{1,80})?", re.IGNORECASE)
RE_REF_THONG_TU = re.compile(r"Thông tư\s+(?:số\s+)?\d+(?:/\d{4})?(?:/[A-ZĐ\-]+)?", re.IGNORECASE)
RE_REF_NGHI_DINH = re.compile(r"Nghị định\s+(?:số\s+)?\d+(?:/\d{4})?(?:/[A-ZĐ\-]+)?", re.IGNORECASE)
RE_REF_LUAT = re.compile(r"Luật\s+[A-ZĐÀ-Ỹ][^,;.]{2,60}?\s*(?:năm\s+)?\d{4}", re.IGNORECASE)
RE_REF_QUYET_DINH = re.compile(r"Quyết định\s+(?:số\s+)?\d+(?:/\d{4})?(?:/[A-ZĐ\-]+)?", re.IGNORECASE)


def _modals(text: str) -> List[str]:
    found: List[str] = []
    for rx in (RE_MODAL_PROHIBIT, RE_MODAL_OBLIGE, RE_MODAL_PERMIT):
        for m in rx.findall(text):
            v = m if isinstance(m, str) else m[0]
            if v and v.lower() not in [x.lower() for x in found]:
                found.append(v.lower())
    return found


def _clause_type_from_modal(text: str) -> ClauseType:
    if RE_MODAL_PROHIBIT.search(text):
        return "prohibition"
    if RE_MODAL_OBLIGE.search(text):
        return "obligation"
    if RE_MODAL_PERMIT.search(text):
        return "permission"
    # Definition heuristic: short clause containing "là" near the start
    head = text[:120]
    if RE_DEFINITION.search(head):
        return "definition"
    return "unknown"


def _references(text: str) -> List[str]:
    refs: List[str] = []
    for rx in (RE_REF_THONG_TU, RE_REF_NGHI_DINH, RE_REF_LUAT, RE_REF_QUYET_DINH, RE_REF_DIEU):
        for m in rx.finditer(text):
            ref = m.group(0).strip().rstrip(".,;")
            if ref and ref not in refs:
                refs.append(ref)
    return refs


def prefill(record: ClauseRecord) -> ClauseRecord:
    """Mutate the record in place with regex-derived fields, return it for chaining."""
    text = record.input_text or ""
    record.normative.modal = _modals(text)
    record.normative.clause_type = _clause_type_from_modal(text)
    record.references = _references(text)
    return record
