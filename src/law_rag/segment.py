"""Segment a flat list of paragraphs into a hierarchical Chương → Mục → Điều → Khoản tree.

Outputs flat ClauseRecord stubs (one per Khoản) ready for rule-based and LLM enrichment.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .schema import ClausePath, ClauseRecord, DocMeta

# --- Regexes for Vietnamese legal structure --------------------------------

RE_CHUONG = re.compile(r"^\s*Chương\s+([IVXLCDM\d]+)\b\.?\s*(.*)$", re.IGNORECASE)
RE_MUC = re.compile(r"^\s*Mục\s+(\d+)\b\.?\s*(.*)$", re.IGNORECASE)
RE_DIEU = re.compile(r"^\s*Điều\s+(\d+)\b\.?\s*(.*)$", re.IGNORECASE)
RE_KHOAN = re.compile(r"^\s*(\d+)\.\s+(.*)$")
RE_DIEM = re.compile(r"^\s*([a-zđ])\)\s+(.*)$", re.IGNORECASE)

RE_SO_HIEU = re.compile(r"\b(\d+)\s*/\s*(\d{4})\s*/\s*([A-ZĐ\-]+)\b")
RE_NGAY = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE)


def slug_doc_id(stem: str) -> str:
    """Build a stable doc_id from filename stem (e.g. '06_2023_TT-NHNN_m_518149' → '06_2023_TT-NHNN')."""
    parts = stem.split("_")
    # Keep parts up to the one that contains the agency code (uppercase or hyphen)
    keep: list[str] = []
    for p in parts:
        keep.append(p)
        if any(c.isupper() for c in p) and "-" in p:
            break
    return "_".join(keep) if keep else stem


def extract_doc_meta(paragraphs: List[str], filename_stem: str) -> DocMeta:
    """Pull số hiệu, ngày, cơ quan, loại văn bản from the document header (first ~30 paragraphs)."""
    meta = DocMeta()
    head = "\n".join(paragraphs[:30])

    m = RE_SO_HIEU.search(head)
    if m:
        meta.so_hieu = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    elif "_" in filename_stem:
        # Fallback: derive from filename like 06_2023_TT-NHNN
        meta.so_hieu = slug_doc_id(filename_stem).replace("_", "/", 2)

    m = RE_NGAY.search(head)
    if m:
        meta.ngay_ban_hanh = f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"

    if "NGÂN HÀNG NHÀ NƯỚC" in head.upper():
        meta.co_quan_ban_hanh = "Ngân hàng Nhà nước"
    elif "BỘ TÀI CHÍNH" in head.upper():
        meta.co_quan_ban_hanh = "Bộ Tài chính"
    elif "CHÍNH PHỦ" in head.upper():
        meta.co_quan_ban_hanh = "Chính phủ"

    return meta


# --- Segmentation ----------------------------------------------------------


@dataclass
class _Khoan:
    chuong: Optional[str] = None
    dieu: Optional[int] = None
    dieu_title: Optional[str] = None
    khoan: Optional[int] = None
    lines: List[str] = field(default_factory=list)

    def text(self) -> str:
        return " ".join(self.lines).strip()


def _flush(current: _Khoan, out: List[_Khoan]) -> None:
    if current.dieu is not None and current.text():
        out.append(current)


def segment_paragraphs(paragraphs: List[str]) -> List[_Khoan]:
    """Walk paragraphs, emit one _Khoan per (Điều, Khoản). Điều with no numbered Khoản
    is emitted as a single Khoản with khoan=1."""
    out: List[_Khoan] = []
    chuong = None
    dieu = dieu_title = None
    current: Optional[_Khoan] = None

    def new_khoan(num: Optional[int], first_line: str) -> _Khoan:
        return _Khoan(
            chuong=chuong,
            dieu=dieu,
            dieu_title=dieu_title,
            khoan=num,
            lines=[first_line] if first_line else [],
        )

    for line in paragraphs:
        if (m := RE_CHUONG.match(line)):
            if current:
                _flush(current, out)
                current = None
            chuong = f"Chương {m.group(1)}"
            dieu = dieu_title = None
            continue

        if RE_MUC.match(line):
            if current:
                _flush(current, out)
                current = None
            continue

        if (m := RE_DIEU.match(line)):
            if current:
                _flush(current, out)
            dieu = int(m.group(1))
            dieu_title = m.group(2).strip().rstrip(".") or None
            # Start an implicit Khoản 1 in case the Điều has no numbered breakdown
            current = new_khoan(num=None, first_line="")
            continue

        if dieu is None:
            continue  # still in header

        if (m := RE_KHOAN.match(line)):
            if current:
                _flush(current, out)
            current = new_khoan(num=int(m.group(1)), first_line=m.group(2).strip())
            continue

        # Continuation line (or Điểm) — append to current
        if current is None:
            current = new_khoan(num=None, first_line=line)
        else:
            current.lines.append(line)

    if current:
        _flush(current, out)

    # Normalize: if a Điều has only the implicit Khoản (num=None), call it Khoản 1
    for k in out:
        if k.khoan is None:
            k.khoan = 1
    return out


def to_clause_records(
    khoans: List[_Khoan], doc_id: str, doc_meta: DocMeta
) -> List[ClauseRecord]:
    """Convert internal _Khoan rows to ClauseRecord stubs (no LLM/rule fields filled)."""
    records: List[ClauseRecord] = []
    seen_ids: set[str] = set()
    for k in khoans:
        rid = f"{doc_id}__D{k.dieu}__K{k.khoan}"
        if rid in seen_ids:
            # Defensive: append a disambiguator if upstream emitted dupes
            i = 2
            while f"{rid}_{i}" in seen_ids:
                i += 1
            rid = f"{rid}_{i}"
        seen_ids.add(rid)

        records.append(
            ClauseRecord(
                id=rid,
                doc_id=doc_id,
                doc_meta=doc_meta,
                path=ClausePath(
                    chuong=k.chuong,
                    dieu=k.dieu,
                    dieu_title=k.dieu_title,
                    khoan=k.khoan,
                ),
                input_text=k.text(),
            )
        )
    return records


def normalize_vi(text: str) -> str:
    """Lowercase + strip Vietnamese diacritics for BM25-friendly text."""
    nfd = unicodedata.normalize("NFD", text)
    no_marks = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return no_marks.replace("đ", "d").replace("Đ", "D").lower()
