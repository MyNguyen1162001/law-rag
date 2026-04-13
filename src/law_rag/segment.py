"""Segment a flat list of paragraphs into a hierarchical Chương → Mục → Điều → Khoản tree.

Also handles Phụ lục (appendices) containing contract templates, producing both
full-text records and per-clause chunk records.

Outputs flat ClauseRecord stubs (one per Khoản) ready for rule-based and LLM enrichment.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .schema import ClausePath, ClauseRecord, ContractMeta, DocMeta, DocType

# --- Regexes for Vietnamese legal structure --------------------------------

RE_CHUONG = re.compile(r"^\s*Chương\s+([IVXLCDM\d]+)\b\.?\s*(.*)$", re.IGNORECASE)
RE_MUC = re.compile(r"^\s*Mục\s+(\d+)\b\.?\s*(.*)$", re.IGNORECASE)
RE_DIEU = re.compile(r"^\s*Điều\s+(\d+)\b\.?\s*(.*)$", re.IGNORECASE)
RE_KHOAN = re.compile(r"^\s*(\d+)\.\s+(.*)$")
RE_DIEM = re.compile(r"^\s*([a-zđ])\)\s+(.*)$", re.IGNORECASE)

RE_SO_HIEU = re.compile(r"\b(\d+)\s*/\s*(\d{4})\s*/\s*([A-ZĐ\-]+)\b")
RE_NGAY = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE)

# --- Regexes for Phụ lục / Mẫu số -----------------------------------------

RE_PHU_LUC = re.compile(
    r"^\s*PH[ỤU]\s*L[ỤU]C\s+([IVXLCDM\d]+)\b",
    re.IGNORECASE,
)
RE_MAU_SO = re.compile(
    r"^\s*(?:Mẫu\s+số|M[ẫa]u\s+s[ốo])\s+([IVXLCDMa-z\d]+)\b",
    re.IGNORECASE,
)

# Contract template markers — if a Phụ lục title contains one of these, it's a
# contract template (Nhóm A) rather than an administrative form (Nhóm B/C).
_CONTRACT_KEYWORDS = [
    "HỢP ĐỒNG",
]


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


def _detect_doc_type(so_hieu: str, filename_stem: str) -> DocType:
    """Infer document type from số hiệu or filename."""
    combined = (so_hieu + " " + filename_stem).upper()
    if "TT-" in combined or "THÔNG TƯ" in combined:
        return "thong_tu"
    if "NĐ-CP" in combined or "ND-CP" in combined or "NGHỊ ĐỊNH" in combined:
        return "nghi_dinh"
    if "QH" in combined or "LUẬT" in combined:
        return "luat"
    if "QĐ-" in combined or "QD-" in combined or "QUYẾT ĐỊNH" in combined:
        return "quyet_dinh"
    return "unknown"


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

    meta.doc_type = _detect_doc_type(meta.so_hieu, filename_stem)

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


# ---------------------------------------------------------------------------
# Phụ lục (appendix) segmentation
# ---------------------------------------------------------------------------

@dataclass
class _PhuLuc:
    """Raw appendix extracted from the paragraph stream."""
    phu_luc: str            # e.g. "Phụ lục I"
    title: str              # first line after PHỤ LỤC header
    mau_so: Optional[str]   # e.g. "Mẫu số Ia" (None for forms/certs)
    lines: List[str] = field(default_factory=list)
    is_contract: bool = False

    def full_text(self) -> str:
        return "\n".join(self.lines).strip()


def _split_phu_luc_sections(paragraphs: List[str]) -> Tuple[List[str], List[_PhuLuc]]:
    """Split paragraph list into (main_body, [appendices]).

    Everything before the first ``PHỤ LỤC`` header is the main body.
    When a single Phụ lục contains multiple ``Mẫu số`` sub-sections (e.g.
    Phụ lục I has Mẫu số Ia, Ib, Ic), each sub-section becomes its own
    ``_PhuLuc`` entry so they get independent FULL + CLAUSE records.
    """
    main: List[str] = []
    appendices: List[_PhuLuc] = []
    current: Optional[_PhuLuc] = None
    cur_phu_luc_label: Optional[str] = None
    cur_phu_luc_title: str = ""
    cur_phu_luc_is_contract: bool = False

    for line in paragraphs:
        m = RE_PHU_LUC.match(line)
        if m:
            if current:
                appendices.append(current)
                current = None
            pl_num = m.group(1)
            cur_phu_luc_label = f"Phụ lục {pl_num}"
            cur_phu_luc_title = ""
            cur_phu_luc_is_contract = False
            # Start a new _PhuLuc (may be replaced if Mẫu số found)
            current = _PhuLuc(
                phu_luc=cur_phu_luc_label,
                title="",
                mau_so=None,
            )
            continue

        if current is None and cur_phu_luc_label is None:
            main.append(line)
            continue

        if current is None:
            # Between a PHỤ LỤC header and first content — shouldn't happen,
            # but be defensive.
            current = _PhuLuc(
                phu_luc=cur_phu_luc_label or "",
                title="",
                mau_so=None,
            )

        # Detect title (first non-empty line after header)
        if not cur_phu_luc_title and line.strip():
            cur_phu_luc_title = line.strip()
            cur_phu_luc_is_contract = any(
                kw in line.upper() for kw in _CONTRACT_KEYWORDS
            )
            current.title = cur_phu_luc_title
            current.is_contract = cur_phu_luc_is_contract

        # Detect Mẫu số — start a new sub-section
        mm = RE_MAU_SO.match(line)
        if mm:
            mau_label = f"Mẫu số {mm.group(1)}"
            if current.mau_so is not None:
                # Already had a Mẫu số → flush previous and start new
                appendices.append(current)
                current = _PhuLuc(
                    phu_luc=cur_phu_luc_label or "",
                    title=cur_phu_luc_title,
                    mau_so=mau_label,
                    is_contract=cur_phu_luc_is_contract,
                )
            else:
                current.mau_so = mau_label
            continue

        current.lines.append(line)

    if current:
        appendices.append(current)

    # De-duplicate: when a Phụ lục has a short TOC section listing Mẫu số
    # labels (just a few lines) followed by the actual content for each Mẫu số,
    # we get two entries with the same (phu_luc, mau_so).  Keep only the longer
    # one per key.
    seen: dict[tuple[str, Optional[str]], int] = {}
    for idx, pl in enumerate(appendices):
        key = (pl.phu_luc, pl.mau_so)
        if key in seen:
            prev_idx = seen[key]
            if len(pl.lines) > len(appendices[prev_idx].lines):
                appendices[prev_idx] = None  # type: ignore[assignment]
                seen[key] = idx
            else:
                appendices[idx] = None  # type: ignore[assignment]
        else:
            seen[key] = idx
    appendices = [pl for pl in appendices if pl is not None]

    return main, appendices


def _extract_contract_parties(text: str) -> List[str]:
    """Best-effort extraction of party labels from contract template text."""
    parties: List[str] = []
    # Look for patterns like "BÊN CHO THUÊ", "BÊN MUA", etc.
    for m in re.finditer(r"(BÊN\s+[\wĐđÀ-ỹ/]+(?:\s+[\wĐđÀ-ỹ/]+){0,3})\s*\(", text.upper()):
        party = m.group(1).strip()
        if party not in parties:
            parties.append(party)
    # Normalize case
    return [p.title() for p in parties] if parties else []


def segment_appendices(
    paragraphs: List[str],
    doc_id: str,
    doc_meta: DocMeta,
) -> Tuple[List[str], List[ClauseRecord]]:
    """Separate appendices from the main body and produce ClauseRecords for them.

    Returns ``(main_body_paragraphs, appendix_records)`` where appendix_records
    contains both ``full_text`` and ``clause`` record types.
    """
    main_body, appendices = _split_phu_luc_sections(paragraphs)
    records: List[ClauseRecord] = []

    for pl in appendices:
        pl_slug = pl.phu_luc.replace(" ", "").replace("ụ", "u").replace("Phụlục", "PL")
        mau_slug = ""
        if pl.mau_so:
            mau_slug = "__" + pl.mau_so.replace(" ", "").replace("ẫ", "a").replace("ố", "o").replace("Mẫusố", "M").replace("Mauso", "M")
            # Simplify: "Mẫu số Ia" → "MIa"
            mm = re.match(r"Mẫu số\s+(.+)", pl.mau_so)
            if mm:
                mau_slug = f"__M{mm.group(1)}"

        full_id = f"{doc_id}__{pl_slug}{mau_slug}__FULL"
        full_text = pl.full_text()

        # --- Build contract_meta (only for contract templates) ---
        contract_meta = None
        if pl.is_contract:
            parties = _extract_contract_parties(full_text)
            contract_meta = ContractMeta(
                ten_hop_dong=pl.title,
                loai_hop_dong=pl.phu_luc,
                ben_lien_quan=parties,
            )

        # --- Tầng 1: FULL record ---
        records.append(
            ClauseRecord(
                id=full_id,
                doc_id=doc_id,
                doc_meta=doc_meta,
                path=ClausePath(phu_luc=pl.phu_luc, mau_so=pl.mau_so),
                record_type="full_text",
                input_text=full_text,
                summary=f"Toàn văn {pl.phu_luc}: {pl.title}",
                contract_meta=contract_meta,
            )
        )

        # --- Tầng 2: CLAUSE records (only for contract templates with Điều) ---
        if pl.is_contract:
            clause_khoans = segment_paragraphs(pl.lines)
            seen_ids: set[str] = set()
            for k in clause_khoans:
                clause_id = f"{doc_id}__{pl_slug}{mau_slug}__D{k.dieu}__K{k.khoan}"
                if clause_id in seen_ids:
                    i = 2
                    while f"{clause_id}_{i}" in seen_ids:
                        i += 1
                    clause_id = f"{clause_id}_{i}"
                seen_ids.add(clause_id)
                records.append(
                    ClauseRecord(
                        id=clause_id,
                        doc_id=doc_id,
                        doc_meta=doc_meta,
                        path=ClausePath(
                            phu_luc=pl.phu_luc,
                            mau_so=pl.mau_so,
                            dieu=k.dieu,
                            dieu_title=k.dieu_title,
                            khoan=k.khoan,
                        ),
                        record_type="clause",
                        parent_id=full_id,
                        input_text=k.text(),
                        contract_meta=contract_meta,
                    )
                )

    return main_body, records
