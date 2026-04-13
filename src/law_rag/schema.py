"""Pydantic models for the law-rag JSON record."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ClauseType = Literal[
    "definition",
    "obligation",
    "permission",
    "prohibition",
    "procedure",
    "sanction",
    "unknown",
]

RecordType = Literal["clause", "full_text"]

DocType = Literal[
    "thong_tu",
    "nghi_dinh",
    "luat",
    "quyet_dinh",
    "unknown",
]


class DocMeta(BaseModel):
    so_hieu: str = ""
    co_quan_ban_hanh: str = ""
    ngay_ban_hanh: str = ""
    doc_type: DocType = "unknown"


class ClausePath(BaseModel):
    chuong: Optional[str] = None
    muc: Optional[str] = None
    phu_luc: Optional[str] = None
    mau_so: Optional[str] = None
    dieu: Optional[int] = None
    dieu_title: Optional[str] = None
    khoan: Optional[int] = None


class ContractMeta(BaseModel):
    """Metadata specific to contract-template appendices (Phụ lục hợp đồng mẫu)."""
    ten_hop_dong: str = ""
    loai_hop_dong: str = ""
    ben_lien_quan: List[str] = Field(default_factory=list)


class Normative(BaseModel):
    clause_type: ClauseType = "unknown"
    modal: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)


class ClauseRecord(BaseModel):
    id: str
    doc_id: str
    doc_meta: DocMeta = Field(default_factory=DocMeta)
    path: ClausePath = Field(default_factory=ClausePath)

    record_type: RecordType = "clause"
    parent_id: Optional[str] = None

    input_text: str = ""
    summary: str = ""

    normative: Normative = Field(default_factory=Normative)
    contract_meta: Optional[ContractMeta] = None

    doi_tuong: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    sanctions: Dict[str, List[str]] = Field(default_factory=dict)

    nhom: Optional[str] = None
    nhom_confidence: float = 0.0
    nhom_source: Literal["manual", "cluster", "classify", "none"] = "none"
    reasoning: str = ""


class LLMEnrichment(BaseModel):
    """Subset of fields the LLM is asked to fill (rules pre-fill the rest)."""

    summary: str = ""
    triggers: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    doi_tuong: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    sanctions: Dict[str, List[str]] = Field(default_factory=dict)
    reasoning: str = ""
    clause_type_override: Optional[ClauseType] = None

    @field_validator("triggers", "actions", "doi_tuong", "keywords", "tags", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v):
        return [] if v is None else v

    @field_validator("sanctions", mode="before")
    @classmethod
    def _none_to_empty_dict(cls, v):
        return {} if v is None else v

    @field_validator("summary", "reasoning", mode="before")
    @classmethod
    def _none_to_empty_str(cls, v):
        return "" if v is None else v
