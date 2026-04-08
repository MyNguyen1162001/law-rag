"""Pydantic models for the law-rag JSON record."""
from __future__ import annotations

from typing import List, Literal, Optional

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


class DocMeta(BaseModel):
    so_hieu: str = ""
    co_quan_ban_hanh: str = ""
    ngay_ban_hanh: str = ""


class ClausePath(BaseModel):
    chuong: Optional[str] = None
    dieu: Optional[int] = None
    dieu_title: Optional[str] = None
    khoan: Optional[int] = None


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

    input_text: str = ""
    summary: str = ""

    normative: Normative = Field(default_factory=Normative)

    doi_tuong: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

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
    reasoning: str = ""
    clause_type_override: Optional[ClauseType] = None

    @field_validator("triggers", "actions", "doi_tuong", "keywords", "tags", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v):
        return [] if v is None else v

    @field_validator("summary", "reasoning", mode="before")
    @classmethod
    def _none_to_empty_str(cls, v):
        return "" if v is None else v
