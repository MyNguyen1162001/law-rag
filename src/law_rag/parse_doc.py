"""Parse .doc / .docx legal documents into plain text preserving paragraph order."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from docx import Document


def _find_soffice() -> str | None:
    """Locate LibreOffice binary on macOS or Linux."""
    cands = [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        shutil.which("soffice"),
        shutil.which("libreoffice"),
    ]
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def doc_to_docx(doc_path: Path) -> Path:
    """Convert legacy .doc to .docx. Tries LibreOffice first, then macOS textutil."""
    out_dir = Path(tempfile.mkdtemp(prefix="lawrag_"))
    out = out_dir / (doc_path.stem + ".docx")

    soffice = _find_soffice()
    if soffice:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "docx", "--outdir", str(out_dir), str(doc_path)],
            check=True,
            capture_output=True,
        )
        if out.exists():
            return out

    textutil = shutil.which("textutil") or "/usr/bin/textutil"
    if Path(textutil).exists():
        subprocess.run(
            [textutil, "-convert", "docx", "-output", str(out), str(doc_path)],
            check=True,
            capture_output=True,
        )
        if out.exists():
            return out

    raise RuntimeError(
        f"Could not convert {doc_path}. Install LibreOffice "
        "(`brew install --cask libreoffice`) or ensure macOS `textutil` is available."
    )


def docx_to_paragraphs(docx_path: Path) -> List[str]:
    """Read a .docx and return a list of non-empty paragraph strings, in order."""
    doc = Document(str(docx_path))
    out: List[str] = []
    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if text:
            out.append(text)
    return out


def file_to_paragraphs(path: Path) -> List[str]:
    """Top-level: accept .doc or .docx, return ordered paragraph list."""
    path = Path(path)
    if path.suffix.lower() == ".docx":
        return docx_to_paragraphs(path)
    if path.suffix.lower() == ".doc":
        return docx_to_paragraphs(doc_to_docx(path))
    raise ValueError(f"Unsupported file type: {path.suffix}")
