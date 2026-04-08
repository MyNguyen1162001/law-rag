# law-rag

Drag-and-drop ingestion pipeline that turns Vietnamese banking-law documents
(`.doc` / `.docx`) into a structured RAG vector database. One database serves
**clustering**, **classification**, and **chatbot** use cases.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# For .doc (legacy) → .docx conversion:
brew install --cask libreoffice
```

`.env` (already populated with Gemini key):

```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
EMBED_MODEL=BAAI/bge-m3
CHROMA_DIR=./chroma_db
```

## Ingest

**Drag-and-drop (recommended):** start the watcher, then drop files into `./inbox/`.

```bash
python -m scripts.watch
# in another terminal:
cp 06_2023_TT-NHNN_m_518149.doc inbox/
```

**Or one-shot CLI:**

```bash
python -m scripts.ingest 06_2023_TT-NHNN_m_518149.doc 39_2016_TT-NHNN_m_338877.doc
# rule-only (no LLM, no API quota):
python -m scripts.ingest --skip-llm 06_2023_TT-NHNN_m_518149.doc
```

Each file produces:

- `data/json/<doc_id>.json` — audit trail (one entry per Khoản, full schema)
- New rows in ChromaDB collections: `clauses`, `articles`, `documents`
- Refreshed BM25 sidecar in `data/bm25/clauses.pkl`
- File moved to `processed/`

## Use cases

### 1. Clustering (discover groups)

```bash
python -m scripts.cluster --algo hdbscan
python -m scripts.cluster --algo kmeans --k 6 --write   # persist nhom = cluster_<i>
```

### 2. Classification (no prompt loops)

```bash
# Label a few examples manually:
python -m scripts.label set 06_2023_TT-NHNN__D38__K1 "QUẢN TRỊ CÔNG TY"
python -m scripts.label rebuild-prototypes
# Then classify new queries instantly:
python -m scripts.classify "TCTD mở chi nhánh ở Hà Nội"
```

### 3. Chatbot

```bash
python -m scripts.chat "Ngân hàng muốn mở chi nhánh, cần làm gì?"
python -m scripts.chat --show-context "Hồ sơ cấp phép gồm những gì?"
```

## Architecture

```
inbox/  ──►  parse_doc  ──►  segment  ──►  rules (regex prefill)
                                                  │
                                                  ▼
                                          extract_llm (Gemini)
                                                  │
                                                  ▼
                          ┌──────────►  data/json/<doc>.json  (audit)
                          │
                          ▼
                       embed (BGE-M3)
                          │
                          ▼
              ┌───────────┴───────────┬─────────────┐
              ▼                       ▼             ▼
          clauses                articles      documents      ← Chroma collections
              │
              ▼
         prototypes  ←─ rebuilt from labelled clauses
              │
              ▼
          classify
```

See [`/Users/tramynguyen/.claude/plans/glittery-forging-ripple.md`](/Users/tramynguyen/.claude/plans/glittery-forging-ripple.md)
for the full design rationale.

## Schema (per Khoản)

See [src/law_rag/schema.py](src/law_rag/schema.py). Notable fields:

- `path.{chuong, muc, dieu, khoan}` — hierarchical citation
- `normative.{clause_type, modal, triggers, actions, scope, exceptions}` — what kind of clause
- `doi_tuong`, `co_quan`, `references`, `keywords`, `tags` — facets
- `nhom`, `nhom_confidence`, `nhom_source` — classification label

## File map

| File | Purpose |
|---|---|
| [src/law_rag/config.py](src/law_rag/config.py) | env + paths |
| [src/law_rag/schema.py](src/law_rag/schema.py) | pydantic models |
| [src/law_rag/parse_doc.py](src/law_rag/parse_doc.py) | `.doc/.docx` → paragraphs |
| [src/law_rag/segment.py](src/law_rag/segment.py) | Chương/Mục/Điều/Khoản segmentation |
| [src/law_rag/rules.py](src/law_rag/rules.py) | regex pre-fill (modal, clause_type, references, co_quan) |
| [src/law_rag/extract_llm.py](src/law_rag/extract_llm.py) | Gemini batched JSON enrichment |
| [src/law_rag/embed.py](src/law_rag/embed.py) | BGE-M3 wrapper |
| [src/law_rag/store.py](src/law_rag/store.py) | Chroma multi-collection helpers |
| [src/law_rag/retriever.py](src/law_rag/retriever.py) | hybrid dense+BM25 RRF |
| [src/law_rag/pipeline.py](src/law_rag/pipeline.py) | end-to-end orchestrator |
| [scripts/watch.py](scripts/watch.py) | watchdog observer on `inbox/` |
| [scripts/ingest.py](scripts/ingest.py) | manual CLI ingest |
| [scripts/cluster.py](scripts/cluster.py) | use case 1 |
| [scripts/classify.py](scripts/classify.py) | use case 2 |
| [scripts/label.py](scripts/label.py) | manage labels + rebuild prototypes |
| [scripts/chat.py](scripts/chat.py) | use case 3 |
