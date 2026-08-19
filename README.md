# Multimodal Bucket-Based RAG

A bucket-aware Retrieval-Augmented Generation system. Documents live in one
of two knowledge bases (**bucket_1** / **bucket_2**); a user picks a
bucket, asks a question, and retrieval is filtered to search **only**
that bucket before a local LLM (via Ollama) generates a grounded,
source-cited answer.

## Pipeline

```
Documents → Parse (PDF/OCR/tables/images) → Preprocess → Chunk → Metadata
→ Embed → Vector DB
                                                              │
User selects bucket → Query → Query Embedding → Bucket Filter
→ Similarity Search → Top-K → Context → Ollama → Answer + Sources
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env            # optional - defaults work out of the box
```

You'll also need:
- **Tesseract OCR** installed system-wide (for scanned PDFs / images):
  `apt install tesseract-ocr` (Linux) or `brew install tesseract` (Mac).
- **Ollama** running locally with two models pulled:
  - `ollama pull llama3.2` — text generation (answers)
  - `ollama pull llava` — vision understanding (charts/diagrams/photos in PDFs and standalone images)

## Usage

**1. Add test documents** (start small — 1 per bucket):

```
data/bucket_1/your_test_doc.pdf
data/bucket_2/another_test_doc.pdf
```

**2. Ingest** (parses → chunks → embeds → stores):

```bash
python scripts/ingest.py                  # both buckets
python scripts/ingest.py --bucket bucket_1
```

**3. Test retrieval directly** (before touching Ollama — recommended):

```bash
python scripts/test_retrieval.py bucket_1 "What is the minimum age required?"
```

**4. Run the API:**

```bash
python run.py
# → http://localhost:8000/docs for interactive Swagger UI
```

**5. Verify with Postman:** import `postman/multimodal-rag.postman_collection.json`
and run requests 1–6 in order (health → ingest → bucket queries → bucket
isolation check).

## Project structure

```
src/
├── ingestion/      # file detection, PDF/image parsing, OCR, tables
├── preprocessing/  # cleaning, normalization, dedup
├── chunking/       # text splitting (respects tables/paragraphs)
├── metadata/       # attaches chunk_id, bucket_id, page_number, etc.
├── embeddings/      # embedding model wrapper (swappable)
├── vector_store/     # abstract interface + Chroma (default) / Qdrant
├── retrieval/       # bucket-filtered similarity search
├── generation/      # prompt building + Ollama client
├── api/              # FastAPI app (/health, /ingest, /query)
└── config/           # all tunables (chunk size, top_k, model names, etc.)
```

Every tunable (chunk size/overlap, embedding model, vector backend, top_k,
Ollama model, OCR engine) lives in `src/config/settings.py` / `.env` —
nothing is hard-coded in the pipeline modules.

## Testing progression (go slow — don't skip stages)

Each stage adds documents incrementally and has a concrete pass/fail
check before moving on. Add docs to `data/bucket_1/` and `data/bucket_2/`,
re-run `python scripts/ingest.py`, then verify with the commands shown.

| Stage | Docs | Goal | How to verify |
|---|---|---|---|
| 🧪 Test 1 | 1 | Prove basic ingestion → retrieval works at all | `ingest.py` runs with no errors → `test_retrieval.py bucket_1 "<a question the doc answers>"` returns a relevant chunk |
| 🧪 Test 2 | 2 | Different document types (e.g. 1 text PDF + 1 scanned/image) | Check ingestion logs show the right path taken per file (text extraction vs. OCR vs. vision) — inspect chunk `content_type` in the printed output |
| 🧪 Test 3 | 5 | Multiple chunks/documents don't collide | `test_retrieval.py` for a few different questions — confirm each returns chunks from the *correct* source document, not a random one |
| 🧪 Test 4 | 10 | Bucket filtering actually isolates | Put docs in both buckets. Ask a bucket_2-only question while querying `bucket_1` — must return "not found," never bucket_2 content |
| 🧪 Test 5 | 25 | Stress retrieval quality at more scale | Spot-check 5–10 varied queries; watch for irrelevant top-K results creeping in (may mean `TOP_K` or chunk size needs tuning) |
| 🧪 Test 6 | 50 | One full bucket complete | Full bucket_1 ingestion completes without errors; retrieval still accurate across the whole set |
| 🚀 Final | 100 | Full project | Both buckets at 50 each; run the full Postman collection end to end |

Only move to the next stage once the current one passes cleanly. If
something breaks, it's much easier to debug at 5 documents than at 100.

## Build order (once past document-count testing)

1. **Ingestion** — `scripts/ingest.py`, inspect chunks look right
2. **Retrieval** — `scripts/test_retrieval.py`, confirm relevant chunks come back
3. **Bucket isolation** — Test 4 above
4. **Generation** — connect Ollama, check grounded answers + "not found" behavior
5. **API** — `python run.py`, hit `/health`, `/ingest`, `/query`
6. **Postman** — run the full collection
7. **UI** (not yet scaffolded — Streamlit, Phase 9)

## What's covered vs. still a stub

| Input | Content | Handling |
|---|---|---|
| PDF | Normal text | ✅ PyMuPDF text extraction |
| PDF | Scanned pages | ✅ Auto-detected (low text density) → OCR fallback |
| PDF | Tables | ✅ Extracted as markdown, kept as one un-split chunk |
| PDF | Embedded images/charts/diagrams | ✅ Extracted via PyMuPDF, run through OCR + `llava` vision description |
| PNG / JPEG (standalone) | Text/visual content | ✅ OCR + `llava` vision description |
| Mixed PDF (text + tables + images) | Combined | ✅ Each page produces separate text / table / image units, all flow into the same chunking → embedding → retrieval path |

Known stub:
- `qdrant_store.py` raises `NotImplementedError` — it's a placeholder so
  the vector-store interface is already backend-agnostic. Switching from
  Chroma later is a one-line change in `vector_store/__init__.py`.

Notes on the image pipeline:
- Small embedded images (under 100×100px) are skipped automatically —
  these are almost always icons/logos/decorative elements, not content.
- If `llava` isn't installed or Ollama isn't reachable, vision description
  fails gracefully (logs a warning, returns `""`) — OCR text is still
  captured, so ingestion never breaks because vision is unavailable.
