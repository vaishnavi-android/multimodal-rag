# Great-INDIA-Query-RAG

## Project Overview

Great-INDIA-Query-RAG is a multimodal Retrieval-Augmented Generation (RAG) application.

The system processes documents, extracts useful information, stores it in a searchable knowledge base, and generates answers to user questions using retrieved document content.

The project currently supports two knowledge buckets:

* `bucket_1`
* `bucket_2`

When a user asks a question, the system automatically searches both buckets, finds the most relevant information, and generates an answer based on the retrieved content.

---

# How the Project Works

## 1. Document Ingestion

Documents are placed inside the data folders:

```text
data/
├── bucket_1/
└── bucket_2/
```

The ingestion pipeline processes the documents through the following stages:

```text
Documents
    ↓
Parsing
    ↓
Text / OCR / Table / Image Extraction
    ↓
Preprocessing
    ↓
Deduplication
    ↓
Chunking
    ↓
Metadata
    ↓
Embeddings
    ↓
Vector Database
```

---

# Supported Content

The project can process:

* Normal PDF text
* Scanned PDF pages using OCR
* Tables
* Embedded images
* Standalone images

Different content types are processed and stored with metadata so they can be retrieved later.

---

# Query Flow

The user asks a question through the Streamlit frontend.

The query follows this flow:

```text
User Question
    ↓
Streamlit Frontend
    ↓
FastAPI Backend
    ↓
Search bucket_1
    +
Search bucket_2
    ↓
Combine Results
    ↓
Reranking
    ↓
Filter Irrelevant Chunks
    ↓
Select Best Chunks
    ↓
Build Prompt
    ↓
Ollama Model
    ↓
Generate Answer
    ↓
Return Answer and Sources
```

The user does not need to select a bucket.

Both buckets are searched automatically.

---

# Retrieval

The system retrieves relevant document chunks from both knowledge buckets.

The retrieved results are then:

1. Combined.
2. Sorted using reranking scores.
3. Checked against the relevance threshold.
4. Filtered to remove weak or irrelevant chunks.
5. Limited to the strongest chunks for answer generation.

This helps ensure that only relevant information is passed to the language model.

---

# Answer Generation

The strongest retrieved chunks are passed to the prompt generator along with the user's question.

The prompt instructs the model to generate a meaningful answer based on the retrieved content.

The system is designed to generate concise answers instead of directly returning entire document chunks.

Answer generation is performed using a local Ollama model.

---

# Unknown Questions

If the system cannot find sufficiently relevant information in the knowledge base, it returns:

```text
Relevant information was not found in the knowledge base.
```

This prevents the model from generating answers without relevant retrieved evidence.

---

# Sources

Every generated answer includes information about the chunks used to generate it.

Source information includes:

* File name
* Bucket ID
* Content type
* Page number
* Reranking score

This information is displayed in the frontend.

---

# Frontend

The project includes a Streamlit frontend.

The frontend allows users to:

* Enter a question
* Send the question to the API
* View the generated answer
* View the sources used for the answer
* See errors if the API is unavailable

---

# Backend

The backend is built using FastAPI.

The frontend sends the user's question to the API.

The API processes the question through the RAG pipeline and returns:

```json
{
    "answer": "Generated answer",
    "sources": []
}
```

---

# Logging

The project includes logging for the RAG query flow.

The log records the major steps involved in answering a question, such as:

```text
Query received
Search started
Bucket 1 searched
Bucket 2 searched
Results combined
Results filtered
Relevant chunks selected
Prompt created
Answer generated
Sources prepared
Response returned
```

This helps with debugging and understanding the complete flow from the user query to the final answer.

---

# Project Structure

```text
multimodal-rag/
│
├── data/
│   ├── bucket_1/
│   └── bucket_2/
│
├── logs/
│
├── scripts/
│   ├── ingest.py
│   ├── test_rag.py
│   └── test scripts
│
├── src/
│   │
│   ├── api/
│   │
│   ├── chunking/
│   │
│   ├── config/
│   │
│   ├── embeddings/
│   │
│   ├── generation/
│   │   ├── rag.py
│   │   ├── prompt.py
│   │   └── ollama_client.py
│   │
│   ├── ingestion/
│   │
│   ├── metadata/
│   │
│   ├── preprocessing/
│   │
│   ├── retrieval/
│   │
│   └── vector_store/
│
├── frontend/
│
├── requirements.txt
├── run.py
└── README.md
```

---

# Technologies Used

The project uses:

* Python
* PyMuPDF
* pdfplumber
* OCR
* RapidOCR
* Pillow
* Embedding models
* Vector database
* BM25 retrieval
* Reranking
* Ollama
* Qwen3
* FastAPI
* Streamlit

---

# Running the Project

## Activate the Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Ingest Documents

```bash
python scripts/ingest.py
```

## Test the RAG Pipeline

```bash
python -m scripts.test_rag
```

## Start the Backend

```bash
python run.py
```

## Start the Frontend

Run the Streamlit command for your frontend Python file:

```bash
streamlit run <frontend_file>.py
```

---

# Example Questions

```text
What is the capital of Karnataka?

Which rivers are important in Karnataka?

Which city is known as the technology hub of Karnataka?

What are the major industries in Kerala?
```

The system also handles questions that are not related to the knowledge base.

Example:

```text
What is the population of Mars?
```

Expected response:

```text
Relevant information was not found in the knowledge base.
```

---

# Current Project Flow Summary

```text
Documents
    ↓
Extract Content
    ↓
Clean Content
    ↓
Chunk Content
    ↓
Create Metadata
    ↓
Generate Embeddings
    ↓
Store in Vector Database

User Question
    ↓
Search Both Buckets
    ↓
Retrieve Results
    ↓
Rerank Results
    ↓
Filter Relevant Chunks
    ↓
Build Prompt
    ↓
Generate Answer
    ↓
Return Answer and Sources
    ↓
Display in Streamlit
```

# Current Status

The project currently has:

* Document ingestion
* Multimodal document processing
* OCR
* Table extraction
* Image processing
* Preprocessing
* Deduplication
* Chunking
* Metadata
* Embeddings
* Vector storage
* Retrieval
* Reranking
* Automatic search across both buckets
* Relevance filtering
* Grounded answer generation
* Unknown-question handling
* Source display
* FastAPI integration
* Streamlit frontend
* Query flow logging

The complete system is currently working as an end-to-end multimodal RAG application.
