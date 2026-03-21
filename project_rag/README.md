# ⚖️ LexAI — Legal Document Assistant

>**Retrieval-Augmented Generation (RAG)** application that lets users upload legal documents and ask questions in natural language — getting precise, source-cited answers in **English or Kannada**, powered by **Endee Vector Database**, **Groq LLM**, and **Streamlit**.

---

## 📽️ What This Project Does

LexAI is an AI-powered legal document assistant. A user uploads any legal document (contract, agreement, constitutional text, etc.) and can immediately:

- **Ask questions** in plain language and get accurate answers backed by the actual document text
- **See inline citations** — every factual claim in the answer links back to the exact passage it came from
- **Read plain-English summaries** of complex legal clauses (Simplify mode)
- **Get answers in Kannada** if the question is typed in Kannada — the system automatically detects the language and responds accordingly
- All of this works without the AI ever "making things up" — it only answers from what is actually in the document

---

## 🗂️ Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [How the RAG Pipeline Works](#-how-the-rag-pipeline-works)
- [How Endee Vector Database is Used](#-how-endee-vector-database-is-used)
- [Component Breakdown](#-component-breakdown)
- [Data Flow — Step by Step](#-data-flow--step-by-step)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Environment Variables](#-environment-variables)
- [Running the Application](#-running-the-application)
- [Interview Q&A Guide](#-interview-qa-guide)

---

## 🧠 Project Overview

### The Problem LexAI Solves

Legal documents are dense, complex, and full of jargon. A non-lawyer reading a 30-page service agreement cannot easily find answers to simple questions like "What is the termination notice period?" or "Am I liable for indirect damages?".

Standard LLMs (like GPT or LLaMA) cannot answer these questions reliably because:
1. They were not trained on your specific document
2. They hallucinate — they confidently fabricate answers
3. They cannot cite the exact sentence they are answering from

**RAG solves all three problems.** Instead of asking the LLM to rely on its training memory, RAG forces it to retrieve the relevant passage from the actual document first, then generate an answer grounded in that passage. The LLM becomes a reasoning engine, not a memory engine.


---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        USER                              │
│              (Browser — Streamlit UI)                    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT APP                          │
│                     (app.py)                             │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐  ┌──────────────┐  │
│  │ Upload Page  │   │  Chat Page   │  │ Simplify Page│  │
│  │ (file input) │   │ (Q&A + cite) │  │ (plain lang) │  │
│  └──────┬───────┘   └──────┬───────┘  └──────┬───────┘  │
│         │                  │                  │          │
│         ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────┐    │
│  │           CORE PIPELINE FUNCTIONS                │    │
│  │  extract_pdf_text() / detect_lang()              │    │
│  │  chunk_text() / ingest_doc()                     │    │
│  │  do_search() / generate_answer()                 │    │
│  └──────────┬──────────────────────┬───────────────┘    │
└─────────────┼──────────────────────┼────────────────────┘
              │                      │
              ▼                      ▼
┌─────────────────────┐   ┌──────────────────────────────┐
│  ENDEE VECTOR DB    │   │      GROQ LLM API            │
│  (localhost:8080)   │   │  (llama-3.3-70b-versatile)   │
│                     │   │                              │
│  • HNSW Index       │   │  • Answer generation         │
│  • 384-dim vectors  │   │  • Simplification            │
│  • Cosine similarity│   │  • Citation-aware prompting  │
│  • msgpack response │   │                              │
└─────────────────────┘   └──────────────────────────────┘
              ▲
              │ embeddings
              │
┌─────────────────────────┐
│  SENTENCE TRANSFORMER   │
│  all-MiniLM-L6-v2       │
│  (384-dim dense vectors) │
└─────────────────────────┘
```

---

## 🔄 How the RAG Pipeline Works

RAG has two distinct phases. Understanding both phases is essential.

### Phase 1 — Ingestion (happens once, when user uploads)

```
PDF/TXT file
     │
     ▼
extract_pdf_text()          ← Clean Unicode, remove garbage chars
     │
     ▼
detect_lang()               ← Count Kannada Unicode codepoints (U+0C80–U+0CFF)
     │
     ▼
chunk_text(size=500 words)  ← Split into overlapping word windows
     │
     ▼  [for each chunk]
SentenceTransformer.encode()← Convert text → 384-dimensional float32 vector
     │
     ▼
Endee /vector/insert API    ← Store {id, vector, meta: {doc, text}} in HNSW index
```

**Why chunk?** The embedding model has a token limit and works best on focused passages. Chunking into 500-word windows means each vector represents one coherent topic/clause, making retrieval more precise.

**Why 384 dimensions?** The `all-MiniLM-L6-v2` model outputs 384-dim vectors. This is a sweet spot — small enough to be fast, large enough to capture semantic meaning.

### Phase 2 — Retrieval + Generation (happens on every question)

```
User question
     │
     ▼
detect_question_lang()      ← Is the question in Kannada or English?
     │
     ▼
SentenceTransformer.encode()← Convert question → 384-dim query vector
     │
     ▼
Endee /search API           ← Find top-4 most similar vectors via HNSW cosine search
     │
     ▼  [returns source passages + similarity scores]
build context string        ← Concatenate retrieved passages as [Source 1]: ..., [Source 2]: ...
     │
     ▼
Groq LLM (llama-3.3-70b)   ← Prompt: "Answer using ONLY these sources. Cite with [1][2]."
     │
     ▼
render_answer()             ← Convert [1][2] markers → chain-link SVG icons
     │
     ▼
Display in chat UI          ← Show answer + source cards below
```

**The critical insight:** The same `all-MiniLM-L6-v2` model is used for **both** encoding the document chunks at ingestion time AND encoding the user's question at query time. This is essential — for cosine similarity to be meaningful, the query vector and the stored vectors **must live in the same embedding space**. If you used different models, the similarity scores would be meaningless.

---

## 🗄️ How Endee Vector Database is Used

Endee is the core retrieval engine of the entire system. Here is every interaction the app has with it:

### 1. Health Check — `GET /api/v1/health`
```python
requests.get(f"{ENDEE_URL}/api/v1/health", timeout=2)
```
Called on every page load to check if Endee is reachable. Determines whether the app uses semantic search or falls back to raw text chunking.

### 2. Index Creation — `POST /api/v1/index/create`
```python
{
  "index_name": "LEGALDOC",
  "dim": 384,              # must match embedding model output
  "space_type": "cosine",  # similarity metric
  "precision": "float32",  # vector precision
  "M": 16,                 # HNSW graph connectivity parameter
  "ef_con": 200            # HNSW construction-time search depth
}
```
Creates a **HNSW (Hierarchical Navigable Small World)** index. HNSW is the most efficient approximate nearest-neighbor algorithm — it builds a multi-layer graph where each node connects to its closest neighbors, allowing O(log n) search even over millions of vectors.

- **`M=16`** controls how many neighbors each node connects to. Higher M = better accuracy, more memory.
- **`ef_con=200`** controls search depth during index construction. Higher = better graph quality, slower build.
- **`cosine`** similarity is used because it measures the angle between vectors (semantic direction) rather than Euclidean distance (magnitude), which is more meaningful for text embeddings.

### 3. Vector Insertion — `POST /api/v1/index/{name}/vector/insert`
```python
[
  {
    "id":     "filename.pdf-1706123456-0",   # unique ID: name + timestamp + chunk_index
    "vector": [0.023, -0.187, 0.441, ...],  # 384 float32 values
    "meta":   '{"doc": "contract.pdf", "text": "The payment shall be made..."}'
  },
  ...
]
```
Inserts all document chunk vectors in a single batch request. The `meta` field is a JSON string that gets stored alongside the vector — this is what gets returned in search results so we can show the actual text to the user.

### 4. Similarity Search — `POST /api/v1/index/{name}/search`
```python
# Request
{"vector": [0.023, -0.187, ...], "k": 4}

# Response (msgpack binary format)
[
  {"meta": '{"doc": "contract.pdf", "text": "..."}', "score": 0.923},
  {"meta": '{"doc": "contract.pdf", "text": "..."}', "score": 0.871},
  ...
]
```
This is the heart of RAG. The query vector is compared against all stored vectors using HNSW approximate nearest-neighbor search. Returns the top-k most semantically similar document chunks.

**Why msgpack?** Endee returns results in MessagePack binary format (not JSON) for performance — binary serialization is faster and smaller than JSON for numeric data. The app decodes it with `msgpack.unpackb(response.content, raw=False)`.

### 5. Auto-Recovery — Delete + Recreate
```python
# If Endee loses index files (common on free cloud hosts after restart):
DELETE /api/v1/index/{name}/delete
POST   /api/v1/index/create
POST   /api/v1/index/{name}/vector/insert  # retry
```
On ephemeral deployments (like Render's free tier), Endee's data directory is wiped on restart. The app detects the `"required files missing for index"` error and silently rebuilds the index, then retries the insertion — fully transparent to the user.

### What Endee Replaces

Without Endee, you would need:
- A traditional database + brute-force cosine similarity (O(n) per query, slow at scale)
- Or Pinecone/Weaviate/Qdrant (paid, cloud-only)
- Or FAISS (local but requires extra glue code)

Endee provides a **self-hosted, REST-API-first vector database** with HNSW indexing, making it ideal for local development and lightweight cloud deployments.

---

## 🧩 Component Breakdown

### `extract_pdf_text(file_bytes)` — PDF Processing
Uses `pypdf` (primary) with `pdfplumber` as fallback. After extraction, applies a Unicode regex filter:
```python
re.sub(r'[^\x20-\x7E\u0C80-\u0CFF\u0900-\u097F\n\r\t]', ' ', text)
```
This keeps printable ASCII, Kannada Unicode, Devanagari Unicode, and whitespace — stripping the garbage byte sequences that cause garbled text in the chat.

### `detect_lang(text)` — Document Language Detection
```python
return "kn" if sum(1 for c in text if '\u0C80' <= c <= '\u0CFF') > 50 else "en"
```
Counts characters in the Kannada Unicode block. If more than 50 Kannada characters are found, the document is classified as Kannada. This threshold prevents false positives from documents with occasional Kannada words.

### `detect_question_lang(question)` — Question Language Detection
```python
return "kn" if sum(1 for c in question if '\u0C80' <= c <= '\u0CFF') > 5 else "en"
```
Same logic but with a lower threshold (5 chars) since questions are short. The answer language follows the question language, not the document language — this is the key design decision that makes the bilingual feature work naturally.

### `chunk_text(text, size=500)` — Document Chunking
```python
words = text.split()
return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]
```
Simple word-window chunking. 500 words ≈ 3-4 paragraphs — enough context for meaningful embedding, small enough for precise retrieval.

### `generate_answer(question, sources, q_lang)` — LLM Prompting
The prompt is carefully engineered:
```
You are a precise legal document assistant.
Answer using ONLY the sources below. Do NOT fabricate.
After each factual statement cite like [1] or [1][2].
[Language rule: reply entirely in Kannada / reply in English]

[Source 1]: <retrieved passage>
[Source 2]: <retrieved passage>

Question: <user question>
Answer:
```
Key design decisions:
- **"ONLY the sources"** — prevents hallucination
- **Citation markers [1][2]** — LLM is trained to follow this pattern reliably
- **Language rule** — explicit instruction overrides the LLM's default behavior

### `render_answer(answer_text, sources)` — Citation Rendering
Converts `[1]` markers to clickable chain-link SVG icons:
```python
re.sub(r'\[(\d+)\]', make_cite_html, answer_text)
```
Each icon: (a) anchors to the corresponding source card `#src-1`, (b) shows a tooltip preview of the source text on hover, (c) is styled with the Feather link SVG icon.

### `ingest_doc(text, model, doc_name)` — Silent Background Ingestion
Called after the user clicks "Start Chatting →". Designed to be fire-and-forget — no loading bar, no status messages. The ingestion happens while the page transitions to the chat view. If Endee is offline, the app falls back to direct text context for the LLM.

---

## 🔢 Data Flow — Step by Step

Here is the complete end-to-end flow from upload to answered question:

```
Step 1: User uploads contract.pdf
        │
        ▼
Step 2: extract_pdf_text() reads all pages, strips garbage chars
        Result: clean string of ~5,000 words
        │
        ▼
Step 3: detect_lang() → "en" (English)
        │
        ▼
Step 4: chunk_text(500) → 10 chunks of ~500 words each
        │
        ▼
Step 5: SentenceTransformer("all-MiniLM-L6-v2").encode(chunks)
        Result: 10 × 384 float32 numpy arrays
        │
        ▼
Step 6: POST /api/v1/index/LEGALDOC/vector/insert
        Payload: [{id, vector[384], meta: {doc, text}}, ...]
        Endee builds HNSW graph, stores on disk
        │
        ▼
Step 7: Session state updated → doc_text, doc_name, doc_lang set
        UI transitions to chat_page()
        │
        ▼
Step 8: User types "What is the payment schedule?"
        │
        ▼
Step 9: detect_question_lang() → "en"
        │
        ▼
Step 10: SentenceTransformer.encode(["What is the payment schedule?"])
         Result: 1 × 384 query vector
         │
         ▼
Step 11: POST /api/v1/index/LEGALDOC/search
         Payload: {vector: [...384 floats...], k: 4}
         Endee runs HNSW approximate nearest-neighbor search
         Result: top 4 chunks with cosine similarity scores
         Response decoded from msgpack binary
         │
         ▼
Step 12: Context string assembled:
         "[Source 1]: 4.1 The Client agrees to pay INR 10,00,000..."
         "[Source 2]: 4.3 Late payments shall incur 2% per month..."
         ...
         │
         ▼
Step 13: Groq API called (llama-3.3-70b-versatile)
         Prompt includes context + question + citation instructions
         Result: "The total fee is INR 10,00,000 [1]. Payments are made
                  in installments [1]. Late payment incurs 2% monthly [2]."
         │
         ▼
Step 14: render_answer() converts [1][2] → chain-link SVG icons
         Source cards rendered below the answer
         │
         ▼
Step 15: User sees formatted answer with clickable source references
```

---

## ✨ Key Features

| Feature | How It Works |
|---|---|
| **Semantic Q&A** | Question embedded → cosine search in Endee → top passages → LLM answer |
| **Inline citations** | LLM instructed to cite `[1][2]` → rendered as chain-link SVG icons |
| **Source cards** | Retrieved passages shown below each answer with keyword highlighting |
| **Auto language** | Kannada Unicode counting → question lang → LLM reply lang |
| **Kannada support** | Full Kannada UI prompts, answer generation, and source display |
| **PDF extraction** | `pypdf` + `pdfplumber` fallback + Unicode garbage stripping |
| **Document simplification** | LLM extracts 8–12 provisions and explains each in plain language |
| **Auto-recovery** | Detects Endee index loss → deletes + recreates → retries insert |
| **No hallucination** | Prompt enforces "answer ONLY from sources" |
| **Single page UX** | Upload and chat in one page — no navigation |
| **Session isolation** | Each browser session has its own state; no cross-user data |

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **UI Framework** | Streamlit | Python-native web interface |
| **Vector Database** | Endee (endee.io) | HNSW-based vector storage and similarity search |
| **Embedding Model** | `all-MiniLM-L6-v2` | Text → 384-dim dense vector conversion |
| **LLM Provider** | Groq API | Fast inference on `llama-3.3-70b-versatile` |
| **PDF Parsing** | pypdf + pdfplumber | Text extraction from PDF files |
| **HTTP Client** | requests | REST calls to Endee and Groq |
| **Binary Decoding** | msgpack | Decode Endee's binary search responses |
| **Env Management** | python-dotenv | Load secrets from `.env` file |
| **Numeric Computing** | numpy | Float32 array handling for vectors |
| **Fonts** | Google Fonts (Inter + JetBrains Mono) | Typography |

### Why These Choices



**Groq over OpenAI:** Groq's hardware (LPU chips) delivers ~10x faster inference than GPU-based APIs. For a chat application where latency is visible, this matters.

**`all-MiniLM-L6-v2` over larger models:** This model is 80MB, loads in under a second, produces 384-dim vectors, and runs on CPU without GPU. For document Q&A, it achieves near state-of-the-art retrieval quality at a fraction of the cost of larger models.

**Streamlit over FastAPI + React:** Legal professionals are the target users, not developers. Streamlit lets the entire application live in a single Python file with no frontend build step.

---

## 📁 Project Structure

```
legal_rag/
│
├── app.py                   # Entire application — UI + pipeline + API calls
│   │
│   ├── inject_theme()       # Full dark CSS — Inter font, token design system
│   ├── init_state()         # Streamlit session state initialization
│   │
│   ├── ── ENDEE FUNCTIONS ──
│   ├── is_endee_ok()        # Health check GET /api/v1/health
│   ├── load_model()         # Cached sentence-transformer model loader
│   ├── create_index()       # POST /api/v1/index/create
│   ├── delete_index()       # DELETE /api/v1/index/{name}/delete
│   ├── chunk_text()         # 500-word window chunker
│   ├── ingest_doc()         # Full ingestion pipeline with auto-recovery
│   ├── do_search()          # POST /api/v1/index/{name}/search + msgpack decode
│   ├── parse_meta()         # Robust multi-format Endee metadata parser
│   │
│   ├── ── LANGUAGE FUNCTIONS ──
│   ├── detect_lang()        # Document language via Kannada Unicode counting
│   ├── detect_question_lang() # Question language detection
│   ├── extract_pdf_text()   # pypdf + pdfplumber + Unicode cleaning
│   │
│   ├── ── LLM FUNCTIONS ──
│   ├── llm_call()           # Groq API wrapper
│   ├── generate_answer()    # Citation-aware RAG prompt construction
│   ├── render_answer()      # [N] → chain-link SVG + markdown→HTML conversion
│   ├── simplify_doc()       # Plain-language provision extraction
│   │
│   ├── ── UI FUNCTIONS ──
│   ├── render_sidebar()     # Nav + doc info sidebar
│   ├── upload_page()        # Welcome + file uploader (no chat widgets)
│   ├── chat_page()          # Chat history + input bar (no upload widgets)
│   ├── home_view()          # Router: upload_page() or chat_page()
│   ├── simplify_view()      # Plain-language cards view
│   └── main()               # App entry point
│
├── ingest.py                # CLI batch ingestor (.txt / .pdf files)
├── query.py                 # CLI query engine (terminal-based Q&A)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9 or newer
- Docker (for running Endee)
- A Groq API key — free at [console.groq.com](https://console.groq.com)

### Step 1 — Start Endee Vector Database

```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```

Verify Endee is running:
```bash
curl http://localhost:8080/api/v1/health
# Expected: 200 OK
```

### Step 2 — Clone and Install

```bash
git clone <your-repo-url>
cd legal_rag
pip install -r requirements.txt
```

### Step 3 — Configure Environment

```bash
cp .env.example .env
# Edit .env — add your GROQ_API_KEY
```

### Step 4 — Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`

---

## 🌐 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | From console.groq.com |
| `ENDEE_URL` | No | `http://localhost:8080` | Full URL to Endee server |
| `ENDEE_HOSTPORT` | No | — | Alternative: host:port (e.g. for Render internal networking) |
| `INDEX_NAME` | No | `LEGALDOC` | Name of the vector index in Endee |
| `ENDEE_AUTH_TOKEN` | No | `""` | Auth token if Endee started with `NDD_AUTH_TOKEN` |

`ENDEE_URL` takes precedence over `ENDEE_HOSTPORT`. If neither is set, defaults to `http://localhost:8080`.

---

## 📦 Requirements

```
streamlit>=1.35.0
sentence-transformers>=2.7.0
groq>=0.9.0
requests>=2.31.0
numpy>=1.26.0
python-dotenv>=1.0.0
msgpack>=1.0.7
pypdf>=4.0.0
reportlab>=4.1.0
```

Optional (better PDF extraction):
```
pdfplumber>=0.11.0
```

---

## 🚀 Deployment on Render

1. Push to GitHub
2. Render → New → Blueprint → connect repo
3. Set `GROQ_API_KEY` environment variable
4. Deploy — auto-recovery handles ephemeral storage restarts



---

## 📄 License

Built on top of the [Endee](https://github.com/endee-io/endee) open-source vector database, licensed under the **Apache License 2.0**.



