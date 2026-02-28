# InsightBot 🔍
### RAG-powered Q&A over market research reports

A complete end-to-end Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural language questions over PDF market research reports — with source-attributed, hallucination-resistant answers. Built entirely on free, local tools (no API keys required).

---

## Architecture

```
User Question
     │
     ▼
[HuggingFace Embeddings]        ← all-MiniLM-L6-v2 (runs locally)
     │
     ▼
[ChromaDB Vector Search]        ← finds top-5 most relevant chunks
     │
     ▼
[Prompt Builder]                ← injects chunks + strict grounding instructions
     │
     ▼
[Ollama LLM — Llama 3.2]       ← generates answer using only retrieved context
     │
     ▼
[Source Citations]              ← file name + page number for every answer
```

**Why this reduces hallucinations:** The LLM is explicitly instructed to answer *only* from the retrieved context. If the documents don't contain the answer, it says so — rather than inventing one.

---

## Setup (one-time)

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/YOUR_USERNAME/insightbot.git
cd insightbot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Install Ollama and pull the LLM

Ollama runs LLMs locally on your machine.

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from https://ollama.com/download

# Pull the model (~2GB, one-time download)
ollama pull llama3.2
```

### 3. Add your PDF reports

Download one or more market research PDFs and place them in the `data/` folder.

**Recommended free reports to start with:**
- [childrens-media-literacy-report-2025](https://www.ofcom.org.uk/siteassets/resources/documents/research-and-data/media-literacy-research/children/childrens-media-use-and-attitudes-report-2025/childrens-media-literacy-report-2025.pdf?v=396621)
- ...

```
insightbot/
└── data/
    ├── childrens-media-literacy-report-2025.pdf
    └── ...
```

### 4. Ingest documents (run once, or when you add new PDFs)

```bash
python ingest.py
```

This will:
- Load and parse all PDFs
- Split them into overlapping chunks
- Download the embedding model (~80MB, first run only)
- Store all embeddings in a local ChromaDB database

Expected output:
```
[1/3] Loading PDFs from 'data/'...
  Loading: childrens-media-literacy-report-2025.pdf
    → <no_pages> pages loaded
[2/3] Chunking documents...
  Total chunks created: <no_chunks>
[3/3] Embedding and storing in ChromaDB...
  ✓ Done! <no_chunks> chunks stored in 'chroma_db/'
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Example questions to try

- *"What devices do children most commonly use to access the internet?"*
- *"How has screen time changed in the past five years?"*
- *"What are the main concerns parents have about their children's online activity?"*
- *"What percentage of children use social media, and at what age do they start?"*

---

## Project structure

```
insightbot/
├── app.py              Streamlit chat UI
├── rag.py              Core RAG pipeline (retrieval + generation)
├── ingest.py           Document ingestion pipeline
├── requirements.txt    Python dependencies
├── data/               Place your PDF reports here
└── chroma_db/          Auto-generated vector store (gitignored)
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| `all-MiniLM-L6-v2` embeddings | 80MB, fast, strong semantic similarity — no API key |
| Chunk size: 800 chars / 150 overlap | Balances context richness vs retrieval precision for report-style text |
| Top-K: 5 chunks | Enough context for nuanced answers without overwhelming the prompt |
| Strict grounding prompt | Explicitly forbids LLM from using outside knowledge — core hallucination mitigation |
| Source attribution | Every answer surfaces file + page number — enables user verification |

---

## Tech stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector database | ChromaDB (local) |
| LLM | Llama 3.2 via Ollama (local) |
| Orchestration | LangChain |
| UI | Streamlit |
| PDF parsing | PyPDF |
