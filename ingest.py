"""
ingest.py
---------
Load PDFs from ./data, chunk them, embed with a local HuggingFace model,
and persist to a local ChromaDB vector store.

Run once (or re-run whenever you add new documents):
    python ingest.py
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

# all-MiniLM-L6-v2: lightweight (80MB), fast, strong semantic similarity
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk size in characters. 800 chars ≈ ~150 words — good balance between
# context richness and retrieval precision for report-style documents.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150  # overlap so sentences don't get cut at boundaries


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_pdfs(data_dir: Path) -> list:
    """Load all PDFs from data_dir. Returns list of LangChain Documents."""
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[ERROR] No PDF files found in '{data_dir}/'.")
        print("  → Download a report (e.g. Ofcom Children's Media Use) and place it there.")
        sys.exit(1)

    all_docs = []
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        # Tag each page with its source filename for citation later
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name
        all_docs.extend(docs)
        print(f"    → {len(docs)} pages loaded")

    return all_docs


def chunk_documents(docs: list) -> list:
    """
    Split documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words — keeping chunks semantically coherent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"\n  Total chunks created: {len(chunks)}")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    """
    Embed chunks with HuggingFace model and persist to ChromaDB.

    First run downloads the model (~80MB). Subsequent runs are instant.
    """
    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}")
    print("  (First run downloads ~80MB — subsequent runs are instant)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"  Embedding {len(chunks)} chunks and saving to '{CHROMA_DIR}/'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="insightbot",
    )

    return vectorstore


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  InsightBot — Document Ingestion Pipeline")
    print("=" * 55)

    # 1. Validate data directory
    if not DATA_DIR.exists():
        DATA_DIR.mkdir()
        print(f"\n[INFO] Created '{DATA_DIR}/' — add your PDFs there and re-run.")
        sys.exit(0)

    # 2. Load PDFs
    print(f"\n[1/3] Loading PDFs from '{DATA_DIR}/'...")
    docs = load_pdfs(DATA_DIR)
    print(f"\n  Total pages loaded: {len(docs)}")

    # 3. Chunk
    print("\n[2/3] Chunking documents...")
    chunks = chunk_documents(docs)

    # 4. Embed + store
    print("\n[3/3] Embedding and storing in ChromaDB...")
    vectorstore = build_vectorstore(chunks)

    print("\n" + "=" * 55)
    print(f"  ✓ Done! {len(chunks)} chunks stored in '{CHROMA_DIR}/'")
    print("  → Now run:  streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
