"""
rag.py
------
Core RAG pipeline: given a user question, retrieve the most relevant
document chunks from ChromaDB, then pass them to a local Ollama LLM
to generate a grounded, source-attributed answer.

This module is imported by app.py (the Streamlit UI).
"""

from pathlib import Path
from dataclasses import dataclass

import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Configuration ────────────────────────────────────────────────────────────

CHROMA_DIR = Path("chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many chunks to retrieve per query.
# More = richer context, but longer prompts and slower responses.
TOP_K = 5

# Ollama model to use. llama3.2 is small (~2GB) and runs well on CPU.
# Other options: mistral, phi3, gemma2
OLLAMA_MODEL = "llama3.2"

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Source:
    """Represents a single source citation."""
    file: str
    page: int
    snippet: str   # short excerpt from the chunk


@dataclass
class RAGResponse:
    """Returned by query(). Contains the answer and its sources."""
    answer: str
    sources: list[Source]
    question: str


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class InsightBotRAG:
    """
    End-to-end RAG pipeline.

    Architecture:
        1. User question → embed with same model used at ingest time
        2. Similarity search in ChromaDB → top-K most relevant chunks
        3. Build a prompt: system instruction + retrieved chunks + question
        4. Send to local Ollama LLM → get grounded answer
        5. Return answer + source citations
    """

    def __init__(self):
        print("[InsightBot] Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Vector store not found at '{CHROMA_DIR}/'. "
                "Please run `python ingest.py` first."
            )

        print("[InsightBot] Connecting to ChromaDB...")
        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embeddings,
            collection_name="insightbot",
        )
        print("[InsightBot] Ready.")

    def _retrieve(self, question: str) -> list:
        """
        Semantic similarity search: returns top-K chunks most relevant
        to the question, along with their relevance scores.
        """
        results = self.vectorstore.similarity_search_with_score(question, k=TOP_K)
        return results  # list of (Document, float_score)

    def _build_prompt(self, question: str, chunks: list) -> str:
        """
        Construct the prompt injected into the LLM.

        Key design decisions:
        - Explicitly instruct the model to ONLY use provided context
          (this is what reduces hallucination)
        - Ask it to cite sources inline
        - Tell it to say "I don't know" if the context doesn't answer
        """
        context_blocks = []
        for i, (doc, score) in enumerate(chunks, 1):
            source_file = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "?")
            context_blocks.append(
                f"[Source {i}: {source_file}, page {page}]\n{doc.page_content.strip()}"
            )
        context_text = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are InsightBot, an expert research analyst assistant.
Your job is to answer questions using ONLY the provided source documents.

STRICT RULES:
1. Base your answer exclusively on the context below — do not use outside knowledge.
2. Cite your sources inline using [Source N] notation.
3. If the context does not contain enough information to answer, say:
   "The provided documents don't contain enough information to answer this question."
4. Be concise and precise. Avoid padding or filler phrases.

--- CONTEXT START ---
{context_text}
--- CONTEXT END ---

QUESTION: {question}

ANSWER (with inline source citations):"""

        return prompt

    def query(self, question: str) -> RAGResponse:
        """
        Full RAG pipeline: retrieve → augment → generate → return.

        Parameters
        ----------
        question : str
            The user's natural language question.

        Returns
        -------
        RAGResponse
            Contains the LLM answer and a list of Source citations.
        """
        # Step 1: Retrieve relevant chunks
        chunks = self._retrieve(question)

        # Step 2: Build augmented prompt
        prompt = self._build_prompt(question, chunks)

        # Step 3: Send to local Ollama LLM
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response["message"]["content"].strip()

        # Step 4: Build source citations from retrieved chunks
        sources = []
        seen = set()
        for doc, score in chunks:
            file = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "?")
            key = (file, page)
            if key not in seen:
                seen.add(key)
                sources.append(Source(
                    file=file,
                    page=int(page) + 1 if isinstance(page, int) else page,
                    snippet=doc.page_content[:200].strip() + "...",
                ))

        return RAGResponse(answer=answer, sources=sources, question=question)
