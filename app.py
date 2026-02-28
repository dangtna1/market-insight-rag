"""
app.py
------
Streamlit chat interface for InsightBot.

Run with:
    streamlit run app.py
"""

import streamlit as st
from rag import InsightBotRAG, RAGResponse

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="InsightBot",
    page_icon="🔍",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .source-card {
        background: #f0f4f8;
        border-left: 3px solid #1F4E79;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.85em;
    }
    .source-label {
        font-weight: 600;
        color: #1F4E79;
    }
    .snippet {
        color: #555;
        font-style: italic;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load RAG pipeline (cached so it only initialises once) ────────────────────

@st.cache_resource(show_spinner="Loading InsightBot...")
def load_rag():
    return InsightBotRAG()


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔍 InsightBot")
st.caption(
    "RAG-powered Q&A over market research reports. "
    "Answers are grounded in your documents with source attribution."
)
st.divider()

# ── Load pipeline ─────────────────────────────────────────────────────────────

try:
    rag = load_rag()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python ingest.py` in your terminal first, then refresh this page.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            _render_sources(msg["sources"]) if False else None  # rendered below

# ── Source renderer ───────────────────────────────────────────────────────────

def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)} documents referenced)", expanded=False):
        for i, src in enumerate(sources, 1):
            st.markdown(
                f"""<div class="source-card">
                    <div class="source-label">Source {i} — {src.file}, page {src.page}</div>
                    <div class="snippet">{src.snippet}</div>
                </div>""",
                unsafe_allow_html=True,
            )

# ── Suggested questions ───────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    suggested = [
        "What devices do children most commonly use to access the internet?",
        "How has screen time changed in the past five years?",
        "What are the key concerns parents have about children's online activity?",
        "What percentage of children use social media, and at what age do they start?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggested):
        if cols[i % 2].button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion
            st.rerun()

# ── Handle pre-filled question from buttons ───────────────────────────────────

if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")
else:
    user_input = st.chat_input("Ask a question about the research reports...")

# ── Main chat logic ───────────────────────────────────────────────────────────

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            response: RAGResponse = rag.query(user_input)

        st.markdown(response.answer)
        render_sources(response.sources)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        })

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About InsightBot")
    st.markdown("""
    **How it works:**

    1. Your PDFs are chunked into ~150-word segments
    2. Each chunk is embedded using a local HuggingFace model (`all-MiniLM-L6-v2`)
    3. Your question is embedded the same way
    4. ChromaDB finds the 5 most semantically similar chunks
    5. Those chunks + your question are passed to a local Llama 3.2 model
    6. The model answers using *only* the retrieved content

    **Why this reduces hallucinations:**
    The LLM is explicitly instructed to use only the provided context.
    If the answer isn't in the documents, it says so.
    """)
    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()
