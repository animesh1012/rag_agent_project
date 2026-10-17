import os
import shutil
import tempfile
from pathlib import Path
from constant import UI_CSS, SYSTEM_PROMPT
from model import load_llm
import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

output_dir = Path("./extracted_visuals")
output_dir.mkdir(parents=True, exist_ok=True)

def cleanup_extracted_visuals():
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Chat",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)



@st.cache_resource(show_spinner=False)
def get_llm():
    return load_llm()


llm = get_llm()

ASSISTANT_AVATAR = "✨"


def doc_to_reference(doc) -> dict:
    meta = doc.metadata or {}
    return {
        "source": meta.get("source_file", "Unknown"),
        "page": meta.get("page_number"),
        "type": meta.get("type", "text"),
        "chunk_id": meta.get("chunk_id"),
        "image_path": meta.get("image_path"),
        "content": doc.page_content,
    }


def render_thinking(searches, notes):
    """Collapsed expander listing the agent's search queries."""
    if not searches and not notes:
        return
    with st.expander(f"🧠 Thinking ({len(searches or [])} search(es))"):
        for note in notes or []:
            st.markdown(f"_{note}_")
        for s in searches or []:
            st.markdown(f"🔍 **Search query:** `{s['query']}`")


def render_references(references):
    """Collapsed expander with one nested section per retrieved chunk."""
    if not references:
        return
    with st.expander(f"📚 References ({len(references)})"):
        for n, ref in enumerate(references, 1):
            page = f", page {ref['page']}" if ref.get("page") else ""
            st.markdown(f"**[{n}] {ref['source']}{page}** · `{ref['type']}`")
            st.text(ref["content"][:1200])
            img = ref.get("image_path")
            if img and os.path.exists(img):
                st.image(img)
            if n < len(references):
                st.divider()


def extract_text(content) -> str:
    """Return text from a message chunk's content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


# ============================================================
# CUSTOM CSS — GEMINI-STYLE UI
# ============================================================

st.html(
    UI_CSS,
)


# ============================================================
# IMPORT YOUR RAG COMPONENTS
# ============================================================

# Change these imports according to your project structure.
#
# Example:
#
# from rag import create_vectorstore, create_agent
#
# OR:
#
# from agent import agent
# from vectorstore import create_vectorstore

from ingestion import create_vectorstore
from langchain.agents import create_agent
from tools import create_retriver_tool


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "document_names" not in st.session_state:
    st.session_state.document_names = []

if "processing_pdf" not in st.session_state:
    st.session_state.processing_pdf = False

if "visuals_cleared" not in st.session_state:
    cleanup_extracted_visuals()
    st.session_state.visuals_cleared = True

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-title">RAG Chat</div>
        <div class="sidebar-description">
            Upload documents and chat with your knowledge base.
        </div>
        """,
    )

    st.markdown("### 📄 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:

        current_names = sorted(f.name for f in uploaded_files)

        if st.session_state.document_names != current_names:

            if st.button(
                f"Create Knowledge Base ({len(uploaded_files)} file"
                f"{'s' if len(uploaded_files) != 1 else ''})",
                use_container_width=True,
            ):

                st.session_state.processing_pdf = True
                pdf_paths = []

                try:

                    with st.status(
                        "Creating vector store...", expanded=True
                    ) as status:

                        # Save uploaded PDFs temporarily
                        for f in uploaded_files:
                            with tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=".pdf",
                            ) as tmp:
                                tmp.write(f.getbuffer())
                                pdf_paths.append((tmp.name, f.name))

                        def on_progress(i, total, name):
                            st.write(f"Processing {i + 1}/{total}: {name}")

                        vectorstore = create_vectorstore(
                            pdf_paths,
                            output_dir,
                            progress_callback=on_progress,
                        )

                        st.session_state.vectorstore = vectorstore

                        retriever = vectorstore.as_retriever()

                        st.session_state.retriever = retriever

                        st.session_state.agent = create_agent(
                            model=llm,
                            tools=[
                                create_retriver_tool(
                                    retriever,
                                    sink=st.session_state.retrieved_docs,
                                )
                            ],
                            system_prompt=SYSTEM_PROMPT,
                        )

                        st.session_state.document_names = current_names

                        st.session_state.messages = []

                        status.update(
                            label="Knowledge base created!",
                            state="complete",
                            expanded=False,
                        )

                except Exception as e:

                    st.error(
                        f"Failed to create knowledge base:\n\n{e}"
                    )

                finally:

                    st.session_state.processing_pdf = False

                    # Remove temporary files
                    for path, _ in pdf_paths:
                        try:
                            os.unlink(path)
                        except Exception:
                            pass

    # ------------------------------------------------------------
    # Current documents
    # ------------------------------------------------------------

    if st.session_state.document_names:

        st.markdown("---")

        st.markdown("### Current Documents")

        for name in st.session_state.document_names:
            st.info(f"📄 {name}")

    # ------------------------------------------------------------
    # Clear conversation
    # ------------------------------------------------------------

    st.markdown("---")

    if st.button(
        "＋ New chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    # ------------------------------------------------------------
    # Reset everything
    # ------------------------------------------------------------

    if st.button(
        "Reset knowledge base",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.session_state.agent = None
        st.session_state.document_names = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div class="app-header">
    <div class="app-logo">✦</div>

    <div>
        <div class="app-title">RAG Assistant</div>
        <div class="app-subtitle">Ask questions about your documents</div>
    </div>
</div>
    """
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.html(
        """
        <div class="welcome-container">

            <div class="welcome-title">
                Hello, how can I help?
            </div>

            <div class="welcome-subtitle">
                Upload one or more PDFs in the sidebar, create the knowledge base, then ask away.
            </div>

        </div>
        """,
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar=ASSISTANT_AVATAR if message["role"] == "assistant" else None,
    ):
        if message["role"] == "assistant":
            render_thinking(message.get("searches"), message.get("notes"))
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_references(message.get("references"))


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask anything about your documents...",
    disabled=st.session_state.agent is None,
)



# ============================================================
# HANDLE USER QUERY
# ============================================================

if prompt and st.session_state.agent is not None:

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    history.append({"role": "user", "content": prompt})

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):

        thinking = st.status("Thinking...", expanded=True)
        response_placeholder = st.empty()

        full_response = ""
        searches = []    # [{"query": str, "k": int}]
        notes = []       # model reasoning text produced before a search
        references = []  # retrieved chunks
        st.session_state.retrieved_docs.clear()
        seen_docs = 0

        try:

            for mode, data in st.session_state.agent.stream(
                {"messages": history},
                stream_mode=["messages", "updates"],
            ):

                if mode == "messages":

                    message_chunk = data[0]

                    # Skip tool output (retrieved docs) - only show model tokens
                    if not isinstance(message_chunk, AIMessageChunk):
                        continue

                    token = extract_text(message_chunk.content)

                    if token:
                        full_response += token
                        response_placeholder.markdown(full_response + " ▌")

                    continue

                # mode == "updates": complete messages after each graph step
                for node_update in data.values():

                    for msg in (node_update or {}).get("messages", []):

                        if isinstance(msg, AIMessage) and msg.tool_calls:

                            # Text before a tool call is reasoning, not the answer
                            if full_response.strip():
                                notes.append(full_response.strip())
                                thinking.markdown(f"_{full_response.strip()}_")
                                full_response = ""
                                response_placeholder.empty()

                            for call in msg.tool_calls:
                                args = call.get("args", {})
                                searches.append(
                                    {"query": args.get("query", ""), "k": args.get("k")}
                                )
                                thinking.markdown(
                                    f"🔍 **Search query:** `{args.get('query', '')}`"
                                )

                        elif isinstance(msg, ToolMessage):

                            # Chunks recorded by the tool since the last result
                            docs = st.session_state.retrieved_docs[seen_docs:]
                            seen_docs += len(docs)
                            references.extend(doc_to_reference(d) for d in docs)
                            thinking.markdown(
                                f"📄 Retrieved {len(docs)} chunk(s)"
                            )

            if not full_response:
                full_response = "I couldn't generate a response. Please try again."

            response_placeholder.markdown(full_response)
            thinking.update(
                label=f"Thought through {len(searches)} search(es)"
                if searches
                else "Answered without searching",
                state="complete",
                expanded=False,
            )

        except Exception as e:

            full_response = (
                f"Sorry, I encountered an error: `{str(e)}`"
            )
            response_placeholder.markdown(full_response)
            thinking.update(label="Error", state="error", expanded=False)

        render_references(references)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "searches": searches,
            "notes": notes,
            "references": references,
        }
    )

    st.rerun()
