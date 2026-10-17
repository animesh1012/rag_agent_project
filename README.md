# RAG Agent

A Retrieval-Augmented Generation (RAG) project that lets you chat with your PDF documents. It parses PDFs with layout awareness (text, tables, and figures), indexes them in a hybrid dense + sparse vector store, and answers questions with an LLM agent that cites its sources.

The repository shows the project's evolution in three versions:

| Version | Folder | What it is | LLM / Embeddings |
|---|---|---|---|
| V1 | `V1_ollama/` | Fully local notebooks (Ollama) | `qwen3:8b` / `bge-m3` via Ollama |
| V2 | `V2_openai/` | Notebooks using OpenAI, with an agentic retrieval tool | `gpt-6-luna` / `text-embedding-3-small` |
| V3 | `V3_RAG_AGENT_APP/` | Streamlit chat app built on the V2 pipeline | `gpt-6-luna` / `text-embedding-3-small` |

V3 is the most complete version and the recommended starting point.

![Ingestion pipeline](ingestion_pipeline.png)

## How it works

### 1. Ingestion

1. **Parse** the PDF with [Docling](https://github.com/docling-project/docling) (OCR, table structure detection, and picture extraction enabled).
2. **Tables** are exported to DataFrames and chunked into row batches (10 rows per chunk). Each chunk repeats the column headers and records its page number.
3. **Figures and charts** are filtered by size and page-area heuristics. Those that pass are saved as PNGs and transcribed by a vision-capable LLM into a Markdown table plus a short summary, grounded in the surrounding page text.
4. **Text** is split with Docling's `HybridChunker` (max 512 tokens, tokenized with `all-MiniLM-L6-v2`) and contextualized with its heading structure.
5. **Index**: every chunk is embedded with OpenAI `text-embedding-3-small` (1024 dimensions) and a BM25 sparse vector (`Qdrant/bm25` via FastEmbed), then stored in an in-memory Qdrant collection for hybrid search.

### 2. Retrieval and answering (V3 app)

- The LLM is wrapped in a LangChain agent that has one tool, `retrieve_docs(query, k)`.
- The tool runs hybrid (semantic + keyword) search over the knowledge base and returns chunks with metadata: `source_file`, `page_number`, and `type` (`text`, `table`, or `chart_or_figure`).
- The agent may search up to a few times with refined queries before answering. Answers must come from retrieved chunks and cite them as `(source_file, p. N)`.
- The UI streams the answer and shows collapsible panels for the agent's search queries and the retrieved references, including extracted figure images.

## Repository layout

```
RAG_AGENT/
├── V1_ollama/                  # Local (Ollama) notebooks
│   ├── Step_1_Ingestion_flow_qdrant.ipynb
│   ├── Step_2_Naive_RAG.ipynb  # LangGraph retrieve -> answer pipeline
│   ├── requirements.txt
│   └── rag_paper.pdf           # Sample document
├── V2_openai/                  # OpenAI notebooks
│   ├── Step_1_Ingestion_flow_qdrant.ipynb
│   ├── Step_2_Agentic_RAG.ipynb  # Agent with a retrieval tool
│   ├── langchain_qdrant/       # Local Qdrant storage from the ingestion run
│   ├── requirements.txt
│   └── rag_paper.pdf
├── V3_RAG_AGENT_APP/           # Streamlit application
│   ├── app.py                  # Streamlit UI, session state, chat loop
│   ├── ingestion.py            # Builds the hybrid Qdrant vector store
│   ├── document_parser.py      # Docling parsing, table/figure/text chunking
│   ├── tools.py                # retrieve_docs tool for the agent
│   ├── model.py                # LLM and embedding model loaders
│   ├── constant.py             # UI CSS and the agent system prompt
│   ├── requirements.txt
│   └── rag_paper.pdf
└── ingestion_pipeline.png      # Ingestion flow diagram
```

## Getting started

### Prerequisites

- Python 3.11
- An OpenAI API key (V2 and V3)
- For V1 only: [Ollama](https://ollama.com) running locally, with the `qwen3:8b` and `bge-m3` models pulled:

  ```bash
  ollama pull qwen3:8b
  ollama pull bge-m3
  ```

### Run the Streamlit app (V3)

```bash
cd V3_RAG_AGENT_APP
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Create a .env file with your API key
echo OPENAI_API_KEY=your-key-here > .env

streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`):

1. Upload one or more PDFs in the sidebar.
2. Click **Create Knowledge Base**. Ingestion can take a while, especially for PDFs with many figures, because each figure is sent to the LLM for transcription.
3. Ask questions in the chat box.

Use **New chat** to clear the conversation, or **Reset knowledge base** to remove the indexed documents.

### Run the notebooks (V1 / V2)

1. Install the dependencies for the version you want:

   ```bash
   cd V2_openai            # or V1_ollama
   pip install -r requirements.txt
   ```

2. For V2, add `OPENAI_API_KEY` to a `.env` file in that folder.
3. Run `Step_1_Ingestion_flow_qdrant.ipynb` to build the index, then run the Step 2 notebook to ask questions.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `OPENAI_API_KEY` | `.env` in the app or notebook folder | required for V2 and V3 |
| Chat model | `load_llm()` in `V3_RAG_AGENT_APP/model.py` | `gpt-6-luna`, with `reasoning_effort="none"` |
| Embedding model | `load_embedding_model()` in `V3_RAG_AGENT_APP/model.py` | `text-embedding-3-small`, 1024 dimensions |
| Table chunk size | `create_langchain_documents()` in `document_parser.py` | 10 rows per chunk |
| Text chunk size | `HybridChunker` in `document_parser.py` | 512 tokens |
| Figure filter thresholds | `is_relevant_picture()` in `document_parser.py` | min 100×80 px, min 12,000 px² area |
| Search depth (`k`) | Agent system prompt in `constant.py` | 4 by default, 6–8 for broad questions |

## Notes and limitations

- **Vector store is in memory.** The V3 app builds a new Qdrant collection for each upload and discards it when the app restarts. Nothing is persisted between sessions.
- **Figure transcriptions are machine-generated.** Values read from charts may be approximate. The agent is instructed to say so.
- **Scanned PDFs** are handled by Docling's OCR, which is slower and less accurate than parsing native text.
- **Keep your API key out of version control.** Add `.env` to `.gitignore` before pushing.
- **V2 notebook mismatch.** The V2 Step 2 notebook imports `langchain_chroma`, but ingestion stores vectors in Qdrant. The V3 app uses Qdrant throughout.

## Tech stack

- [Docling](https://github.com/docling-project/docling): PDF parsing, OCR, table and figure extraction, hybrid chunking
- [LangChain](https://www.langchain.com/) and LangGraph: agents, tools, and retrieval
- [Qdrant](https://qdrant.tech/) (`qdrant-client`, `langchain-qdrant`): hybrid dense + sparse vector search
- [FastEmbed](https://github.com/qdrant/fastembed): BM25 sparse embeddings
- [OpenAI](https://platform.openai.com/) and [Ollama](https://ollama.com): LLM and embedding providers
- [Streamlit](https://streamlit.io/): chat UI

## Sample data

Each version includes `rag_paper.pdf`, a research paper you can use to try the pipeline without any setup.
