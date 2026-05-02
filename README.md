# 🤖 Agentic AI Research Assistant

A production-grade AI research assistant built from scratch using Python and the Claude API — no LangChain, no heavy frameworks.

Built as a hands-on extension of the **"Working with the Claude API"** course by Anthropic.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF Ingestion | Load PDFs, chunk with overlap, embed and store in FAISS |
| 🔍 Hybrid Search | Semantic (FAISS) + lexical (BM25) combined with weighted scoring |
| 🎯 Re-ranking | Cross-encoder model re-ranks candidates for higher precision |
| ✍️ Query Rewriting | Claude rewrites vague questions into precise search queries |
| 💬 Cited Answers | Answers grounded in documents with `[N]` inline citations |
| 🌐 Web Search | Claude autonomously searches the web via Tavily when needed |
| 🔗 RAG + Tools | Unified pipeline combining document context and web search |
| 🖼️ Vision Support | Analyze images, charts, and figures from PDFs or uploads |
| ⚡ Streaming | Real-time token-by-token streaming responses |

---

## 🏗️ Architecture
research_assistant/
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── api/routes/
│   │   ├── chat.py                # /chat — streaming conversation
│   │   ├── ingest.py              # /ingest — PDF processing
│   │   ├── search.py              # /search — hybrid retrieval
│   │   ├── rag.py                 # /ask — cited answers
│   │   ├── agent.py               # /agent — autonomous tool use
│   │   ├── research.py            # /research — RAG + tools unified
│   │   └── vision.py              # /vision — multimodal analysis
│   ├── core/
│   │   └── config.py              # settings and env vars
│   ├── rag/
│   │   ├── pdf_loader.py          # PDF → pages
│   │   ├── chunker.py             # pages → overlapping chunks
│   │   ├── embedder.py            # chunks → vectors (MiniLM)
│   │   ├── vector_store.py        # FAISS index management
│   │   ├── semantic_search.py     # vector similarity search
│   │   ├── bm25_search.py         # keyword search
│   │   ├── hybrid_retriever.py    # combined search + reranking
│   │   ├── reranker.py            # cross-encoder reranking
│   │   ├── compressor.py          # context compression
│   │   └── image_extractor.py     # PDF image extraction
│   ├── services/
│   │   ├── claude_service.py      # Claude API streaming
│   │   ├── query_rewriter.py      # query rewriting/expansion
│   │   ├── tool_executor.py       # Claude tool use loop
│   │   ├── research_service.py    # unified research pipeline
│   │   └── vision_service.py      # vision pipeline
│   ├── tools/
│   │   ├── base.py                # BaseTool interface
│   │   ├── web_search.py          # Tavily web search tool
│   │   └── registry.py            # tool registry
│   ├── models/
│   │   └── schemas.py             # Pydantic schemas
│   └── utils/
│       └── prompt_builder.py      # prompt construction
├── data/
│   ├── pdfs/                      # place your PDFs here
│   ├── index/                     # FAISS index (auto-generated)
│   └── images/                    # uploaded images
├── .env                           # API keys (not committed)
└── requirements.txt

---

## 🔄 How It Works
PDF
↓ pdf_loader → chunker → embedder
FAISS Index + BM25 Corpus + Metadata
User Question
↓ query_rewriter (Claude)
↓ hybrid_search (FAISS + BM25)
↓ reranker (cross-encoder)
↓ compressor (Claude, optional)
↓ build_research_system_prompt (inject context)
↓ Claude tool loop (web search if needed)
↓ streaming answer with citations

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/harshithh-j/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```

Edit `.env` with your keys:
```env
ANTHROPIC_API_KEY=your_anthropic_key
TAVILY_API_KEY=your_tavily_key
MODEL_NAME=claude-sonnet-4-5
MAX_TOKENS=4096
```

### 5. Start the server
```bash
uvicorn app.main:app --reload
```

### 6. Visit the API docs
http://localhost:8000/docs

---

## 📡 API Endpoints

### Ingest a PDF
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"filename": "your_paper.pdf"}'
```

### Ask a question (with citations)
```bash
curl -N -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what are the main findings?", "k": 5}'
```

### Full research (documents + web)
```bash
curl -N -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"question": "what are recent developments?", "k": 5}'
```

### Analyze an image
```bash
curl -N -X POST http://localhost:8000/api/v1/vision/analyze \
  -F "question=What does this diagram show?" \
  -F "files=@/path/to/image.png"
```

### Analyze figures from a PDF
```bash
curl -N -X POST http://localhost:8000/api/v1/vision/pdf-images \
  -F "question=What do the figures show?" \
  -F "filename=your_paper.pdf" \
  -F "max_images=3"
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| LLM | Claude (Anthropic) |
| Vector Store | FAISS |
| Lexical Search | BM25 (rank-bm25) |
| Embeddings | all-MiniLM-L6-v2 |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Web Search | Tavily |
| PDF Processing | PyMuPDF |
| Image Processing | Pillow |
| Streaming | FastAPI StreamingResponse |

---

## 📚 Phases Built

| Phase | What Was Built |
|---|---|
| Phase 1 | FastAPI app + Claude streaming |
| Phase 2 | PDF ingestion + FAISS vector store |
| Phase 3 | Hybrid search (semantic + BM25) |
| Phase 4 | Answer generation with citations |
| Phase 5 | Query rewriting + expansion |
| Phase 6 | Tool system + web search |
| Phase 7 | RAG + tools unified pipeline |
| Phase 8 | Cross-encoder reranking + compression |
| Phase 9 | Multimodal vision support |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Claude API key |
| `TAVILY_API_KEY` | ⚠️ Optional | Web search (Phase 6+) |
| `MODEL_NAME` | ✅ | Default: `claude-sonnet-4-5` |
| `MAX_TOKENS` | ✅ | Default: `4096` |

---

## 📖 Course

Built as a hands-on project after completing:

**"Working with the Claude API"** — Anthropic (Skilljar)

🎓https://verify.skilljar.com/c/vkhxo9nu83wq

---

## 📄 License

MIT License — free to use, modify, and distribute.