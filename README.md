# ContextCore

> A persistent personal context layer that grounds AI conversations in verified user truth — and flags when AI contradicts it.

## The problem

Every AI conversation starts from zero. Users re-explain their job, goals, constraints, and context every single time. And when AI does "remember" — it hallucinates details confidently. There's no accountability layer. No source of truth.

## What ContextCore does

| Module | What it does |
|---|---|
| **Truth Document Builder** | Define your personal context once — career, projects, goals, constraints |
| **Context-Grounded Conversations** | Every AI conversation is automatically grounded via RAG |
| **Contradiction Detection** | AI output that contradicts your truth document gets flagged in real time |

## Tech stack

- **Language**: Python
- **LLM**: Groq (`openai/gpt-oss-20b`) — free tier
- **Embeddings**: local `sentence-transformers` (`all-MiniLM-L6-v2`) — no API key required
- **Vector DB**: ChromaDB (local)
- **RAG Framework**: LangChain
- **Contradiction Detection**: LLM-as-judge pattern
- **Frontend**: Streamlit

## Project structure

```
ContextCore/
├── data/
│   └── truth_document.json   # Your personal context source of truth
├── src/
│   ├── schema.py             # Pydantic models for truth document validation
│   ├── chunker.py            # Splits truth doc into RAG-ready chunks
│   ├── vector_store.py       # ChromaDB setup, local embedding pipeline, and semantic search
│   ├── rag_pipeline.py       # LangChain retrieval chain (conversational RAG via Groq)
│   ├── contradiction.py      # LLM-as-judge contradiction detector
│   └── app.py                # Streamlit UI — live demo
├── .env.example
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/shrijita24/ContextCore
cd ContextCore
pip install -r requirements.txt
cp .env.example .env  # add your free Groq API key (console.groq.com — no card required)

# Validate your truth document
python src/schema.py

# Preview RAG chunks
python src/chunker.py

# Index chunks + run a test semantic search
python src/vector_store.py

# Run the full conversational RAG pipeline
python src/rag_pipeline.py

# Run contradiction detection tests
python src/contradiction.py
```

## Build status

- [x] Architecture design
- [x] Truth document schema
- [x] Pydantic validation layer
- [x] RAG chunker
- [x] ChromaDB vector store + semantic search (local, free embeddings)
- [x] Streamlit UI (live demo)
- [x] LangChain RAG pipeline (full conversational retrieval, via Groq)
- [x] Contradiction detection module (LLM-as-judge, via Groq)
- [ ] README demo GIF

## Live demo

Try it: **[contextcore-6qfpxktgczy5akqmcvzryz.streamlit.app](https://contextcore-6qfpxktgczy5akqmcvzryz.streamlit.app/)**


---

Built by [Shrijita Bhattacharyya](https://github.com/shrijita24) · Final Year Project, IEM Kolkata
