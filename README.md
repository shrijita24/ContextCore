> Live demo: https://contextcore-6qfpxktgczy5akqmcvzryz.streamlit.app/

# ContextCore

> A persistent personal context layer that grounds AI conversations in verified user truth — and flags when AI contradicts it.

## The problem

Every AI conversation starts from zero. Users re-explain their job, goals, constraints, and context every single time. And when AI does "remember" — it hallucinates details confidently. There's no accountability layer. No source of truth.

## What ContextCore does

| Module | What it does |
| --- | --- |
| **Truth Document Builder** | Define your personal context once — career, projects, goals, constraints |
| **RAG Chunker** | Splits your truth document into retrieval-ready chunks |
| **Ask AI** | Every conversation is automatically grounded via RAG against your truth document |
| **Contradiction Check** | Paste any AI-generated statement about you — it's checked against your truth document and flagged in real time if it conflicts |

## How it works

1. Load and validate your truth document (Pydantic schema)
2. The chunker splits it into RAG-ready pieces and embeds them into ChromaDB
3. Ask AI retrieves relevant chunks via LangChain and answers grounded in your real context
4. Contradiction Check runs an LLM-as-judge pass on any statement you paste in, comparing it against your truth document and flagging conflicts

## Tech stack

- **Language**: Python
- **LLM**: Groq
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
│   ├── vector_store.py       # ChromaDB setup and embedding pipeline
│   ├── rag_pipeline.py       # LangChain retrieval chain
│   ├── contradiction.py      # LLM-as-judge contradiction detector
│   └── app.py                # Streamlit UI
├── .env.example
├── requirements.txt
└── README.md
```

## Build status

- [x] Architecture design
- [x] Truth document schema
- [x] Pydantic validation layer
- [x] RAG chunker
- [x] ChromaDB vector store
- [x] LangChain RAG pipeline
- [x] Contradiction detection module
- [x] Streamlit UI

## Getting started

```
git clone https://github.com/shrijita24/ContextCore
cd ContextCore
pip install -r requirements.txt
cp .env.example .env  # add your Groq API key

# Validate your truth document
python src/schema.py

# Preview RAG chunks
python src/chunker.py
```

---

Built by [Shrijita Bhattacharyya](https://github.com/shrijita24) · Final Year Project, IEM Kolkata
