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
- **LLM**: OpenAI GPT-4
- **Embeddings**: OpenAI `text-embedding-ada-002`
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
│   ├── vector_store.py       # ChromaDB setup, embedding pipeline, and semantic search
│   ├── rag_pipeline.py       # LangChain retrieval chain (coming soon)
│   ├── contradiction.py      # LLM-as-judge contradiction detector (coming soon)
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
cp .env.example .env  # add your OpenAI key

# Validate your truth document
python src/schema.py

# Preview RAG chunks
python src/chunker.py
```

## Build status

- [x] Architecture design
- [x] Truth document schema
- [x] Pydantic validation layer
- [x] RAG chunker
- [x] ChromaDB vector store + semantic search
- [x] Streamlit UI (live demo)
- [ ] LangChain RAG pipeline (full conversational retrieval)
- [ ] Contradiction detection module
- [ ] README demo GIF

## Live demo

Try it: **[contextcore.streamlit.app](https://contextcore.streamlit.app)** *(update this link once deployed)*


---

Built by [Shrijita Bhattacharyya](https://github.com/shrijita24) · Final Year Project, IEM Kolkata
