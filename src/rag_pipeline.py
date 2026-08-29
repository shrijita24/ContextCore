"""
ContextCore — RAG Pipeline
LangChain conversational retrieval chain that grounds every answer in the
user's truth document. Uses free, no-billing infrastructure end to end:
- Embeddings: local sentence-transformers model (same one vector_store.py
  uses), so the LangChain retriever reads the exact same ChromaDB index.
- Generation: Groq's free-tier hosted LLMs (fast, no credit card required).
"""

from __future__ import annotations
import os
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from vector_store import DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, index_chunks
from chunker import chunk_truth_document
from schema import TruthDocument

load_dotenv()

# Free-tier Groq model — fast and more than capable for grounded Q&A
# over a small truth document. See https://console.groq.com/docs/models
# for current options (Groq periodically retires older Llama model ids).
CHAT_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are ContextCore, a grounding layer that answers questions \
strictly using the user's verified truth document below. Rules:

1. Only use facts present in the provided context. Never invent, assume, or \
extrapolate details that aren't stated.
2. If the context doesn't contain enough information to answer, say so plainly \
instead of guessing.
3. Keep answers concise and factual — this is a source-of-truth lookup, not a \
creative or conversational response.

Context from the truth document:
{context}
"""


class RAGAnswer(TypedDict):
    answer: str
    sources: List[Document]


def _get_vectorstore() -> Chroma:
    """
    Connects to the same persistent ChromaDB collection that vector_store.py
    writes to, wrapped as a LangChain VectorStore. Uses the identical local
    embedding model as vector_store.py so query vectors line up with the
    ones already stored in the index.
    """
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{EMBEDDING_MODEL}")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('section', 'unknown')}] {d.page_content}" for d in docs
    )


def build_chain(top_k: int = 3):
    """
    Builds an LCEL runnable: retrieve relevant chunks -> stuff into prompt
    -> ask the chat model -> parse to a plain string answer.
    """
    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGroq(model=CHAT_MODEL, temperature=0, api_key=os.environ["GROQ_API_KEY"])

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def answer(query: str, top_k: int = 3) -> RAGAnswer:
    """
    Runs the full RAG pipeline for a single query: retrieves grounding
    chunks and generates an answer constrained to them. Returns both the
    answer text and the source chunks so callers (e.g. the Streamlit UI)
    can show what the answer was grounded in.
    """
    chain, retriever = build_chain(top_k=top_k)
    sources = retriever.invoke(query)
    generated = chain.invoke(query)
    return {"answer": generated, "sources": sources}


if __name__ == "__main__":
    # 1. Make sure the truth document is indexed (safe to re-run — upsert).
    doc = TruthDocument.from_json("data/truth_document.json")
    chunks = chunk_truth_document(doc)
    index_chunks(chunks)

    # 2. Ask a grounded question through the full LangChain pipeline.
    test_query = "What is her current focus and what internship tracks is she targeting?"
    print(f"\nQuery: {test_query}\n")
    result = answer(test_query)

    print("Answer:")
    print(result["answer"])
    print("\nGrounded in:")
    for src in result["sources"]:
        print(f"  [{src.metadata.get('section')}] {src.page_content[:90]}...")
