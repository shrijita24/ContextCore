"""
ContextCore — Vector Store
Embeds truth-document chunks with OpenAI embeddings and stores them
in a local ChromaDB collection so they can be retrieved by relevance.
"""

from __future__ import annotations
import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from schema import TruthDocument
from chunker import Chunk, chunk_truth_document

load_dotenv()

EMBEDDING_MODEL = "text-embedding-ada-002"
COLLECTION_NAME = "truth_document"
DB_PATH = "./chroma_db"  # local, persistent on disk


def get_collection():
    """
    Connects to (or creates) a local, persistent ChromaDB collection,
    configured to embed text using OpenAI's embedding model.
    """
    client = chromadb.PersistentClient(path=DB_PATH)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBEDDING_MODEL,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
    )
    return collection


def index_chunks(chunks: List[Chunk]) -> None:
    """
    Embeds each chunk and upserts it into the ChromaDB collection.
    Upsert = insert if new, overwrite if the same id already exists —
    this makes re-running the script safe (no duplicate chunks).
    """
    collection = get_collection()

    ids = [f"{c.section}-{i}" for i, c in enumerate(chunks)]
    documents = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"✓ Indexed {len(chunks)} chunks into ChromaDB collection '{COLLECTION_NAME}'")


def search(query: str, top_k: int = 3):
    """
    Embeds the query and returns the top_k most relevant chunks
    from the truth document, ranked by semantic similarity.
    """
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    matches = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        matches.append({"text": doc, "metadata": meta, "distance": dist})
    return matches


if __name__ == "__main__":
    # 1. Load and validate the truth document
    doc = TruthDocument.from_json("data/truth_document.json")

    # 2. Split it into retrieval-ready chunks
    chunks = chunk_truth_document(doc)

    # 3. Embed + store those chunks in ChromaDB
    index_chunks(chunks)

    # 4. Quick sanity-check search
    test_query = "What is her CGPA and where does she study?"
    print(f"\nTest query: {test_query}")
    for match in search(test_query, top_k=2):
        print(f"  [{match['metadata'].get('section')}] {match['text'][:100]}...")
