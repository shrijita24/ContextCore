"""
ContextCore — Vector Store
Embeds truth-document chunks with a free, local sentence-transformers model
and stores them in a local ChromaDB collection so they can be retrieved by
relevance. No API key or billing required for embeddings.
"""

from __future__ import annotations
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from schema import TruthDocument
from chunker import Chunk, chunk_truth_document

# Small, fast, free model that runs on CPU — good enough for a
# truth-document-sized dataset like this one. Downloads once, then caches.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "truth_document_v2"  # new name — old collection was built with OpenAI embeddings, incompatible with local ones
DB_PATH = "./chroma_db"  # local, persistent on disk


def get_collection():
    """
    Connects to (or creates) a local, persistent ChromaDB collection,
    configured to embed text using a free local sentence-transformers
    model — no external API calls, no billing.
    """
    client = chromadb.PersistentClient(path=DB_PATH)

    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=local_ef,
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

    # 3. Embed + store those chunks in ChromaDB (free, local)
    index_chunks(chunks)

    # 4. Quick sanity-check search
    test_query = "What is her CGPA and where does she study?"
    print(f"\nTest query: {test_query}")
    for match in search(test_query, top_k=2):
        print(f"  [{match['metadata'].get('section')}] {match['text'][:100]}...")
