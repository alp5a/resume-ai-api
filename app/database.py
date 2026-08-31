"""
Thin wrapper around a ChromaDB collection used to index and search the
resume chunks with sentence-transformer embeddings.
"""

from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "resume_chunks"

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# In-memory Chroma client. Swap for chromadb.PersistentClient(path="./chroma_db")
# if you want the index to survive restarts.
_client = chromadb.Client()
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=_embedding_fn
        )
    return _collection


def reset_collection():
    global _collection
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=_embedding_fn
    )
    return _collection


def index_chunks(chunks: List[Dict]) -> int:
    """Embeds and stores resume chunks. Returns the number indexed."""
    collection = reset_collection()

    ids = [str(i) for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{"section": c["section"]} for c in chunks]

    if documents:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return len(documents)


def query_chunks(question: str, top_k: int = 3) -> List[Dict]:
    """Returns the top_k most relevant chunks for a question."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[question], n_results=top_k)

    out = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        # Chroma returns a distance (lower = closer); convert to a
        # 0-1 "confidence" style similarity score for the API response.
        similarity = max(0.0, 1.0 - dist)
        out.append({"text": doc, "section": meta.get("section", "unknown"), "score": similarity})

    return out


def chunk_count() -> int:
    return get_collection().count()
