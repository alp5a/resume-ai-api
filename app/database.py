"""
Thin wrapper around a ChromaDB collection used to index and search the
resume chunks. Uses ChromaDB's built-in lightweight ONNX embedding model
(no PyTorch/sentence-transformers needed) to keep memory usage low.
"""

from typing import Dict, List

import chromadb

COLLECTION_NAME = "resume_chunks"

_client = chromadb.Client()
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def reset_collection():
    global _collection
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def index_chunks(chunks: List[Dict]) -> int:
    collection = reset_collection()
    ids = [str(i) for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{"section": c["section"]} for c in chunks]
    if documents:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(documents)


def query_chunks(question: str, top_k: int = 3) -> List[Dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[question], n_results=top_k)
    out = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        similarity = max(0.0, 1.0 - dist)
        out.append({"text": doc, "section": meta.get("section", "unknown"), "score": similarity})
    return out


def chunk_count() -> int:
    return get_collection().count()
