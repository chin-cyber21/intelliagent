import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

INDEX_PATH = Path("rag/index/faiss.index")
DOCS_PATH = Path("rag/index/documents.pkl")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_index = None
_documents = None
_model = None


def _load_index():
    global _index, _documents, _model
    if _index is None:
        if not INDEX_PATH.exists():
            from rag.indexer import build_index
            build_index()
        _index = faiss.read_index(str(INDEX_PATH))
        with open(DOCS_PATH, "rb") as f:
            _documents = pickle.load(f)
        _model = SentenceTransformer(EMBED_MODEL)


def dense_search(query: str, k: int = 5) -> list[tuple[str, float]]:
    """Dense vector search using FAISS + BGE embeddings."""
    _load_index()
    query_vec = _model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(query_vec, k)
    return [(_documents[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i < len(_documents)]


def sparse_search(query: str, k: int = 5) -> list[tuple[str, float]]:
    """Sparse BM25 keyword search."""
    _load_index()
    tokenized_docs = [doc.lower().split() for doc in _documents]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:k]
    return [(_documents[i], float(scores[i])) for i in top_indices]


def retrieve_context(query: str, k: int = 4) -> str:
    """
    Hybrid retrieval — combines dense and sparse search results.
    Dense retrieval finds semantically similar docs.
    Sparse retrieval finds exact keyword matches.
    Both are merged and deduplicated.
    """
    dense_results = dense_search(query, k=k)
    sparse_results = sparse_search(query, k=k)

    # Merge and deduplicate by document text
    seen = set()
    merged = []
    for doc, score in dense_results + sparse_results:
        if doc not in seen:
            seen.add(doc)
            merged.append(doc)

    # Return top k unique results as context string
    context = "\n\n---\n\n".join(merged[:k])
    return context if context else "No relevant context found."
