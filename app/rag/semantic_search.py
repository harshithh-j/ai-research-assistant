from typing import List, Dict
from app.rag.embedder import embed_texts
from app.rag.vector_store import search_index

def semantic_search(query: str, k: int = 10) -> List[Dict]:
    """
    Embeds the query and searches FAISS for similar chunks.
    Returns top-k chunks with semantic similarity scores.
    """
    query_embedding = embed_texts([query])[0]
    results = search_index(query_embedding, k=k)

    # Normalize scores: FAISS returns L2 distance (lower = better)
    # Convert to similarity score (higher = better)
    for r in results:
        r["semantic_score"] = 1 / (1 + r["score"])  # convert distance to similarity
        del r["score"]

    return results