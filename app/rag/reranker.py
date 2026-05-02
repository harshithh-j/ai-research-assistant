from sentence_transformers import CrossEncoder
from typing import List, Dict

# Load once at module level
# This model scores (query, passage) pairs for relevance
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Re-ranks chunks using a cross-encoder model.
    More accurate than bi-encoder (FAISS) because it sees
    query and chunk together, not separately.

    Args:
        query:  the user's question
        chunks: candidate chunks from hybrid search
        top_k:  how many to keep after re-ranking

    Returns:
        top_k chunks sorted by cross-encoder score
    """
    if not chunks:
        return []

    # Build (query, chunk_text) pairs for the cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score all pairs — higher score = more relevant
    scores = model.predict(pairs)

    # Attach scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # Sort by rerank score descending and return top_k
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]