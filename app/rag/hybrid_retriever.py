from typing import List, Dict
from app.rag.semantic_search import semantic_search
from app.rag.bm25_search import bm25_search
from app.rag.reranker import rerank

def normalize_scores(results: List[Dict], score_key: str) -> List[Dict]:
    if not results:
        return results

    scores = [r[score_key] for r in results]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    for r in results:
        if score_range == 0:
            r[score_key] = 1.0
        else:
            r[score_key] = (r[score_key] - min_score) / score_range

    return results

def hybrid_search(
    query: str,
    k: int = 5,
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3,
    use_reranking: bool = True,
    rerank_candidates: int = 20,
) -> List[Dict]:
    """
    Hybrid search with optional re-ranking.
    - Retrieves rerank_candidates from semantic + BM25
    - Re-ranks with cross-encoder if use_reranking=True
    - Returns top k
    """
    # Retrieve more candidates for re-ranking
    candidate_k = rerank_candidates if use_reranking else k

    semantic_results = semantic_search(query, k=candidate_k)
    bm25_results = bm25_search(query, k=candidate_k)

    semantic_results = normalize_scores(semantic_results, "semantic_score")
    bm25_results = normalize_scores(bm25_results, "bm25_score")

    scores_map: Dict[str, Dict] = {}

    for r in semantic_results:
        cid = r["chunk_id"]
        scores_map[cid] = r.copy()
        scores_map[cid]["bm25_score"] = 0.0

    for r in bm25_results:
        cid = r["chunk_id"]
        if cid in scores_map:
            scores_map[cid]["bm25_score"] = r["bm25_score"]
        else:
            entry = r.copy()
            entry["semantic_score"] = 0.0
            scores_map[cid] = entry

    for cid, entry in scores_map.items():
        entry["hybrid_score"] = (
            semantic_weight * entry["semantic_score"] +
            bm25_weight * entry["bm25_score"]
        )

    ranked = sorted(
        scores_map.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    candidates = ranked[:candidate_k]

    # Re-rank with cross-encoder if enabled
    if use_reranking and candidates:
        candidates = rerank(query, candidates, top_k=k)
    else:
        candidates = candidates[:k]

    return candidates