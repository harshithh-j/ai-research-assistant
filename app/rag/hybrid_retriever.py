from typing import List, Dict
from app.rag.semantic_search import semantic_search
from app.rag.bm25_search import bm25_search

def normalize_scores(results: List[Dict], score_key: str) -> List[Dict]:
    """
    Normalizes scores to 0-1 range using min-max normalization.
    """
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
    bm25_weight: float = 0.3
) -> List[Dict]:
    """
    Combines semantic and BM25 search results into a single ranked list.
    Uses weighted combination of normalized scores.
    """
    # Run both searches
    semantic_results = semantic_search(query, k=10)
    bm25_results = bm25_search(query, k=10)

    # Normalize each set of scores to 0-1
    semantic_results = normalize_scores(semantic_results, "semantic_score")
    bm25_results = normalize_scores(bm25_results, "bm25_score")

    # Build lookup by chunk_id
    scores_map: Dict[str, Dict] = {}

    for r in semantic_results:
        cid = r["chunk_id"]
        scores_map[cid] = r.copy()
        scores_map[cid]["bm25_score"] = 0.0  # default if not in BM25 results

    for r in bm25_results:
        cid = r["chunk_id"]
        if cid in scores_map:
            scores_map[cid]["bm25_score"] = r["bm25_score"]
        else:
            entry = r.copy()
            entry["semantic_score"] = 0.0  # default if not in semantic results
            scores_map[cid] = entry

    # Compute hybrid score
    for cid, entry in scores_map.items():
        entry["hybrid_score"] = (
            semantic_weight * entry["semantic_score"] +
            bm25_weight * entry["bm25_score"]
        )

    # Sort by hybrid score and return top-k
    ranked = sorted(scores_map.values(), key=lambda x: x["hybrid_score"], reverse=True)
    return ranked[:k]