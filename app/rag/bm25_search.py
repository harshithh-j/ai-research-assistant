from rank_bm25 import BM25Okapi
from typing import List, Dict
import json

METADATA_PATH = "data/index/metadata.json"

def load_chunks() -> List[Dict]:
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

def bm25_search(query: str, k: int = 10) -> List[Dict]:
    """
    Searches chunks using BM25 keyword matching.
    Returns top-k chunks with BM25 scores.
    """
    chunks = load_chunks()

    # Tokenize all chunks
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    tokenized_query = query.lower().split()

    # Build BM25 index and score
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Attach scores to chunks and sort
    scored_chunks = []
    for idx, score in enumerate(scores):
        if score > 0:  # skip chunks with zero relevance
            chunk = chunks[idx].copy()
            chunk["bm25_score"] = float(score)
            scored_chunks.append(chunk)

    # Sort by score descending and return top-k
    scored_chunks.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored_chunks[:k]