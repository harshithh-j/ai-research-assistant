import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict

INDEX_PATH = "data/index/faiss.index"
METADATA_PATH = "data/index/metadata.json"

def save_index(embeddings: np.ndarray, metadata: List[Dict]):
    """
    Creates a FAISS index from embeddings and saves it to disk.
    Also saves metadata as JSON.
    """
    Path("data/index").mkdir(parents=True, exist_ok=True)

    dimension = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {index.ntotal} vectors to FAISS index")

def load_index():
    """
    Loads FAISS index and metadata from disk.
    Returns (index, metadata)
    """
    if not Path(INDEX_PATH).exists():
        raise FileNotFoundError("No FAISS index found. Run /ingest first.")

    index = faiss.read_index(INDEX_PATH)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return index, metadata

def search_index(query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
    """
    Searches FAISS for the k most similar chunks.
    Returns list of metadata dicts with similarity scores.
    """
    index, metadata = load_index()

    query = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        result = metadata[idx].copy()
        result["score"] = float(dist)
        results.append(result)

    return results
