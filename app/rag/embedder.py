from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Load once at module level — expensive operation
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Converts a list of strings into a 2D numpy array of embeddings.
    Shape: (num_texts, 384)
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float32)