from typing import List, Dict

def chunk_text(
    pages: List[Dict],
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict]:
    """
    Splits page text into overlapping chunks.
    Each chunk carries metadata: source, page, chunk_id.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        words = text.split()
        total_words = len(words)

        start = 0
        while start < total_words:
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": f"chunk_{chunk_id:04d}",
                "source": page["source"],
                "page": page["page"],
                "text": chunk_text
            })

            chunk_id += 1
            start += chunk_size - overlap  # move forward with overlap

    return chunks