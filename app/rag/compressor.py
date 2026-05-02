import anthropic
from typing import List, Dict
from app.core.config import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

def compress_chunks(query: str, chunks: List[Dict], max_tokens: int = 150) -> List[Dict]:
    """
    Compresses each chunk by extracting only the parts relevant to the query.
    Reduces token usage when sending context to Claude.

    Args:
        query:      the user's question
        chunks:     re-ranked chunks to compress
        max_tokens: max tokens per compressed chunk

    Returns:
        chunks with compressed 'text' field
    """
    compressed = []

    for chunk in chunks:
        original_text = chunk["text"]

        # Skip compression for short chunks — not worth the API call
        if len(original_text.split()) < 100:
            compressed.append(chunk)
            continue

        try:
            response = client.messages.create(
                model=settings.model_name,
                max_tokens=max_tokens,
                system=(
                    "You are a text compression assistant. "
                    "Extract ONLY the sentences from the provided text that are "
                    "directly relevant to answering the given question. "
                    "Return only the extracted sentences, nothing else. "
                    "Do not summarize or paraphrase — use exact sentences from the text. "
                    "If the entire text is relevant, return it as-is."
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Text to compress:\n{original_text}"
                    )
                }]
            )

            compressed_text = response.content[0].text.strip()

            # Safety check — if compression makes it longer, keep original
            if len(compressed_text.split()) < len(original_text.split()):
                chunk = chunk.copy()
                chunk["text"] = compressed_text
                chunk["compressed"] = True
            else:
                chunk["compressed"] = False

        except Exception:
            chunk["compressed"] = False

        compressed.append(chunk)

    return compressed