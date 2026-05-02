import anthropic
from typing import Generator, List, Dict, Optional
from app.core.config import settings
from app.rag.hybrid_retriever import hybrid_search
from app.rag.reranker import rerank
from app.utils.prompt_builder import format_chunks_as_context
from app.services.query_rewriter import rewrite_query

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

def build_vision_messages(
    question: str,
    images: List[Dict],
    chunks: List[Dict]
) -> List[Dict]:
    """
    Builds the messages array with images + text for Claude.
    Images are embedded as base64 in the content array.
    """
    content = []

    # Add all images first
    for i, img in enumerate(images):
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img["media_type"],
                "data": img["data"]
            }
        })
        content.append({
            "type": "text",
            "text": f"[Image {i+1} from {img['source']} page {img.get('page', 'N/A')}]"
        })

    # Add document context if available
    if chunks:
        context = format_chunks_as_context(chunks)
        content.append({
            "type": "text",
            "text": f"\nDocument Context:\n{context}\n"
        })

    # Add the question
    content.append({
        "type": "text",
        "text": f"\nQuestion: {question}"
    })

    return [{"role": "user", "content": content}]


def run_vision(
    question: str,
    images: List[Dict],
    use_rag: bool = True,
    k: int = 3
) -> Generator[str, None, None]:
    """
    Vision pipeline:
    1. Optionally retrieve relevant document chunks
    2. Build message with images + context + question
    3. Stream Claude's visual analysis
    """

    chunks = []

    # Step 1: Retrieve relevant chunks if RAG is enabled
    if use_rag:
        search_query = rewrite_query(question)
        yield f"Search query: {search_query}\n\n"

        chunks = hybrid_search(query=search_query, k=k, use_reranking=True)

        if chunks:
            yield f"Found {len(chunks)} relevant document chunks\n\n"

    # Step 2: Build messages with images + context
    yield f"Analyzing {len(images)} image(s)...\n\n"

    messages = build_vision_messages(question, images, chunks)

    # Step 3: Build system prompt
    system_prompt = (
        "You are an expert AI Research Assistant with vision capabilities. "
        "You can analyze images, charts, diagrams, and figures. "
        "When analyzing images:\n"
        "- Describe what you see clearly and precisely\n"
        "- Connect visual information to document context when available\n"
        "- Cite document sources as [N] when referencing context\n"
        "- If the image contains text, read and include it\n"
        "- If the image is a chart/graph, extract key data points\n"
        "- If the image is a diagram, explain the relationships shown\n"
    )

    # Step 4: Stream response
    with client.messages.stream(
        model=settings.model_name,
        max_tokens=settings.max_tokens,
        system=system_prompt,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            yield text

    # Step 5: Append sources if chunks were used
    if chunks:
        yield "\n\nDocument Sources:\n" + "\n".join(
            f"[{i}] {c['source']} (page {c['page']})"
            for i, c in enumerate(chunks, 1)
        )