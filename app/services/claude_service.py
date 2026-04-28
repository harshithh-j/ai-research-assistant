import anthropic
from typing import Generator, List
from app.core.config import settings
from app.models.schemas import Message

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

def chat_stream(
    messages: List[Message],
    system_prompt: str,
) -> Generator[str, None, None]:
    formatted = [{"role": m.role, "content": m.content} for m in messages]

    with client.messages.stream(
        model=settings.model_name,
        max_tokens=settings.max_tokens,
        system=system_prompt,
        messages=formatted,
    ) as stream:
        for text in stream.text_stream:
            yield text

def rag_stream(
    question: str,
    system_prompt: str,
) -> Generator[str, None, None]:
    """
    Streams a RAG response — question answered using retrieved context.
    """
    with client.messages.stream(
        model=settings.model_name,
        max_tokens=settings.max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Question: {question}"}],
    ) as stream:
        for text in stream.text_stream:
            yield text