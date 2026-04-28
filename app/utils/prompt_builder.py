from typing import List, Dict

def build_system_prompt(context: str = "") -> str:
    base = (
        "You are an expert AI Research Assistant. "
        "Answer questions clearly and concisely. "
        "When you use information from provided documents, cite your sources."
    )
    if context:
        return f"{base}\n\n--- Context ---\n{context}"
    return base

def format_chunks_as_context(chunks: List[Dict]) -> str:
    """
    Formats retrieved chunks into numbered context blocks.
    Each chunk gets a [N] reference number for citations.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] Source: {chunk['source']} (page {chunk['page']})\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(context_parts)

def build_rag_prompt(question: str, chunks: List[Dict]) -> tuple[str, str]:
    """
    Builds system prompt with context and user message.
    Returns (system_prompt, user_message)
    """
    context = format_chunks_as_context(chunks)

    system_prompt = (
        "You are an expert AI Research Assistant. "
        "Answer the user's question using ONLY the provided context below. "
        "Cite sources using [N] notation inline. "
        "Do NOT add a sources list at the end — sources will be appended automatically.\n\n"
        "If the context doesn't contain enough information, say so clearly.\n\n"
        "--- Context ---\n"
        f"{context}"
    )

    user_message = f"Question: {question}"

    return system_prompt, user_message