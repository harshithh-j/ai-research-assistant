import anthropic
from app.core.config import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

def rewrite_query(question: str) -> str:
    """
    Uses Claude to rewrite a vague user question into
    a precise search query for better retrieval.
    """
    response = client.messages.create(
        model=settings.model_name,
        max_tokens=256,
        system=(
            "You are a search query optimizer. "
            "Your job is to rewrite user questions into precise, "
            "specific search queries that will retrieve the most relevant documents. "
            "Rules:\n"
            "- Expand abbreviations and vague terms\n"
            "- Add relevant technical keywords\n"
            "- Keep it concise — one sentence max\n"
            "- Return ONLY the rewritten query, nothing else\n"
            "- Do not answer the question, just rewrite it"
        ),
        messages=[
            {"role": "user", "content": f"Rewrite this question as a search query:\n{question}"}
        ]
    )

    rewritten = response.content[0].text.strip()
    return rewritten

def rewrite_with_expansion(question: str) -> list[str]:
    """
    Generates the original query + 2 alternative phrasings.
    Useful for retrieving a broader set of relevant chunks.
    """
    response = client.messages.create(
        model=settings.model_name,
        max_tokens=512,
        system=(
            "You are a search query optimizer. "
            "Generate 3 different search queries for the given question. "
            "Each query should approach the topic from a slightly different angle. "
            "Return ONLY the 3 queries, one per line, no numbering, no extra text."
        ),
        messages=[
            {"role": "user", "content": f"Generate 3 search queries for:\n{question}"}
        ]
    )

    queries = response.content[0].text.strip().split("\n")
    queries = [q.strip() for q in queries if q.strip()]
    return queries[:3]  # ensure max 3