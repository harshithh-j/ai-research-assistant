def build_system_prompt(context: str = "") -> str:
    base = (
        "You are an expert AI Research Assistant. "
        "Answer questions clearly and concisely. "
        "When you use information from provided documents, cite your sources."
    )
    if context:
        return f"{base}\n\n--- Context ---\n{context}"
    return base