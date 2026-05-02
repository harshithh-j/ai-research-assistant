import anthropic
import json
from typing import Generator, List, Dict
from app.core.config import settings
from app.tools.registry import get_claude_tools, get_tool
from app.rag.hybrid_retriever import hybrid_search
from app.rag.compressor import compress_chunks
from app.utils.prompt_builder import format_chunks_as_context
from app.services.query_rewriter import rewrite_query

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

def build_research_system_prompt(chunks: List[Dict]) -> str:
    context = format_chunks_as_context(chunks)

    return (
        "You are an expert AI Research Assistant with access to both "
        "ingested documents and web search.\n\n"
        "DOCUMENT CONTEXT (from ingested PDFs):\n"
        "Use [1], [2], etc. to cite these sources inline.\n\n"
        f"{context}\n\n"
        "---\n\n"
        "TOOLS AVAILABLE:\n"
        "- web_search: Use this for current information not covered in the documents above.\n\n"
        "GUIDELINES:\n"
        "1. Always check the document context first\n"
        "2. Use web_search only if documents don't fully answer the question\n"
        "3. Cite document sources as [N] inline\n"
        "4. Cite web sources as URLs inline\n"
        "5. Clearly distinguish between document-based and web-based information\n"
        "6. Do NOT add a sources list at the end — it will be appended automatically\n"
    )

def run_research(
    question: str,
    k: int = 5,
    use_reranking: bool = True,
    use_compression: bool = True,
) -> Generator[str, None, None]:
    """
    Full research pipeline:
    1. Query rewriting
    2. Hybrid search with re-ranking
    3. Context compression
    4. RAG + tool loop
    5. Streaming answer with citations
    """

    # Step 1: Rewrite query
    search_query = rewrite_query(question)
    yield f"Search query: {search_query}\n\n"

    # Step 2: Retrieve + re-rank
    chunks = hybrid_search(
        query=search_query,
        k=k,
        use_reranking=use_reranking,
        rerank_candidates=20
    )

    if chunks:
        yield f"Retrieved {len(chunks)} chunks"
        if use_reranking:
            yield " (re-ranked)"
        yield "\n\n"
    else:
        yield "No relevant chunks found — will rely on web search\n\n"

    # Step 3: Compress chunks
    if use_compression and chunks:
        chunks = compress_chunks(question, chunks)
        yield "Context compressed\n\n"

    # Step 4: Build system prompt
    system_prompt = build_research_system_prompt(chunks)

    # Step 5: Pre-build document sources
    doc_sources = "\n\nDocument Sources:\n" + "\n".join(
        f"[{i}] {c['source']} (page {c['page']})"
        for i, c in enumerate(chunks, 1)
    ) if chunks else ""

    # Step 6: Tool loop
    tools = get_claude_tools()
    messages = [{"role": "user", "content": question}]

    response = client.messages.create(
        model=settings.model_name,
        max_tokens=settings.max_tokens,
        system=system_prompt,
        tools=tools,
        messages=messages
    )

    web_sources = []

    while response.stop_reason == "tool_use":
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                yield f"Using tool: {tool_name}({json.dumps(tool_input)})\n\n"

                tool = get_tool(tool_name)
                result = tool.run(**tool_input)

                if tool_name == "web_search":
                    web_sources.append({
                        "query": tool_input.get("query", ""),
                    })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=settings.model_name,
            max_tokens=settings.max_tokens,
            system=system_prompt,
            tools=tools,
            messages=messages
        )

    # Step 7: Stream final answer
    with client.messages.stream(
        model=settings.model_name,
        max_tokens=settings.max_tokens,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text

    # Step 8: Append sources
    if doc_sources:
        yield doc_sources

    if web_sources:
        yield "\n\nWeb Sources Used:\n" + "\n".join(
            f"- Query: {ws['query']}"
            for ws in web_sources
        )