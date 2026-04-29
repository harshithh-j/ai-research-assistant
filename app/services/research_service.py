import anthropic
import json
from typing import Generator, List, Dict
from app.core.config import settings
from app.tools.registry import get_claude_tools, get_tool
from app.rag.hybrid_retriever import hybrid_search
from app.utils.prompt_builder import format_chunks_as_context
from app.services.query_rewriter import rewrite_query

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

def build_research_system_prompt(chunks: List[Dict]) -> str:
    """
    Builds a system prompt that includes:
    - Document context from RAG
    - Instructions for tool use
    - Citation guidelines
    """
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

def run_research(question: str, k: int = 5) -> Generator[str, None, None]:
    """
    Unified research pipeline:
    1. Rewrite query
    2. Retrieve document chunks
    3. Give Claude chunks as context + web_search as tool
    4. Claude answers from docs, web, or both
    5. Stream final answer with citations
    """

    # Step 1: Rewrite query for better retrieval
    search_query = rewrite_query(question)
    yield f"Search query: {search_query}\n\n"

    # Step 2: Retrieve relevant chunks
    chunks = hybrid_search(query=search_query, k=k)

    if chunks:
        yield f"Found {len(chunks)} relevant document chunks\n\n"
    else:
        yield "No relevant document chunks found — will rely on web search\n\n"

    # Step 3: Build system prompt with document context
    system_prompt = build_research_system_prompt(chunks)

    # Step 4: Build source list from documents
    doc_sources = "\n\nDocument Sources:\n" + "\n".join(
        f"[{i}] {c['source']} (page {c['page']})"
        for i, c in enumerate(chunks, 1)
    ) if chunks else ""

    # Step 5: Run tool loop — Claude decides if web search is needed
    tools = get_claude_tools()
    messages = [{"role": "user", "content": question}]

    response = client.messages.create(
        model=settings.model_name,
        max_tokens=settings.max_tokens,
        system=system_prompt,
        tools=tools,
        messages=messages
    )

    # Step 6: Handle tool calls if Claude requests them
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

                # Track web sources for citation
                if tool_name == "web_search":
                    web_sources.append({
                        "query": tool_input.get("query", ""),
                        "results": result
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

    # Step 8: Append all sources
    if doc_sources:
        yield doc_sources

    if web_sources:
        yield "\n\nWeb Sources Used:\n" + "\n".join(
            f"- Query: {ws['query']}"
            for ws in web_sources
        )