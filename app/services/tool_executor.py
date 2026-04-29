import anthropic
import json
from typing import Generator
from app.core.config import settings
from app.tools.registry import get_claude_tools, get_tool

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

AGENT_SYSTEM_PROMPT = """You are an expert AI Research Assistant with access to tools.

You have access to:
1. A web_search tool for finding current information
2. Your knowledge from ingested documents (used automatically)

Guidelines:
- Use web_search when asked about recent events, latest updates, or time-sensitive information
- Answer directly from context when the question is about the provided documents
- Always be clear about where your information comes from
- If you use web search, cite the URLs in your response
"""

def run_agent(question: str) -> Generator[str, None, None]:
    """
    Runs the Claude tool use loop.
    Claude decides whether to use tools, uses them if needed,
    then generates a final answer. Streams the final response.
    """
    tools = get_claude_tools()
    messages = [{"role": "user", "content": question}]

    # Step 1: Ask Claude — does it want to use a tool?
    response = client.messages.create(
        model=settings.model_name,
        max_tokens=settings.max_tokens,
        system=AGENT_SYSTEM_PROMPT,
        tools=tools,
        messages=messages
    )

    # Step 2: Tool use loop
    while response.stop_reason == "tool_use":
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                yield f"Using tool: {tool_name}({json.dumps(tool_input)})\n\n"

                # Execute the tool
                tool = get_tool(tool_name)
                result = tool.run(**tool_input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        # Add Claude's response and tool results to message history
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        # Step 3: Ask Claude again with tool results
        response = client.messages.create(
            model=settings.model_name,
            max_tokens=settings.max_tokens,
            system=AGENT_SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )

    # Step 4: Stream the final answer
    final_messages = messages + [{"role": "assistant", "content": response.content}]

    with client.messages.stream(
        model=settings.model_name,
        max_tokens=settings.max_tokens,
        system=AGENT_SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text