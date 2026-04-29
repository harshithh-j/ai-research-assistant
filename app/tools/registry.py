from typing import Dict
from app.tools.base import BaseTool
from app.tools.web_search import WebSearchTool

# Register all available tools here
_tools: Dict[str, BaseTool] = {
    "web_search": WebSearchTool(),
}

def get_all_tools() -> list[BaseTool]:
    return list(_tools.values())

def get_tool(name: str) -> BaseTool:
    if name not in _tools:
        raise ValueError(f"Tool '{name}' not found in registry")
    return _tools[name]

def get_claude_tools() -> list[dict]:
    """Returns all tools in Claude API format."""
    return [t.to_claude_format() for t in _tools.values()]