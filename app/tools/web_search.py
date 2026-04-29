from typing import Any, Dict
from app.tools.base import BaseTool
from app.core.config import settings

class WebSearchTool(BaseTool):

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current information. "
            "Use this when the user asks about recent events, "
            "latest updates, or information that may not be in the documents. "
            "Input: a search query string."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                }
            },
            "required": ["query"]
        }

    def run(self, query: str) -> str:
        if not settings.tavily_api_key:
            return "Web search is not configured. No TAVILY_API_KEY found."

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.tavily_api_key)

            response = client.search(
                query=query,
                max_results=5,
                search_depth="basic"
            )

            results = response.get("results", [])
            if not results:
                return "No results found."

            formatted = []
            for i, r in enumerate(results[:5], 1):
                formatted.append(
                    f"[{i}] {r.get('title', 'No title')}\n"
                    f"URL: {r.get('url', '')}\n"
                    f"{r.get('content', 'No content')}"
                )

            return "\n\n".join(formatted)

        except Exception as e:
            return f"Web search failed: {str(e)}"