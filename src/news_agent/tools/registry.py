from dataclasses import dataclass
from typing import Callable, Any

from news_agent.tools.guardian import search_news
from news_agent.tools.purify import purify_query
from news_agent.tools.summarize import summarize_articles


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]


def get_tool_registry() -> dict[str, ToolSpec]:
    return {
        "purifyquery": ToolSpec(
            name="purifyquery",
            description=(
                "Rewrites a vague or conversational user request into a short, precise search query. "
                "tool_input: plain text — the raw user question. "
                'Example: tool_input="what is going on with the war in ukraine lately" '
                "returns: Ukraine Russia war 2026"
            ),
            fn=purify_query,
        ),
        "searchnews": ToolSpec(
            name="searchnews",
            description=(
                "Searches The Guardian for recent news articles. "
                "tool_input: a JSON object with fields: "
                "q (required, string), "
                "section (optional — one of: world, politics, business, technology, science, "
                "environment, sport, culture, us-news, uk-news, australia-news, global-development), "
                "from_days (optional, integer — limit to articles from the last N days). "
                'Example: {"q": "Ukraine Russia war", "section": "world", "from_days": 7}. '
                "Returns up to 8 articles with title, date, url, and snippet."
            ),
            fn=search_news,
        ),
        "summarizearticles": ToolSpec(
            name="summarizearticles",
            description=(
                "Synthesizes the articles from the last searchnews call into a structured briefing. "
                "tool_input: the user's original question as plain text. "
                "Returns a briefing with: key developments, main actors, contradictions, open questions."
            ),
            fn=summarize_articles,
        ),
    }


def get_tool_descriptions() -> str:
    registry = get_tool_registry()
    lines = []
    for tool in registry.values():
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)
