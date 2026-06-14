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
            description="Convert vague raw user input into a short clean news search query.",
            fn=purify_query,
        ),
        "searchnews": ToolSpec(
            name="searchnews",
            description="Search news articles using the Guardian-backed cache and return normalized article records.",
            fn=search_news,
        ),
        "summarizearticles": ToolSpec(
            name="summarizearticles",
            description="Summarize a list of articles into a structured briefing.",
            fn=summarize_articles,
        ),
    }


def get_tool_descriptions() -> str:
    registry = get_tool_registry()
    lines = []
    for tool in registry.values():
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)
