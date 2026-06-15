from dataclasses import dataclass
from typing import Callable, Any

from news_agent.tools.guardian import VALID_SECTIONS, search_news
from news_agent.tools.purify import purify_query
from news_agent.tools.requery import suggest_requery_tool
from news_agent.tools.summarize import summarize_articles


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    status: str = "Working..."  # live message shown while this tool runs


def get_tool_registry() -> dict[str, ToolSpec]:
    return {
        "purifyquery": ToolSpec(
            name="purifyquery",
            description=(
                "Turns a vague or conversational user question into one or more Guardian search queries.\n"
                "  tool_input — the raw user question as plain text.\n"
                "  Output — a JSON array of query strings to use in searchnews queries.\n"
                "           Usually one query; multiple only when the topic has distinct sub-angles.\n"
                "           Queries may use Guardian operators: AND, OR, NOT, \"exact phrase\", (grouping).\n"
                "  Skip this tool if the user's question is already a clear, specific query.\n"
                "  Example:\n"
                '    tool_input: "what is going on with the war in ukraine lately"\n'
                '    output:     ["Ukraine war military offensive", "Ukraine peace negotiations"]'
            ),
            fn=purify_query,
            status="Refining the search query...",
        ),
        "searchnews": ToolSpec(
            name="searchnews",
            description=(
                "Searches The Guardian for news articles (newest first), then filters them for relevance.\n"
                "  tool_input — a JSON string with these fields:\n"
                "    q        (array of strings, required) — one or more search queries (always an array,\n"
                "              even for a single query); results from all are merged and deduplicated.\n"
                "              Pass the purifyquery output here. Results are already newest-first, so do NOT\n"
                "              add time words ('latest', 'recent', 'today') or words that just restate it is\n"
                "              news ('news', 'update'). DO keep subject-matter words that say what the user\n"
                "              wants ('race', 'result', 'winner', 'final'). 'NBA finals winner' is good;\n"
                "              'latest NBA news' is bad.\n"
                "    section  (optional, string) — narrow to one Guardian section. If given, it must be\n"
                "              EXACTLY one of these values; no other value or abbreviation is accepted:\n"
                "              " + ", ".join(sorted(VALID_SECTIONS)) + ".\n"
                "              Use 'sport' for any sport (F1, NBA, football, tennis, etc.); 'technology' for tech.\n"
                "              Omit section entirely if none fits. Each query may use Guardian operators:\n"
                "              AND, OR, NOT, \"exact phrase\", (grouping).\n"
                "    order_by (optional, string) — 'newest' (default) or 'relevance'. Use 'relevance'\n"
                "              when a newest-ordered search missed the articles you expected.\n"
                "  Output — the relevant articles that survived filtering (title, date, url, snippet),\n"
                "           plus a coverage verdict (good / partial / none) and how many were dropped.\n"
                "           You read this and decide what to do next.\n"
                "  Example:\n"
                '    tool_input: {"q": ["NBA finals 2026", "OKC Thunder championship"], "section": "sport"}'
            ),
            fn=search_news,
            status="Searching The Guardian and filtering for relevance...",
        ),
        "suggestrequery": ToolSpec(
            name="suggestrequery",
            description=(
                "When a search returned weak or irrelevant results, proposes alternative search queries\n"
                "from a different angle (synonyms, broader/narrower terms, related entities).\n"
                "  tool_input — the user's original question as plain text.\n"
                "  Output — a JSON array of 1-2 new query strings to use in searchnews queries.\n"
                "  Optional: only call this if you cannot think of a better query yourself."
            ),
            fn=suggest_requery_tool,
            status="Thinking of alternative search queries...",
        ),
        "summarizearticles": ToolSpec(
            name="summarizearticles",
            description=(
                "Synthesizes the articles from the last searchnews call into a structured briefing.\n"
                "  tool_input — the user's original question as plain text.\n"
                "  Output — a briefing with four sections: Key developments, Main actors,\n"
                "           Contradictions or uncertainties, Open questions.\n"
                "           Followed by a Sources block with each article's title, date, and URL.\n"
                "  Call this once you have relevant articles. Do not call it if 0 were found.\n"
                "  Example:\n"
                '    tool_input: "Who won the NBA finals?"\n'
                "    output: Key developments\n"
                "            - OKC Thunder defeated the Indiana Pacers 4-1 to win the 2026 NBA title.\n"
                "            ...\n"
                "            Sources:\n"
                "            - OKC Thunder win NBA title (2026-06-10)\n"
                "              https://..."
            ),
            fn=summarize_articles,
            status="Writing the news briefing...",
        ),
    }


def get_tool_descriptions() -> str:
    registry = get_tool_registry()
    lines = []
    for tool in registry.values():
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)
