"""Suggest a refined search query when the current results have weak coverage.

This is the one place the decoder LLM is used for the relevance loop: given the
user's question and the queries already tried, it proposes a different angle so
the agent can search again. All keep/coverage decisions remain deterministic.
"""

from news_agent.llm.ollama_client import generate
from news_agent.tools.purify import _parse_queries

SYSTEM_PROMPT = """You refine failed news searches.

The queries already tried returned weak or no relevant results for the user's question.
Propose a DIFFERENT angle: alternate terms, synonyms, broader or narrower phrasing,
or related entities.

Keep queries focused — drop noise, keep meaning:
- Results are already newest-first, so NEVER add time words ("latest", "recent", "today", "now").
- Drop words that only restate it is news ("news", "update", "story"). Bad: "latest NBA news".
- KEEP subject-matter words that describe what the user wants ("race", "result", "winner",
  "final", "trial", etc.) — these narrow the topic. Good: "NBA finals winner", "F1 race result".
- When a search fails it usually has time/news noise or too many quotes — remove those, not the
  meaningful words. Trying a related entity or synonym is also good.

Quoting:
- Default to PLAIN keywords. Quotes force an exact phrase match and usually return FEWER results.
- If a tried query used quoted phrases, drop the quotes and broaden — that is often why it failed.

Rules:
- Output ONLY a JSON array of 1-2 plain-keyword query strings.
- Do NOT repeat any query that was already tried.
- If you cannot think of a genuinely different query, output [].
"""


def suggest_requery(user_question: str, tried_queries: list[str], model: str) -> list[str]:
    """Return 1-2 new queries different from those already tried, or [] if none."""
    prompt = (
        f"User question:\n{user_question}\n\n"
        f"Queries already tried:\n{tried_queries}\n\n"
        "New search queries (JSON array):"
    )
    result = generate(
        model=model,
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.3,
    )

    tried_lower = {q.strip().lower() for q in tried_queries}
    suggestions = _parse_queries(result)
    return [q for q in suggestions if q.strip().lower() not in tried_lower]


def suggest_requery_tool(raw_input: str, model: str) -> list[str]:
    """Registry-callable adapter: the agent passes the user's question as query_text."""
    return suggest_requery(raw_input, tried_queries=[], model=model)
