"""Suggest a refined search query when the current results have weak coverage.

This is the one place the decoder LLM is used for the relevance loop: given the
user's question and the queries already tried, it proposes a different angle so
the agent can search again. All keep/coverage decisions remain deterministic.
"""

from news_agent.llm.ollama_client import generate
from news_agent.tools.purify import _parse_queries

SYSTEM_PROMPT = """You refine failed news searches.

The queries already tried returned weak or no relevant results for the user's question.
Propose a DIFFERENT angle: alternate terms, synonyms, broader phrasing, or related entities.

HOW THE GUARDIAN SEARCH WORKS:
The API matches SHORT keyword tokens (1 word, occasionally 2-3 adjacent words for a single name like
Formula One), not long phrases. Build each query by JOINING short tokens with boolean operators:
- AND   — require both tokens:    Hamilton AND Ferrari
- OR    — allow either token:     F1 OR Formula One
- NOT   — exclude a token:        Apple AND NOT fruit
- (...) — group tokens:           (NBA OR basketball) AND finals

NEVER use double-quote characters inside a query. Adjacent words are already treated as a phrase
(write Formula One, not quoted). Double quotes break the JSON and must not appear.

When a search fails, the usual fixes are:
- BROADEN with OR — add synonyms/alternatives of the same concept: (race OR grand prix).
- Remove an over-restrictive AND token.
- Swap in a related entity or synonym.
- NEVER add time words (latest, recent, today) or it-is-news words (news, update).
  KEEP subject-matter tokens (race, result, winner, final, trial, ...).

Rules:
- Output ONLY a JSON array of 1-2 query strings, each a boolean expression of short tokens.
- NEVER put double-quote characters inside a query.
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
