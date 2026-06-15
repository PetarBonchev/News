"""Relevance filtering for news articles.

Scores articles against the query with a cross-encoder, boosted by a HyDE
hypothetical answer, and decides which to keep. HyDE lives here, not in the agent
loop — the loop just asks "which of these articles are relevant?".
"""

from news_agent.tools.hyde import hypothetical_answer
from news_agent.tools.relevance import (
    assess_coverage,
    score_articles,
    select_relevant,
)


def filter_articles(
    articles: list[dict], query: str, model: str
) -> tuple[list[dict], list[str], str]:
    """Score articles against the query and decide which to keep.

    Scoring also considers a HyDE hypothetical answer for the query and keeps the
    higher score per article. Returns (relevant_articles, dropped_titles, coverage)
    where coverage is one of "good", "partial", or "none".
    """
    if not articles:
        return [], [], "none"

    hyde_text = hypothetical_answer(query, model=model)
    scores = score_articles(query, articles, hyde_text=hyde_text)
    relevant, dropped = select_relevant(articles, scores)
    coverage = assess_coverage(scores)
    return relevant, dropped, coverage
