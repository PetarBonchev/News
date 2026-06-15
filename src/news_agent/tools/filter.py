import json

from news_agent.llm.ollama_client import generate


SYSTEM_PROMPT = """You are a relevance filter for news articles.

Given a user query and a list of articles, return a JSON array containing only the
indices (0-based) of articles that are genuinely relevant to the query.

An article is relevant if its topic, subject, or key entities directly relate to
what the user is asking about. Be strict — exclude articles that merely mention
the topic in passing or as background context.

Return only a JSON array of integers, e.g. [0, 2, 5]. Return [] if none are relevant.
Do not explain. Do not wrap in markdown."""


def _build_prompt(query: str, articles: list[dict]) -> str:
    items = []
    for i, a in enumerate(articles):
        items.append(
            f"{i}. {a.get('title', '')} — {a.get('snippet', '')[:150]}"
        )
    return f"Query: {query}\n\nArticles:\n" + "\n".join(items) + "\n\nRelevant indices:"


def filter_articles(articles: list[dict], query: str, model: str) -> tuple[list[dict], int]:
    """Return (relevant_articles, n_dropped) using an LLM relevance judgement."""
    if not articles:
        return [], 0

    prompt = _build_prompt(query, articles)
    raw = generate(
        model=model,
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.0
    )

    try:
        indices = json.loads(raw.strip())
        if not isinstance(indices, list):
            raise ValueError("Expected a list")
        relevant = [articles[i] for i in indices if isinstance(i, int) and 0 <= i < len(articles)]
    except (json.JSONDecodeError, ValueError):
        return articles, 0

    return relevant, len(articles) - len(relevant)
