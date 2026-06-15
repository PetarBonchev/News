import json
import os
from datetime import datetime, timedelta

import requests

from news_agent.db.cache_repo import get_fresh_cached_articles, upsert_cache
from news_agent.db.session import SessionLocal
from news_agent.models.article import Article

GUARDIAN_URL = "https://content.guardianapis.com/search"

VALID_SECTIONS = {
    "world", "politics", "business", "technology", "science", "environment",
    "sport", "culture", "society", "us-news", "uk-news", "australia-news",
    "global-development", "law", "education", "media", "film", "music",
    "books", "travel", "food", "fashion", "lifeandstyle",
}


def parse_search_input(tool_input: str) -> dict:
    """Parse tool_input as JSON params or fall back to treating it as a plain query string."""
    stripped = tool_input.strip()

    # Try standard JSON first
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Handle Python dict repr with single quotes (common LLM mistake)
    try:
        import ast
        data = ast.literal_eval(stripped)
        if isinstance(data, dict):
            return data
    except (ValueError, SyntaxError):
        pass

    # Fall back to treating the whole string as a plain query
    return {"q": stripped}


def normalize_articles(results: list[dict]) -> list[dict]:
    normalized: list[dict] = []

    for item in results:
        fields = item.get("fields", {})

        article = Article(
            title=item.get("webTitle", "") or fields.get("headline", ""),
            date=item.get("webPublicationDate", ""),
            url=item.get("webUrl", ""),
            snippet=fields.get("trailText", "") or "",
            body=fields.get("bodyText", "") or "",
        )
        normalized.append(article.model_dump())

    return normalized


def fetch_guardian(
    q: str,
    section: str | None = None,
    from_days: int | None = None,
    page_size: int = 8,
) -> dict:
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        raise RuntimeError("GUARDIAN_API_KEY is not set")

    params: dict = {
        "q": q,
        "page-size": page_size,
        "show-fields": "headline,trailText,bodyText",
        "order-by": "newest",
        "api-key": api_key,
    }

    if section and section in VALID_SECTIONS:
        params["section"] = section

    if from_days and isinstance(from_days, int) and from_days > 0:
        from_date = (datetime.now() - timedelta(days=from_days)).strftime("%Y-%m-%d")
        params["from-date"] = from_date

    response = requests.get(GUARDIAN_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def search_news(tool_input: str, page_size: int = 8) -> tuple[list[dict], bool]:
    params = parse_search_input(tool_input)

    q = params.get("q", "").strip()
    if not q or len(q) < 2 or not any(c.isalpha() for c in q):
        return [], False

    section = params.get("section")
    from_days = params.get("from_days")

    # Normalize cache key so single-quote and double-quote representations match
    cache_key = json.dumps({"q": q, "section": section, "from_days": from_days}, sort_keys=True)

    with SessionLocal() as session:
        cached = get_fresh_cached_articles(session, cache_key)
        if cached is not None:
            return cached, True

        raw = fetch_guardian(q=q, section=section, from_days=from_days, page_size=page_size)
        results = raw["response"]["results"]
        articles = normalize_articles(results)

        upsert_cache(session, query=cache_key, articles=articles)
        return articles, False

