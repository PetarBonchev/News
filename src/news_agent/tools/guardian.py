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


SEARCH_INPUT_SCHEMA = (
    'searchnews tool_input must be a JSON object: '
    '{"q": ["query one", "query two"], "section": "sport", "order_by": "newest"}. '
    '"q" is a required non-empty array of query strings. '
    '"order_by" is optional: "newest" (default) or "relevance". '
    '"section" is optional and must be exactly one of: '
    + ", ".join(sorted(VALID_SECTIONS)) + "."
)


def parse_search_input(tool_input: str) -> dict:
    """Parse and validate the searchnews tool_input. Raises ValueError with the expected
    schema if the input is not a JSON object with a valid 'q' array (and section, if given)."""
    try:
        params = json.loads(tool_input)
    except (json.JSONDecodeError, ValueError):
        raise ValueError(f"Invalid JSON. {SEARCH_INPUT_SCHEMA}")

    if not isinstance(params, dict):
        raise ValueError(f"Expected a JSON object. {SEARCH_INPUT_SCHEMA}")

    q = params.get("q")
    if not isinstance(q, list) or not q or not all(isinstance(s, str) and s.strip() for s in q):
        raise ValueError(f'"q" must be a non-empty array of query strings. {SEARCH_INPUT_SCHEMA}')

    section = params.get("section")
    if section is not None and section not in VALID_SECTIONS:
        raise ValueError(f'"{section}" is not a valid section. {SEARCH_INPUT_SCHEMA}')

    return params


def normalize_articles(results: list[dict]) -> list[dict]:
    normalized: list[dict] = []

    for item in results:
        fields = item.get("fields", {})

        article = Article(
            id=item.get("id", "") or item.get("webUrl", ""),
            title=item.get("webTitle", "") or fields.get("headline", ""),
            date=item.get("webPublicationDate", ""),
            url=item.get("webUrl", ""),
            snippet=fields.get("trailText", "") or "",
            body=fields.get("bodyText", "") or "",
        )
        normalized.append(article.model_dump())

    return normalized


VALID_ORDER_BY = {"newest", "oldest", "relevance"}


def fetch_guardian(
    query: str,
    section: str | None = None,
    from_days: int | None = None,
    page_size: int = 8,
    order_by: str = "newest",
) -> dict:
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        raise RuntimeError("GUARDIAN_API_KEY is not set")

    params: dict = {
        "q": query,
        "page-size": page_size,
        "show-fields": "headline,trailText,bodyText",
        "order-by": order_by if order_by in VALID_ORDER_BY else "newest",
        "api-key": api_key,
    }

    if section:
        params["section"] = section

    if from_days and isinstance(from_days, int) and from_days > 0:
        from_date = (datetime.now() - timedelta(days=from_days)).strftime("%Y-%m-%d")
        params["from-date"] = from_date

    response = requests.get(GUARDIAN_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _is_valid_query(query: str) -> bool:
    return bool(query) and len(query) >= 2 and any(c.isalpha() for c in query)


def search_single_query(
    query: str, section: str | None, page_size: int = 8, order_by: str = "newest"
) -> tuple[list[dict], bool]:
    """Run one Guardian query, using the per-query cache. Returns (articles, from_cache)."""
    query = query.strip()
    if not _is_valid_query(query):
        return [], False

    cache_key = json.dumps({"query": query, "section": section, "order_by": order_by}, sort_keys=True)

    with SessionLocal() as session:
        cached = get_fresh_cached_articles(session, cache_key)
        if cached is not None:
            return cached, True

        raw = fetch_guardian(query=query, section=section, page_size=page_size, order_by=order_by)
        results = raw["response"]["results"]
        articles = normalize_articles(results)

        upsert_cache(session, query=cache_key, articles=articles)
        return articles, False


def dedupe_articles(articles: list[dict]) -> list[dict]:
    """Drop duplicate articles, keying on the Guardian id (falling back to url)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for article in articles:
        key = article.get("id") or article.get("url", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(article)
    return unique


def search_news(tool_input: str, page_size: int = 8) -> tuple[list[dict], bool]:
    """Search The Guardian. tool_input is a JSON object with 'q' (a non-empty array of
    query strings). Results from all queries are merged and deduplicated by id.

    Raises ValueError (with the expected schema) if tool_input is invalid.
    """
    params = parse_search_input(tool_input)
    section = params.get("section")
    order_by = params.get("order_by", "newest")

    merged: list[dict] = []
    all_cached = True
    for query in params["q"]:
        articles, from_cache = search_single_query(query, section, page_size=page_size, order_by=order_by)
        all_cached = all_cached and from_cache
        merged.extend(articles)

    return dedupe_articles(merged), all_cached

