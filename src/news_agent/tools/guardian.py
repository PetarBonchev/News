import os

import requests

from news_agent.db.cache_repo import get_fresh_cached_articles, upsert_cache
from news_agent.db.session import SessionLocal
from news_agent.models.article import Article

GUARDIAN_URL = "https://content.guardianapis.com/search"


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


def fetch_guardian(query: str, page_size: int = 8) -> dict:
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        raise RuntimeError("GUARDIAN_API_KEY is not set")

    params = {
        "q": query,
        "page-size": page_size,
        "show-fields": "headline,trailText,bodyText",
        "order-by": "newest",
        "api-key": api_key,
    }

    response = requests.get(GUARDIAN_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def search_news(query: str, page_size: int = 8) -> tuple[list[dict], bool]:
    with SessionLocal() as session:
        cached = get_fresh_cached_articles(session, query)
        if cached is not None:
            return cached, True

        raw = fetch_guardian(query=query, page_size=page_size)
        results = raw["response"]["results"]
        articles = normalize_articles(results)

        upsert_cache(session, query=query, articles=articles)
        return articles, False

