import hashlib
import json
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from news_agent.models.cache import CacheEntry

CACHE_HOURS = 6


def hash_query(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()


def get_fresh_cached_articles(session: Session, query: str) -> list[dict] | None:
    qh = hash_query(query)
    cutoff = datetime.now() - timedelta(hours=CACHE_HOURS)

    stmt = select(CacheEntry).where(
        CacheEntry.query_hash == qh,
        CacheEntry.fetched_at >= cutoff,
    )
    entry = session.scalar(stmt)

    if not entry:
        return None

    payload = json.loads(entry.articles_json)

    if isinstance(payload, list):
        return payload

    return None


def upsert_cache(session: Session, query: str, articles: list[dict]) -> None:
    qh = hash_query(query)
    entry = session.get(CacheEntry, qh)
    articles_json = json.dumps(articles, ensure_ascii=False)

    if entry:
        entry.query_text = query
        entry.articles_json = articles_json
        entry.fetched_at = datetime.now()
    else:
        session.add(
            CacheEntry(
                query_hash=qh,
                query_text=query,
                articles_json=articles_json,
                fetched_at=datetime.now(),
            )
        )

    session.commit()

