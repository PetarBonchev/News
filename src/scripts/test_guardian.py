import hashlib
import json
import os
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

if not GUARDIAN_API_KEY:
    raise RuntimeError("GUARDIAN_API_KEY is not set in .env")

import sys
sys.path.insert(0, os.path.abspath("src"))

from news_agent.models.cache import CacheEntry

engine = create_engine(DATABASE_URL)


def query_hash(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()


def fetch_guardian(query: str) -> dict:
    url = "https://content.guardianapis.com/search"
    params = {
        "q": query,
        "page-size": 5,
        "show-fields": "headline,trailText,bodyText",
        "order-by": "newest",
        "api-key": GUARDIAN_API_KEY,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def search_news(query: str) -> tuple[dict, bool]:
    qh = query_hash(query)
    cutoff = datetime.now() - timedelta(hours=6)

    with Session(engine) as session:
        stmt = select(CacheEntry).where(
            CacheEntry.query_hash == qh,
            CacheEntry.fetched_at >= cutoff,
        )
        cached = session.scalar(stmt)

        if cached:
            return json.loads(cached.articles_json), True

        data = fetch_guardian(query)
        payload = json.dumps(data)

        existing = session.get(CacheEntry, qh)
        if existing:
            existing.query_text = query
            existing.articles_json = payload
            existing.fetched_at = datetime.now()
        else:
            session.add(
                CacheEntry(
                    query_hash=qh,
                    query_text=query,
                    articles_json=payload,
                    fetched_at=datetime.now(),
                )
            )

        session.commit()
        return data, False


def print_results(data: dict, from_cache: bool) -> None:
    print(f"from_cache={from_cache}")
    results = data["response"]["results"]

    for item in results[:5]:
        fields = item.get("fields", {})
        print("-" * 80)
        print("title:", item.get("webTitle"))
        print("url:", item.get("webUrl"))
        print("headline:", fields.get("headline"))
        print("trail:", fields.get("trailText"))


if __name__ == "__main__":
    query = "climate"
    data, from_cache = search_news(query)
    print_results(data, from_cache)

