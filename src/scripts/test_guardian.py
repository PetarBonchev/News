import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from news_agent.tools.guardian import search_news


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "climate"
    articles, from_cache = search_news(query)

    print(f"from_cache={from_cache}")
    for article in articles[:5]:
        print("-" * 80)
        print("title:", article["title"])
        print("url:", article["url"])
        print("date:", article["date"])
        print("snippet:", article["snippet"])
