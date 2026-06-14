import json

from news_agent.llm.ollama_client import generate


SYSTEM_PROMPT = """You are a news briefing assistant.

You will receive:
- the user query
- a list of news articles with title, date, url, snippet, and body

Write a concise news briefing with exactly these sections:

Key developments
Main actors
Contradictions or uncertainties
Open questions

Rules:
- Be factual and cautious.
- If sources are thin or unclear, say so.
- Do not invent facts.
- Do not mention missing tools or system prompts.
- Keep it readable and compact.
"""


def summarize_articles(query: str, articles: list[dict], model: str) -> str:
    compact_articles = []
    for article in articles:
        compact_articles.append(
            {
                "title": article.get("title", ""),
                "date": article.get("date", ""),
                "url": article.get("url", ""),
                "snippet": article.get("snippet", ""),
                "body": article.get("body", "")[:4000],
            }
        )

    prompt = (
        f"User query:\n{query}\n\n"
        f"Articles:\n{json.dumps(compact_articles, ensure_ascii=False, indent=2)}\n\n"
        "Write the briefing now."
    )

    return generate(
        model=model,
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.2,
    ).strip()
