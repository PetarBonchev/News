import json

from news_agent.llm.ollama_client import generate


SYSTEM_PROMPT = """You are a news briefing assistant.

You will receive:
- the user query
- a list of news articles with title, date, url, snippet, and body

Read the FULL body of each article, not just its title or snippet. An article whose
title is about a different topic may still contain a passage relevant to the query —
extract and use that passage. Focus the briefing on the parts of each article that
actually address the user's query, and ignore unrelated sections.

Write a concise news briefing with exactly these sections:

Key developments
Main actors
Contradictions or uncertainties
Open questions

Rules:
- Be factual and cautious. Ground every statement in the article bodies.
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
                "body": article.get("body", "")[:6000],
            }
        )

    prompt = (
        f"User query:\n{query}\n\n"
        f"Articles:\n{json.dumps(compact_articles, ensure_ascii=False, indent=2)}\n\n"
        "Write the briefing now."
    )

    briefing = generate(
        model=model,
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.2,
    ).strip()

    sources = "\n".join(
        f"- {a['title']} ({a['date'][:10]})\n  {a['url']}"
        for a in compact_articles
        if a.get("url")
    )
    if sources:
        briefing += f"\n\nSources:\n{sources}"

    return briefing
