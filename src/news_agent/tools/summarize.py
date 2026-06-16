import json

from news_agent.llm.ollama_client import generate


_BASE_RULES = """
You will receive the user query and a list of news articles (title, date, url, snippet, body).
Read the FULL body of each article. Focus only on parts relevant to the user query.
Be factual. Ground every statement in the article bodies. Do not invent facts.
Do not mention missing tools or system issues.
"""

_STYLE_PROMPTS = {
    "abstract": _BASE_RULES + """
Write EXACTLY 3 sentences.
No headers. No bullets. No bold text. Plain prose only.
Casual tone, like telling a friend what happened.
Stop after the 3rd sentence.
""",
    "structured": _BASE_RULES + """
Write a markdown briefing with exactly these sections and bullet points:
## Key Developments
## Main Actors
## Contradictions or Uncertainties
## Open Questions
Be terse and factual.
""",
    "chainofthought": _BASE_RULES + """
Write multiple paragraphs with full context: background, what happened, who is involved, why it matters, what comes next.
Include specific facts, numbers, and quotes from the articles.
End with a paragraph on open questions or what to watch for.
""",
}

_DEFAULT_STYLE = "structured"


def summarize_articles(query: str, articles: list[dict], model: str, style: str = _DEFAULT_STYLE) -> str:
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

    system_prompt = _STYLE_PROMPTS.get(style, _STYLE_PROMPTS[_DEFAULT_STYLE])
    briefing = generate(
        model=model,
        prompt=prompt,
        system=system_prompt,
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
