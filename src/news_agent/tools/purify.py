import json

from news_agent.llm.ollama_client import generate


SYSTEM_PROMPT = """You turn a raw user news request into one or more Guardian search queries.

The Guardian 'q' parameter supports operators:
- AND   — both terms must appear:      Ferrari AND Hamilton
- OR    — either term may appear:      EVs OR electric cars
- NOT   — exclude a term:              Apple AND NOT fruit
- "..." — exact phrase:                "Lewis Hamilton"
- (...) — group terms:                 climate AND (policy OR summit)

Output a JSON array of search query strings.

Keep queries focused — drop noise, keep meaning:
- Results are already returned newest-first, so NEVER add time words: drop "latest", "recent",
  "today", "this week", "current", "now". They add nothing and hurt matching.
- Drop words that only restate it is news: "news", "update", "story", "article". They add nothing.
- KEEP words that describe what the user wants — these narrow the topic and are valuable:
  "race", "result", "winner", "final", "score", "report", "trial", "election", etc.
  Example: for "who won the f1 race" keep "race" → "Formula One race result", NOT just "Formula One".
- So strip recency/"it's news" words, but keep the subject-matter words.

Quoting:
- Default to PLAIN keywords with no quotes. Plain keywords match broadly and find more articles.
- Quotes force an EXACT phrase match and usually return FEWER or ZERO results — use them rarely.
- Only quote a genuine proper noun that must stay together, e.g. a person or place: "Lewis Hamilton".

How many queries:
- Most requests need exactly ONE query. Return a single-element array.
- Use OR inside one query to cover synonyms or alternate phrasings of the SAME thing.
- Return MULTIPLE queries (2-3) only when the topic has genuinely distinct sub-angles
  that a single query cannot capture well, or when broader coverage clearly helps.

Rules:
- Output ONLY a JSON array of strings. No prose, no markdown, no keys.
- Each query: short, specific, English keywords.
- Do not invent facts or add a date unless the user mentioned one.

Examples:
User: "what's the latest with electric cars"
["electric cars OR electric vehicles OR EVs"]

User: "tell me about the ukraine war"
["Ukraine war", "Ukraine peace negotiations"]

User: "who won the nba finals"
["NBA finals winner"]

User: "what happened at the latest f1 grand prix"
["Formula One grand prix result", "F1 race report"]
"""


def _parse_queries(raw: str) -> list[str]:
    """Parse the model output into a list of query strings, tolerating common mistakes."""
    stripped = raw.strip()

    if stripped.startswith("```"):
        stripped = stripped.strip("`").lstrip("json").strip()

    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        cleaned = stripped.strip().strip('"')
        return [cleaned] if cleaned else []

    if isinstance(data, str):
        return [data.strip()] if data.strip() else []

    if isinstance(data, list):
        return [q.strip() for q in data if isinstance(q, str) and q.strip()]

    return []


def purify_query(raw_input: str, model: str) -> list[str]:
    prompt = f"Raw user input:\n{raw_input}\n\nSearch queries (JSON array):"
    result = generate(
        model=model,
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.1,
    )

    queries = _parse_queries(result)
    return queries or [raw_input.strip()]
