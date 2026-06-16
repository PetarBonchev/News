import json

from news_agent.llm.ollama_client import generate


SYSTEM_PROMPT = """You turn a raw user news request into one or more Guardian search queries.

HOW THE GUARDIAN SEARCH WORKS — read this carefully:
The API matches SHORT keyword tokens, not long phrases. A token is 1 word, occasionally 2-3 adjacent
words that form a single name (e.g. Lewis Hamilton, Formula One). Long flat queries like
'latest Formula One grand prix race result' match poorly. Instead, build a query by JOINING short
tokens with boolean operators:
- AND   — both tokens must appear:     Hamilton AND Ferrari
- OR    — either token may appear:     F1 OR Formula One
- NOT   — exclude a token:             Apple AND NOT fruit
- (...) — group tokens:                (NBA OR basketball) AND finals

So: take the core tokens, then connect them. Use OR for synonyms/alternatives of the SAME thing,
AND to require multiple distinct concepts, NOT to exclude. Example:
  'what happened at the f1 race' → tokens: F1/Formula One, race → (F1 OR Formula One) AND race

NEVER use double-quote characters anywhere in a query. Adjacent words are already treated as a phrase,
so write Formula One, not quoted. Double quotes break the JSON and must not appear.

Choosing tokens — drop noise, keep meaning:
- Results are already newest-first, so NEVER include time words: latest, recent, today, now.
- Drop words that only restate it is news: news, update, story, article.
- KEEP subject-matter tokens that say what the user wants: race, result, winner, final, trial,
  election, etc. These narrow the topic and are valuable.

How many queries:
- Most requests need exactly ONE query (one boolean expression). Return a single-element array.
- Return MULTIPLE queries (2-3) only when the topic has genuinely distinct sub-angles that one
  expression cannot capture, or when broader coverage clearly helps.

Rules:
- Output ONLY a JSON array of strings. No prose, no markdown, no keys.
- Each query is a boolean expression of short tokens. Avoid long flat word sequences.
- NEVER put double-quote characters inside a query.
- Do not invent facts or add a date unless the user mentioned one.

Examples:
User: 'what's the latest with electric cars'
["electric cars OR electric vehicles OR EVs"]

User: 'tell me about the ukraine war'
["Ukraine AND (war OR conflict OR offensive)", "Ukraine AND (peace OR ceasefire OR negotiations)"]

User: 'who won the nba finals'
["(NBA OR basketball) AND finals AND (winner OR champion)"]

User: 'what happened at the latest f1 grand prix'
["(Formula One OR F1) AND (race OR grand prix) AND (result OR winner)"]
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
