from news_agent.llm.ollama_client import generate


SYSTEM_PROMPT = """You convert raw user news requests into a short, clean English search query.

Rules:
- Output only the rewritten search query.
- Do not explain anything.
- Keep it under 12 words if possible.
- Preserve the user's topic and intent.
- If the input is already clear, return it with minimal cleanup.
- If the input is vague, make it specific enough for a news search.
"""


def purify_query(raw_input: str, model: str) -> str:
    prompt = f"Raw user input:\n{raw_input}\n\nClean search query:"
    result = generate(
        model=model,
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.1,
    )
    return result.strip().strip('"')
