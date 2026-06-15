"""HyDE (Hypothetical Document Embeddings) query transform.

A user question and a news article are asymmetric: the question is short and
interrogative, the article is long and declarative. Scoring them directly
understates relevance. HyDE bridges the gap by asking the LLM to write a short
hypothetical news snippet that *would* answer the question, then scoring that
answer-shaped text against the real articles.

The hypothetical answer may be factually wrong — that is fine. It is never shown
to the user or fed to the summarizer; it is used only to compute relevance, where
having the right entities and vocabulary is what matters.
"""

from news_agent.llm.ollama_client import generate

SYSTEM_PROMPT = """You write a short hypothetical news snippet that would answer the user's question.

This text is only used to find matching articles by topical similarity, so it must
describe the GENERAL shape of an answer, not invented specifics.

Rules:
- Write 1-3 sentences in a factual news style, on the topic of the question.
- Use the general vocabulary and entity types a real answer would contain
  (e.g. "a driver", "a team", "the race") rather than guessing specific names.
- Do NOT invent specific names, dates, years, scores, or places. Wrong specifics
  hurt matching; generic topical wording matches the right articles better.
- Output only the snippet. No preamble, no markdown, no explanation.
"""


def hypothetical_answer(question: str, model: str) -> str:
    """Generate an answer-shaped snippet for the question, or '' on failure."""
    prompt = f"Question:\n{question}\n\nHypothetical news snippet:"
    result = generate(
        model=model,
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.3,
    )
    return result.strip()
