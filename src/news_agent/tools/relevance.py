"""Cross-encoder relevance scoring for news articles.

The reranker scores each (query, article) pair; the coverage verdict and which
articles to keep are derived deterministically from those scores.
"""

import math
import os

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"
BODY_CHARS = 3000

GOOD_THRESHOLD = 0.6
PARTIAL_THRESHOLD = 0.3
KEEP_THRESHOLD = 0.3

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Load the cross-encoder once and reuse it for the rest of the process.

    Tries an offline load first (no network ping when the model is already
    cached); falls back to a normal load that downloads on first run.
    """
    global _model
    if _model is None:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        try:
            _model = CrossEncoder(MODEL_NAME, local_files_only=True)
        except (OSError, ValueError):
            _model = CrossEncoder(MODEL_NAME)
    return _model


def _article_text(article: dict) -> str:
    text = (article.get("body", "") or article.get("snippet", "")).strip()
    return text[:BODY_CHARS]


def _predict(left: str, bodies: list[str]) -> list[float]:
    """Score (left, body) pairs, mapping the model's raw logits to 0-1 probabilities."""
    logits = _get_model().predict([(left, b) for b in bodies])
    return [1.0 / (1.0 + math.exp(-float(z))) for z in logits]


def score_articles(query: str, articles: list[dict], hyde_text: str = "") -> list[float]:
    """Return a relevance score for each article against the query.

    If hyde_text is provided, each article is scored against both the original
    query and the hypothetical answer, and the higher of the two is kept. This
    rescues relevant articles the bare question scores poorly while leaving
    irrelevant ones low.
    """
    if not articles:
        return []

    bodies = [_article_text(a) for a in articles]
    query_scores = _predict(query, bodies)
    if not hyde_text:
        return query_scores

    hyde_scores = _predict(hyde_text, bodies)
    return [max(q, h) for q, h in zip(query_scores, hyde_scores)]


def assess_coverage(scores: list[float]) -> str:
    """Map the best score to a coverage verdict: good, partial, or none."""
    if not scores:
        return "none"
    top = max(scores)
    if top >= GOOD_THRESHOLD:
        return "good"
    if top >= PARTIAL_THRESHOLD:
        return "partial"
    return "none"


def select_relevant(
    articles: list[dict], scores: list[float]
) -> tuple[list[dict], list[str]]:
    """Keep articles scoring at or above KEEP_THRESHOLD, preserving order.

    Returns (kept_articles, dropped_titles).
    """
    kept: list[dict] = []
    dropped: list[str] = []
    for article, score in zip(articles, scores):
        if score >= KEEP_THRESHOLD:
            kept.append(article)
        else:
            dropped.append(article.get("title", ""))
    return kept, dropped
