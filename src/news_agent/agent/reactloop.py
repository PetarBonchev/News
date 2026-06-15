import json
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError
from rich.console import Console

from news_agent.llm.ollama_client import generate
from news_agent.tools.filter import filter_articles
from news_agent.tools.guardian import parse_search_input
from news_agent.tools.registry import get_tool_descriptions, get_tool_registry
from news_agent.tools.requery import suggest_requery

_console = Console()

DECIDING_STATUS = "Deciding the next step..."

MAX_SEARCHES = 5

def search_budget_exceeded(context: dict) -> bool:
    """Count this searchnews call and report whether the per-run budget is spent."""
    context["search_count"] = context.get("search_count", 0) + 1
    return context["search_count"] > MAX_SEARCHES


def accumulate_articles(context: dict, new_articles: list[dict]) -> None:
    """Add newly found relevant articles to the running set, deduped by url."""
    kept = context.setdefault("last_articles", [])
    seen = {a.get("url") for a in kept}
    for article in new_articles:
        if article.get("url") not in seen:
            kept.append(article)
            seen.add(article.get("url"))


def record_tried_queries(context: dict, queries: list[str]) -> None:
    """Remember the queries a searchnews call used, so requery can avoid repeating them."""
    context.setdefault("tried_queries", []).extend(queries)


def tool_status(tool_name: str) -> str:
    """The live message to show while a tool runs (defined per tool in the registry)."""
    spec = get_tool_registry().get(tool_name)
    return spec.status if spec else f"Running {tool_name}..."

from news_agent.prompts.abstract import SYSTEM_PROMPT as ABSTRACT_PROMPT
from news_agent.prompts.structured import SYSTEM_PROMPT as STRUCTURED_PROMPT
from news_agent.prompts.chainofthought import SYSTEM_PROMPT as COT_PROMPT

class ReActStep(BaseModel):
    kind: str = Field(description="Either 'action' or 'final'")
    thought: str = Field(default="")
    tool_name: str = Field(default="")
    tool_input: str = Field(default="")
    final_answer: str = Field(default="")


REACT_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["action", "final"],
        },
        "thought": {
            "type": "string",
        },
        "tool_name": {
            "type": "string",
        },
        "tool_input": {
            "type": "string",
        },
        "final_answer": {
            "type": "string",
        },
    },
    "required": ["kind", "thought", "tool_name", "tool_input", "final_answer"],
}


@dataclass
class ReActResult:
    final_answer: str
    trace: str
    tool_calls: int

def get_style_prompt(prompt_style: str) -> str:
    mapping = {
        "abstract": ABSTRACT_PROMPT,
        "structured": STRUCTURED_PROMPT,
        "chainofthought": COT_PROMPT,
    }
    return mapping.get(prompt_style, ABSTRACT_PROMPT)


def build_react_system_prompt(prompt_style: str) -> str:
    tools_text = get_tool_descriptions()
    style_prompt = get_style_prompt(prompt_style)

    return f"""{style_prompt}

You are working inside a ReAct-style tool loop. Each turn you return exactly one JSON object, and YOU
decide what to do next based on what the tools return.

Available tools:
{tools_text}

Every action step has a single tool_input string. What goes in it depends on the tool:
- purifyquery / summarizearticles / suggestrequery → tool_input is plain text (the user's question).
- searchnews → tool_input is a JSON string with the search parameters, for example:
  {{"q": ["NBA finals result", "OKC Thunder championship"], "section": "sport", "order_by": "newest"}}
  "q" is required and is always an array of query strings (use one element for a single query).
  "section" and "order_by" ("newest" or "relevance") are optional.

Typical flow:
1. purifyquery — turn the user's raw question into clean search queries.
2. searchnews — run those queries. It filters the results for relevance and returns the relevant
   articles plus a coverage verdict (good / partial / none). Articles you find are KEPT across searches.
3. YOU judge the result:
   - If you have ANY relevant articles (coverage good or partial) → call summarizearticles. Even one
     on-topic article is enough to answer; do not keep searching just because coverage is "partial".
   - Only if coverage is "none" (zero relevant articles found so far) → call suggestrequery to get
     better queries, then searchnews again. Try a different section if it helps. On your LAST search,
     if you still have nothing, use order_by="relevance" to prioritise topical matches over recency.
   - If coverage stays "none" after a few tries → return kind="final" with an honest answer saying
     you could not find articles on the topic.
4. After summarizearticles, return kind="final" with its output as final_answer.

You are in control: you may call searchnews more than once with different queries. Judge relevance
yourself — do not summarize off-topic articles, and do not invent an answer when nothing fits.
You get a limited number of searches; if searchnews reports the search limit is reached, stop searching
and give an honest final answer.

For kind="action": fill thought, tool_name, tool_input. Leave final_answer "".
For kind="final": fill thought and final_answer. Leave tool_name and tool_input "".

Rules:
- Never return a real briefing as final_answer before summarizearticles has run.
- Never invent tool results or facts not present in the articles.
- Output only the JSON object — no markdown, no text outside it.
"""


def format_articles_for_observation(articles: list[dict], from_cache: bool) -> str:
    lines = [f"Returned {len(articles)} articles. from_cache={from_cache}"]
    for article in articles[:5]:
        lines.append(
            f"- {article.get('date', '')} | {article.get('title', '')} | {article.get('url', '')}"
        )
        snippet = article.get("snippet", "")
        if snippet:
            lines.append(f"  snippet: {snippet[:200]}")
    return "\n".join(lines)


def call_tool(tool_name: str, tool_input: str, model: str, context: dict) -> str:
    registry = get_tool_registry()

    if tool_name not in registry:
        return f"Unknown tool: {tool_name}"

    if tool_name == "purifyquery":
        queries = registry[tool_name].fn(tool_input, model=model)
        return f"Purified into search queries: {json.dumps(queries)}"

    if tool_name == "suggestrequery":
        tried = context.get("tried_queries", [])
        queries = suggest_requery(tool_input, tried_queries=tried, model=model)
        if not queries:
            return "No alternative queries suggested. Try a final answer if you cannot find anything."
        return f"Suggested alternative search queries (different from {json.dumps(tried)}): {json.dumps(queries)}"

    if tool_name == "searchnews":
        if search_budget_exceeded(context):
            if context.get("last_articles"):
                return (
                    f"Search limit reached ({MAX_SEARCHES} searches). Stop searching. "
                    "Call summarizearticles to summarize the relevant articles you already found."
                )
            return (
                f"Search limit reached ({MAX_SEARCHES} searches). No relevant articles were found for "
                "this query. Return a final answer saying you could not find articles on the topic."
            )

        try:
            params = parse_search_input(tool_input)
        except ValueError as e:
            return str(e)

        record_tried_queries(context, params["q"])
        articles, from_cache = registry[tool_name].fn(tool_input)
        query = context.get("user_input", "")
        relevant, dropped_titles, coverage = filter_articles(articles, query=query, model=model)
        accumulate_articles(context, relevant)

        observation = format_articles_for_observation(context["last_articles"], from_cache)
        observation += f"\nCoverage: {coverage}. {len(dropped_titles)} articles filtered out as irrelevant."
        if coverage == "none" and not context["last_articles"]:
            searches_left = MAX_SEARCHES - context["search_count"]
            if searches_left <= 1:
                observation += (
                    "\nNo relevant articles yet, and this is your last search. Call searchnews one more time "
                    'with order_by="relevance" to prioritise topical matches over recency.'
                )
            else:
                observation += (
                    "\nNo relevant articles yet. Call suggestrequery to get better queries, then searchnews again."
                )
        else:
            observation += "\nYou have relevant articles. Call summarizearticles, or call suggestrequery to get better queries, then searchnews again if you want broader coverage."
        return observation

    if tool_name == "summarizearticles":
        articles = context.get("last_articles")
        query = tool_input or context.get("user_input", "")

        if not articles:
            return "No relevant articles available. Search first, or give a final answer that nothing was found."

        summary = registry[tool_name].fn(query=query, articles=articles, model=model)
        context["last_summary"] = summary
        return summary

    return "Tool call path not implemented."


def parse_step(raw_text: str) -> ReActStep:
    data = json.loads(raw_text)
    step = ReActStep.model_validate(data)

    if step.kind == "action":
        if not step.tool_name or not step.tool_input:
            raise ValueError("Action step must include tool_name and tool_input.")

    if step.kind == "final":
        if not step.final_answer:
            raise ValueError("Final step must include final_answer.")

    return step


def run_react_loop(
    user_input: str,
    model: str,
    prompt_style: str = "abstract",
    max_iterations: int = 15,
    temperature: float = 0.0,
) -> ReActResult:
    system_prompt = build_react_system_prompt(prompt_style)

    history_parts = [
        f"User request: {user_input}",
        "",
        "Return the next step as JSON.",
    ]
    context: dict = {"user_input": user_input}
    trace_lines: list[str] = []
    tool_calls = 0

    for _ in range(max_iterations):
        prompt = "\n".join(history_parts)

        with _console.status(f"[cyan]{DECIDING_STATUS}[/cyan]", spinner="dots"):
            raw_response = generate(
                model=model,
                prompt=prompt,
                system=system_prompt,
                temperature=temperature,
                response_format=REACT_STEP_SCHEMA,
            ).strip()

        trace_lines.append(raw_response)

        try:
            step = parse_step(raw_response)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            observation = f"Invalid structured output: {str(e)}"
            history_parts.append(f"Observation: {observation}")
            trace_lines.append(f"Observation: {observation}")
            continue

        if step.kind == "final":
            return ReActResult(
                final_answer=step.final_answer.strip() or "I could not generate a useful final answer.",
                trace="\n".join(trace_lines),
                tool_calls=tool_calls,
            )

        with _console.status(f"[cyan]{tool_status(step.tool_name)}[/cyan]", spinner="dots"):
            observation = call_tool(
                tool_name=step.tool_name,
                tool_input=step.tool_input,
                model=model,
                context=context,
            )
        tool_calls += 1

        history_parts.append(f"Tool used: {step.tool_name}")
        history_parts.append(f"Tool input: {step.tool_input}")
        history_parts.append(f"Observation: {observation}")
        trace_lines.append(f"Observation: {observation}")

    raw_final = context.get("last_summary")
    if raw_final:
        fallback = raw_final
    else:
        fallback = "I could not complete the loop within the maximum number of iterations."

    return ReActResult(
        final_answer=fallback,
        trace="\n".join(trace_lines),
        tool_calls=tool_calls,
    )