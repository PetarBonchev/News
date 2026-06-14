import json
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from news_agent.llm.ollama_client import generate
from news_agent.tools.registry import get_tool_descriptions, get_tool_registry


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


def build_react_system_prompt() -> str:
    tools_text = get_tool_descriptions()
    return f"""You are a local news research agent working in a ReAct loop.

Available tools:
{tools_text}

You must return exactly one JSON object per turn.

Rules:
- If you need to use a tool, return kind="action".
- If you have enough information to answer the user, return kind="final".
- Use only the listed tool names.
- Keep thought short.
- Do not invent tool results.
- Do not wrap JSON in markdown.
- Do not output any text outside the JSON object.

Tool usage guidance:
- Use purifyquery when the user's request is vague or messy.
- Use searchnews when you need current articles.
- Use summarizearticles after you have article results.

For kind="action":
- fill thought
- fill tool_name
- fill tool_input
- leave final_answer as empty string

For kind="final":
- fill thought
- fill final_answer
- leave tool_name and tool_input as empty strings
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
        result = registry[tool_name].fn(tool_input, model=model)
        context["last_query"] = result
        return result

    if tool_name == "searchnews":
        articles, from_cache = registry[tool_name].fn(tool_input)
        context["last_query"] = tool_input
        context["last_articles"] = articles
        return format_articles_for_observation(articles, from_cache)

    if tool_name == "summarizearticles":
        articles = context.get("last_articles")
        query = context.get("last_query", tool_input)

        if not articles:
            return "No articles available. Use searchnews first."

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
    max_iterations: int = 8,
    temperature: float = 0.0,
) -> ReActResult:
    system_prompt = build_react_system_prompt()

    history_parts = [
        f"User request: {user_input}",
        "",
        "Return the next step as JSON.",
    ]
    context: dict = {}
    trace_lines: list[str] = []
    tool_calls = 0

    for _ in range(max_iterations):
        prompt = "\n".join(history_parts)

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
                final_answer=step.final_answer,
                trace="\n".join(trace_lines),
                tool_calls=tool_calls,
            )

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

    fallback = context.get("last_summary") or "I could not complete the loop within the maximum number of iterations."
    return ReActResult(
        final_answer=fallback,
        trace="\n".join(trace_lines),
        tool_calls=tool_calls,
    )
