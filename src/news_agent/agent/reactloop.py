import json
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from news_agent.llm.ollama_client import generate
from news_agent.tools.filter import filter_articles
from news_agent.tools.registry import get_tool_descriptions, get_tool_registry

from news_agent.prompts.abstract import SYSTEM_PROMPT as ABSTRACT_PROMPT
from news_agent.prompts.structured import SYSTEM_PROMPT as STRUCTURED_PROMPT
from news_agent.prompts.chainofthought import SYSTEM_PROMPT as COT_PROMPT

class SearchParams(BaseModel):
    q: str = Field(default="")
    section: str = Field(default="")
    from_days: int = Field(default=0)


class ReActStep(BaseModel):
    kind: str = Field(description="Either 'action' or 'final'")
    thought: str = Field(default="")
    tool_name: str = Field(default="")
    query_text: str = Field(default="", description="Plain text input for purifyquery and summarizearticles")
    search_params: SearchParams = Field(default_factory=SearchParams, description="Structured input for searchnews")
    final_answer: str = Field(default="")


REACT_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["action", "final"]},
        "thought": {"type": "string"},
        "tool_name": {"type": "string"},
        "query_text": {"type": "string"},
        "search_params": {
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "section": {"type": "string"},
                "from_days": {"type": "integer"},
            },
            "required": ["q"],
        },
        "final_answer": {"type": "string"},
    },
    "required": ["kind", "thought", "tool_name", "query_text", "search_params", "final_answer"],
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

You are also working inside a ReAct-style tool loop.

Available tools:
{tools_text}

You must return exactly one JSON object per turn.

Critical rules:
- NEVER treat the number of articles or a raw list of titles as the final answer.
- If you have article results, you MUST call summarizearticles BEFORE returning a final answer.
- Only return kind="final" AFTER summarizearticles has been used (unless no articles were found).
- If you need to use a tool, return kind="action".
- If you have enough information to answer the user, return kind="final".
- Use only the listed tool names.
- Keep thought short.
- Do not invent tool results.
- Do not wrap JSON in markdown.
- Do not output any text outside the JSON object.

Tool sequence — follow this order every time:
1. purifyquery — only if the user input is vague or conversational.
   Set query_text to the raw user question. Leave search_params empty.
   Example: query_text="what's going on with the war in ukraine lately"
   Returns a clean query. Use it as search_params.q in step 2.

2. searchnews — search for articles.
   Set search_params.q (required), search_params.section (optional), search_params.from_days (optional).
   Example: search_params={{"q": "Ukraine Russia war", "section": "world", "from_days": 7}}
   The observation will tell you how many relevant articles were found.
   - If relevant articles were found: proceed to step 3.
   - If 0 relevant articles: call searchnews again with a different q, section, or from_days.
   - Do NOT call searchnews again with the exact same search_params.q.

3. summarizearticles — call this once you have relevant articles.
   Set query_text to the user's original question. Leave search_params empty.
   Example: query_text="What is happening in Ukraine?"

4. Return kind="final" with final_answer set to the summary from step 3.

For kind="action": fill thought, tool_name, and either query_text or search_params. Set final_answer to "".
For kind="final": fill thought, final_answer. Set tool_name and query_text to "" and search_params to empty.
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


def call_tool(step: ReActStep, model: str, context: dict) -> str:
    registry = get_tool_registry()
    tool_name = step.tool_name

    if tool_name not in registry:
        return f"Unknown tool: {tool_name}"

    if tool_name == "purifyquery":
        result = registry[tool_name].fn(step.query_text, model=model)
        context["last_query"] = result
        return f"Purified query: {result}\nNext: call searchnews with this query."

    if tool_name == "searchnews":
        params = step.search_params
        tool_input_str = json.dumps({"q": params.q, "section": params.section, "from_days": params.from_days})
        articles, from_cache = registry[tool_name].fn(tool_input_str)
        query = context.get("last_query") or params.q
        context["last_query"] = query
        relevant, dropped = filter_articles(articles, query=query, model=model)
        context["last_articles"] = relevant
        observation = format_articles_for_observation(relevant, from_cache)
        if relevant:
            observation += f"\n{dropped} irrelevant articles filtered out. Proceed to summarizearticles."
        else:
            observation += "\n0 relevant articles found. Call searchnews again with a different query or section."
        return observation

    if tool_name == "summarizearticles":
        articles = context.get("last_articles")
        query = context.get("last_query") or step.query_text

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
        if not step.tool_name:
            raise ValueError("Action step must include tool_name.")
        if step.tool_name == "searchnews" and not step.search_params.q:
            raise ValueError("searchnews requires search_params.q to be set.")
        if step.tool_name in ("purifyquery", "summarizearticles") and not step.query_text:
            raise ValueError(f"{step.tool_name} requires query_text to be set.")

    if step.kind == "final" and not step.final_answer:
        raise ValueError("Final step must include final_answer.")

    return step


def run_react_loop(
    user_input: str,
    model: str,
    prompt_style: str = "abstract",
    max_iterations: int = 8,
    temperature: float = 0.0,
) -> ReActResult:
    system_prompt = build_react_system_prompt(prompt_style)

    history_parts = [
        f"User request: {user_input}",
        "",
        "Return the next step as JSON.",
    ]
    context: dict = {}
    seen_calls: set[tuple[str, str]] = set()
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
            answer = step.final_answer.strip()
            if len(answer) <= 3 and context.get("last_articles") and not context.get("last_summary"):
                fallback_step = ReActStep(kind="action", tool_name="summarizearticles", query_text=context.get("last_query", user_input))
                observation = call_tool(fallback_step, model, context)
                tool_calls += 1
                trace_lines.append(f"Observation: [auto-summarize fallback]\n{observation}")
                return ReActResult(final_answer=observation, trace="\n".join(trace_lines), tool_calls=tool_calls)

            return ReActResult(
                final_answer=answer or "I could not generate a useful final answer.",
                trace="\n".join(trace_lines),
                tool_calls=tool_calls,
            )

        call_key = (step.tool_name, step.search_params.q if step.tool_name == "searchnews" else step.query_text)
        if call_key in seen_calls:
            if step.tool_name == "purifyquery" and context.get("last_query"):
                observation = f"Query already purified: {context['last_query']}. Now call searchnews with this query."
            elif step.tool_name == "searchnews" and context.get("last_articles"):
                fallback_step = ReActStep(kind="action", tool_name="summarizearticles", query_text=context.get("last_query", user_input))
                observation = call_tool(fallback_step, model, context)
                tool_calls += 1
                trace_lines.append(f"Observation: {observation}")
                return ReActResult(final_answer=observation, trace="\n".join(trace_lines), tool_calls=tool_calls)
            else:
                observation = f"You already called '{step.tool_name}' with that input. Try a different approach."
            history_parts.append(f"Observation: {observation}")
            trace_lines.append(f"Observation: {observation}")
            continue

        seen_calls.add(call_key)
        observation = call_tool(step, model, context)
        tool_calls += 1

        tool_input_display = step.search_params.q if step.tool_name == "searchnews" else step.query_text
        history_parts.append(f"Tool used: {step.tool_name}")
        history_parts.append(f"Tool input: {tool_input_display}")
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
