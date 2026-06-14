from dataclasses import dataclass
from time import perf_counter

from news_agent.agent.reactloop import run_react_loop
from news_agent.llm.specs import AgentSpec


@dataclass
class PipelineResult:
    agent_name: str
    model: str
    prompt_style: str
    final_answer: str
    trace: str
    tool_calls: int
    elapsed_seconds: float


def run_pipeline(
    user_input: str,
    agent_spec: AgentSpec,
    max_iterations: int = 8,
    temperature: float = 0.0,
) -> PipelineResult:
    started = perf_counter()

    result = run_react_loop(
        user_input=user_input,
        model=agent_spec.model,
        prompt_style=agent_spec.prompt_style,
        max_iterations=max_iterations,
        temperature=temperature,
    )
    
    elapsed = perf_counter() - started

    return PipelineResult(
        agent_name=agent_spec.name,
        model=agent_spec.model,
        prompt_style=agent_spec.prompt_style,
        final_answer=result.final_answer,
        trace=result.trace,
        tool_calls=result.tool_calls,
        elapsed_seconds=elapsed,
    )
