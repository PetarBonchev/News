import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from news_agent.config import (
    DEFAULT_MODEL,
    DEFAULT_MODELS,
    DEFAULT_NUM_AGENTS,
    DEFAULT_PROMPTS,
    DEFAULT_TEMPERATURE,
)
from news_agent.llm.ollama_client import generate, list_models
from news_agent.llm.specs import AgentSpec
from news_agent.prompts.abstract import SYSTEM_PROMPT as ABSTRACT_PROMPT
from news_agent.prompts.structured import SYSTEM_PROMPT as STRUCTURED_PROMPT
from news_agent.prompts.chainofthought import SYSTEM_PROMPT as COT_PROMPT


PROMPT_MAP = {
    "abstract": ABSTRACT_PROMPT,
    "structured": STRUCTURED_PROMPT,
    "chainofthought": COT_PROMPT,
}


def build_agent_specs(
    models_arg: str | None,
    prompts_arg: str | None,
    num_agents_arg: int | None,
) -> list[AgentSpec]:
    models = [m.strip() for m in (models_arg or DEFAULT_MODELS).split(",") if m.strip()]
    prompts = [p.strip() for p in (prompts_arg or DEFAULT_PROMPTS).split(",") if p.strip()]
    num_agents = num_agents_arg or DEFAULT_NUM_AGENTS

    if not models:
        models = [DEFAULT_MODEL]

    specs: list[AgentSpec] = []
    for i in range(num_agents):
        model = models[i % len(models)]
        prompt_style = prompts[i % len(prompts)]
        specs.append(
            AgentSpec(
                name=chr(65 + i),
                model=model,
                prompt_style=prompt_style,
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="What is happening with climate policy?")
    parser.add_argument("--models", help="Comma-separated Ollama models")
    parser.add_argument("--prompts", help="Comma-separated prompt styles")
    parser.add_argument("--num-agents", type=int, help="Number of agents to run")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--list-models", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        for model_name in list_models():
            print(model_name)
        return

    specs = build_agent_specs(
        models_arg=args.models,
        prompts_arg=args.prompts,
        num_agents_arg=args.num_agents,
    )

    print("Running agents sequentially...\n")

    for spec in specs:
        system_prompt = PROMPT_MAP.get(spec.prompt_style)
        if not system_prompt:
            raise ValueError(f"Unknown prompt style: {spec.prompt_style}")

        response = generate(
            model=spec.model,
            prompt=args.query,
            system=system_prompt,
            temperature=args.temperature,
        )

        print("=" * 80)
        print(f"Agent: {spec.name}")
        print(f"Model: {spec.model}")
        print(f"Prompt style: {spec.prompt_style}")
        print("-" * 80)
        print(response.strip())
        print()


if __name__ == "__main__":
    main()
