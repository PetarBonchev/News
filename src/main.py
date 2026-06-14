import argparse

from rich.console import Console
from rich.panel import Panel

from news_agent.config import DEFAULT_MODELS, DEFAULT_NUM_AGENTS, DEFAULT_PROMPTS
from news_agent.llm.specs import AgentSpec
from news_agent.pipeline import run_pipeline


console = Console()


def build_agent_specs(
    models_arg: str | None,
    prompts_arg: str | None,
    num_agents_arg: int | None,
) -> list[AgentSpec]:
    models = [m.strip() for m in (models_arg or DEFAULT_MODELS).split(",") if m.strip()]
    prompts = [p.strip() for p in (prompts_arg or DEFAULT_PROMPTS).split(",") if p.strip()]
    num_agents = num_agents_arg or DEFAULT_NUM_AGENTS

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
    parser.add_argument("query", nargs="?", help="User query")
    parser.add_argument("--models", help="Comma-separated Ollama models")
    parser.add_argument("--prompts", help="Comma-separated prompt styles")
    parser.add_argument("--num-agents", type=int, help="Number of agents to run")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--show-trace", action="store_true")
    args = parser.parse_args()

    user_input = args.query or input("Enter your query: ").strip()
    if not user_input:
        console.print("[red]No query provided.[/red]")
        return

    specs = build_agent_specs(
        models_arg=args.models,
        prompts_arg=args.prompts,
        num_agents_arg=args.num_agents,
    )

    console.print("\n[bold]Running pipelines sequentially...[/bold]\n")

    results = []
    for spec in specs:
        console.print(
            f"[cyan]Running agent {spec.name}[/cyan] "
            f"(model={spec.model}, prompt={spec.prompt_style})"
        )
        result = run_pipeline(
            user_input=user_input,
            agent_spec=spec,
            max_iterations=args.max_iterations,
            temperature=args.temperature,
        )
        results.append(result)

    console.print()
    for result in results:
        body = (
            f"[bold]Model:[/bold] {result.model}\n"
            f"[bold]Prompt:[/bold] {result.prompt_style}\n"
            f"[bold]Tool calls:[/bold] {result.tool_calls}\n"
            f"[bold]Time:[/bold] {result.elapsed_seconds:.2f}s\n\n"
            f"{result.final_answer}"
        )
        console.print(
            Panel(
                body,
                title=f"Agent {result.agent_name}",
                expand=False,
            )
        )
        console.print()
        if args.show_trace:
          console.print(Panel(result.trace, title=f"Trace {result.agent_name}", expand=False))
          console.print()


    console.print("[bold]Done.[/bold]")


if __name__ == "__main__":
    main()
