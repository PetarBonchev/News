import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from news_agent.agent.reactloop import run_react_loop
from news_agent.config import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="What is happening in Ukraine?")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    result = run_react_loop(
        user_input=args.query,
        model=args.model,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
    )

    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result.final_answer)
    print()
    print("=" * 80)
    print("TOOL CALLS")
    print("=" * 80)
    print(result.tool_calls)
    print()
    print("=" * 80)
    print("TRACE")
    print("=" * 80)
    print(result.trace)


if __name__ == "__main__":
    main()
