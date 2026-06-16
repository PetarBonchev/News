"""Lightweight debug logging gated by the DEBUG_LOGS env flag.

When DEBUG_LOGS is enabled, debug() prints labelled, dimmed messages to the
console; otherwise it is a no-op. Used to trace each tool's input and output.
"""

from rich.console import Console

from news_agent.config import DEBUG_LOGS

_console = Console()


def debug(label: str, value: object = "") -> None:
    """Print a debug line if DEBUG_LOGS is enabled, otherwise do nothing."""
    if not DEBUG_LOGS:
        return
    text = str(value)
    if len(text) > 500:
        text = text[:500] + "…"
    suffix = f" {text}" if text else ""
    _console.print(f"[dim][debug] {label}:{suffix}[/dim]")
