from dataclasses import dataclass


@dataclass
class AgentSpec:
    name: str
    model: str
    prompt_style: str
