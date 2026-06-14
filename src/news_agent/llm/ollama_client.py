import requests

from news_agent.config import OLLAMA_BASE_URL


def list_models() -> list[str]:
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30)
    response.raise_for_status()
    data = response.json()
    return [model["name"] for model in data.get("models", [])]


def generate(
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.2,
    response_format: dict | None = None,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    if response_format is not None:
        payload["format"] = response_format

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["response"]
