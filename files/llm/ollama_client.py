"""
Minimal Ollama client.

Hard rules from the manager:
  * If Ollama is not running, do NOT silently fall back. Show a precise error.
  * If the model is missing, tell the user to run `ollama pull qwen2.5:3b`.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from app.config import (
    ERR_MODEL_MISSING,
    ERR_OLLAMA_UNAVAILABLE,
    OLLAMA_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_TAGS_URL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
)


class OllamaError(RuntimeError):
    """Raised whenever Ollama is unreachable or returns an error."""


class OllamaModelMissingError(OllamaError):
    """Raised when the requested model is not installed locally."""


def _ollama_is_up() -> bool:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _list_installed_models() -> list[str]:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        r.raise_for_status()
        data = r.json()
        return [m.get("name", "") for m in data.get("models", [])]
    except requests.RequestException:
        return []


def ensure_ollama_ready(model: str = OLLAMA_MODEL) -> None:
    """
    Verify that Ollama is reachable AND that the requested model is installed.
    Raises a precise OllamaError otherwise. No silent fallback.
    """
    if not _ollama_is_up():
        raise OllamaError(ERR_OLLAMA_UNAVAILABLE)

    installed = _list_installed_models()
    # Ollama tags include the tag suffix; match on prefix or exact name.
    if not any(m == model or m.startswith(f"{model}") for m in installed):
        raise OllamaModelMissingError(ERR_MODEL_MISSING)


def generate(
    prompt: str,
    *,
    model: str = OLLAMA_MODEL,
    format_json: bool = True,
    options: dict[str, Any] | None = None,
) -> str:
    """
    Call /api/generate and return the raw text response.
    `format_json=True` asks Ollama to constrain output to valid JSON.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {**OLLAMA_OPTIONS, **(options or {})},
    }
    if format_json:
        payload["format"] = "json"

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise OllamaError(ERR_OLLAMA_UNAVAILABLE) from e

    if r.status_code == 404:
        # Ollama returns 404 when the model isn't pulled
        raise OllamaModelMissingError(ERR_MODEL_MISSING)
    if r.status_code != 200:
        raise OllamaError(
            f"Ollama returned HTTP {r.status_code}: {r.text[:300]}"
        )

    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise OllamaError(f"Ollama returned non-JSON envelope: {r.text[:300]}") from e

    return data.get("response", "")
