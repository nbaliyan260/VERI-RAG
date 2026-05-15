"""Probe LLM providers before long experiment runs."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMProbeResult:
    ok: bool
    message: str


def probe_openai(model_name: str = "gpt-4o-mini") -> LLMProbeResult:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return LLMProbeResult(False, "OPENAI_API_KEY is not set in .env")

    try:
        from openai import OpenAI
    except ImportError:
        return LLMProbeResult(False, "openai package not installed (pip install -e '.[llm]')")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        return LLMProbeResult(True, "OpenAI API reachable")
    except Exception as exc:  # noqa: BLE001 — surface provider message to CLI
        err = str(exc).lower()
        if "insufficient_quota" in err or "exceeded your current quota" in err:
            return LLMProbeResult(
                False,
                "OpenAI quota exhausted — add billing at platform.openai.com/account/billing",
            )
        if "invalid_api_key" in err or "incorrect api key" in err:
            return LLMProbeResult(False, "Invalid OPENAI_API_KEY")
        return LLMProbeResult(False, f"OpenAI probe failed: {exc}")


def probe_ollama(
    model_name: str = "llama3",
    base_url: str = "http://localhost:11434",
) -> LLMProbeResult:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status != 200:
                return LLMProbeResult(False, f"Ollama returned HTTP {resp.status}")
    except urllib.error.URLError as exc:
        return LLMProbeResult(
            False,
            f"Ollama not running at {base_url} (start with: ollama serve)",
        )
    except Exception as exc:  # noqa: BLE001
        return LLMProbeResult(False, f"Ollama probe failed: {exc}")

    return LLMProbeResult(True, f"Ollama reachable (model {model_name} not verified)")


def resolve_profile_with_fallback(
    profile: str,
    *,
    fallback_mock: bool = True,
    model_name: str = "gpt-4o-mini",
) -> tuple[str, str | None]:
    """Return (effective_profile, warning_message)."""
    if profile == "mock":
        return profile, None

    if profile.startswith("openai"):
        result = probe_openai(model_name=model_name)
        if result.ok:
            return profile, None
        if fallback_mock:
            return "mock", result.message
        raise RuntimeError(result.message)

    if profile.startswith("ollama"):
        result = probe_ollama()
        if result.ok:
            return profile, None
        if fallback_mock:
            return "mock", result.message
        raise RuntimeError(result.message)

    return profile, None
