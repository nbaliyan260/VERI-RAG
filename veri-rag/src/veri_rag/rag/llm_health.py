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
    except Exception as exc:  # noqa: BLE001
        err = str(exc).lower()
        if "insufficient_quota" in err or "exceeded your current quota" in err:
            return LLMProbeResult(
                False,
                "OpenAI quota exhausted — add billing at platform.openai.com/account/billing",
            )
        if "invalid_api_key" in err or "incorrect api key" in err:
            return LLMProbeResult(False, "Invalid OPENAI_API_KEY")
        return LLMProbeResult(False, f"OpenAI probe failed: {exc}")


def probe_anthropic(model_name: str = "claude-haiku-4-5-20251001") -> LLMProbeResult:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return LLMProbeResult(False, "ANTHROPIC_API_KEY is not set in .env")

    try:
        from anthropic import Anthropic
    except ImportError:
        return LLMProbeResult(
            False, "anthropic package not installed (pip install -e '.[llm]')"
        )

    try:
        client = Anthropic(api_key=api_key)
        client.messages.create(
            model=model_name,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        return LLMProbeResult(True, "Anthropic (Claude) API reachable")
    except Exception as exc:  # noqa: BLE001
        err = str(exc).lower()
        if "credit" in err or "balance" in err or "billing" in err:
            return LLMProbeResult(
                False,
                "Anthropic credits exhausted — check console.anthropic.com billing",
            )
        if "authentication" in err or "invalid" in err and "api" in err:
            return LLMProbeResult(False, "Invalid ANTHROPIC_API_KEY")
        return LLMProbeResult(False, f"Anthropic probe failed: {exc}")


def probe_ollama(
    model_name: str = "llama3",
    base_url: str = "http://localhost:11434",
) -> LLMProbeResult:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status != 200:
                return LLMProbeResult(False, f"Ollama returned HTTP {resp.status}")
    except urllib.error.URLError:
        return LLMProbeResult(
            False,
            f"Ollama not running at {base_url} (start with: ollama serve)",
        )
    except Exception as exc:  # noqa: BLE001
        return LLMProbeResult(False, f"Ollama probe failed: {exc}")

    return LLMProbeResult(True, f"Ollama reachable (model {model_name} not verified)")


def _is_anthropic_profile(profile: str) -> bool:
    return profile.startswith("anthropic") or profile.startswith("claude")


def _anthropic_model_for_profile(profile: str, default: str) -> str:
    if profile in ("claude_haiku", "anthropic_haiku"):
        return "claude-haiku-4-5-20251001"
    if profile in ("claude_sonnet", "anthropic_sonnet"):
        return "claude-sonnet-4-20250514"
    return default


def resolve_auto_profile() -> str:
    """Pick best available profile: Claude → OpenAI → mock."""
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return "claude_haiku"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    return "mock"


def resolve_profile_with_fallback(
    profile: str,
    *,
    fallback_mock: bool = True,
    model_name: str = "gpt-4o-mini",
) -> tuple[str, str | None]:
    """Return (effective_profile, warning_message)."""
    if profile == "auto":
        profile = resolve_auto_profile()
        if profile == "mock":
            return "mock", "No ANTHROPIC_API_KEY or OPENAI_API_KEY in .env — using mock"

    if profile == "mock":
        return profile, None

    if _is_anthropic_profile(profile):
        anthropic_model = _anthropic_model_for_profile(profile, model_name)
        result = probe_anthropic(model_name=anthropic_model)
        if result.ok:
            return profile, None
        if fallback_mock:
            openai_result = probe_openai(model_name=model_name)
            if openai_result.ok:
                return "openai", f"Claude unavailable ({result.message}); using OpenAI"
            return "mock", result.message
        raise RuntimeError(result.message)

    if profile.startswith("openai"):
        result = probe_openai(model_name=model_name)
        if result.ok:
            return profile, None
        if fallback_mock:
            anthropic_model = _anthropic_model_for_profile("claude_haiku", model_name)
            anthropic_result = probe_anthropic(model_name=anthropic_model)
            if anthropic_result.ok:
                return (
                    "claude_haiku",
                    f"OpenAI unavailable ({result.message}); using Claude",
                )
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
