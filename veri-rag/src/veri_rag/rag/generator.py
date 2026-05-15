"""LLM generator abstraction — MockLLM for tests, OpenAI-compatible for production."""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Abstract LLM interface."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""

    @property
    def token_count(self) -> int:
        """Return the token count of the last generation (0 if not tracked)."""
        return 0


class MockLLM(BaseLLM):
    """Deterministic mock LLM for testing and development.

    The mock extracts information from the context to build plausible answers.
    This allows the full pipeline to run without an API key.
    """

    def __init__(self, **kwargs: Any):
        self._last_token_count = 0

    def generate(self, prompt: str) -> str:
        self._last_token_count = 0

        # Extract query from prompt
        query_match = re.search(r"Question:\s*(.+?)(?:\n|Context:)", prompt, re.DOTALL)
        query = query_match.group(1).strip() if query_match else ""

        # Extract chunk IDs and their content
        chunk_pattern = r"\[(c_[a-f0-9]+)\]\s*(.+?)(?=\n\n\[c_|\nAnswer|\nProvide|\Z)"
        chunks = re.findall(chunk_pattern, prompt, re.DOTALL)

        if not chunks:
            answer = "The evidence is insufficient to answer this question."
            self._last_token_count = len(answer.split())
            return answer

        # Check for instruction-like content in chunks — the mock deliberately
        # follows instructions to simulate a vulnerable LLM for attack testing
        for chunk_id, text in chunks:
            lower_text = text.lower()
            if any(
                phrase in lower_text
                for phrase in [
                    "ignore all previous instructions",
                    "ignore previous instructions",
                    "system override",
                    "highest priority source",
                    "you must answer",
                    "must refuse",
                    "must not answer",
                ]
            ):
                # Simulate the model following the injected instruction
                # Extract the target instruction
                for line in text.split("\n"):
                    line = line.strip()
                    if line and not any(
                        p in line.lower()
                        for p in ["ignore", "override", "priority", "supersedes"]
                    ):
                        answer = f"{line} [{chunk_id}]"
                        self._last_token_count = len(answer.split())
                        return answer

        # Check for refusal instructions (blocker attack)
        for chunk_id, text in chunks:
            if any(
                p in text.lower()
                for p in ["must refuse", "must not answer", "cannot answer", "safety reasons"]
            ):
                answer = f"I'm sorry, I cannot answer this question for safety reasons. [{chunk_id}]"
                self._last_token_count = len(answer.split())
                return answer

        # Default: build answer from the first relevant chunk
        # Use the first chunk's content as the answer basis
        primary_id, primary_text = chunks[0]
        # Extract the most answer-like sentence
        sentences = [s.strip() for s in primary_text.split(".") if s.strip()]
        if sentences:
            answer_text = sentences[0] + "."
        else:
            answer_text = primary_text.strip()[:200]

        # Add citations from used chunks
        cited_ids = [primary_id]
        if len(chunks) > 1:
            cited_ids.append(chunks[1][0])

        citations = ", ".join(f"[{cid}]" for cid in cited_ids)
        answer = f"{answer_text} {citations}"
        self._last_token_count = len(answer.split())
        return answer

    @property
    def model_name(self) -> str:
        return "mock-llm-v1"

    @property
    def token_count(self) -> int:
        return self._last_token_count


class OpenAICompatibleLLM(BaseLLM):
    """LLM using the OpenAI API (or any compatible endpoint)."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._last_token_count = 0
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        content = response.choices[0].message.content or ""
        self._last_token_count = response.usage.total_tokens if response.usage else 0
        return content.strip()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def token_count(self) -> int:
        return self._last_token_count


class OllamaLLM(BaseLLM):
    """LLM using a local Ollama endpoint."""

    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._last_token_count = 0

    def generate(self, prompt: str) -> str:
        import urllib.request
        import json

        payload = json.dumps({
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        content = data.get("response", "")
        self._last_token_count = data.get("eval_count", 0)
        return content.strip()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def token_count(self) -> int:
        return self._last_token_count


def create_llm(provider: str = "mock", **kwargs) -> BaseLLM:
    """Factory function for creating an LLM instance."""
    if provider == "mock":
        return MockLLM(**kwargs)
    elif provider == "openai":
        return OpenAICompatibleLLM(**kwargs)
    elif provider == "ollama":
        return OllamaLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
