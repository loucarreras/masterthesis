from __future__ import annotations
import json
import os
import requests
import re

class LLMBackend:
    """Base class — subclass and implement generate()."""

    def generate(self, prompt: str) -> str:
        """Send prompt, return raw response text (should be valid JSON per classifier contract)."""
        raise NotImplementedError

class OllamaBackend(LLMBackend):
    """
    Runs any model served by Ollama (medgemma, biomistral, llama3, etc.).

    Args:
        model:   Ollama model tag, e.g. "medgemma1.5", "biomistral", "llama3.1:8b"
        base_url: Override if Ollama is running on a different host/port.
    """

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", strip_thinking: bool = False):
        self.model = model
        self.url = f"{base_url}/api/generate"
        self.strip_thinking = strip_thinking

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        text = response.json()["response"]
        if self.strip_thinking:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

# class OllamaBackend(LLMBackend):
#     def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", strip_thinking: bool = False):
#         self.model = model
#         self.base_url = base_url
#         self.strip_thinking = strip_thinking

#     def generate(self, prompt: str) -> str:
#         payload = {
#             "model": self.model,
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": "You are a clinical terminology expert. Always respond with valid JSON only — no preamble, no markdown fences."
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             "stream": False,
#             "format": "json",
#         }
#         response = requests.post(f"{self.base_url}/api/chat", json=payload)
#         response.raise_for_status()
#         text = response.json()["message"]["content"]  # note: different response path than /api/generate
#         if self.strip_thinking:
#             text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
#         return text
# ---------------------------------------------------------------------------
# Anthropic  (Claude family)
# ---------------------------------------------------------------------------

class AnthropicBackend(LLMBackend):
    """
    Uses the Anthropic Messages API directly.

    Args:
        model:   e.g. "claude-haiku-4-5-20251001", "claude-sonnet-4-6"
        api_key: Falls back to ANTHROPIC_API_KEY env var.

    Note: Instructs the model to reply with JSON only so the classifier can
          parse it the same way it parses Ollama responses.
    """

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def generate(self, prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": 512,
            "system": "You are a clinical terminology expert. Always respond with valid JSON only — no preamble, no markdown fences.",
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages", headers=headers, json=body
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


# ---------------------------------------------------------------------------
# OpenAI-compatible  (OpenAI, OpenRouter, vLLM, Together AI, Groq, …)
# ---------------------------------------------------------------------------

class OpenAICompatibleBackend(LLMBackend):
    """
    Any endpoint that speaks the OpenAI /v1/chat/completions protocol.

    Examples
    --------
    # OpenAI
    OpenAICompatibleBackend("gpt-4o-mini", api_key="sk-...")

    # OpenRouter  (access to Med42, BioMistral, Llama, Qwen, DeepSeek, …)
    OpenAICompatibleBackend(
        model="m42-health/med42-70b",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-...",
    )

    # Local vLLM / LM Studio
    OpenAICompatibleBackend(
        model="meditron-7b",
        base_url="http://localhost:8000/v1",
        api_key="none",          # vLLM ignores the key
    )
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
    ):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a clinical terminology expert. Always respond with valid JSON only — no preamble, no markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "response_format": {"type": "json_object"},  # ignored silently if unsupported
        }
        response = requests.post(self.url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]