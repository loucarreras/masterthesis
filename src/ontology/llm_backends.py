from __future__ import annotations
import json
import os
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class LLMBackend:

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class OllamaBackend(LLMBackend):

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", strip_thinking: bool = False, num_ctx: int = 8192):
        self.model = model
        self.url = f"{base_url}/api/generate"
        self.strip_thinking = strip_thinking
        self.num_ctx = num_ctx

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_ctx": self.num_ctx,
            }
        }
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        data = response.json()

        total_time = data.get("total_duration")
        text = data.get("response", "") or ""
        thinking = data.get("thinking", "") or ""

        if not text.strip() and thinking.strip():
            text = thinking
        
        # print(json.dumps(response.json(), indent=2)) 

        if self.strip_thinking:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text, total_time

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
        print(response.json())
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class HuggingFaceBackend(LLMBackend):
    def __init__(self, model_id: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)