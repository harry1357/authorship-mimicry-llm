# src/llm_client.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI


@dataclass
class LLMRequest:
    prompt_id: str
    author_id: str
    run_id: int          # here, run_id = full_run (1 or 2)
    prompt_text: str
    max_tokens: int = 1024
    temperature: float = 0.8
    seed: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    prompt_id: str
    author_id: str
    run_id: int
    llm_key: str
    generated_text: str
    usage: Dict[str, Any]
    raw_response: Dict[str, Any]


class BaseLLMClient:
    def generate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError


class GPT51Client(BaseLLMClient):
    def __init__(self, model: str = "gpt-5.1") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please export it before running."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.llm_key = model

    def generate(self, req: LLMRequest) -> LLMResponse:
        response = self.client.responses.create(
            model=self.model,
            input=req.prompt_text,
            temperature=req.temperature,
            max_output_tokens=req.max_tokens,
            seed=req.seed,
        )

        text = ""
        if hasattr(response, "output_text") and response.output_text:
            text = response.output_text
        else:
            try:
                text = response.output[0].content[0].text
            except Exception:
                text = ""

        usage_dict: Dict[str, Any] = {}
        if hasattr(response, "usage") and response.usage is not None:
            usage = response.usage
            usage_dict = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        return LLMResponse(
            prompt_id=req.prompt_id,
            author_id=req.author_id,
            run_id=req.run_id,
            llm_key=self.llm_key,
            generated_text=text.strip(),
            usage=usage_dict,
            raw_response={},  # keep small; can expand if needed
        )


class MockLLMClient(BaseLLMClient):
    def __init__(self, llm_key: str = "mock-llm") -> None:
        self.llm_key = llm_key

    def generate(self, req: LLMRequest) -> LLMResponse:
        base = req.prompt_text[:400]
        fake_text = (
            f"[MOCK GENERATION for author {req.author_id}, run {req.run_id}]\n\n"
            f"{base}\n\n"
            "[... truncated mock output ...]"
        )
        usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }
        return LLMResponse(
            prompt_id=req.prompt_id,
            author_id=req.author_id,
            run_id=req.run_id,
            llm_key=self.llm_key,
            generated_text=fake_text,
            usage=usage,
            raw_response={"mock": True},
        )


def get_llm_client(llm_key: str) -> BaseLLMClient:
    key = llm_key.lower()
    if key in {"gpt-5.1", "gpt5.1", "gpt5_1", "gpt-5"}:
        return GPT51Client(model="gpt-5.1")
    if key in {"mock", "fake"}:
        return MockLLMClient()
    raise ValueError(f"Unknown llm_key: {llm_key}")