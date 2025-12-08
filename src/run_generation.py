# src/run_generation.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from generation_config import GENERATED_DIR, PROMPTS_DIR, DEFAULT_LLM_KEY
from llm_client import get_llm_client, LLMRequest, LLMResponse


def get_prompts_path(full_run: int, prompt_variant: str) -> Path:
    if prompt_variant == "complex":
        return PROMPTS_DIR / f"generation_prompts_fullrun{full_run}.jsonl"
    else:
        return PROMPTS_DIR / f"generation_prompts_simple_fullrun{full_run}.jsonl"


def get_output_path(llm_key: str, full_run: int, prompt_variant: str) -> Path:
    out_dir = GENERATED_DIR / llm_key
    out_dir.mkdir(parents=True, exist_ok=True)

    if prompt_variant == "complex":
        return out_dir / f"generations_fullrun{full_run}.jsonl"
    else:
        return out_dir / f"generations_simple_fullrun{full_run}.jsonl"


def run_generation(full_run: int, llm_key: str, prompt_variant: str) -> Path:
    prompts_path = get_prompts_path(full_run, prompt_variant)
    output_path = get_output_path(llm_key, full_run, prompt_variant)

    print(f"[run_generation] full_run={full_run} llm={llm_key} variant={prompt_variant}")
    print(f"[run_generation] prompts: {prompts_path}")
    print(f"[run_generation] output: {output_path}")

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    client = get_llm_client(llm_key)

    written = 0

    with prompts_path.open("r", encoding="utf-8") as pf, output_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line in pf:
            line = line.strip()
            if not line:
                continue

            prompt_record: Dict[str, Any] = json.loads(line)
            author_id = prompt_record.get("author_id")
            prompt_id = prompt_record.get("prompt_id")
            prompt_index = prompt_record.get("prompt_index")
            generation_topic = prompt_record.get("generation_topic")
            temp = prompt_record.get("temperature", 0.7)
            max_tokens = prompt_record.get("max_tokens", 1200)

            print(
                f"[run_generation] author={author_id} prompt_id={prompt_id} "
                f"full_run={full_run} p={prompt_index} variant={prompt_variant}"
            )

            req = LLMRequest(
                prompt_id=prompt_id,
                author_id=author_id,
                run_id=full_run,
                prompt_text=prompt_record["prompt_text"],
                max_tokens=max_tokens,
                temperature=temp,
                seed=None,  # seed not supported by Responses API
                metadata={
                    "prompt_index": prompt_index,
                    "generation_topic": generation_topic,
                    "prompt_variant": prompt_variant,
                },
            )

            resp: LLMResponse = client.generate(req)

            out_record: Dict[str, Any] = {
                "llm_key": resp.llm_key,
                "prompt_variant": prompt_variant,
                "full_run": full_run,
                "prompt_id": prompt_id,
                "author_id": author_id,
                "prompt_index": prompt_index,
                "generation_topic": generation_topic,
                "temperature": temp,
                "max_tokens": max_tokens,
                "prompt_text": prompt_record["prompt_text"],
                "training_reviews": prompt_record.get("training_reviews"),
                "metadata": prompt_record.get("metadata", {}),
                "response": {
                    "generated_text": resp.generated_text,
                    "usage": resp.usage,
                    "raw_response": resp.raw_response,
                },
            }

            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[run_generation] wrote {written} generations to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-run",
        type=int,
        choices=[1, 2],
        required=True,
        help="Which full run (1 or 2).",
    )
    parser.add_argument(
        "--llm-key",
        type=str,
        default=DEFAULT_LLM_KEY,
        help="Which LLM key to use (e.g. gpt-5.1).",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        choices=["complex", "simple"],
        default="complex",
        help="Prompt variant to use.",
    )
    args = parser.parse_args()

    run_generation(args.full_run, args.llm_key, args.prompt_variant)


if __name__ == "__main__":
    main()