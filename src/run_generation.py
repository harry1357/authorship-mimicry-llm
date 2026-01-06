# src/run_generation.py
"""
Text Generation Orchestration Module

This module coordinates the text generation process for the authorship mimicry
research project. It loads generation prompts, interfaces with LLM clients,
and persists generated outputs in a structured format for subsequent analysis.

The module supports multiple experimental runs and prompt variants (simple/complex)
to facilitate comparative evaluation of different generation strategies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from generation_config import GENERATED_DIR, PROMPTS_DIR, DEFAULT_LLM_KEY
from llm_client import get_llm_client, LLMRequest, LLMResponse


def get_prompts_path(full_run: int, prompt_variant: str) -> Path:
    """
    Determine the file path for generation prompts based on run parameters.
    
    Args:
        full_run: Experimental run identifier (1 or 2)
        prompt_variant: Type of prompt structure ("simple" or "complex")
        
    Returns:
        Path object pointing to the appropriate prompts file
    """
    if prompt_variant == "complex":
        return PROMPTS_DIR / f"generation_prompts_fullrun{full_run}.jsonl"
    else:
        return PROMPTS_DIR / f"generation_prompts_simple_fullrun{full_run}.jsonl"


def get_output_path(llm_key: str, full_run: int, prompt_variant: str) -> Path:
    """
    Construct the output file path for generated texts.
    
    Creates the necessary directory structure if it does not exist.
    
    Args:
        llm_key: Identifier for the LLM model being used
        full_run: Experimental run identifier
        prompt_variant: Type of prompt structure used
        
    Returns:
        Path object for the output JSONL file
    """
    out_dir = GENERATED_DIR / llm_key
    out_dir.mkdir(parents=True, exist_ok=True)

    if prompt_variant == "complex":
        return out_dir / f"generations_fullrun{full_run}.jsonl"
    else:
        return out_dir / f"generations_simple_fullrun{full_run}.jsonl"


def run_generation(full_run: int, llm_key: str, prompt_variant: str) -> Path:
    """
    Execute the text generation pipeline for a complete experimental run.
    
    This function processes all prompts for the specified configuration, generates
    text using the designated LLM, and writes results to a JSONL output file.
    Each line in the output contains the prompt, generated text, and associated
    metadata for downstream analysis.
    
    Args:
        full_run: Experimental run identifier (1 or 2)
        llm_key: LLM model identifier (e.g., "gpt-5.1")
        prompt_variant: Prompt structure type ("simple" or "complex")
        
    Returns:
        Path to the generated output file
        
    Raises:
        FileNotFoundError: If the specified prompts file does not exist
    """
    prompts_path = get_prompts_path(full_run, prompt_variant)
    output_path = get_output_path(llm_key, full_run, prompt_variant)

    print(f"[run_generation] Executing generation pipeline")
    print(f"[run_generation]   Run: {full_run}, Model: {llm_key}, Variant: {prompt_variant}")
    print(f"[run_generation]   Input prompts: {prompts_path}")
    print(f"[run_generation]   Output file: {output_path}")

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    # Load existing generations to support resume
    existing_prompt_ids = set()
    if output_path.exists():
        print(f"[run_generation] Found existing output, loading completed prompts...")
        with output_path.open("r", encoding="utf-8") as existing_f:
            for line in existing_f:
                try:
                    record = json.loads(line.strip())
                    existing_prompt_ids.add(record["prompt_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[run_generation] {len(existing_prompt_ids)} prompts already completed, will skip")

    client = get_llm_client(llm_key)

    written = 0
    skipped = 0

    # Open in append mode to preserve existing generations
    with prompts_path.open("r", encoding="utf-8") as pf, output_path.open(
        "a", encoding="utf-8"
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
            max_tokens = prompt_record.get("max_tokens", 2000)

            # Skip if already generated
            if prompt_id in existing_prompt_ids:
                skipped += 1
                if skipped % 10 == 0:  # Print every 10 skips to avoid spam
                    print(f"[run_generation] Skipped {skipped} already-completed prompts...")
                continue

            print(
                f"[run_generation] Processing prompt for author {author_id} "
                f"(ID: {prompt_id}, Index: {prompt_index}, Variant: {prompt_variant})"
            )

            req = LLMRequest(
                prompt_id=prompt_id,
                author_id=author_id,
                run_id=full_run,
                prompt_text=prompt_record["prompt_text"],
                max_tokens=max_tokens,
                temperature=temp,
                seed=None,  # Note: seed parameter not supported by OpenAI Responses API
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

    print(f"[run_generation] Completed: wrote {written} new generations to {output_path}")
    if skipped > 0:
        print(f"[run_generation] Skipped {skipped} already-completed prompts")
    print(f"[run_generation] Total in file: {len(existing_prompt_ids) + written}")
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