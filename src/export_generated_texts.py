# src/export_generated_texts.py
"""
Generated Text Export Module

This module provides functionality for extracting and organizing generated texts from
the raw LLM output files. It processes JSONL generation records, extracts the generated
text content, and exports it in a structured format suitable for stylometric analysis.

The export process organizes texts by author and generation topic, facilitating
systematic comparison between genuine author texts and LLM-generated mimicry attempts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from generation_config import GENERATED_DIR, PROMPTS_DIR


def extract_text(record: Dict[str, Any]) -> str:
    """
    Extract generated text from a generation record.
    
    This function handles various record formats and nested structures that may
    appear in the LLM API responses, providing robust text extraction across
    different API versions and response formats.
    
    Args:
        record: Dictionary containing the generation record
        
    Returns:
        The generated text string
        
    Raises:
        KeyError: If no recognized text field is found in the record
    """
    # Check for direct text fields in the record
    for key in ["generated_text", "output_text", "text"]:
        if key in record:
            return record[key]

    # Check for text fields within a nested response object
    resp = record.get("response")
    if isinstance(resp, Dict):
        for key in ["generated_text", "output_text", "text"]:
            if key in resp:
                return resp[key]

    raise KeyError("Could not find generated text field in record.")


def get_prompts_path(full_run: int, prompt_variant: str) -> Path:
    """
    Determine the file path for generation prompts.
    
    Args:
        full_run: Experimental run identifier
        prompt_variant: Prompt complexity type ("simple" or "complex")
        
    Returns:
        Path to the prompts JSONL file
    """
    if prompt_variant == "complex":
        return PROMPTS_DIR / f"generation_prompts_fullrun{full_run}.jsonl"
    else:
        return PROMPTS_DIR / f"generation_prompts_simple_fullrun{full_run}.jsonl"


def get_generations_path(llm_key: str, full_run: int, prompt_variant: str) -> Path:
    """
    Determine the file path for generated texts.
    
    Args:
        llm_key: LLM model identifier
        full_run: Experimental run identifier
        prompt_variant: Prompt complexity type
        
    Returns:
        Path to the generations JSONL file
    """
    if prompt_variant == "complex":
        return GENERATED_DIR / llm_key / f"generations_fullrun{full_run}.jsonl"
    else:
        return GENERATED_DIR / llm_key / f"generations_simple_fullrun{full_run}.jsonl"


def load_prompt_topics(full_run: int, prompt_variant: str) -> Dict[str, str]:
    """
    Load the mapping of prompt IDs to generation topics.
    
    This mapping is used to organize exported texts by their target generation topic,
    enabling topic-specific analysis and comparison.
    
    Args:
        full_run: Experimental run identifier
        prompt_variant: Prompt complexity type
        
    Returns:
        Dictionary mapping prompt IDs to topic names
        
    Raises:
        FileNotFoundError: If the prompts file does not exist
    """
    path = get_prompts_path(full_run, prompt_variant)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("prompt_id")
            topic = rec.get("generation_topic", "UNKNOWN_TOPIC")
            if pid is not None:
                mapping[pid] = topic
    return mapping


def clean_topic_for_filename(topic: str) -> str:
    topic_clean = "".join(c for c in topic if c.isalnum() or c in "_-")
    return topic_clean or "UNKNOWN_TOPIC"


def export_generations(
    llm_key: str,
    full_run: int,
    prompt_variant: str,
    author_filter: str | None = None,
) -> Path:
    input_path = get_generations_path(llm_key, full_run, prompt_variant)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    prompt_topics = load_prompt_topics(full_run, prompt_variant)

    output_root = GENERATED_DIR / llm_key / f"texts_{prompt_variant}_fullrun{full_run}"
    output_root.mkdir(parents=True, exist_ok=True)

    count = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            author_id = record.get("author_id", "UNKNOWN_AUTHOR")
            if author_filter and author_id != author_filter:
                continue

            full_run_rec = record.get("full_run", full_run)
            prompt_index = record.get("prompt_index")
            prompt_id = record.get("prompt_id")

            topic = "UNKNOWN_TOPIC"
            if prompt_id is not None and prompt_id in prompt_topics:
                topic = prompt_topics[prompt_id]

            topic_clean = clean_topic_for_filename(topic)

            text = extract_text(record)

            author_dir = output_root / author_id
            author_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{author_id}_run{full_run_rec}_p{prompt_index}_{topic_clean}.txt"
            out_path = author_dir / filename

            out_path.write_text(text, encoding="utf-8")
            count += 1

    print(f"[export] Wrote {count} {prompt_variant} text files under {output_root}")
    return output_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-key",
        type=str,
        default="gpt-5.1",
        help="Which LLM key subfolder to read from (e.g. gpt-5.1).",
    )
    parser.add_argument(
        "--full-run",
        type=int,
        choices=[1, 2],
        required=True,
        help="Which full run to export (1 or 2).",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        choices=["complex", "simple"],
        default="complex",
        help="Prompt variant to export.",
    )
    parser.add_argument(
        "--author-id",
        type=str,
        default=None,
        help="If provided, only export generations for this author.",
    )
    args = parser.parse_args()

    export_generations(args.llm_key, args.full_run, args.prompt_variant, args.author_id)


if __name__ == "__main__":
    main()