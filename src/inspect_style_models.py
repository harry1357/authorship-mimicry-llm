#!/usr/bin/env python3
"""
Inspect max_count / max sequence length for all style-embedding models.

This script:
- Iterates over STYLE_MODEL_KEYS from generation_config.py
- Uses MODEL_CONFIGS from model_configs.py to get the HF model name (and tokenizer if needed)
- Loads tokenizer + config from Hugging Face
- Prints a table with a sensible `max_count` per model

Usage:
    python src/inspect_style_model_max_lengths.py
"""

from __future__ import annotations

from transformers import AutoConfig, AutoTokenizer

from generation_config import STYLE_MODEL_KEYS
from model_configs import MODEL_CONFIGS


def get_model_config(model_key: str) -> dict:
    """
    Resolve your internal model_key to its config dict in MODEL_CONFIGS.
    """
    if model_key not in MODEL_CONFIGS:
        raise KeyError(f"model_key '{model_key}' not found in MODEL_CONFIGS")
    return MODEL_CONFIGS[model_key]


def infer_max_count(hf_name: str, tokenizer_name: str | None = None) -> int:
    """
    Infer a sensible max_count from HF config + tokenizer.

    Priority:
        1. config.max_position_embeddings (architectural limit)
        2. tokenizer.model_max_length, if it's smaller and reasonable
    """
    tok_id = tokenizer_name or hf_name

    print(f"\n[INFO] Inspecting model:")
    print(f"       config:   {hf_name}")
    print(f"       tokenizer:{tok_id}")

    config = AutoConfig.from_pretrained(hf_name)
    tokenizer = AutoTokenizer.from_pretrained(tok_id)

    # 1) Architectural limit
    max_pos = getattr(config, "max_position_embeddings", None)

    # 2) Tokenizer limit (sometimes absurdly large when "unset")
    tok_max = getattr(tokenizer, "model_max_length", None)

    # Start with config limit if present
    if isinstance(max_pos, int) and max_pos > 0:
        max_count = max_pos
    else:
        max_count = None

    # Refine with tokenizer limit if it looks sane
    if isinstance(tok_max, int) and 0 < tok_max < 10_000:
        if max_count is None:
            max_count = tok_max
        else:
            max_count = min(max_count, tok_max)

    # Fallback if both look weird
    if max_count is None:
        # Very safe default; you can adjust this
        max_count = 256

    return int(max_count)


def main():
    rows = []

    for model_key in STYLE_MODEL_KEYS:
        try:
            cfg = get_model_config(model_key)
        except Exception as e:
            print(f"[ERROR] Could not find config for {model_key}: {e}")
            continue

        try:
            hf_name = cfg["hf_name"]
        except KeyError:
            print(f"[ERROR] Config for {model_key} has no 'hf_name' field: {cfg}")
            continue

        tokenizer_name = cfg.get("tokenizer_name")  # only set for STAR in your config

        try:
            max_count = infer_max_count(hf_name, tokenizer_name)
        except Exception as e:
            print(f"[ERROR] Could not inspect {hf_name} (tokenizer={tokenizer_name}): {e}")
            continue

        # Also record any manually-configured max_length in MODEL_CONFIGS (if present)
        configured_max = cfg.get("max_length")

        rows.append((model_key, hf_name, tokenizer_name, max_count, configured_max))

    # Pretty print as a table
    print("\n" + "=" * 100)
    print("Per-model max_count (sequence length) for style-embedding models")
    print("=" * 100)
    print(f"{'model_key':20} {'hf_name':35} {'tokenizer':20} {'hf_max_count':>12} {'cfg_max_length':>14}")
    print("-" * 100)

    for model_key, hf_name, tokenizer_name, max_count, configured_max in rows:
        short_hf = hf_name if len(hf_name) <= 33 else hf_name[:30] + "..."
        short_tok = (
            tokenizer_name
            if tokenizer_name and len(tokenizer_name) <= 18
            else (tokenizer_name[:15] + "..." if tokenizer_name else "-")
        )
        cfg_max_str = str(configured_max) if configured_max is not None else "-"
        print(
            f"{model_key:20} "
            f"{short_hf:35} "
            f"{short_tok:20} "
            f"{max_count:12d} "
            f"{cfg_max_str:14}"
        )

    print("=" * 100)
    print(
        "\nYou can now plug these hf_max_count values into your MODEL_CONFIGS "
        "as 'max_length' per model.\n"
        "For models where you already set max_length (e.g., LUAR orig = 256, STAR = 512), "
        "the 'cfg_max_length' column shows what you currently use."
    )


if __name__ == "__main__":
    main()