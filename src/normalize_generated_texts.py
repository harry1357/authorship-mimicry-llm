# src/normalize_generated_texts.py

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

from generation_config import GENERATED_DIR

# Default target size ~4 KB
DEFAULT_TARGET_BYTES = 4096


def normalize_whitespace(text: str) -> str:
    """
    Collapse all whitespace (spaces, newlines, tabs) into single spaces,
    and strip leading/trailing spaces. Result is a single paragraph.
    """
    # Normalise newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse all whitespace sequences into a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter based on punctuation patterns.
    
    Splits text on sentence-ending punctuation (. ! ?) followed by whitespace.
    This is not a perfect linguistic sentence splitter, but sufficient for
    truncation purposes in this normalization pipeline.
    
    Returns:
        List of sentence strings
    """
    # Split on . ! ? followed by whitespace
    # Using a simpler pattern that doesn't require lookbehind
    parts = re.split(r'([.!?])\s+', text)
    
    # Reconstruct sentences by pairing text with punctuation
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        if i + 1 < len(parts):
            sentence = parts[i] + parts[i + 1]
            sentences.append(sentence.strip())
    
    # Add the last part if it exists and isn't just punctuation
    if parts and len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    
    return [s for s in sentences if s]


def hard_truncate(text: str, target_bytes: int, encoding: str = "utf-8") -> str:
    """
    Fallback truncation: cut at target_bytes, then backtrack to the last space
    so we don't leave an obviously broken word at the end.
    """
    raw = text.encode(encoding)[:target_bytes]
    cut = raw.decode(encoding, errors="ignore")
    last_space = cut.rfind(" ")
    if last_space != -1:
        cut = cut[:last_space]
    return cut.strip()


def truncate_to_target_bytes(text: str, target_bytes: int = DEFAULT_TARGET_BYTES) -> str:
    """
    Truncate text so that its UTF-8 byte length is <= target_bytes.
    Prefer truncating at sentence boundaries; if that fails, fall back
    to cutting at the last whitespace before the limit.
    """
    enc = "utf-8"
    if len(text.encode(enc)) <= target_bytes:
        return text

    sentences = split_into_sentences(text)
    if not sentences:
        # Fallback: raw truncation
        return hard_truncate(text, target_bytes, enc)

    kept: List[str] = []
    for sent in sentences:
        candidate = (" ".join(kept + [sent])).strip()
        if len(candidate.encode(enc)) > target_bytes:
            break
        kept.append(sent)

    if not kept:
        # Even the first sentence is too long; hard truncate
        return hard_truncate(text, target_bytes, enc)

    return " ".join(kept).strip()


def normalize_file(in_path: Path, out_path: Path, target_bytes: int = DEFAULT_TARGET_BYTES) -> None:
    """
    Read a .txt file, normalize whitespace, truncate to target size,
    and write to out_path.
    """
    text = in_path.read_text(encoding="utf-8")
    text = normalize_whitespace(text)
    text = truncate_to_target_bytes(text, target_bytes=target_bytes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def normalize_generated_texts(
    llm_key: str,
    full_run: int,
    prompt_variant: str,
    target_bytes: int = DEFAULT_TARGET_BYTES,
) -> Path:
    """
    Normalize all generated .txt reviews for a given (llm_key, full_run, variant).

    Input:
      data/generated/<llm_key>/texts_<variant>_fullrun<run>/AUTHOR_ID/*.txt

    Output:
      data/generated/<llm_key>/normalized/texts_<variant>_fullrun<run>/AUTHOR_ID/*.txt
    """
    variant_suffix = f"texts_{prompt_variant}_fullrun{full_run}"
    input_root = GENERATED_DIR / llm_key / variant_suffix
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    output_root = GENERATED_DIR / llm_key / "normalized" / variant_suffix
    output_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for in_path in input_root.rglob("*.txt"):
        # Preserve relative path inside author directories
        rel = in_path.relative_to(input_root)
        out_path = output_root / rel

        print(f"[normalize] {in_path} -> {out_path}")
        normalize_file(in_path, out_path, target_bytes=target_bytes)
        count += 1

    print(
        f"[normalize] Normalized {count} files for "
        f"llm_key={llm_key}, full_run={full_run}, variant={prompt_variant}, "
        f"target_bytes={target_bytes}"
    )
    return output_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-key",
        type=str,
        default="gpt-5.1",
        help="LLM key subfolder under data/generated (e.g. gpt-5.1).",
    )
    parser.add_argument(
        "--full-run",
        type=int,
        choices=[1, 2],
        required=True,
        help="Which full run (1 or 2).",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        choices=["complex", "simple"],
        required=True,
        help="Prompt variant to normalize.",
    )
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=DEFAULT_TARGET_BYTES,
        help="Target maximum size in bytes (default: 4096).",
    )
    args = parser.parse_args()

    normalize_generated_texts(
        llm_key=args.llm_key,
        full_run=args.full_run,
        prompt_variant=args.prompt_variant,
        target_bytes=args.target_bytes,
    )


if __name__ == "__main__":
    main()