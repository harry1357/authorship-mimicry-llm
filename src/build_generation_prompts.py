# src/build_generation_prompts.py
"""
Prompt Construction Module for Authorship Mimicry Experiments

This module constructs generation prompts for the authorship mimicry research project.
It processes author-specific corpora, selects appropriate training examples based on
topic similarity and consistency metrics, and formats prompts according to experimental
requirements (simple vs. complex variants).

The prompt construction process involves:
1. Loading author metadata and topic assignments
2. Selecting training reviews based on embedding similarity
3. Formatting prompts with appropriate instructions and examples
4. Generating output files for downstream text generation

Key Functions:
    load_author_list: Retrieves the list of authors included in the study
    load_topics: Loads topic assignments for training and generation phases
    build_prompts_for_full_run: Main orchestration function for prompt generation
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np

from generation_config import (
    AUTHOR_LIST_FILE,
    TOPICS_FILE,
    AUTHOR_CATEGORIES_FILE,
    CORPUS_DIR,
    EMBEDDINGS_DIR,
    PROMPTS_DIR,
    REFERENCE_MODEL_KEY,
    REFERENCE_CONSISTENCY_CSV,
    DEFAULT_GEN_PARAMS,
)


def load_author_list() -> List[str]:
    """
    Load the list of author identifiers included in the experimental dataset.
    
    Returns:
        List of author ID strings
        
    Note:
        The function handles files with or without a header line containing "author_id"
    """
    ids: List[str] = []
    with AUTHOR_LIST_FILE.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                if line.lower() == "author_id":
                    continue
            ids.append(line)
    return ids


def load_topics() -> Dict[str, Dict[str, str]]:
    """
    Load topic assignments for each author in the dataset.
    
    The topic file specifies which product categories are assigned to each author
    for training examples and generation targets across different experimental phases.
    
    Returns:
        Dictionary mapping author IDs to their topic assignments
        
    Format:
        {author_id: {topic_key: category_name, ...}, ...}
    """
    topics: Dict[str, Dict[str, str]] = {}
    with TOPICS_FILE.open("r", encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if header is None:
                header = parts
                continue
            author_id = parts[0]
            row = dict(zip(header[1:], parts[1:]))
            topics[author_id] = row
    return topics


def load_author_categories() -> Dict[str, List[str]]:
    """
    Load product category associations for each author.
    
    Returns:
        Dictionary mapping author IDs to lists of product categories they reviewed
        
    Note:
        Filters out 'NA' entries from the category lists
    """
    mapping: Dict[str, List[str]] = {}
    with AUTHOR_CATEGORIES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            author_id = parts[0]
            cats = [c for c in parts[1:] if c and c != "NA"]
            mapping[author_id] = cats
    return mapping


def load_selected_indices() -> Dict[str, List[int]]:
    """
    Load pre-computed indices of high-consistency reviews for each author.
    
    These indices identify reviews that demonstrate high stylistic consistency
    with the author's typical writing patterns, as determined by previous analysis.
    
    Returns:
        Dictionary mapping author IDs to lists of selected review indices
        
    Note:
        Handles both list and string representations of indices in the CSV file
    """
    mapping: Dict[str, List[int]] = {}
    with REFERENCE_CONSISTENCY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            author_id = row["author_id"]
            raw = row.get("selected_indices", "").strip()
            if not raw:
                continue
            try:
                indices = ast.literal_eval(raw)
            except Exception:
                indices = [
                    int(x)
                    for x in raw.replace("[", "").replace("]", "").split(",")
                    if x.strip()
                ]
            indices = [int(i) for i in indices]
            if indices:
                mapping[author_id] = indices
    return mapping


def read_review_text(path: Path) -> str:
    """
    Read the text content of a single review file.
    
    Args:
        path: Path to the review text file
        
    Returns:
        String containing the review text
    """
    return path.read_text(encoding="utf-8")


def build_complex_prompt_text(
    author_id: str,
    training_reviews: List[Dict[str, Any]],
    generation_topic: str,
    include_author_id: bool = True,
) -> str:
    lines: List[str] = []

    lines.append("You are an expert at mimicking writing style while changing the topic.")
    if include_author_id:
        lines.append(
            f"Below are several example Amazon product reviews written by the SAME anonymous author (ID: {author_id})."
        )
    else:
        lines.append(
            "Below are several example Amazon product reviews written by the SAME anonymous author."
        )
    lines.append(
        "Your task is to write a NEW product review in the same writing style "
        "(sentence rhythm, vocabulary, level of detail, tone), "
        f"but about a product in the following category: {generation_topic}."
    )
    lines.append("")
    lines.append("Constraints:")
    lines.append("1. Do not copy any specific sentences or phrases from the examples.")
    lines.append("2. Write around 600–900 words.")
    lines.append(
        "3. Write a single coherent review with natural structure "
        "(introduction, details, evaluation, conclusion)."
    )
    lines.append(
        "4. Do not mention that you are an AI model or that you are imitating anyone."
    )
    lines.append("")
    lines.append("==== EXAMPLE REVIEWS START ====")
    lines.append("")

    for i, rev in enumerate(training_reviews, start=1):
        lines.append(
            f"### Example review {i} (Category: {rev.get('category', 'Unknown')})"
        )
        lines.append(rev["text"].strip())
        lines.append("")

    lines.append("==== EXAMPLE REVIEWS END ====")
    lines.append("")
    lines.append("Now write the new review.")

    return "\n".join(lines)


def build_simple_prompt_text(
    author_id: str,
    training_reviews: List[Dict[str, Any]],
    generation_topic: str,
) -> str:
    """
    Ultra-minimal 'simple' condition:
    No extra constraints, no word counts, just:
    - here are example reviews
    - learn the style
    - write about this category
    """
    lines: List[str] = []

    lines.append(
        "Below are example Amazon product reviews written by the same person."
    )
    lines.append(
        f"Learn their writing style and write a new Amazon product review "
        f"about a product in the category: {generation_topic}."
    )
    lines.append("")
    lines.append("Constraints:")
    lines.append("Write around 600–900 words.")
    lines.append("")
    lines.append("==== EXAMPLE REVIEWS START ====")
    lines.append("")

    for i, rev in enumerate(training_reviews, start=1):
        lines.append(f"### Example review {i}")
        lines.append(rev["text"].strip())
        lines.append("")

    lines.append("==== EXAMPLE REVIEWS END ====")
    lines.append("")
    lines.append("Now write the new review.")

    return "\n".join(lines)


def deterministic_sample(author_id: str, candidates: List[str], k: int) -> List[str]:
    if not candidates:
        return []
    rng = random.Random(sum(ord(c) for c in author_id) + 12345)
    if len(candidates) <= k:
        return list(candidates)
    return rng.sample(candidates, k)


def choose_generation_topics(
    author_id: str,
    topic_row: Dict[str, str] | None,
    training_categories: List[str],
    author_categories: List[str] | None,
) -> Tuple[str, str, Dict[str, Any]]:
    training_set = set(c for c in training_categories if c and c != "NA")

    if author_categories:
        base_cats = [c for c in author_categories if c and c != "NA"]
    else:
        base_cats = []

    if not base_cats and topic_row:
        base_cats = [v for v in topic_row.values() if v and v != "NA"]

    if not base_cats:
        base_cats = ["HomeandKitchen", "MoviesandTV", "Electronics", "Beauty"]

    non_training = [c for c in base_cats if c not in training_set]

    g1_pref = topic_row.get("generation1") if topic_row else None
    g2_pref = topic_row.get("generation2") if topic_row else None

    g1 = None
    g2 = None

    if g1_pref and g1_pref in non_training:
        g1 = g1_pref
    if g2_pref and g2_pref in non_training and g2_pref != g1:
        g2 = g2_pref

    remaining = [c for c in non_training if c not in {g1, g2}]

    if g1 is None:
        pick = deterministic_sample(author_id, remaining, 1)
        if pick:
            g1 = pick[0]
            remaining = [c for c in remaining if c != g1]

    if g2 is None:
        pick = deterministic_sample(author_id + "_g2", remaining, 1)
        if pick:
            g2 = pick[0]

    if g1 is None or g2 is None or g1 == g2:
        extras = [c for c in base_cats if c not in {g1, g2}]
        picks = deterministic_sample(author_id + "_fallback", extras, 2)
        for p in picks:
            if g1 is None:
                g1 = p
            elif g2 is None or g2 == g1:
                g2 = p

    if g1 is None:
        g1 = base_cats[0]
    if g2 is None:
        g2 = base_cats[1 if len(base_cats) > 1 else 0]

    meta = {
        "training_categories": sorted(training_set),
        "base_categories": base_cats,
        "non_training_candidates": non_training,
        "g1_pref": g1_pref,
        "g2_pref": g2_pref,
    }
    return g1, g2, meta


def build_prompts_for_full_run(full_run: int, prompt_variant: str) -> Path:
    """
    prompt_variant: 'complex' or 'simple'.

    For 'complex', we keep the original filename:
      data/prompts/generation_prompts_fullrun{full_run}.jsonl

    For 'simple', we write:
      data/prompts/generation_prompts_simple_fullrun{full_run}.jsonl
    """
    author_ids = load_author_list()
    topics = load_topics()
    author_cats_map = load_author_categories()
    selected_idx_map = load_selected_indices()

    if prompt_variant == "complex":
        output_path = PROMPTS_DIR / f"generation_prompts_fullrun{full_run}.jsonl"
    else:
        output_path = PROMPTS_DIR / f"generation_prompts_simple_fullrun{full_run}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for author_id in author_ids:
            topic_row = topics.get(author_id)
            if not topic_row:
                print(f"[build_prompts] WARNING: no topics row for {author_id}, skipping.")
                continue

            npz_path = EMBEDDINGS_DIR / REFERENCE_MODEL_KEY / f"{author_id}.npz"
            if not npz_path.exists():
                print(f"[build_prompts] WARNING: missing embeddings npz for {author_id}, skipping.")
                continue

            data = np.load(npz_path, allow_pickle=True)
            files_arr = data["files"]

            if len(files_arr) < 6:
                print(
                    f"[build_prompts] WARNING: author {author_id} has only {len(files_arr)} files, "
                    "need at least 6; skipping."
                )
                continue

            idx_list = selected_idx_map.get(author_id, [])
            use_luar_indices = False
            idx_sorted: List[int] = []

            if idx_list and len(idx_list) >= 6:
                idx_sorted = sorted(idx_list)[:6]
                if max(idx_sorted) < len(files_arr) and min(idx_sorted) >= 0:
                    use_luar_indices = True
                else:
                    print(
                        f"[build_prompts] WARNING: LUAR indices out of range for {author_id}, "
                        "falling back to first 6 files."
                    )

            if use_luar_indices:
                selected_paths: List[Path] = []
                for i in idx_sorted:
                    raw_path_str = str(files_arr[i])
                    p = Path(raw_path_str)
                    if not p.is_absolute():
                        p = CORPUS_DIR / p
                    selected_paths.append(p)
                selected_source = "luar_selected_indices"
            else:
                idx_sorted = list(range(6))
                selected_paths = []
                for i in idx_sorted:
                    raw_path_str = str(files_arr[i])
                    p = Path(raw_path_str)
                    if not p.is_absolute():
                        p = CORPUS_DIR / p
                    selected_paths.append(p)
                selected_source = "fallback_first6"
                print(
                    f"[build_prompts] INFO: using fallback first-6 files for {author_id} "
                    f"(no valid selected_indices)."
                )

            group1_paths = selected_paths[:3]
            group2_paths = selected_paths[3:6]

            def make_training_reviews(paths: List[Path]) -> List[Dict[str, Any]]:
                trs: List[Dict[str, Any]] = []
                for p in paths:
                    try:
                        text = read_review_text(p)
                    except FileNotFoundError:
                        print(f"[build_prompts] WARNING: file not found {p}, skipping author {author_id}.")
                        return []
                    stem = p.stem
                    if "_" in stem:
                        category = stem.split("_", 1)[1]
                    else:
                        category = "Unknown"
                    trs.append(
                        {
                            "path": str(p),
                            "text": text,
                            "category": category,
                        }
                    )
                return trs

            training_group1 = make_training_reviews(group1_paths)
            training_group2 = make_training_reviews(group2_paths)

            if not training_group1 or not training_group2:
                continue

            training_categories = [r["category"] for r in (training_group1 + training_group2)]

            author_categories = author_cats_map.get(author_id)
            g1, g2, topic_meta = choose_generation_topics(
                author_id=author_id,
                topic_row=topic_row,
                training_categories=training_categories,
                author_categories=author_categories,
            )

            configs = [
                (1, training_group1, g1),
                (2, training_group2, g2),
            ]

            for prompt_index, training_reviews, generation_topic in configs:
                if prompt_variant == "complex":
                    prompt_text = build_complex_prompt_text(
                        author_id=author_id,
                        training_reviews=training_reviews,
                        generation_topic=generation_topic,
                        include_author_id=True,
                    )
                else:
                    prompt_text = build_simple_prompt_text(
                        author_id=author_id,
                        training_reviews=training_reviews,
                        generation_topic=generation_topic,
                    )

                seed = full_run * 100 + prompt_index

                record: Dict[str, Any] = {
                    "prompt_id": f"{author_id}_run{full_run}_p{prompt_index}_{prompt_variant}",
                    "author_id": author_id,
                    "full_run": full_run,
                    "prompt_index": prompt_index,
                    "prompt_variant": prompt_variant,
                    "generation_topic": generation_topic,
                    "training_reviews": training_reviews,
                    "prompt_text": prompt_text,
                    "max_tokens": DEFAULT_GEN_PARAMS.max_tokens,
                    "temperature": DEFAULT_GEN_PARAMS.temperature,
                    "seed": seed,
                    "metadata": {
                        "topics_row": topic_row,
                        "selected_indices": idx_sorted,
                        "selected_source": selected_source,
                        "topic_meta": topic_meta,
                        "group": f"G{prompt_index}",
                    },
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(
        f"[build_prompts] Wrote {count} {prompt_variant} prompts for full_run={full_run} to {output_path}"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-run",
        type=int,
        choices=[1, 2],
        default=1,
        help="Which full run to build prompts for (1 or 2).",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        choices=["complex", "simple"],
        default="complex",
        help="Prompt variant to build.",
    )
    args = parser.parse_args()
    build_prompts_for_full_run(args.full_run, args.prompt_variant)


if __name__ == "__main__":
    main()