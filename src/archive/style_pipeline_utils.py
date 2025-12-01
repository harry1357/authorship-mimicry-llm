import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# All six models
MODEL_CONFIG = {
    "luar_crud_orig": {
        "type": "luar_orig",
        "hf_name": "rrivera1849/LUAR-CRUD",
    },
    "luar_mud_orig": {
        "type": "luar_orig",
        "hf_name": "rrivera1849/LUAR-MUD",
    },
    "luar_crud_st": {
        "type": "st",
        "hf_name": "gabrielloiseau/LUAR-CRUD-sentence-transformers",
    },
    "luar_mud_st": {
        "type": "st",
        "hf_name": "gabrielloiseau/LUAR-MUD-sentence-transformers",
    },
    "style_embedding": {
        "type": "st",
        "hf_name": "AnnaWegmann/Style-Embedding",
    },
    "star": {
        "type": "star",
        "hf_name": "AIDA-UPM/star",
    },
}


def load_author_ids(author_ids_file: str) -> list[str]:
    """
    The author_ids_three_training_topics_x_two_two_generation_topics.txt file:

      header_line_with_topics
      A1A1BM6N28X9J0  Automotive  Beauty  ...

    We take the first token of each non-empty line after the header.
    """
    ids: list[str] = []
    with open(author_ids_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"No non-empty lines in {author_ids_file}")
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        ids.append(parts[0])
    return ids


def find_author_review_files(aavc_root: str, author_id: str) -> list[Path]:
    """
    Robust lookup:

      1) Prefer folder <root>/<author_id>/*.txt  (your current layout)
      2) Fallback: any <root>/**/<author_id>_*.txt

    We never *skip* an author here.
    """
    root = Path(aavc_root)
    author_dir = root / author_id
    files: list[Path] = []

    if author_dir.is_dir():
        files = sorted(author_dir.glob(f"{author_id}_*.txt"))

    if not files:
        # Fallback – search anywhere
        files = sorted(root.glob(f"**/{author_id}_*.txt"))

    return files


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore").strip()


def load_reviews_for_author(
    aavc_root: str,
    author_id: str,
) -> Tuple[list[str], list[str]]:
    """
    Load *all* reviews for an author.
    We do NOT enforce 6 here – that is only enforced later when
    we compute consistency.
    If a review is empty, we still include an empty string instead of skipping.
    """
    paths = find_author_review_files(aavc_root, author_id)
    texts: list[str] = []
    str_paths: list[str] = []
    for p in paths:
        txt = read_text(p)
        texts.append(txt)
        str_paths.append(str(p))
    return texts, str_paths


def ensure_six_reviews(
    texts: list[str],
    paths: list[str],
) -> Tuple[list[str], list[str], int]:
    """
    For the consistency step we want exactly 6 reviews per author.

    - If len(texts) == 6: keep as is.
    - If > 6: take the first 6 (sorted).
    - If 0 < len(texts) < 6: repeat the last review until 6.
    - If 0: create six empty strings.

    Returns (texts6, paths6, original_num_reviews).
    """
    original_n = len(texts)

    if original_n == 0:
        texts = ["" for _ in range(6)]
        paths = ["" for _ in range(6)]
    elif original_n < 6:
        # repeat last
        last_t = texts[-1]
        last_p = paths[-1]
        while len(texts) < 6:
            texts.append(last_t)
            paths.append(last_p)
    elif original_n > 6:
        texts = texts[:6]
        paths = paths[:6]

    return texts, paths, original_n


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)