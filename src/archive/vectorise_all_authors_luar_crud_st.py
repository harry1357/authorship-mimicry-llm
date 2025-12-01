#!/usr/bin/env python
"""
vectorise_all_authors_luar_crud_st.py

Corpus layout:

  amazon_product_data_corpus_mixed_topics_per_author_reformatted/
      A1A1BM6N28X9J0/
          A1A1BM6N28X9J0_Automotive.txt
          A1A1BM6N28X9J0_Beauty.txt
          A1A1BM6N28X9J0_Electronics.txt
          ...

I.e. one subfolder per author_id, containing that author's review .txt files.

Goal:
  For each author in the 2k-author list, if there is a folder with that author_id
  and at least 6 review files, embed the FIRST 6 reviews using:

      gabrielloiseau/LUAR-CRUD-sentence-transformers

and save:

  data/style_embeddings_luar_crud_st/<AUTHORID>.npz

with keys:
  - author_id   (scalar string)
  - embeddings  (shape [6, dim], float32)
  - num_reviews (int, 6)
  - files       (list of 6 file paths)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_author_ids(author_ids_file: str) -> list[str]:
    """
    Load author IDs from the 2k-author text file.

    File format (space-separated), e.g.:

        author_id training11 training12 training13 generation1 training21 training22 training23 generation2
        A100UD67AHFODS HealthandPersonalCare ...

    We only care about the first column.
    """
    df = pd.read_csv(
        author_ids_file,
        sep=r"\s+",
        engine="python",
        usecols=[0],   # first column only
        header=0,      # header line present
        names=["author_id"],
    )
    ids = df["author_id"].astype(str).drop_duplicates().tolist()
    return ids


def find_author_review_files(aavc_root: str, author_id: str) -> list[Path]:
    """
    Find review files for a given author assuming structure:

      AAVC_ROOT/<author_id>/<author_id>_*.txt
    """
    root = Path(aavc_root)
    author_dir = root / author_id
    if not author_dir.exists() or not author_dir.is_dir():
        return []

    # e.g. A1A1BM6N28X9J0_Automotive.txt, A1A1BM6N28X9J0_Beauty.txt, ...
    paths = sorted(author_dir.glob(f"{author_id}_*.txt"))
    return paths


def load_first_six_reviews(aavc_root: str, author_id: str) -> tuple[list[str], list[str]]:
    """
    Load the first 6 reviews for an author (if available).

    Returns:
      texts: list of 6 review strings
      files: list of 6 file paths (as strings)

    If fewer than 6 usable reviews are found, returns ([], []).
    """
    paths = find_author_review_files(aavc_root, author_id)
    if len(paths) < 6:
        return [], []

    paths = paths[:6]
    texts: list[str] = []
    files: list[str] = []

    for p in paths:
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            raw = p.read_text(encoding="latin-1", errors="ignore")

        raw = raw.strip()
        if not raw:
            # Enforce 6 non-empty reviews
            return [], []

        texts.append(raw)
        files.append(str(p))

    if len(texts) != 6:
        return [], []

    return texts, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aavc_root",
        type=str,
        required=True,
        help="Path to 'amazon_product_data_corpus_mixed_topics_per_author_reformatted'.",
    )
    parser.add_argument(
        "--author_ids_file",
        type=str,
        required=True,
        help="Path to 'author_ids_three_training_topics_x_two_two_generation_topics.txt'.",
    )
    parser.add_argument(
        "--emb_dir",
        type=str,
        default="data/style_embeddings_luar_crud_st",
        help="Output directory for NPZ embeddings.",
    )
    parser.add_argument(
        "--max_authors",
        type=int,
        default=None,
        help="Optional: limit to first N authors (for quick tests).",
    )
    args = parser.parse_args()

    aavc_root = args.aavc_root
    author_ids_file = args.author_ids_file
    emb_dir = Path(args.emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    author_ids = load_author_ids(author_ids_file)
    if args.max_authors is not None:
        author_ids = author_ids[: args.max_authors]

    print(f"AAVC root:       {aavc_root}")
    print(f"Author IDs file: {author_ids_file}")
    print(f"Num authors in list: {len(author_ids)}")
    print(f"Output dir:      {emb_dir}")

    model_name = "gabrielloiseau/LUAR-CRUD-sentence-transformers"
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    written = 0
    skipped_no_reviews = 0
    skipped_not_enough = 0

    for author_id in tqdm(author_ids, desc="Authors"):
        out_path = emb_dir / f"{author_id}.npz"
        if out_path.exists():
            written += 1
            continue

        texts, files = load_first_six_reviews(aavc_root, author_id)

        if len(files) == 0:
            # Either no folder or fewer than 6 usable reviews
            all_files = find_author_review_files(aavc_root, author_id)
            if len(all_files) == 0:
                skipped_no_reviews += 1
            else:
                skipped_not_enough += 1
            continue

        embeddings = model.encode(
            texts,
            batch_size=6,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        np.savez_compressed(
            out_path,
            author_id=author_id,
            embeddings=embeddings,
            num_reviews=6,
            files=np.array(files, dtype=object),
        )
        written += 1

    print("Done.")
    print(f"Wrote embeddings for {written} authors.")
    print(f"Skipped {skipped_no_reviews} authors (no review folder/files found).")
    print(f"Skipped {skipped_not_enough} authors (had some reviews but < 6 usable).")


if __name__ == "__main__":
    main()