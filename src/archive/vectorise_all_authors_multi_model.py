#!/usr/bin/env python
"""
vectorise_all_authors_multi_model.py

Goal:
  For ALL authors in the 2k-author list, compute style embeddings
  using one of several models:

    - style_embedding  -> AnnaWegmann/Style-Embedding
    - luar_crud_st     -> gabrielloiseau/LUAR-CRUD-sentence-transformers
    - luar_crud_raw    -> rrivera1849/LUAR-CRUD (episode_length = 1 per review)
    - star             -> AIDA-UPM/star

For each author:
  - Read the author/topic file
      author_ids_three_training_topics_x_two_two_generation_topics.txt
    and get the author_id column.
  - Find all their review .txt files in the AAVC folder (recursively).
  - Require exactly 6 reviews per author:
      * if < 6 reviews: skip this author
      * if >= 6 reviews: take the first 6 by filename
  - Encode those 6 reviews to a matrix [6, dim]
  - Save as data/<output_subdir>/<AUTHOR_ID>.npz
      keys: author_id, texts, files, embeddings
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------- Backends -------------------------------------------------------


class SentenceTransformersBackend:
    """
    Wrapper for sentence-transformers style models:
      - AnnaWegmann/Style-Embedding
      - gabrielloiseau/LUAR-CRUD-sentence-transformers
    """

    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings


class LuarRawBackend:
    """
    Wrapper for rrivera1849/LUAR-CRUD using its official episode interface.

    We treat each review as an "episode" of length 1, so for an author with 6
    reviews we get a [6, 512] embedding matrix (one style vector per review).
    """

    def __init__(self, model_name: str = "rrivera1849/LUAR-CRUD", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 4,
        max_length: int = 256,
    ) -> np.ndarray:
        """
        texts: list of review strings for a SINGLE author.
        Returns: [num_reviews, 512] numpy array.
        """
        all_embeds = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].unsqueeze(1).to(self.device)       # [B,1,L]
            attention_mask = encoded["attention_mask"].unsqueeze(1).to(self.device)

            with self.torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # LUAR returns [batch_size, 512]
                embeds = out.detach().cpu().numpy()
            all_embeds.append(embeds)

        return np.vstack(all_embeds)


class StarBackend:
    """
    Wrapper for AIDA-UPM/star.

    The model card indicates we should use pooler_output as style embeddings.
    """

    def __init__(self, model_name: str = "AIDA-UPM/star", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 4,
        max_length: int = 512,
    ) -> np.ndarray:
        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with self.torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.pooler_output  # [batch, dim]
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)


def make_backend(model_key: str):
    """
    Map a short model key to a backend instance.
    """
    key = model_key.lower()
    if key == "style_embedding":
        return SentenceTransformersBackend("AnnaWegmann/Style-Embedding")
    elif key == "luar_crud_st":
        return SentenceTransformersBackend("gabrielloiseau/LUAR-CRUD-sentence-transformers")
    elif key == "luar_crud_raw":
        return LuarRawBackend("rrivera1849/LUAR-CRUD")
    elif key == "star":
        return StarBackend("AIDA-UPM/star")
    else:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Expected one of: style_embedding, luar_crud_st, luar_crud_raw, star."
        )


# ---------- Author list helpers -------------------------------------------


def load_all_author_ids(author_ids_file: Path) -> list[str]:
    """
    Load author IDs from the 2k-author text file.

    File format (space-separated):
      author_id training11 training12 training13 generation1 ...
      A100UD67AHFODS HealthandPersonalCare ...

    We only care about the first column (author_id).
    """
    df = pd.read_csv(author_ids_file, sep=r"\s+", engine="python")
    if "author_id" not in df.columns:
        # In case pandas didn't infer the header for some reason
        df.columns = [
            "author_id",
            "training11", "training12", "training13",
            "generation1",
            "training21", "training22", "training23",
            "generation2",
        ]
    ids = df["author_id"].astype(str).drop_duplicates().tolist()
    return ids


# ---------- AAVC helpers ---------------------------------------------------


def find_author_files(aavc_root: Path, author_id: str) -> List[Path]:
    """
    Recursively search for review files for a given author.

    We assume filenames start with '<AUTHOR_ID>_' and end with '.txt'
    (case-insensitive), e.g. A1A1BM6N28X9J0_Automotive.txt, possibly
    inside subfolders.
    """
    matches: List[Path] = []
    prefix = f"{author_id}_"
    for root, _, files in os.walk(aavc_root):
        for fname in files:
            lower = fname.lower()
            if lower.endswith(".txt") and fname.startswith(prefix):
                matches.append(Path(root) / fname)
    return sorted(matches)


def load_author_texts(
    aavc_root: Path,
    author_id: str,
    n_docs: int = 6,
) -> Tuple[List[str], List[str]]:
    """
    Load exactly n_docs reviews for a given author.

    - If the author has fewer than n_docs reviews: return empty lists (skip).
    - If the author has >= n_docs reviews: take the first n_docs sorted by filename.
    """
    paths = find_author_files(aavc_root, author_id)
    if len(paths) < n_docs:
        return [], []

    paths = paths[:n_docs]
    texts: List[str] = []
    files: List[str] = []
    for path in paths:
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            continue
        texts.append(raw)
        files.append(str(path))

    # if any were empty => not enough usable docs
    if len(texts) != n_docs:
        return [], []

    return texts, files


# ---------- Main -----------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aavc_root",
        type=str,
        required=True,
        help="Path to 'amazon_product_data_corpus_mixed_topics_per_author_reformatted'",
    )
    parser.add_argument(
        "--author_ids_file",
        type=str,
        required=True,
        help="TXT file listing all author IDs and topics, "
        "e.g. author_ids_three_training_topics_x_two_two_generation_topics.txt",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["style_embedding", "luar_crud_st", "luar_crud_raw", "star"],
        help="Which embedding model to use.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        required=True,
        help="Subdirectory under data/ to store .npz files, e.g. 'style_embeddings_anna'.",
    )
    parser.add_argument(
        "--max_authors",
        type=int,
        default=None,
        help="Optional: limit to first N authors (for quick tests).",
    )
    args = parser.parse_args()

    aavc_root = Path(args.aavc_root).expanduser().resolve()
    if not aavc_root.exists():
        raise FileNotFoundError(f"AAVC root not found: {aavc_root}")

    author_ids_path = Path(args.author_ids_file).expanduser().resolve()
    if not author_ids_path.exists():
        raise FileNotFoundError(f"Author IDs file not found: {author_ids_path}")

    author_ids = load_all_author_ids(author_ids_path)
    if args.max_authors is not None:
        author_ids = author_ids[: args.max_authors]

    out_dir = Path("data") / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"AAVC root:       {aavc_root}")
    print(f"Author IDs file: {author_ids_path}")
    print(f"Num authors:     {len(author_ids)}")
    print(f"Model key:       {args.model}")
    print(f"Output dir:      {out_dir}")

    backend = make_backend(args.model)

    skipped = 0
    written = 0

    for author_id in tqdm(author_ids, desc="Authors"):
        texts, files = load_author_texts(aavc_root, author_id, n_docs=6)
        if len(texts) == 0:
            skipped += 1
            continue

        embeddings = backend.encode(texts)

        if embeddings.shape[0] != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch for {author_id}: "
                f"{embeddings.shape[0]} embeddings vs {len(texts)} texts."
            )

        out_path = out_dir / f"{author_id}.npz"
        np.savez_compressed(
            out_path,
            author_id=author_id,
            texts=np.array(texts, dtype=object),
            files=np.array(files, dtype=object),
            embeddings=embeddings,
        )
        written += 1

    print(f"Done. Wrote embeddings for {written} authors; skipped {skipped} authors.")


if __name__ == "__main__":
    main()