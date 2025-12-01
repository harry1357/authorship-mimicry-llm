#!/usr/bin/env python
"""
vectorise_all_authors_style_models.py

Vectorise ALL authors (exactly 6 reviews each) for different style models:

  - luar_crud_orig      -> rrivera1849/LUAR-CRUD          (LUAR backend)
  - luar_mud_orig       -> rrivera1849/LUAR-MUD           (LUAR backend)
  - luar_crud_st        -> gabrielloiseau/LUAR-CRUD-sentence-transformers
  - luar_mud_st         -> gabrielloiseau/LUAR-MUD-sentence-transformers
  - style_embedding     -> AnnaWegmann/Style-Embedding
  - star                -> AIDA-UPM/star

Corpus layout:

  amazon_product_data_corpus_mixed_topics_per_author_reformatted/
      <author_id>/
          <author_id>_Automotive.txt
          <author_id>_Beauty.txt
          ...

For each (model_key, author_id):

  - load exactly 6 non-empty reviews (if not exactly 6, skip author)
  - embed with the chosen model
  - save NPZ: data/style_embeddings_<model_key>/<author_id>.npz

Keys in NPZ:
  - author_id   (scalar string)
  - embeddings  (6, dim) float32
  - num_reviews (int, 6)
  - files       (6,) object (paths as strings)
  - model_key   (scalar string)
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ----------------------------------------------------------------------
# Model mapping
# ----------------------------------------------------------------------

MODEL_MAP = {
    "luar_crud_orig": "rrivera1849/LUAR-CRUD",
    "luar_mud_orig": "rrivera1849/LUAR-MUD",
    "luar_crud_st": "gabrielloiseau/LUAR-CRUD-sentence-transformers",
    "luar_mud_st": "gabrielloiseau/LUAR-MUD-sentence-transformers",
    "style_embedding": "AnnaWegmann/Style-Embedding",
    "star": "AIDA-UPM/star",
}


# ----------------------------------------------------------------------
# Helpers to read author list and reviews
# ----------------------------------------------------------------------

def load_author_ids(author_ids_file: str) -> list[str]:
    """
    Read the full 9-column author table and return a list of author_ids.

    File format (whitespace-separated, header on first line):
      author_id training11 training12 training13 generation1 training21 training22 training23 generation2
    We only need the first column.
    """
    df = pd.read_csv(
        author_ids_file,
        sep=r"\s+",
        engine="python",
        usecols=[0],
        header=0,
        names=["author_id"],
    )
    return df["author_id"].astype(str).drop_duplicates().tolist()


def find_author_review_files(aavc_root: str, author_id: str) -> list[Path]:
    """
    AAVC layout in your Dropbox:

      <aavc_root>/<author_id>/<author_id>_*.txt
    """
    root = Path(aavc_root)
    author_dir = root / author_id
    if not author_dir.exists() or not author_dir.is_dir():
        return []
    paths = sorted(author_dir.glob(f"{author_id}_*.txt"))
    return paths


def read_text_file(path: Path) -> str:
    """
    Read a text file as UTF-8, fall back to latin-1 if needed.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="latin-1", errors="ignore")
    return raw.strip()


def load_exact_six_reviews(aavc_root: str, author_id: str):
    """
    Professor's request: use authors with EXACTLY 6 reviews,
    not more, not less.

    - If there are no files: return ([], []).
    - If there are fewer than 6 OR more than 6: skip this author.
    - If exactly 6: read all, ensure non-empty.

    Returns:
      texts (list[str]) length 6
      files (list[str]) length 6
    """
    paths = find_author_review_files(aavc_root, author_id)
    if len(paths) == 0:
        return [], []

    if len(paths) != 6:
        # not exactly 6 -> skip
        return [], []

    texts, files = [], []
    for p in paths:
        raw = read_text_file(p)
        if not raw:
            # any empty review -> skip author entirely
            return [], []
        texts.append(raw)
        files.append(str(p))

    if len(texts) != 6:
        return [], []

    return texts, files


# ----------------------------------------------------------------------
# SentenceTransformer backend for 4 models
# ----------------------------------------------------------------------

def embed_with_sentence_transformer(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Embed each review separately with a SentenceTransformer-compatible model.
    """
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    emb = model.encode(
        texts,
        batch_size=6,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


# ----------------------------------------------------------------------
# LUAR backend for original LUAR-CRUD / LUAR-MUD
# ----------------------------------------------------------------------

def simple_sentence_chunks(text: str, episode_length: int = 16) -> list[str]:
    """
    Very simple segmentation into pseudo-sentences for LUAR:

      - Replace newlines with space
      - Split on '.'
      - Strip and drop empties
      - If fewer than episode_length, pad by repeating last
      - If more, truncate to episode_length
    """
    raw = text.replace("\n", " ").split(".")
    sentences = [s.strip() for s in raw if s.strip()]
    if not sentences:
        sentences = ["dummy text"]

    if len(sentences) >= episode_length:
        return sentences[:episode_length]
    else:
        last = sentences[-1]
        while len(sentences) < episode_length:
            sentences.append(last)
        return sentences


def embed_with_luar_original(
    model,
    tokenizer,
    texts: list[str],
    episode_length: int = 16,
    max_token_length: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """
    Embed 6 reviews using original LUAR models.

    Each review -> one "episode" (episode_length segments).
    We get one embedding per episode (so 6 embeddings total).

    Returns: (6, dim) float32
    """
    if not texts:
        return np.zeros((0, 512), dtype=np.float32)

    # Build episodes: one episode per review
    episodes = [simple_sentence_chunks(t, episode_length=episode_length) for t in texts]
    batch_size = len(episodes)  # should be 6
    episode_len = len(episodes[0])

    # Flatten all segments from all episodes into one list
    flat_segments = [seg for ep in episodes for seg in ep]

    tokenized = tokenizer(
        flat_segments,
        max_length=max_token_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Reshape to (batch_size, episode_len, seq_len)
    input_ids = tokenized["input_ids"].reshape(batch_size, episode_len, -1)
    attention_mask = tokenized["attention_mask"].reshape(batch_size, episode_len, -1)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Try to be robust to different output types
        if isinstance(outputs, torch.Tensor):
            emb = outputs
        elif hasattr(outputs, "logits"):
            emb = outputs.logits
        elif hasattr(outputs, "last_hidden_state"):
            # fallback: mean over tokens and segments
            # outputs.last_hidden_state: (batch, episode_len, seq_len, hidden)
            hidden = outputs.last_hidden_state
            emb = hidden.mean(dim=-2).mean(dim=-2)  # mean over seq_len and segments
        else:
            # Try first element as tensor
            emb = outputs[0]

    emb = emb.detach().cpu().numpy().astype(np.float32)

    # Make sure we end up with (6, dim)
    if emb.ndim == 3:
        # e.g. (batch, episode_len, dim) -> average over segments
        emb = emb.mean(axis=1)
    if emb.ndim == 2 and emb.shape[0] == batch_size:
        return emb
    if emb.ndim == 2 and emb.shape[0] == 1:
        # one vector for all -> repeat 6 times (last resort)
        emb = np.repeat(emb, batch_size, axis=0)
        return emb.astype(np.float32)

    # As a final fallback, reshape to (batch_size, -1)
    emb = emb.reshape(batch_size, -1)
    return emb.astype(np.float32)


def embed_with_star(model, tokenizer, texts: list[str], max_length: int = 512) -> np.ndarray:
    """
    Embed texts using AIDA-UPM/star (RoBERTa-based) via pooler_output.
    """
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)  # 1024 for roberta-large

    tokenized = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        tokenized = {k: v.cuda() for k, v in tokenized.items()}
        model.cuda()

    with torch.no_grad():
        outputs = model(**tokenized)
        # Many RoBERTa models expose pooler_output
        if hasattr(outputs, "pooler_output"):
            emb = outputs.pooler_output
        else:
            # fallback: CLS token
            emb = outputs.last_hidden_state[:, 0, :]

    emb = emb.detach().cpu().numpy().astype(np.float32)
    return emb


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aavc_root", type=str, required=True)
    parser.add_argument("--author_ids_file", type=str, required=True)
    parser.add_argument(
        "--model_key",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
        help="Which style model to use.",
    )
    parser.add_argument(
        "--emb_dir",
        type=str,
        default=None,
        help="Output dir for embeddings; default is data/style_embeddings_<model_key>",
    )
    parser.add_argument(
        "--max_authors",
        type=int,
        default=None,
        help="Optional: limit to first N authors for quick tests.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for LUAR backend (cpu/cuda). If not set, auto-detect.",
    )
    args = parser.parse_args()

    aavc_root = args.aavc_root
    author_ids_file = args.author_ids_file
    model_key = args.model_key

    if args.emb_dir is None:
        emb_dir = Path(f"data/style_embeddings_{model_key}")
    else:
        emb_dir = Path(args.emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    author_ids = load_author_ids(author_ids_file)
    if args.max_authors is not None:
        author_ids = author_ids[: args.max_authors]

    print(f"AAVC root:       {aavc_root}")
    print(f"Author IDs file: {author_ids_file}")
    print(f"Num authors:     {len(author_ids)}")
    print(f"Model key:       {model_key}")
    print(f"HF model:        {MODEL_MAP[model_key]}")
    print(f"Output dir:      {emb_dir}")

    model_name = MODEL_MAP[model_key]
    is_luar_original = model_key in ["luar_crud_orig", "luar_mud_orig"]
    is_star = model_key == "star"

    # Device for LUAR
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    if is_luar_original:
        print(f"Loading original LUAR model with AutoModel on device={device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(device)
        model.eval()
    elif is_star:
        print("Loading STAR model with AutoModel (RoBERTa-based)...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        model = AutoModel.from_pretrained(model_name)
        model.eval()
    else:
        print("Loading SentenceTransformer model...")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        tokenizer = None

    written = 0
    skipped_no_reviews = 0
    skipped_not_exact_six = 0

    for author_id in tqdm(author_ids, desc="Authors"):
        out_path = emb_dir / f"{author_id}.npz"
        if out_path.exists():
            written += 1
            continue

        texts, files = load_exact_six_reviews(aavc_root, author_id)
        if not files:
            all_files = find_author_review_files(aavc_root, author_id)
            if len(all_files) == 0:
                skipped_no_reviews += 1
            else:
                skipped_not_exact_six += 1
            continue

        if is_luar_original:
            emb = embed_with_luar_original(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                episode_length=16,
                max_token_length=32,
                device=device,
            )
        elif is_star:
            emb = embed_with_star(model, tokenizer, texts)
        else:
            emb = embed_with_sentence_transformer(model, texts)

        # Safety check: we expect (6, dim)
        if emb.ndim != 2 or emb.shape[0] != 6:
            skipped_not_exact_six += 1
            continue

        np.savez_compressed(
            out_path,
            author_id=author_id,
            embeddings=emb.astype(np.float32),
            num_reviews=6,
            files=np.array(files, dtype=object),
            model_key=model_key,
        )
        written += 1

    print("Done.")
    print(f"Wrote embeddings for {written} authors.")
    print(f"Skipped {skipped_no_reviews} authors (no review folder/files).")
    print(f"Skipped {skipped_not_exact_six} authors (not exactly 6 usable reviews or embedding shape issues).")


if __name__ == "__main__":
    main()