# src/embed_all_models.py

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# Map short keys to HF model IDs
MODEL_MAP = {
    "luar_crud_orig": "rrivera1849/LUAR-CRUD",
    "luar_mud_orig": "rrivera1849/LUAR-MUD",
    "luar_crud_st": "gabrielloiseau/LUAR-CRUD-sentence-transformers",
    "luar_mud_st": "gabrielloiseau/LUAR-MUD-sentence-transformers",
    "style_embedding": "AnnaWegmann/Style-Embedding",
    "star": "AIDA-UPM/star",
}


def load_author_ids(author_ids_file: str) -> List[str]:
    """Read first column (author_id) from the txt list, skipping header."""
    df = pd.read_csv(
        author_ids_file,
        sep=r"\s+",
        engine="python",
        usecols=[0],
        header=0,
        names=["author_id"],
    )
    ids = df["author_id"].astype(str).drop_duplicates().tolist()
    return ids


def find_author_review_files(aavc_root: Path, author_id: str) -> List[Path]:
    """Return all review files for an author: <root>/<author_id>/<author_id>_*.txt"""
    author_dir = aavc_root / author_id
    if not author_dir.exists() or not author_dir.is_dir():
        return []
    paths = sorted(author_dir.glob(f"{author_id}_*.txt"))
    return paths


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def embed_with_sentence_transformer(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 8,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def embed_with_autotransformer(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 8,
) -> np.ndarray:
    """Generic transformer embedding: mean-pool last_hidden_state."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    all_embeddings = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tok = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**tok)
            if hasattr(outputs, "last_hidden_state"):
                token_emb = outputs.last_hidden_state  # (B, L, H)
            else:
                token_emb = outputs[0]

            sent_emb = token_emb.mean(dim=1)  # (B, H)
            all_embeddings.append(sent_emb.cpu().numpy())

    emb = np.vstack(all_embeddings).astype(np.float32)
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aavc_root", type=str, required=True)
    parser.add_argument("--author_ids_file", type=str, required=True)
    parser.add_argument(
        "--model_key",
        type=str,
        choices=list(MODEL_MAP.keys()),
        required=True,
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/embeddings",
        help="Base folder for embeddings; per-model subfolders are created inside.",
    )
    parser.add_argument(
        "--max_authors",
        type=int,
        default=None,
        help="Optional limit for quick tests.",
    )
    args = parser.parse_args()

    aavc_root = Path(args.aavc_root)
    author_ids_file = Path(args.author_ids_file)
    model_key = args.model_key
    model_name = MODEL_MAP[model_key]

    out_dir = Path(args.out_root) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    author_ids = load_author_ids(str(author_ids_file))
    if args.max_authors is not None:
        author_ids = author_ids[: args.max_authors]

    print(f"AAVC root:       {aavc_root}")
    print(f"Author IDs file: {author_ids_file}")
    print(f"Num authors:     {len(author_ids)}")
    print(f"Model key:       {model_key}")
    print(f"HF model:        {model_name}")
    print(f"Output dir:      {out_dir}")

    # Decide how to load the model
    st_keys = {"luar_crud_st", "luar_mud_st", "style_embedding"}
    use_sentence_transformer = model_key in st_keys

    if use_sentence_transformer:
        print("Loading as SentenceTransformer...")
        model_st = SentenceTransformer(model_name, trust_remote_code=True)
        model_auto = None
        tokenizer = None
    else:
        print("Loading with AutoModel + AutoTokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        model_auto = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )
        model_st = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    written = 0
    skipped_no_files = 0

    for author_id in tqdm(author_ids, desc="Authors"):
        out_path = out_dir / f"{author_id}.npz"
        if out_path.exists():
            written += 1
            continue

        files = find_author_review_files(aavc_root, author_id)
        if not files:
            skipped_no_files += 1
            # still continue; we promised not to crash, but no embeddings to write
            continue

        texts = [read_text(p).strip() for p in files]
        texts = [t for t in texts if t]

        if not texts:
            skipped_no_files += 1
            continue

        if use_sentence_transformer:
            emb = embed_with_sentence_transformer(model_st, texts)
        else:
            emb = embed_with_autotransformer(
                model_auto,
                tokenizer,
                texts,
                device=device,
            )

        np.savez_compressed(
            out_path,
            author_id=author_id,
            embeddings=emb,
            files=np.array([str(p) for p in files], dtype=object),
        )
        written += 1

    print("Embedding finished.")
    print(f"Wrote embeddings for {written} authors.")
    print(f"Authors with no usable files: {skipped_no_files}")


if __name__ == "__main__":
    main()