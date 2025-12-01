# src/embed_authors.py
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .model_configs import (
    CORPUS_DIR,
    AUTHOR_LIST_FILE,
    EMBEDDINGS_DIR,
    MODEL_CONFIGS,
)

# Lazy imports for optional libraries
_sentence_transformers = None
_transformers = None


def load_author_ids(author_list_path: Path):
    """Read author IDs from the given file (skip header)."""
    author_ids = []
    with author_list_path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                # header line
                first = False
                continue
            parts = line.split()
            if len(parts) == 0:
                continue
            author_ids.append(parts[0])
    return author_ids


def load_reviews_for_author(author_id: str):
    """Load all non-empty .txt reviews for a given author."""
    author_dir = CORPUS_DIR / author_id
    if not author_dir.is_dir():
        return [], []

    txt_files = sorted(author_dir.glob("*.txt"))
    texts = []
    files = []
    for path in txt_files:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            content = ""
        if content:
            texts.append(content)
            # Store relative path from corpus root for reproducibility
            files.append(str(path.relative_to(CORPUS_DIR)))
    return texts, files


def get_sentence_transformer(model_name: str):
    global _sentence_transformers
    if _sentence_transformers is None:
        from sentence_transformers import SentenceTransformer

        _sentence_transformers = SentenceTransformer  # type: ignore
    return _sentence_transformers(model_name)


def get_transformers():
    global _transformers
    if _transformers is None:
        from transformers import AutoTokenizer, AutoModel

        _transformers = (AutoTokenizer, AutoModel)
    return _transformers


def embed_sentence_transformers(model, texts, batch_size):
    """Embed texts using a SentenceTransformer-style model."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def embed_luar_orig(model, tokenizer, texts, max_length, batch_size, device):
    """
    Embed texts using original LUAR models (episode-based).
    We treat each review as its own episode of length 1, following the
    official usage pattern but with episode_length=1. [oai_citation:4‡Hugging Face](https://huggingface.co/rrivera1849/LUAR-CRUD/blame/353504a9269dc58c38f380e4d9b2d03a86444169/README.md)
    """
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tokenized = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].unsqueeze(1).to(device)  # (B,1,L)
            attention_mask = tokenized["attention_mask"].unsqueeze(1).to(device)  # (B,1,L)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            # LUAR custom code returns a tensor of shape (batch_size, 512) [oai_citation:5‡Hugging Face](https://huggingface.co/rrivera1849/LUAR-CRUD/blame/353504a9269dc58c38f380e4d9b2d03a86444169/README.md)
            if isinstance(out, torch.Tensor):
                emb = out
            else:
                # Fallback in case custom code wraps output
                emb = out[0]
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def embed_star(model, tokenizer, texts, max_length, batch_size, device):
    """
    Embed texts using the STAR model, taking pooler_output as style embeddings. [oai_citation:6‡Hugging Face](https://huggingface.co/AIDA-UPM/star?utm_source=chatgpt.com)
    """
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            # Use pooled output as style embedding
            emb = out.pooler_output  # (B, hidden_dim)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def embed_for_model(model_key: str, overwrite: bool = False):
    """Embed all authors for a single model_key."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key: {model_key}")

    cfg = MODEL_CONFIGS[model_key]
    family = cfg["family"]
    model_name = cfg["hf_name"]
    batch_size = cfg.get("batch_size", 16)
    max_length = cfg.get("max_length", 256)
    tokenizer_name = cfg.get("tokenizer_name", None)

    # Device selection: prefer MPS (Apple Silicon) > CUDA > CPU
    # NOTE: Original LUAR models have MPS compatibility issues, force CPU for them
    if family == "luar_orig":
        # LUAR original models use custom operations (einops repeat) that fail on MPS
        device = "cpu"
        print("Note: Using CPU for original LUAR models (MPS incompatibility)")
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    out_dir = EMBEDDINGS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    author_ids = load_author_ids(AUTHOR_LIST_FILE)

    print(f"Embedding {len(author_ids)} authors for model '{model_key}' on {device}.")
    
    # Load model ONCE before processing all authors
    model = None
    tokenizer = None
    
    if family == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        print(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name, device=device)
    elif family == "luar_orig":
        AutoTokenizer, AutoModel = get_transformers()
        print(f"Loading LUAR original model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(device)
        model.eval()
    elif family == "star":
        AutoTokenizer, AutoModel = get_transformers()
        print(f"Loading STAR model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
    else:
        raise ValueError(f"Unsupported family type: {family}")

    for author_id in tqdm(author_ids, desc=f"{model_key}"):
        out_path = out_dir / f"{author_id}.npz"
        if out_path.exists() and not overwrite:
            continue

        texts, files = load_reviews_for_author(author_id)
        if len(texts) == 0:
            # Only authors with literally no non-empty files are skipped
            continue

        if family == "sentence_transformers":
            embeddings = embed_sentence_transformers(
                model, texts, batch_size=batch_size
            )
        elif family == "luar_orig":
            embeddings = embed_luar_orig(
                model,
                tokenizer,
                texts,
                max_length=max_length,
                batch_size=batch_size,
                device=device,
            )
        elif family == "star":
            embeddings = embed_star(
                model,
                tokenizer,
                texts,
                max_length=max_length,
                batch_size=batch_size,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported family type: {family}")

        np.savez_compressed(
            out_path,
            author_id=author_id,
            model_key=model_key,
            embeddings=embeddings,
            files=np.array(files),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Embed all authors for all style models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Which model_keys to run; 'all' runs every model in MODEL_CONFIGS.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute embeddings even if .npz already exists.",
    )
    args = parser.parse_args()

    if "all" in args.models:
        model_keys = list(MODEL_CONFIGS.keys())
    else:
        model_keys = args.models

    torch.manual_seed(42)
    np.random.seed(42)

    for mk in model_keys:
        embed_for_model(mk, overwrite=args.overwrite)


if __name__ == "__main__":
    main()