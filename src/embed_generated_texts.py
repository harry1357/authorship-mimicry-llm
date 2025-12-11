# src/embed_generated_texts.py
"""
Embedding Module for Generated Review Texts

This module generates style embeddings for LLM-generated review texts that have been
normalized. It processes generated texts using the same embedding models employed for
real author texts, enabling direct comparison in the embedding space.

The module supports all style embedding models defined in model_configs.py, including:
- LUAR (CRUD and MUD variants, both original and sentence-transformers)
- Style-Embedding
- STAR (Style Transformer for Authorship Representations)

Input:
    Normalized generated texts from: data/generated/<llm-key>/normalized/texts_<variant>_fullrun<N>/

Output:
    Per-author embedding files (.npz) saved to:
    data/embeddings/generated/<model-key>/<llm-key>/<variant>/fullrun<N>/<AUTHOR_ID>.npz

Usage:
    python src/embed_generated_texts.py --model-key style_embedding --full-run 1 --prompt-variant simple
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from generation_config import GENERATED_DIR, EMBEDDINGS_DIR, STYLE_MODEL_KEYS
from model_configs import MODEL_CONFIGS


# Lazy imports for optional libraries
_sentence_transformers = None
_transformers = None


def get_sentence_transformer(model_name: str, device: str):
    """Lazy load SentenceTransformer model."""
    global _sentence_transformers
    if _sentence_transformers is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformers = SentenceTransformer
    return _sentence_transformers(model_name, device=device)


def get_transformers():
    """Lazy load transformers library components."""
    global _transformers
    if _transformers is None:
        from transformers import AutoTokenizer, AutoModel
        _transformers = (AutoTokenizer, AutoModel)
    return _transformers


def embed_sentence_transformers(model, texts: List[str], batch_size: int) -> np.ndarray:
    """
    Embed texts using a SentenceTransformer-style model.
    
    Args:
        model: Loaded SentenceTransformer model
        texts: List of text strings to embed
        batch_size: Batch size for encoding
        
    Returns:
        2D numpy array of embeddings (n_texts × embedding_dim)
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def embed_luar_orig(
    model, 
    tokenizer, 
    texts: List[str], 
    max_length: int, 
    batch_size: int, 
    device: str
) -> np.ndarray:
    """
    Embed texts using original LUAR models with episode-based approach.
    
    Each text is treated as its own episode of length 1. The input tensors
    need to have shape (batch_size, 1, max_length) to represent single-episode batches.
    
    Args:
        model: Loaded LUAR model
        tokenizer: Associated tokenizer
        texts: List of text strings to embed
        max_length: Maximum sequence length for tokenization
        batch_size: Number of texts to process at once
        device: Device to run inference on
        
    Returns:
        2D numpy array of embeddings (n_texts × 512)
    """
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize the batch
            tokenized = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            # Add episode dimension: (B, L) -> (B, 1, L)
            input_ids = tokenized["input_ids"].unsqueeze(1).to(device)
            attention_mask = tokenized["attention_mask"].unsqueeze(1).to(device)
            
            # Forward pass through LUAR model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # LUAR returns embeddings of shape (batch_size, 512)
            if isinstance(outputs, torch.Tensor):
                batch_embeddings = outputs
            else:
                # Fallback in case custom code wraps output
                batch_embeddings = outputs[0]
            
            all_embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def embed_star(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: str
) -> np.ndarray:
    """
    Embed texts using STAR (Style Transformer for Authorship Representations).
    
    Uses the pooler_output from the model for authorship representation.
    
    Args:
        model: Loaded STAR model
        tokenizer: Associated tokenizer
        texts: List of text strings to embed
        max_length: Maximum sequence length for tokenization
        batch_size: Number of texts to process at once
        device: Device to run inference on
        
    Returns:
        2D numpy array of embeddings (n_texts × hidden_size)
    """
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize and move entire encoded dict to device
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            
            # Forward pass
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            
            # Use pooler_output for STAR model
            batch_embeddings = outputs.pooler_output.cpu().numpy()
            all_embeddings.append(batch_embeddings)
    
    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def load_generated_texts_for_author(
    author_dir: Path,
) -> Tuple[List[str], List[str]]:
    """
    Load all normalized generated texts for a single author.
    
    Args:
        author_dir: Path to the author's directory containing .txt files
        
    Returns:
        Tuple of (texts, file_paths) where file_paths are relative to the author directory
    """
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
            files.append(path.name)  # Store just the filename
    
    return texts, files


def embed_generated_for_model(
    model_key: str,
    llm_key: str,
    full_run: int,
    prompt_variant: str,
    overwrite: bool = False,
) -> None:
    """
    Generate embeddings for all generated texts using a specific model.
    
    Args:
        model_key: Style embedding model identifier (e.g., 'style_embedding')
        llm_key: LLM identifier (e.g., 'gpt-5.1')
        full_run: Experimental run number (1 or 2)
        prompt_variant: Prompt type ('simple' or 'complex')
        overwrite: Whether to recompute existing embeddings
        
    Raises:
        ValueError: If model_key is not recognized
        FileNotFoundError: If input directory does not exist
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key: {model_key}")
    
    cfg = MODEL_CONFIGS[model_key]
    family = cfg["family"]
    model_name = cfg["hf_name"]
    batch_size = cfg.get("batch_size", 16)
    max_length = cfg.get("max_length", 256)
    tokenizer_name = cfg.get("tokenizer_name", None)
    
    # Determine input directory
    variant_suffix = f"texts_{prompt_variant}_fullrun{full_run}"
    input_root = GENERATED_DIR / llm_key / "normalized" / variant_suffix
    
    if not input_root.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_root}\n"
            f"Please run normalization first: "
            f"python src/normalize_generated_texts.py --llm-key {llm_key} "
            f"--full-run {full_run} --prompt-variant {prompt_variant}"
        )
    
    # Determine output directory
    output_root = (
        EMBEDDINGS_DIR / "generated" / model_key / llm_key / 
        prompt_variant / f"fullrun{full_run}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Device selection
    if family == "luar_orig":
        device = "cpu"
        print(f"[embed_generated] Using CPU for {model_key} (MPS incompatibility)")
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"[embed_generated] Model: {model_key}, LLM: {llm_key}, "
          f"Run: {full_run}, Variant: {prompt_variant}")
    print(f"[embed_generated] Input: {input_root}")
    print(f"[embed_generated] Output: {output_root}")
    print(f"[embed_generated] Device: {device}")
    
    # Get list of author directories
    author_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    
    if not author_dirs:
        print(f"[embed_generated] WARNING: No author directories found in {input_root}")
        return
    
    print(f"[embed_generated] Processing {len(author_dirs)} authors")
    
    # Load model ONCE before processing all authors
    model = None
    tokenizer = None
    
    if family == "sentence_transformers":
        print(f"[embed_generated] Loading SentenceTransformer: {model_name}")
        model = get_sentence_transformer(model_name, device)
    elif family == "luar_orig":
        AutoTokenizer, AutoModel = get_transformers()
        print(f"[embed_generated] Loading LUAR original: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(device)
        model.eval()
    elif family == "star":
        AutoTokenizer, AutoModel = get_transformers()
        print(f"[embed_generated] Loading STAR: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
    else:
        raise ValueError(f"Unsupported family type: {family}")
    
    # Process each author
    skipped = 0
    processed = 0
    
    for author_dir in tqdm(author_dirs, desc=f"Embedding {model_key}"):
        author_id = author_dir.name
        output_path = output_root / f"{author_id}.npz"
        
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        
        texts, files = load_generated_texts_for_author(author_dir)
        
        if len(texts) == 0:
            print(f"[embed_generated] WARNING: No texts found for {author_id}")
            continue
        
        # Generate embeddings based on model family
        if family == "sentence_transformers":
            embeddings = embed_sentence_transformers(model, texts, batch_size)
        elif family == "luar_orig":
            embeddings = embed_luar_orig(
                model, tokenizer, texts, max_length, batch_size, device
            )
        elif family == "star":
            embeddings = embed_star(
                model, tokenizer, texts, max_length, batch_size, device
            )
        else:
            raise ValueError(f"Unsupported family type: {family}")
        
        # Save embeddings with metadata
        np.savez_compressed(
            output_path,
            author_id=author_id,
            llm_key=llm_key,
            model_key=model_key,
            prompt_variant=prompt_variant,
            full_run=full_run,
            files=np.array(files),
            embeddings=embeddings,
        )
        processed += 1
    
    print(f"[embed_generated] Complete: {processed} authors processed, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for LLM-generated texts"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        required=True,
        choices=STYLE_MODEL_KEYS,
        help="Which style embedding model to use",
    )
    parser.add_argument(
        "--llm-key",
        type=str,
        default="gpt-5.1",
        help="LLM identifier (default: gpt-5.1)",
    )
    parser.add_argument(
        "--full-run",
        type=int,
        required=True,
        choices=[1, 2],
        help="Experimental run number (1 or 2)",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        required=True,
        choices=["simple", "complex"],
        help="Prompt variant to process",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute embeddings even if they already exist",
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    embed_generated_for_model(
        model_key=args.model_key,
        llm_key=args.llm_key,
        full_run=args.full_run,
        prompt_variant=args.prompt_variant,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
