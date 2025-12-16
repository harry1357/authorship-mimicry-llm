#!/usr/bin/env python3
"""
Generate embeddings using split-and-average strategy.

For documents longer than max_length:
1. Split document into 2 halves (by character count)
2. Embed each half separately
3. Average the two embeddings

This reduces truncation by splitting long docs into two halves.
Note: Some truncation may still occur if individual halves exceed max_length.

Usage:
    # Generate for all corpus authors
    python src/generate_split_embeddings.py --model-key star --corpus all
    
    # Generate for specific authors only
    python src/generate_split_embeddings.py --model-key luar_crud_orig --author-ids author1 author2
    
    # Dry run to see what would be generated
    python src/generate_split_embeddings.py --model-key star --corpus all --dry-run
"""

import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
from tqdm import tqdm
import torch

from generation_config import STYLE_MODEL_KEYS
from model_configs import MODEL_CONFIGS, EMBEDDINGS_DIR, CORPUS_DIR, AUTHOR_LIST_FILE


# Device selection: prefer MPS (Apple Silicon GPU), then CUDA, then CPU
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

print(f"[INFO] Using device: {DEVICE}")


def load_style_model(model_key: str):
    """Load style embedding model and tokenizer."""
    config = MODEL_CONFIGS[model_key]
    hf_name = config['hf_name']
    
    if model_key in ['luar_crud_st', 'luar_mud_st', 'style_embedding']:
        # Sentence-transformer style models
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(hf_name, device=DEVICE)
        tokenizer = model.tokenizer
        return model, tokenizer
    
    elif model_key in ['luar_crud_orig', 'luar_mud_orig']:
        # Original LUAR models with custom code
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(hf_name, trust_remote_code=True)
        return model, tokenizer
    
    elif model_key == 'star':
        # STAR model
        from transformers import AutoTokenizer, AutoModel
        tokenizer_name = config.get('tokenizer_name', 'roberta-large')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModel.from_pretrained(hf_name)
        return model, tokenizer
    
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def split_text_in_half(text: str) -> tuple[str, str]:
    """Split text into two halves by character count."""
    text = text.strip()
    if not text:
        return "", ""
    
    mid_point = len(text) // 2
    
    # Try to split at a sentence boundary near the midpoint
    # Look for '. ' within 10% of midpoint
    search_start = max(0, mid_point - len(text) // 10)
    search_end = min(len(text), mid_point + len(text) // 10)
    search_text = text[search_start:search_end]
    
    sentence_end = search_text.rfind('. ')
    if sentence_end != -1:
        split_point = search_start + sentence_end + 2  # +2 for '. '
    else:
        split_point = mid_point
    
    left, right = text[:split_point].strip(), text[split_point:].strip()
    
    # Safeguard: if one half is empty, fall back to simple split
    if not left or not right:
        left, right = text[:mid_point].strip(), text[mid_point:].strip()
    
    return left, right


def load_corpus_texts(corpus_dir: Path) -> dict[str, list[str]]:
    """Load all review texts from corpus."""
    author_texts = {}
    
    for author_dir in tqdm(list(corpus_dir.iterdir()), desc="Loading corpus"):
        if not author_dir.is_dir():
            continue
        
        author_id = author_dir.name
        texts = []
        
        for review_file in sorted(author_dir.glob("*.txt")):
            with open(review_file, 'r', encoding='utf-8') as f:
                texts.append(f.read().strip())
        
        if texts:
            author_texts[author_id] = texts
    
    return author_texts


def generate_split_embeddings(
    model_key: str,
    author_ids: Optional[List[str]] = None,
    corpus: str = "all",
    dry_run: bool = False
):
    """
    Generate embeddings using split-and-average strategy.
    
    Args:
        model_key: Style embedding model to use
        author_ids: Specific authors to process (None = all)
        corpus: Which corpus to use ("all" = full corpus, "experimental" = 157 authors)
        dry_run: If True, just show what would be generated
    """
    print(f"\n{'='*80}")
    print(f"Generating Split-Average Embeddings: {model_key}")
    print(f"Corpus: {corpus}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*80}\n")
    
    # Load model
    print(f"[INFO] Loading model: {model_key}")
    model, tokenizer = load_style_model(model_key)
    
    # LUAR original models don't support MPS, force CPU for them
    if model_key in ['luar_crud_orig', 'luar_mud_orig'] and DEVICE == 'mps':
        device = 'cpu'
        print(f"[WARNING] {model_key} doesn't support MPS, falling back to CPU")
    else:
        device = DEVICE
    
    if model_key not in ['luar_crud_st', 'luar_mud_st', 'style_embedding']:
        model = model.to(device)
        model.eval()
    
    # Get max_length for this model
    config = MODEL_CONFIGS[model_key]
    max_length = config.get('max_length', 512)
    print(f"[INFO] Using max_length: {max_length}")
    
    # Load corpus
    corpus_path = CORPUS_DIR
    
    print(f"[INFO] Loading corpus from: {corpus_path}")
    author_texts = load_corpus_texts(corpus_path)
    
    # Filter to specific authors if requested
    if author_ids:
        author_texts = {aid: texts for aid, texts in author_texts.items() if aid in author_ids}
    
    # Filter to experimental authors if requested
    if corpus == "experimental":
        if AUTHOR_LIST_FILE.exists():
            with open(AUTHOR_LIST_FILE, 'r') as f:
                # Skip header line
                exp_ids = set(line.split()[0] for line in f if line.strip() and not line.startswith('author_id'))
            author_texts = {aid: texts for aid, texts in author_texts.items() if aid in exp_ids}
    
    print(f"[INFO] Processing {len(author_texts)} authors")
    
    # Output directory
    output_dir = EMBEDDINGS_DIR / "split_average" / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        print(f"\n[DRY RUN] Would generate embeddings for {len(author_texts)} authors")
        print(f"[DRY RUN] Output directory: {output_dir}")
        return
    
    # Generate embeddings
    stats = {
        'total_docs': 0,
        'split_docs': 0,
        'normal_docs': 0
    }
    
    # Get batch_size from config
    batch_size = config.get('batch_size', 32)
    print(f"[INFO] Using batch_size: {batch_size}")
    
    for author_id, texts in tqdm(author_texts.items(), desc="Generating embeddings"):
        doc_lengths = []
        split_flags = []
        
        # Prepare all texts and determine which need splitting
        texts_to_embed = []
        
        for idx, text in enumerate(texts):
            # Check if text will be split
            tokens_check = tokenizer(text, truncation=False, add_special_tokens=True)
            num_tokens = len(tokens_check['input_ids'])
            doc_lengths.append(num_tokens)
            
            if num_tokens > max_length:
                # Need to split
                half1, half2 = split_text_in_half(text)
                texts_to_embed.append(half1)
                texts_to_embed.append(half2)
                split_flags.append(True)
                stats['split_docs'] += 1
            else:
                # Can embed directly
                texts_to_embed.append(text)
                split_flags.append(False)
                stats['normal_docs'] += 1
            
            stats['total_docs'] += 1
        
        # Now embed all texts in batches (much faster!)
        all_embeddings = []
        
        if model_key in ['luar_crud_st', 'luar_mud_st', 'style_embedding']:
            # Sentence transformers - use their batch encoding
            all_embeddings = model.encode(texts_to_embed, batch_size=batch_size, show_progress_bar=False)
        else:
            # LUAR original or STAR - manual batching
            with torch.no_grad():
                for i in range(0, len(texts_to_embed), batch_size):
                    batch_texts = texts_to_embed[i:i+batch_size]
                    
                    if model_key in ['luar_crud_orig', 'luar_mud_orig']:
                        tokenized = tokenizer(
                            batch_texts,
                            max_length=max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        input_ids = tokenized['input_ids'].unsqueeze(1).to(device)  # (B, 1, L)
                        attention_mask = tokenized['attention_mask'].unsqueeze(1).to(device)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        
                        if isinstance(outputs, torch.Tensor):
                            batch_emb = outputs.cpu().numpy()
                        else:
                            batch_emb = outputs[0].cpu().numpy()
                            
                    elif model_key == 'star':
                        tokenized = tokenizer(
                            batch_texts,
                            max_length=max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        ).to(device)
                        outputs = model(**tokenized)
                        batch_emb = outputs.pooler_output.cpu().numpy()
                    
                    all_embeddings.append(batch_emb)
            
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Now reconstruct final embeddings (averaging splits where needed)
        final_embeddings = []
        cursor = 0
        
        for is_split in split_flags:
            if is_split:
                # Average the two halves
                emb = (all_embeddings[cursor] + all_embeddings[cursor + 1]) / 2
                cursor += 2
            else:
                # Use as-is
                emb = all_embeddings[cursor]
                cursor += 1
            final_embeddings.append(emb)
        
        # Save
        output_path = output_dir / f"{author_id}.npz"
        np.savez_compressed(
            output_path,
            embeddings=np.array(final_embeddings),
            doc_lengths=np.array(doc_lengths),
            split_flags=np.array(split_flags)
        )
    
    print(f"\n[SUCCESS] Generated embeddings for {len(author_texts)} authors")
    print(f"[STATS] Total documents: {stats['total_docs']}")
    print(f"[STATS] Split documents: {stats['split_docs']} ({stats['split_docs']/stats['total_docs']*100:.1f}%)")
    print(f"[STATS] Normal documents: {stats['normal_docs']} ({stats['normal_docs']/stats['total_docs']*100:.1f}%)")
    print(f"[SAVED] Embeddings to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings using split-and-average strategy"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        required=True,
        choices=STYLE_MODEL_KEYS,
        help="Style embedding model to use"
    )
    parser.add_argument(
        "--author-ids",
        type=str,
        nargs='+',
        help="Specific author IDs to process (default: all)"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="all",
        choices=["all", "experimental"],
        help="Which corpus to use (all = full corpus, experimental = 157 authors)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without actually generating"
    )
    
    args = parser.parse_args()
    
    generate_split_embeddings(
        model_key=args.model_key,
        author_ids=args.author_ids,
        corpus=args.corpus,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
