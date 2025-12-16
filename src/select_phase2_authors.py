#!/usr/bin/env python3
"""
Select authors for Phase 2 (LLM Mimicry Detection) based on authorship verification threshold.

Criteria:
1. Author must have multiple documents (for training/testing split)
2. Same-author cosine distances must be consistently below threshold
3. This ensures the model can reliably identify this author's style

Usage:
    python src/select_phase2_authors.py --model-key luar_mud_orig --threshold 0.2648 --min-docs 4
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
from scipy.spatial.distance import cosine
from tqdm import tqdm
import json

from model_configs import EMBEDDINGS_DIR, MODEL_CONFIGS


def load_author_embeddings(embeddings_dir: Path, author_id: str) -> np.ndarray:
    """Load embeddings for a single author."""
    npz_path = embeddings_dir / f"{author_id}.npz"
    if not npz_path.exists():
        return None
    
    data = np.load(npz_path, allow_pickle=True)
    return data['embeddings']


def compute_all_sa_distances(embeddings: np.ndarray) -> List[float]:
    """Compute all pairwise same-author distances."""
    n = len(embeddings)
    distances = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = cosine(embeddings[i], embeddings[j])
            distances.append(dist)
    
    return distances


def evaluate_author(embeddings: np.ndarray, threshold: float) -> dict:
    """
    Evaluate if an author meets Phase 2 criteria.
    
    Returns dict with:
        - n_docs: number of documents
        - sa_distances: all pairwise SA distances
        - mean_sa_dist: mean SA distance
        - max_sa_dist: maximum SA distance
        - qualifies: whether author meets threshold
    """
    if embeddings is None or len(embeddings) < 2:
        return None
    
    sa_distances = compute_all_sa_distances(embeddings)
    
    stats = {
        'n_docs': len(embeddings),
        'n_pairs': len(sa_distances),
        'sa_distances': sa_distances,
        'mean_sa_dist': np.mean(sa_distances),
        'std_sa_dist': np.std(sa_distances),
        'max_sa_dist': np.max(sa_distances),
        'min_sa_dist': np.min(sa_distances),
        'qualifies': np.max(sa_distances) < threshold  # All pairs must be below threshold
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Select authors for Phase 2 based on verification threshold"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="luar_mud_orig",
        help="Model to use (default: luar_mud_orig)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2648,
        help="CosD threshold from Phase 1 (default: 0.2648 for luar_mud_orig)"
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=4,
        help="Minimum number of documents per author (default: 4)"
    )
    parser.add_argument(
        "--use-split-embeddings",
        action="store_true",
        help="Use split-average embeddings"
    )
    parser.add_argument(
        "--max-authors",
        type=int,
        default=None,
        help="Maximum number of qualifying authors to select (optional)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE 2: AUTHOR SELECTION FOR LLM MIMICRY DETECTION")
    print("="*80)
    print(f"Model:              {args.model_key}")
    print(f"CosD Threshold:     {args.threshold}")
    print(f"Min documents:      {args.min_docs}")
    print(f"Split embeddings:   {args.use_split_embeddings}")
    print("="*80)
    print()
    
    # Load embeddings directory
    if args.use_split_embeddings:
        embeddings_dir = EMBEDDINGS_DIR / "split_average" / args.model_key
    else:
        embeddings_dir = EMBEDDINGS_DIR / args.model_key
    
    if not embeddings_dir.exists():
        print(f"[ERROR] Embeddings directory not found: {embeddings_dir}")
        return
    
    print(f"[INFO] Loading embeddings from: {embeddings_dir}")
    
    # Get all author files
    author_files = sorted(embeddings_dir.glob("*.npz"))
    print(f"[INFO] Found {len(author_files)} authors with embeddings")
    
    # Evaluate each author
    qualifying_authors = []
    disqualified = {
        'too_few_docs': 0,
        'above_threshold': 0
    }
    
    for author_file in tqdm(author_files, desc="Evaluating authors"):
        author_id = author_file.stem
        embeddings = load_author_embeddings(embeddings_dir, author_id)
        
        if embeddings is None or len(embeddings) < args.min_docs:
            disqualified['too_few_docs'] += 1
            continue
        
        stats = evaluate_author(embeddings, args.threshold)
        
        if stats and stats['qualifies']:
            qualifying_authors.append({
                'author_id': author_id,
                **stats
            })
        else:
            disqualified['above_threshold'] += 1
    
    # Sort by mean SA distance (most consistent first)
    qualifying_authors.sort(key=lambda x: x['mean_sa_dist'])
    
    # Convert numpy types to Python types for JSON serialization
    for author in qualifying_authors:
        author['n_docs'] = int(author['n_docs'])
        author['n_pairs'] = int(author['n_pairs'])
        author['mean_sa_dist'] = float(author['mean_sa_dist'])
        author['std_sa_dist'] = float(author['std_sa_dist'])
        author['max_sa_dist'] = float(author['max_sa_dist'])
        author['min_sa_dist'] = float(author['min_sa_dist'])
        author['qualifies'] = bool(author['qualifies'])
        author['sa_distances'] = [float(d) for d in author['sa_distances']]
    
    # Apply max_authors limit if specified
    if args.max_authors and len(qualifying_authors) > args.max_authors:
        print(f"\n[INFO] Limiting to top {args.max_authors} most consistent authors")
        qualifying_authors = qualifying_authors[:args.max_authors]
    
    # Save results
    output_dir = Path("data/phase2_selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full details as JSON
    output_json = output_dir / f"selected_authors_{args.model_key}_threshold{args.threshold:.4f}.json"
    with open(output_json, 'w') as f:
        json.dump({
            'model_key': args.model_key,
            'threshold': args.threshold,
            'min_docs': args.min_docs,
            'n_qualifying': len(qualifying_authors),
            'n_disqualified_few_docs': disqualified['too_few_docs'],
            'n_disqualified_threshold': disqualified['above_threshold'],
            'authors': qualifying_authors
        }, f, indent=2)
    
    # Save simple list of author IDs
    output_txt = output_dir / f"selected_author_ids_{args.model_key}.txt"
    with open(output_txt, 'w') as f:
        f.write("# Phase 2 Selected Authors\n")
        f.write(f"# Model: {args.model_key}\n")
        f.write(f"# Threshold: {args.threshold}\n")
        f.write(f"# Min docs: {args.min_docs}\n")
        f.write(f"# Total selected: {len(qualifying_authors)}\n")
        f.write("#\n")
        f.write("author_id\tn_docs\tmean_sa_dist\tmax_sa_dist\n")
        for author in qualifying_authors:
            f.write(f"{author['author_id']}\t{author['n_docs']}\t{author['mean_sa_dist']:.4f}\t{author['max_sa_dist']:.4f}\n")
    
    # Print summary
    print("\n" + "="*80)
    print("SELECTION RESULTS")
    print("="*80)
    print(f"✅ Qualifying authors:           {len(qualifying_authors)}")
    print(f"❌ Disqualified (< {args.min_docs} docs):  {disqualified['too_few_docs']}")
    print(f"❌ Disqualified (> threshold):   {disqualified['above_threshold']}")
    print(f"📊 Total evaluated:              {len(author_files)}")
    print("="*80)
    
    if qualifying_authors:
        print("\n📋 Top 10 Most Consistent Authors:")
        print("-" * 80)
        print(f"{'Author ID':<20} {'Docs':>6} {'Mean SA':>10} {'Max SA':>10} {'Std SA':>10}")
        print("-" * 80)
        for author in qualifying_authors[:10]:
            print(f"{author['author_id']:<20} {author['n_docs']:>6} "
                  f"{author['mean_sa_dist']:>10.4f} {author['max_sa_dist']:>10.4f} "
                  f"{author['std_sa_dist']:>10.4f}")
        print("-" * 80)
        
        print(f"\n💾 Saved to:")
        print(f"   - Full details: {output_json}")
        print(f"   - Author IDs:   {output_txt}")
        
        print(f"\n🎯 Next steps for Phase 2:")
        print(f"   1. Select subset of these {len(qualifying_authors)} authors for generation")
        print(f"   2. Split their texts into training/testing sets")
        print(f"   3. Generate LLM-mimicked reviews using training texts")
        print(f"   4. Test if model can distinguish real vs generated")
    else:
        print("\n⚠️  No authors qualify with current threshold!")
        print("   Consider:")
        print("   - Lowering --threshold")
        print("   - Reducing --min-docs")


if __name__ == "__main__":
    main()
