#!/usr/bin/env python3
"""
Global t-SNE: All 157 Experimental Authors (Training + Generated)

Creates t-SNE visualizations showing:
- All 157 experimental authors with TRAINING + SIMPLE + COMPLEX generated docs
- Each author gets a unique color
- Shows whether generated texts cluster with their own training docs

This is like plot_global_157_vs_all.py but ensures it includes generated docs.

Usage:
    python src/plot_157_with_generated_vs_all.py --model-key luar_crud_orig --full-run 1
    python src/plot_157_with_generated_vs_all.py --all-models --full-run 1
"""

import argparse
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import pandas as pd
import csv
import ast

from generation_config import (
    EMBEDDINGS_DIR,
    STYLE_MODEL_KEYS,
    CONSISTENCY_DIR,
    REFERENCE_CONSISTENCY_CSV,
)
from model_configs import PLOTS_DIR


def load_selected_indices() -> Dict[str, list]:
    """
    Load pre-computed indices of the 6 most consistent reviews for each author.
    These are the EXACT training reviews used to generate the prompts and analysis.
    """
    mapping = {}
    
    if not REFERENCE_CONSISTENCY_CSV.exists():
        print(f"[WARNING] Reference consistency CSV not found: {REFERENCE_CONSISTENCY_CSV}")
        return mapping
    
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
                indices = [int(x) for x in raw.replace("[", "").replace("]", "").split(",") if x.strip()]
            indices = [int(i) for i in indices]
            if indices:
                mapping[author_id] = indices
    
    return mapping


def load_experimental_authors_with_generated(
    model_key: str,
    llm_key: str,
    full_run: int,
    experimental_ids: set,
    selected_indices_map: Dict[str, list]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load training + generated embeddings for 157 experimental authors.
    
    Uses the EXACT 6 training docs that were used for prompts (via selected_indices).
    
    Returns dict with structure:
        {author_id: {'training': array, 'simple': array, 'complex': array}}
    """
    result = {}
    
    for author_id in tqdm(experimental_ids, desc="Loading experimental"):
        # Training - USE SELECTED INDICES (same as prompts/analysis)
        train_path = EMBEDDINGS_DIR / model_key / f"{author_id}.npz"
        if not train_path.exists():
            continue
        
        train_data = np.load(train_path, allow_pickle=True)
        
        # Priority: Use selected_indices from CSV (same as prompts)
        if author_id in selected_indices_map:
            selected_idx = selected_indices_map[author_id]
            training = train_data['embeddings'][selected_idx]
        elif 'selected_indices' in train_data:
            # Fallback: indices in npz (unlikely)
            training = train_data['embeddings'][train_data['selected_indices']]
        else:
            # Last resort: first 6 (inconsistent with prompts!)
            print(f"[WARNING] {author_id}: Using first 6 (no selected_indices found)")
            training = train_data['embeddings'][:6]
        
        # Simple generated
        simple_path = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "simple" / f"fullrun{full_run}" / f"{author_id}.npz"
        if not simple_path.exists():
            continue
        simple = np.load(simple_path, allow_pickle=True)['embeddings']
        
        # Complex generated
        complex_path = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "complex" / f"fullrun{full_run}" / f"{author_id}.npz"
        if not complex_path.exists():
            continue
        complex = np.load(complex_path, allow_pickle=True)['embeddings']
        
        result[author_id] = {
            'training': training,
            'simple': simple,
            'complex': complex
        }
    
    return result


def create_tsne_plot(
    model_key: str,
    llm_key: str,
    full_run: int,
    output_dir: Path
):
    """Create t-SNE plot with all 157 experimental authors (training + generated)."""
    
    print(f"\n{'='*80}")
    print(f"Generating t-SNE: All 157 Experimental Authors (Training + Generated)")
    print(f"Model: {model_key}, Run: {full_run}")
    print(f"{'='*80}\n")
    
    # Get list of experimental authors
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_fullrun{full_run}.csv"
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    experimental_ids = set(df['author_id'].tolist())
    print(f"[INFO] Found {len(experimental_ids)} experimental authors")
    
    # Load selected indices (same as used for prompts/analysis)
    print(f"[INFO] Loading selected training indices...")
    selected_indices_map = load_selected_indices()
    print(f"[INFO] Loaded selected indices for {len(selected_indices_map)} authors")
    
    # Load experimental authors with generated
    print(f"[INFO] Loading experimental authors with generated texts...")
    experimental_with_gen = load_experimental_authors_with_generated(
        model_key, llm_key, full_run, experimental_ids, selected_indices_map
    )
    print(f"[INFO] Loaded {len(experimental_with_gen)} experimental authors with generated texts")
    
    # Prepare data for t-SNE - each author gets unique ID
    all_embeddings = []
    doc_types = []  # 'training', 'simple', 'complex'
    author_ids = []
    
    # Add experimental authors (training + generated)
    for author_id, data in experimental_with_gen.items():
        for train_emb in data['training']:
            all_embeddings.append(train_emb)
            doc_types.append('training')
            author_ids.append(author_id)
        
        for simple_emb in data['simple']:
            all_embeddings.append(simple_emb)
            doc_types.append('simple')
            author_ids.append(author_id)
        
        for complex_emb in data['complex']:
            all_embeddings.append(complex_emb)
            doc_types.append('complex')
            author_ids.append(author_id)
    
    all_embeddings = np.array(all_embeddings)
    print(f"\n[INFO] Total documents for t-SNE: {len(all_embeddings)}")
    print(f"  - Training: {doc_types.count('training')}")
    print(f"  - Simple: {doc_types.count('simple')}")
    print(f"  - Complex: {doc_types.count('complex')}")
    
    # Run t-SNE
    print(f"\n[INFO] Running t-SNE (this may take a few minutes)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=1)
    coords_2d = tsne.fit_transform(all_embeddings)
    
    # Create plot with unique color per author
    print(f"[INFO] Creating plot...")
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Get unique author list and assign colors
    unique_authors = sorted(list(set(author_ids)))
    n_authors = len(unique_authors)
    colors = plt.cm.tab20c(np.linspace(0, 1, n_authors)) if n_authors <= 20 else plt.cm.rainbow(np.linspace(0, 1, n_authors))
    author_to_color = {author: colors[i] for i, author in enumerate(unique_authors)}
    
    markers = {
        'training': 'o',
        'simple': 's',
        'complex': '^',
    }
    
    sizes = {
        'training': 60,
        'simple': 80,
        'complex': 80,
    }
    
    # Plot each author
    for author_id in unique_authors:
        author_color = author_to_color[author_id]
        
        # Get indices for this author
        author_mask = np.array([aid == author_id for aid in author_ids])
        
        # Plot each doc type for this author
        for doc_type in ['training', 'simple', 'complex']:
            mask = author_mask & np.array([dt == doc_type for dt in doc_types])
            
            if np.sum(mask) > 0:
                ax.scatter(
                    coords_2d[mask, 0],
                    coords_2d[mask, 1],
                    c=[author_color],
                    marker=markers[doc_type],
                    s=sizes[doc_type],
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=1,
                    label=f'{author_id[:15]}... ({doc_type})' if author_id == unique_authors[0] else ''
                )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Global t-SNE: All 157 Authors - Training + Simple + Complex\n'
        f'Each author has a unique color. Model: {model_key}, Run: {full_run}',
        fontsize=16, fontweight='bold', pad=20
    )
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, 
               label='Training (6 per author)', markeredgecolor='black', markeredgewidth=1),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10,
               label='Simple (2 per author)', markeredgecolor='black', markeredgewidth=1),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10,
               label='Complex (2 per author)', markeredgecolor='black', markeredgewidth=1),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add interpretation box
    interpretation = (
        "INTERPRETATION:\n"
        "• Each author = unique color (157 colors)\n"
        "• Circles = Training, Squares = Simple, Triangles = Complex\n"
        "\n"
        "GOOD MIMICRY: Same-color points cluster together\n"
        "POOR MIMICRY: Same-color points separate by shape"
    )
    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tsne_157_all_with_generated_{model_key}_run{full_run}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SAVED] {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot 157 experimental authors (with generated) vs all other authors"
    )
    parser.add_argument("--model-key", type=str, help="Style embedding model")
    parser.add_argument("--all-models", action="store_true", help="Generate for all models")
    parser.add_argument("--llm-key", type=str, default="gpt-5.1")
    parser.add_argument("--full-run", type=int, default=1, choices=[1, 2])
    
    args = parser.parse_args()
    
    # Determine which models to process
    if args.all_models:
        model_keys = STYLE_MODEL_KEYS
    elif args.model_key:
        model_keys = [args.model_key]
    else:
        print("[ERROR] Must specify --model-key or --all-models")
        return
    
    # Generate plots
    for model_key in model_keys:
        output_dir = PLOTS_DIR / model_key / f"fullrun{args.full_run}_global"
        create_tsne_plot(model_key, args.llm_key, args.full_run, output_dir)


if __name__ == "__main__":
    main()
