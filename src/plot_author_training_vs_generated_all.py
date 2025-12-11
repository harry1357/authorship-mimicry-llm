#!/usr/bin/env python3
"""
Generate Global t-SNE Plot: All 157 Authors (Training + Simple + Complex)

Creates ONE t-SNE visualization showing ALL documents from all 157 authors:
- 6 training documents per author (most consistent) = 942 docs
- 2 simple-generated documents per author = 314 docs
- 2 complex-generated documents per author = 314 docs
Total: 1,570 documents

Color-coded by author to visualize:
- Whether generated texts cluster with their author's training documents
- Overall separation between authors
- Relative performance of simple vs complex prompts

Usage:
    python src/plot_author_training_vs_generated_all.py --model-key style_embedding --full-run 1
    
    # Generate for all models
    python src/plot_author_training_vs_generated_all.py --all-models --full-run 1
"""

import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from tqdm import tqdm

from generation_config import (
    EMBEDDINGS_DIR,
    STYLE_MODEL_KEYS,
    REFERENCE_MODEL_KEY,
)
from model_configs import PLOTS_DIR


def load_author_embeddings(
    author_id: str,
    model_key: str,
    llm_key: str,
    full_run: int,
) -> Dict:
    """
    Load all embeddings for an author: training + simple + complex.
    
    Returns:
        Dict with keys: 'training', 'simple', 'complex'
        Each value is a 2D numpy array of embeddings
    """
    # Load training embeddings (6 most consistent docs)
    train_path = EMBEDDINGS_DIR / model_key / f"{author_id}.npz"
    if not train_path.exists():
        return None
    
    train_data = np.load(train_path, allow_pickle=True)
    
    # Get the 6 most consistent training docs
    if 'selected_indices' in train_data:
        selected_idx = train_data['selected_indices']
        training_embs = train_data['embeddings'][selected_idx]
    else:
        # Fallback: use first 6
        training_embs = train_data['embeddings'][:6]
    
    # Load simple generated embeddings
    simple_path = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "simple" / f"fullrun{full_run}" / f"{author_id}.npz"
    if simple_path.exists():
        simple_data = np.load(simple_path, allow_pickle=True)
        simple_embs = simple_data['embeddings']
    else:
        simple_embs = None
    
    # Load complex generated embeddings
    complex_path = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "complex" / f"fullrun{full_run}" / f"{author_id}.npz"
    if complex_path.exists():
        complex_data = np.load(complex_path, allow_pickle=True)
        complex_embs = complex_data['embeddings']
    else:
        complex_embs = None
    
    if simple_embs is None or complex_embs is None:
        return None
    
    return {
        'training': training_embs,
        'simple': simple_embs,
        'complex': complex_embs,
    }


def plot_global_tsne(
    all_data: Dict,
    model_key: str,
    full_run: int,
    output_dir: Path,
    top_n: int = None,
    author_order: List[str] = None,
):
    """
    Create ONE global t-SNE plot with all authors.
    Each author gets a unique color.
    
    Args:
        all_data: Dict mapping author_id -> embeddings dict
        model_key: Style embedding model
        full_run: Experimental run number
        output_dir: Where to save the plot
        top_n: Number of top authors (for title), or None for all
        author_order: List of author IDs in ranking order (best to worst)
    """
    # Collect all embeddings and metadata
    all_embeddings = []
    all_labels = []
    all_types = []  # 'training', 'simple', 'complex'
    all_authors = []
    
    # Create author to index mapping using the provided order (not alphabetical!)
    # This way A1 = best performer, A2 = second best, etc.
    if author_order is not None:
        author_list = author_order
    else:
        author_list = sorted(all_data.keys())
    
    author_to_idx = {author: idx for idx, author in enumerate(author_list)}
    
    for author_id, embs in all_data.items():
        # Training
        for emb in embs['training']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('training')
            all_authors.append(author_to_idx[author_id])
        
        # Simple
        for emb in embs['simple']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('simple')
            all_authors.append(author_to_idx[author_id])
        
        # Complex
        for emb in embs['complex']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('complex')
            all_authors.append(author_to_idx[author_id])
    
    all_embeddings = np.array(all_embeddings)
    all_authors = np.array(all_authors)
    n_docs = len(all_embeddings)
    n_authors = len(all_data)
    
    print(f"[INFO] Total documents: {n_docs}")
    print(f"[INFO] Total authors: {n_authors}")
    print(f"[INFO] Running t-SNE (this may take a few minutes)...")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(50, n_docs // 4),
        max_iter=1000,
        verbose=1,
    )
    coords_2d = tsne.fit_transform(all_embeddings)
    
    print("[INFO] Creating visualization...")
    
    # Create plot with larger figure
    fig, ax = plt.subplots(figsize=(24, 20))
    
    # Get colormap for authors
    colors = cm.get_cmap('tab20', n_authors)
    if n_authors > 20:
        # Use hsv for many authors
        colors = cm.get_cmap('hsv', n_authors)
    
    # Convert types to arrays for indexing
    types_array = np.array(all_types)
    
    # Plot each author with their unique color
    training_mask = types_array == 'training'
    simple_mask = types_array == 'simple'
    complex_mask = types_array == 'complex'
    
    # Plot training documents
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = training_mask & author_mask
        
        if combined_mask.sum() > 0:
            ax.scatter(
                coords_2d[combined_mask, 0],
                coords_2d[combined_mask, 1],
                c=[colors(author_idx)],
                marker='o',
                s=120,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5
            )
    
    # Plot simple generated (larger, more visible)
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = simple_mask & author_mask
        
        if combined_mask.sum() > 0:
            ax.scatter(
                coords_2d[combined_mask, 0],
                coords_2d[combined_mask, 1],
                c=[colors(author_idx)],
                marker='s',
                s=200,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5
            )
    
    # Plot complex generated (larger, more visible)
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = complex_mask & author_mask
        
        if combined_mask.sum() > 0:
            ax.scatter(
                coords_2d[combined_mask, 0],
                coords_2d[combined_mask, 1],
                c=[colors(author_idx)],
                marker='^',
                s=200,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5
            )
    
    # Add text labels at the TRAINING centroid of each author
    # This shows the "real" author position, and you can see how far generated texts drift
    print("[INFO] Adding author labels at training centroids...")
    for author_idx, author_id in enumerate(author_list):
        author_mask = all_authors == author_idx
        training_author_mask = training_mask & author_mask
        
        if training_author_mask.sum() > 0:
            # Calculate centroid of ONLY training documents
            training_centroid = np.mean(coords_2d[training_author_mask], axis=0)
            
            # Add text label with background box
            ax.text(
                training_centroid[0], training_centroid[1],
                f'A{author_idx+1}',
                fontsize=11,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor='white',
                    edgecolor=colors(author_idx),
                    linewidth=2,
                    alpha=0.9
                ),
                zorder=100
            )
    
    # Create legend with document types only (not individual authors)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=12, alpha=0.6, label='Training (6 per author)', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=14, alpha=0.9, label='Simple (2 per author)', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=14, alpha=0.9, label='Complex (2 per author)', linestyle='None'),
    ]
    
    # Styling
    ax.set_xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
    
    title_prefix = f"Top {top_n}" if top_n else f"All {n_authors}"
    ax.set_title(
        f'Global t-SNE: {title_prefix} Authors - Training + Simple + Complex\n'
        f'Each author has a unique color. Model: {model_key}, Run: {full_run}',
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(handles=legend_elements, loc='best', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_top{top_n}" if top_n else "_all_157"
    output_path = output_dir / f"global_tsne{suffix}_authors_{model_key}_run{full_run}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Global plot: {output_path}")
    
    # Save author mapping to a text file
    mapping_path = output_dir / f"author_mapping{suffix}_{model_key}_run{full_run}.txt"
    with open(mapping_path, 'w') as f:
        f.write(f"Author Label Mapping for {model_key} (Run {full_run})\n")
        f.write(f"Ranked by best mimicry performance (min of simple/complex distance)\n")
        f.write(f"NOTE: Distances shown are in embedding space, not t-SNE 2D space\n")
        f.write("=" * 60 + "\n\n")
        for idx, author_id in enumerate(author_list):
            f.write(f"A{idx+1:2d} -> {author_id} (Rank {idx+1})\n")
    
    print(f"[SAVED] Author mapping: {mapping_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate global t-SNE plot with all 157 authors (training + simple + complex)"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="style_embedding",
        help="Style embedding model to use",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Generate plots for all models",
    )
    parser.add_argument(
        "--llm-key",
        type=str,
        default="gpt-5.1",
        help="LLM used for generation",
    )
    parser.add_argument(
        "--full-run",
        type=int,
        default=1,
        choices=[1, 2],
        help="Experimental run number",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only plot top N authors (by mimicry performance). Default: all authors",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default="best",
        choices=["simple", "complex", "best", "average"],
        help="How to rank authors: 'simple' (simple only), 'complex' (complex only), 'best' (min of both), 'average' (avg of both). Default: best",
    )
    
    args = parser.parse_args()
    
    models = STYLE_MODEL_KEYS if args.all_models else [args.model_key]
    
    for model_key in models:
        print(f"\n{'='*80}")
        print(f"Generating global plot for model: {model_key}")
        print(f"{'='*80}\n")
        
        # Get list of authors
        if args.top_n:
            # Load from analysis CSV to get top performers
            from generation_config import CONSISTENCY_DIR
            import pandas as pd
            
            csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_fullrun{args.full_run}.csv"
            if not csv_path.exists():
                print(f"[ERROR] CSV not found: {csv_path}")
                print(f"Run: python src/analyse_simple_vs_complex.py --model-key {model_key} --full-run {args.full_run}")
                continue
            
            df = pd.read_csv(csv_path)
            
            # Sort by chosen ranking method
            if args.rank_by == "simple":
                df_sorted = df.sort_values('dist_real_centroid_simple')
                rank_desc = "simple prompt performance"
            elif args.rank_by == "complex":
                df_sorted = df.sort_values('dist_real_centroid_complex')
                rank_desc = "complex prompt performance"
            elif args.rank_by == "best":
                df['best_mimicry_dist'] = df[['dist_real_centroid_simple', 'dist_real_centroid_complex']].min(axis=1)
                df_sorted = df.sort_values('best_mimicry_dist')
                rank_desc = "best mimicry performance (min of simple/complex)"
            else:  # average
                df['avg_mimicry_dist'] = df[['dist_real_centroid_simple', 'dist_real_centroid_complex']].mean(axis=1)
                df_sorted = df.sort_values('avg_mimicry_dist')
                rank_desc = "average mimicry performance"
            
            author_ids = df_sorted.head(args.top_n)['author_id'].tolist()
            
            print(f"[INFO] Using top {args.top_n} authors by {rank_desc}")
            
            # Show which authors and their distances
            for idx, row in df_sorted.head(args.top_n).iterrows():
                better = "simple" if row['dist_real_centroid_simple'] < row['dist_real_centroid_complex'] else "complex"
                print(f"  A{author_ids.index(row['author_id'])+1}: {row['author_id']} "
                      f"(simple={row['dist_real_centroid_simple']:.4f}, "
                      f"complex={row['dist_real_centroid_complex']:.4f}, "
                      f"best={better})")
        else:
            # Get all authors with generated texts
            gen_simple_dir = EMBEDDINGS_DIR / "generated" / model_key / args.llm_key / "simple" / f"fullrun{args.full_run}"
            
            if not gen_simple_dir.exists():
                print(f"[ERROR] Generated embeddings not found: {gen_simple_dir}")
                continue
            
            author_files = sorted(gen_simple_dir.glob("*.npz"))
            author_ids = [f.stem for f in author_files]
        
        print(f"[INFO] Found {len(author_ids)} authors to plot")
        
        # Load all author data
        all_data = {}
        missing_count = 0
        
        for author_id in tqdm(author_ids, desc="Loading embeddings"):
            embeddings = load_author_embeddings(
                author_id, model_key, args.llm_key, args.full_run
            )
            
            if embeddings is None:
                missing_count += 1
                continue
            
            all_data[author_id] = embeddings
        
        print(f"[INFO] Loaded {len(all_data)} authors successfully")
        if missing_count > 0:
            print(f"[WARNING] {missing_count} authors skipped (missing data)")
        
        if len(all_data) == 0:
            print("[ERROR] No data to plot!")
            continue
        
        # Output directory
        suffix = f"_top{args.top_n}" if args.top_n else "_all"
        output_dir = PLOTS_DIR / model_key / f"fullrun{args.full_run}_global{suffix}"
        
        # Generate global plot
        try:
            # Pass the author order to the plot function
            plot_global_tsne(all_data, model_key, args.full_run, output_dir, args.top_n, author_ids)
            print(f"\n[SUCCESS] Global plot generated for {model_key}\n")
        except Exception as e:
            print(f"[ERROR] Failed to generate plot: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
