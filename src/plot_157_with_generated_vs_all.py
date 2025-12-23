#!/usr/bin/env python3
"""
Global t-SNE/UMAP: All 157 Experimental Authors (Training + Generated)

Creates visualizations showing:
- All 157 experimental authors with TRAINING + SIMPLE + COMPLEX generated docs
- Each author gets a unique color
- Shows whether generated texts cluster with their own training docs

Supports:
- t-SNE (2D/3D)
- UMAP (2D/3D)
- Interactive 3D plots (Plotly)

Usage:
    python src/plot_157_with_generated_vs_all.py --model-key luar_mud_orig --full-run 1
    python src/plot_157_with_generated_vs_all.py --model-key luar_mud_orig --full-run 1 --viz-type umap --plot-3d --interactive
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

# Optional imports
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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


def create_visualization(
    model_key: str,
    llm_key: str,
    full_run: int,
    output_dir: Path,
    viz_type: str = "tsne",
    n_dims: int = 2,
    interactive: bool = False,
    top_n: int = None
):
    """Create visualization with experimental authors (training + generated)."""
    
    viz_name = viz_type.upper()
    dims_str = "3D" if n_dims == 3 else "2D"
    print(f"\n{'='*80}")
    print(f"Generating {dims_str} {viz_name}: All 157 Experimental Authors (Training + Generated)")
    print(f"Model: {model_key}, Run: {full_run}")
    print(f"{'='*80}\n")
    
    # Get list of experimental authors
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_fullrun{full_run}.csv"
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Filter out comment rows (summary metrics at bottom of CSV)
    df = df[~df['author_id'].astype(str).str.startswith('#')]
    
    # Filter by top_n if specified (CSV is already sorted by mimicry performance)
    if top_n is not None:
        df = df.head(top_n)
        print(f"[INFO] Using top {top_n} authors by mimicry performance")
    
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
    
    # Run dimensionality reduction
    if viz_type == "tsne":
        print(f"\n[INFO] Running {dims_str} t-SNE (this may take a few minutes)...")
        reducer = TSNE(n_components=n_dims, random_state=42, perplexity=min(30, len(all_embeddings)-1), max_iter=1000, verbose=1)
        coords = reducer.fit_transform(all_embeddings)
    elif viz_type == "umap":
        if not UMAP_AVAILABLE:
            print("[ERROR] UMAP not installed. Run: pip install umap-learn")
            return None
        print(f"\n[INFO] Running {dims_str} UMAP (this may take a few minutes)...")
        reducer = UMAP(n_components=n_dims, random_state=42, n_neighbors=min(15, len(all_embeddings)-1), verbose=True)
        coords = reducer.fit_transform(all_embeddings)
    else:
        print(f"[ERROR] Unknown viz_type: {viz_type}")
        return None
    
    # Generate interactive plot if requested
    if interactive and n_dims == 3:
        if not PLOTLY_AVAILABLE:
            print("[WARNING] Plotly not installed. Skipping interactive plot. Run: pip install plotly")
        else:
            return create_interactive_plot(coords, author_ids, doc_types, model_key, llm_key, full_run, output_dir, viz_type)
    
    # Create static plot with unique color per author
    print(f"[INFO] Creating {dims_str} plot...")
    if n_dims == 3:
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
    else:
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
                if n_dims == 3:
                    ax.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        coords[mask, 2],
                        c=[author_color],
                        marker=markers[doc_type],
                        s=sizes[doc_type],
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=1,
                        label=f'{author_id[:15]}... ({doc_type})' if author_id == unique_authors[0] else ''
                    )
                else:
                    ax.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        c=[author_color],
                        marker=markers[doc_type],
                        s=sizes[doc_type],
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=1,
                        label=f'{author_id[:15]}... ({doc_type})' if author_id == unique_authors[0] else ''
                    )
    
    ax.set_xlabel(f'{viz_name} Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{viz_name} Dimension 2', fontsize=14, fontweight='bold')
    if n_dims == 3:
        ax.set_zlabel(f'{viz_name} Dimension 3', fontsize=14, fontweight='bold')
    
    ax.set_title(
        f'Global {dims_str} {viz_name}: All 157 Authors - Training + Simple + Complex\n'
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
    
    # Add interpretation box (only for 2D)
    if n_dims == 2:
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
    suffix = f"_{n_dims}d" if n_dims == 3 else ""
    output_path = output_dir / f"{viz_type}{suffix}_157_all_with_generated_{model_key}_run{full_run}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SAVED] {output_path}")
    return output_path


def create_interactive_plot(
    coords: np.ndarray,
    author_ids: list,
    doc_types: list,
    model_key: str,
    llm_key: str,
    full_run: int,
    output_dir: Path,
    viz_type: str
):
    """Create interactive 3D plot using Plotly."""
    print(f"[INFO] Creating interactive 3D {viz_type.upper()} plot...")
    
    # Get unique authors
    unique_authors = sorted(list(set(author_ids)))
    n_authors = len(unique_authors)
    
    # Generate colors
    if n_authors <= 24:
        color_map = px.colors.qualitative.Dark24
    else:
        color_map = px.colors.sample_colorscale(
            px.colors.sequential.Rainbow,
            [i / (n_authors - 1) for i in range(n_authors)]
        )
    
    author_to_color = {author: color_map[i % len(color_map)] for i, author in enumerate(unique_authors)}
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'z': coords[:, 2],
        'author_id': author_ids,
        'doc_type': doc_types
    })
    
    # Create figure
    fig = go.Figure()
    
    markers_plotly = {
        'training': 'circle',
        'simple': 'square',
        'complex': 'diamond'
    }
    
    # Add traces for each author
    for author_id in unique_authors:
        author_df = df[df['author_id'] == author_id]
        color = author_to_color[author_id]
        
        for doc_type in ['training', 'simple', 'complex']:
            type_df = author_df[author_df['doc_type'] == doc_type]
            if len(type_df) == 0:
                continue
            
            fig.add_trace(go.Scatter3d(
                x=type_df['x'],
                y=type_df['y'],
                z=type_df['z'],
                mode='markers',
                name=f'{author_id[:12]}... ({doc_type})',
                marker=dict(
                    size=6 if doc_type == 'training' else 8,
                    color=color,
                    symbol=markers_plotly[doc_type],
                    line=dict(color='black', width=1)
                ),
                hovertemplate=f'<b>{author_id}</b><br>Type: {doc_type}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>',
                showlegend=(author_id == unique_authors[0])  # Only show legend for first author
            ))
    
    fig.update_layout(
        title=dict(
            text=(
                f"Interactive 3D {viz_type.upper()}: All 157 Authors<br>"
                f"<sub>Training (circles) + Simple (squares) + Complex (diamonds)</sub><br>"
                f"<sub>Model: {model_key}, Run: {full_run}</sub>"
            ),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title=f'{viz_type.upper()} Dimension 1',
            yaxis_title=f'{viz_type.upper()} Dimension 2',
            zaxis_title=f'{viz_type.upper()} Dimension 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=1400,
        height=1000,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
        font=dict(size=10)
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{viz_type}_3d_interactive_157_all_with_generated_{model_key}_run{full_run}.html"
    fig.write_html(output_path)
    
    print(f"\n[SAVED] Interactive plot: {output_path}")
    print(f"[INFO] Open in browser: file://{output_path.absolute()}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot 157 experimental authors (with generated) with various visualization methods"
    )
    parser.add_argument("--model-key", type=str, help="Style embedding model")
    parser.add_argument("--all-models", action="store_true", help="Generate for all models")
    parser.add_argument("--llm-key", type=str, default="gpt-5.1", help="LLM identifier")
    parser.add_argument("--full-run", type=int, default=1, choices=[1, 2], help="Experimental run number")
    parser.add_argument("--viz-type", type=str, default="tsne", choices=["tsne", "umap", "all"], 
                       help="Visualization type (default: tsne)")
    parser.add_argument("--plot-3d", action="store_true", help="Generate 3D plots in addition to 2D")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive 3D plots using Plotly")
    parser.add_argument("--top-n", type=int, default=None, 
                       help="Only plot top N authors by mimicry performance (default: all)")
    
    args = parser.parse_args()
    
    # Determine which models to process
    if args.all_models:
        model_keys = STYLE_MODEL_KEYS
    elif args.model_key:
        model_keys = [args.model_key]
    else:
        print("[ERROR] Must specify --model-key or --all-models")
        return
    
    # Determine visualization types
    viz_types = ["tsne", "umap"] if args.viz_type == "all" else [args.viz_type]
    
    # Generate plots
    for model_key in model_keys:
        output_dir = PLOTS_DIR / model_key / f"fullrun{args.full_run}_global_all"
        
        for viz_type in viz_types:
            # Skip UMAP if not available
            if viz_type == "umap" and not UMAP_AVAILABLE:
                print(f"[WARNING] UMAP not available, skipping. Install with: pip install umap-learn")
                continue
            
            # 2D visualization
            print(f"\n{'='*80}")
            print(f"Model: {model_key}, Viz: {viz_type.upper()}, Dims: 2D")
            print(f"{'='*80}")
            create_visualization(model_key, args.llm_key, args.full_run, output_dir, 
                               viz_type=viz_type, n_dims=2, interactive=False, top_n=args.top_n)
            
            # 3D visualization if requested
            if args.plot_3d:
                print(f"\n{'='*80}")
                print(f"Model: {model_key}, Viz: {viz_type.upper()}, Dims: 3D")
                print(f"{'='*80}")
                create_visualization(model_key, args.llm_key, args.full_run, output_dir,
                                   viz_type=viz_type, n_dims=3, interactive=False, top_n=args.top_n)
            
            # Interactive 3D if requested
            if args.interactive:
                if not PLOTLY_AVAILABLE:
                    print(f"[WARNING] Plotly not available. Install with: pip install plotly")
                else:
                    print(f"\n{'='*80}")
                    print(f"Model: {model_key}, Viz: {viz_type.upper()}, Interactive 3D")
                    print(f"{'='*80}")
                    create_visualization(model_key, args.llm_key, args.full_run, output_dir,
                                       viz_type=viz_type, n_dims=3, interactive=True, top_n=args.top_n)


if __name__ == "__main__":
    main()
