#!/usr/bin/env python3
"""
Visualize how different LLMs' generated texts are distributed in embedding space.

Creates t-SNE and UMAP plots showing all LLMs' generations together to compare
their stylistic differences.

Usage:
    python src/plot_llm_distribution_comparison.py --model-key luar_mud_orig --full-run 1
    python src/plot_llm_distribution_comparison.py --model-key luar_mud_orig --full-run 1 --prompt simple
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List


def load_all_generated_embeddings(model_key: str, prompt_variant: str, full_run: int, base_path: Path) -> Dict:
    """
    Load generated embeddings from all LLMs + human training texts.
    
    Args:
        prompt_variant: "simple", "complex", or "both" (loads both prompts)
    
    Returns:
        Dictionary with structure:
        {
            'embeddings': np.ndarray,  # (n_total_docs, embedding_dim)
            'llm_labels': List[str],    # LLM name or "Human (training)" for each document
            'author_ids': List[str],    # Author ID for each document
            'llm_names': List[str]      # Unique LLM names + "Human (training)"
        }
    """
    llm_models = [
        "claude-opus-4-5-20251101",
        "deepseek-reasoner",
        "gemini-3-pro-preview",
        "gpt-5.2-2025-12-11",
        "gpt-5.2-pro",
        "grok-4-1-fast-reasoning"
    ]
    
    # Determine which prompts to load
    if prompt_variant == "both":
        prompts_to_load = ["simple", "complex"]
    else:
        prompts_to_load = [prompt_variant]
    
    all_embeddings = []
    llm_labels = []
    author_ids = []
    
    print(f"Loading generated embeddings from all LLMs (prompt: {prompt_variant})...")
    
    for llm_key in llm_models:
        n_docs = 0
        for prompt in prompts_to_load:
            embeddings_dir = (base_path / "data" / "embeddings" / "generated" / 
                             model_key / llm_key / prompt / f"fullrun{full_run}")
            
            if not embeddings_dir.exists():
                print(f"  [SKIP] {llm_key}/{prompt}: directory not found")
                continue
            
            for author_file in sorted(embeddings_dir.glob("*.npz")):
                author_id = author_file.stem
                data = np.load(author_file)
                embeddings = data['embeddings']  # Shape: (2, embedding_dim)
                
                # Add each document
                for emb in embeddings:
                    all_embeddings.append(emb)
                    llm_labels.append(llm_key)
                    author_ids.append(author_id)
                    n_docs += 1
        
        print(f"  ✓ {llm_key}: {n_docs} documents")
    
    # Load human-written training texts
    print("\nLoading human-written training texts...")
    human_embeddings_dir = base_path / "data" / "embeddings" / model_key
    if human_embeddings_dir.exists():
        # Get the set of authors that have generated texts (only load those)
        authors_with_generated = set()
        for prompt in prompts_to_load:
            for llm_key in llm_models:
                embeddings_dir = (base_path / "data" / "embeddings" / "generated" / 
                                 model_key / llm_key / prompt / f"fullrun{full_run}")
                if embeddings_dir.exists():
                    for author_file in embeddings_dir.glob("*.npz"):
                        authors_with_generated.add(author_file.stem)
        
        print(f"  Found {len(authors_with_generated)} authors with generated texts")
        
        n_human = 0
        for author_file in sorted(human_embeddings_dir.glob("*.npz")):
            author_id = author_file.stem
            
            # Only load authors that have generated texts
            if author_id not in authors_with_generated:
                continue
            
            data = np.load(author_file)
            embeddings = data['embeddings']  # Shape: (6, embedding_dim)
            
            # Add all 6 training documents
            for emb in embeddings:
                all_embeddings.append(emb)
                llm_labels.append("Human (training)")
                author_ids.append(author_id)
                n_human += 1
        
        print(f"  ✓ Human (training): {n_human} documents")
    else:
        print(f"  [SKIP] Human training embeddings not found")
    
    if not all_embeddings:
        raise ValueError("No embeddings found for any LLM!")
    
    embeddings_array = np.array(all_embeddings)
    print(f"\nTotal: {len(embeddings_array)} documents from {len(set(llm_labels))} sources")
    
    return {
        'embeddings': embeddings_array,
        'llm_labels': llm_labels,
        'author_ids': author_ids,
        'llm_names': sorted(set(llm_labels))
    }


def create_tsne_projection(embeddings: np.ndarray, perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """Create 2D t-SNE projection."""
    print(f"\nComputing t-SNE projection (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                max_iter=1000, verbose=1)
    projection = tsne.fit_transform(embeddings)
    print("✓ t-SNE complete")
    return projection


def create_tsne_3d_projection(embeddings: np.ndarray, perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """Create 3D t-SNE projection."""
    print(f"\nComputing 3D t-SNE projection (perplexity={perplexity})...")
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=random_state,
                max_iter=1000, verbose=1)
    projection = tsne.fit_transform(embeddings)
    print("✓ 3D t-SNE complete")
    return projection


def create_umap_projection(embeddings: np.ndarray, n_neighbors: int = 15, random_state: int = 42) -> np.ndarray:
    """Create 2D UMAP projection."""
    print(f"\nComputing UMAP projection (n_neighbors={n_neighbors})...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state, verbose=True)
    projection = reducer.fit_transform(embeddings)
    print("✓ UMAP complete")
    return projection


def create_umap_3d_projection(embeddings: np.ndarray, n_neighbors: int = 15, random_state: int = 42) -> np.ndarray:
    """Create 3D UMAP projection."""
    print(f"\nComputing 3D UMAP projection (n_neighbors={n_neighbors})...")
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=random_state, verbose=True)
    projection = reducer.fit_transform(embeddings)
    print("✓ 3D UMAP complete")
    return projection


def plot_2d_comparison(projection: np.ndarray, llm_labels: List[str], llm_names: List[str],
                       title: str, output_path: Path):
    """Create 2D scatter plot comparing all LLMs with centroid labels."""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Create color palette
    colors = sns.color_palette("husl", len(llm_names))
    color_map = dict(zip(llm_names, colors))
    
    # Plot each LLM and compute centroids
    centroids = {}
    for llm_name in llm_names:
        mask = np.array(llm_labels) == llm_name
        points = projection[mask]
        
        ax.scatter(points[:, 0], points[:, 1],
                  alpha=0.6, s=30, label=llm_name,
                  color=color_map[llm_name], edgecolors='white', linewidth=0.5)
        
        # Compute centroid
        centroid = points.mean(axis=0)
        centroids[llm_name] = centroid
        
        # Plot centroid as larger marker
        ax.scatter(centroid[0], centroid[1], 
                  color=color_map[llm_name], s=200, marker='*',
                  edgecolors='black', linewidth=2, zorder=10)
    
    # Add text labels at centroids with nice formatting
    for llm_name, centroid in centroids.items():
        # Shorten label for display
        display_name = llm_name.replace('claude-opus-4-5-20251101', 'Claude Opus 4.5')
        display_name = display_name.replace('deepseek-reasoner', 'DeepSeek R1')
        display_name = display_name.replace('gemini-3-pro-preview', 'Gemini 3 Pro')
        display_name = display_name.replace('gpt-5.2-2025-12-11', 'GPT-5.2')
        display_name = display_name.replace('gpt-5.2-pro', 'GPT-5.2 Pro')
        display_name = display_name.replace('grok-4-1-fast-reasoning', 'Grok 4.1')
        
        ax.annotate(display_name, 
                   xy=(centroid[0], centroid[1]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color_map[llm_name], 
                            alpha=0.7, edgecolor='black', linewidth=1.5),
                   color='white' if llm_name != "Human (training)" else 'black',
                   zorder=11)
    
    ax.set_xlabel('Dimension 1', fontsize=13, fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.legend(title='Source', loc='best', framealpha=0.9, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved 2D plot to: {output_path}")


def plot_3d_interactive(projection: np.ndarray, llm_labels: List[str], author_ids: List[str],
                       llm_names: List[str], title: str, output_path: Path):
    """Create interactive 3D plot using Plotly with better labels."""
    # Create shortened display names
    display_labels = []
    for llm in llm_labels:
        if llm == 'claude-opus-4-5-20251101':
            display_labels.append('Claude Opus 4.5')
        elif llm == 'deepseek-reasoner':
            display_labels.append('DeepSeek R1')
        elif llm == 'gemini-3-pro-preview':
            display_labels.append('Gemini 3 Pro')
        elif llm == 'gpt-5.2-2025-12-11':
            display_labels.append('GPT-5.2')
        elif llm == 'gpt-5.2-pro':
            display_labels.append('GPT-5.2 Pro')
        elif llm == 'grok-4-1-fast-reasoning':
            display_labels.append('Grok 4.1')
        else:
            display_labels.append(llm)
    
    # Create DataFrame for plotly
    df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'z': projection[:, 2],
        'Source': display_labels,
        'Author': author_ids
    })
    
    # Create 3D scatter plot
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='Source',
                       hover_data=['Author'],
                       title=title,
                       color_discrete_sequence=px.colors.qualitative.Set2)
    
    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        height=800,
        font=dict(size=12),
        legend=dict(title='Source', font=dict(size=10))
    )
    
    fig.write_html(output_path)
    print(f"✓ Saved interactive 3D plot to: {output_path}")


def create_all_visualizations(model_key: str, prompt_variant: str, full_run: int, base_path: Path):
    """Create all visualization types."""
    print(f"\n{'='*80}")
    print(f"LLM Distribution Comparison Visualization")
    print(f"{'='*80}")
    print(f"Model: {model_key}")
    print(f"Prompt: {prompt_variant}")
    print(f"Full run: {full_run}\n")
    
    # Load data
    data = load_all_generated_embeddings(model_key, prompt_variant, full_run, base_path)
    embeddings = data['embeddings']
    llm_labels = data['llm_labels']
    author_ids = data['author_ids']
    llm_names = data['llm_names']
    
    # Create output directory
    output_dir = base_path / "data" / "plots" / model_key / "llm_distribution" / prompt_variant / f"fullrun{full_run}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. t-SNE 2D
    tsne_2d = create_tsne_projection(embeddings)
    plot_2d_comparison(
        tsne_2d, llm_labels, llm_names,
        f't-SNE: LLM Generation Distribution ({prompt_variant} prompt)',
        output_dir / "tsne_2d_llm_comparison.png"
    )
    
    # 2. t-SNE 3D interactive
    tsne_3d = create_tsne_3d_projection(embeddings)
    plot_3d_interactive(
        tsne_3d, llm_labels, author_ids, llm_names,
        f't-SNE 3D: LLM Generation Distribution ({prompt_variant} prompt)',
        output_dir / "tsne_3d_llm_comparison.html"
    )
    
    # 3. UMAP 2D
    umap_2d = create_umap_projection(embeddings)
    plot_2d_comparison(
        umap_2d, llm_labels, llm_names,
        f'UMAP: LLM Generation Distribution ({prompt_variant} prompt)',
        output_dir / "umap_2d_llm_comparison.png"
    )
    
    # 4. UMAP 3D interactive
    umap_3d = create_umap_3d_projection(embeddings)
    plot_3d_interactive(
        umap_3d, llm_labels, author_ids, llm_names,
        f'UMAP 3D: LLM Generation Distribution ({prompt_variant} prompt)',
        output_dir / "umap_3d_llm_comparison.html"
    )
    
    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize how different LLMs' generations are distributed in embedding space"
    )
    parser.add_argument("--model-key", type=str, default="luar_mud_orig",
                       help="Embedding model key")
    parser.add_argument("--prompt", type=str, default="simple",
                       choices=["simple", "complex", "both"],
                       help="Prompt variant to visualize (use 'both' to combine simple+complex)")
    parser.add_argument("--full-run", type=int, default=1,
                       help="Full run number")
    parser.add_argument("--all-prompts", action="store_true",
                       help="Create plots for simple, complex, AND both prompts combined")
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    
    if args.all_prompts:
        # Create plots for simple, complex, and combined
        for prompt in ["simple", "complex", "both"]:
            create_all_visualizations(args.model_key, prompt, args.full_run, base_path)
    else:
        create_all_visualizations(args.model_key, args.prompt, args.full_run, base_path)


if __name__ == "__main__":
    main()
