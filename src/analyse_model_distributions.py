#!/usr/bin/env python3
"""
Analyze how model-generated texts relate to human texts.

1. Visualizes distributions (t-SNE/UMAP) of human vs generated texts
2. Performs clustering analysis
3. Runs classification experiments (human vs generated, model vs model)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

def load_embeddings(npz_path):
    """Load embeddings from npz file."""
    data = np.load(npz_path)
    return data['embeddings'], data['labels']

def create_labels_dataframe(labels):
    """Parse labels and create a structured dataframe."""
    records = []
    
    for i, label in enumerate(labels):
        parts = label.split('_')
        
        if label.startswith('run'):
            # Generated text: run1_p1_simple or author_run1_p1_simple
            record = {
                'index': i,
                'label': label,
                'type': 'generated',
                'author_id': parts[0] if not parts[0].startswith('run') else None,
                'category': None
            }
        else:
            # Training text: author_category
            record = {
                'index': i,
                'label': label,
                'type': 'human',
                'author_id': parts[0],
                'category': parts[1] if len(parts) > 1 else None
            }
        
        records.append(record)
    
    return pd.DataFrame(records)

def visualize_distributions(embeddings, labels_df, output_dir, method='tsne', perplexity=30):
    """Create 2D visualization of embeddings."""
    print(f"[visualize] Creating {method.upper()} projection...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        print(f"Method {method} not available, using PCA")
        reducer = PCA(n_components=2, random_state=42)
    
    coords = reducer.fit_transform(embeddings)
    labels_df['x'] = coords[:, 0]
    labels_df['y'] = coords[:, 1]
    
    # Plot 1: Human vs Generated
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for doc_type, color in [('human', 'blue'), ('generated', 'red')]:
        mask = labels_df['type'] == doc_type
        axes[0].scatter(labels_df.loc[mask, 'x'], labels_df.loc[mask, 'y'], 
                       c=color, label=doc_type.capitalize(), alpha=0.6, s=50)
    
    axes[0].set_xlabel(f'{method.upper()} Dimension 1')
    axes[0].set_ylabel(f'{method.upper()} Dimension 2')
    axes[0].set_title('Human vs Generated Texts')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Per-author view (sample of authors)
    sample_authors = labels_df['author_id'].dropna().unique()[:10]  # First 10 authors
    
    for author_id in sample_authors:
        author_mask = labels_df['author_id'] == author_id
        axes[1].scatter(labels_df.loc[author_mask, 'x'], labels_df.loc[author_mask, 'y'],
                       label=author_id, alpha=0.6, s=30)
    
    axes[1].set_xlabel(f'{method.upper()} Dimension 1')
    axes[1].set_ylabel(f'{method.upper()} Dimension 2')
    axes[1].set_title('Sample Authors (Human + Generated)')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / f'distribution_{method}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[visualize] Saved plot to {plot_file}")
    plt.close()
    
    return labels_df

def clustering_analysis(embeddings, labels_df, output_dir):
    """Perform clustering analysis."""
    print("[clustering] Running clustering analysis...")
    
    # KMeans with k=2 (human vs generated)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Map cluster labels to majority type
    cluster_0_human = (labels_df[cluster_labels == 0]['type'] == 'human').sum()
    cluster_1_human = (labels_df[cluster_labels == 1]['type'] == 'human').sum()
    
    if cluster_0_human > cluster_1_human:
        predicted_type = ['human' if c == 0 else 'generated' for c in cluster_labels]
    else:
        predicted_type = ['generated' if c == 0 else 'human' for c in cluster_labels]
    
    # Compute accuracy
    accuracy = (labels_df['type'] == predicted_type).mean()
    
    # Silhouette score
    silhouette = silhouette_score(embeddings, cluster_labels)
    
    print(f"[clustering] KMeans (k=2) clustering accuracy: {accuracy:.3f}")
    print(f"[clustering] Silhouette score: {silhouette:.3f}")
    
    # HDBSCAN
    try:
        from sklearn.cluster import HDBSCAN as HDBSCAN_sklearn
        hdbscan_available = True
    except ImportError:
        hdbscan_available = False
    
    if hdbscan_available:
        print("[clustering] Running HDBSCAN...")
        clusterer = HDBSCAN_sklearn(min_cluster_size=5, min_samples=3)
        hdbscan_labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
        n_noise = list(hdbscan_labels).count(-1)
        
        print(f"[clustering] HDBSCAN found {n_clusters} clusters, {n_noise} noise points")
    
    # Save results
    results = {
        'method': 'KMeans',
        'n_clusters': 2,
        'accuracy': accuracy,
        'silhouette_score': silhouette
    }
    
    results_df = pd.DataFrame([results])
    results_file = output_dir / 'clustering_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"[clustering] Saved results to {results_file}")
    
    return results

def classification_human_vs_generated(embeddings, labels_df, output_dir):
    """Classify human vs generated texts."""
    print("[classification] Running human vs generated classification...")
    
    X = embeddings
    y = (labels_df['type'] == 'generated').astype(int)
    
    # Logistic Regression with cross-validation
    lr = LogisticRegression(max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
    
    print(f"[classification] Logistic Regression CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    
    # Train on full data and get predictions
    lr.fit(X, y)
    y_pred = lr.predict(X)
    
    # Classification report
    print("\n[classification] Classification Report:")
    print(classification_report(y, y_pred, target_names=['Human', 'Generated']))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'Generated'],
                yticklabels=['Human', 'Generated'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Human vs Generated Classification')
    
    plt.tight_layout()
    plot_file = output_dir / 'confusion_matrix_human_vs_generated.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[classification] Saved confusion matrix to {plot_file}")
    plt.close()
    
    # Save results
    results = {
        'classifier': 'LogisticRegression',
        'cv_accuracy_mean': scores.mean(),
        'cv_accuracy_std': scores.std(),
        'train_accuracy': accuracy_score(y, y_pred)
    }
    
    results_df = pd.DataFrame([results])
    results_file = output_dir / 'classification_human_vs_generated.csv'
    results_df.to_csv(results_file, index=False)
    print(f"[classification] Saved results to {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze model-generated vs human text distributions')
    parser.add_argument('--model-key', default='luar_mud_orig', help='Embedding model key')
    parser.add_argument('--llm-key', required=True, help='LLM key (e.g., deepseek-reasoner)')
    parser.add_argument('--full-run', type=int, default=1, help='Full run number')
    parser.add_argument('--prompt-variant', default='simple', choices=['simple', 'complex'])
    parser.add_argument('--viz-method', default='tsne', choices=['tsne', 'umap', 'pca'])
    parser.add_argument('--output-dir', default='data/model_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    llm_output_dir = output_dir / f"{args.llm_key}_fullrun{args.full_run}_{args.prompt_variant}"
    llm_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[model_analysis] Analyzing {args.llm_key}")
    print(f"[model_analysis] Model: {args.model_key}, Run: {args.full_run}, Variant: {args.prompt_variant}")
    
    # Load embeddings
    npz_path = Path(f"data/embeddings/{args.model_key}_{args.llm_key}_fullrun{args.full_run}_{args.prompt_variant}.npz")
    
    if not npz_path.exists():
        print(f"ERROR: Embeddings not found at {npz_path}")
        print(f"Please run embed_generated_texts.py first")
        return
    
    embeddings, labels = load_embeddings(npz_path)
    print(f"[model_analysis] Loaded {len(embeddings)} embeddings")
    
    # Create labels dataframe
    labels_df = create_labels_dataframe(labels)
    print(f"[model_analysis] Human texts: {(labels_df['type'] == 'human').sum()}")
    print(f"[model_analysis] Generated texts: {(labels_df['type'] == 'generated').sum()}")
    
    # 1. Visualize distributions
    labels_df = visualize_distributions(embeddings, labels_df, llm_output_dir, 
                                       method=args.viz_method)
    
    # 2. Clustering analysis
    clustering_results = clustering_analysis(embeddings, labels_df, llm_output_dir)
    
    # 3. Classification: Human vs Generated
    classification_results = classification_human_vs_generated(embeddings, labels_df, llm_output_dir)
    
    print(f"\n[model_analysis] Analysis complete! Results saved to {llm_output_dir}")

if __name__ == '__main__':
    main()
