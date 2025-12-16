#!/usr/bin/env python3
"""
Analyze the difference between:
1. Clustering quality (how close generated docs are to EACH OTHER)
2. Mimicry quality (how close generated docs are to TRAINING CENTROID)

This explains why a lower-ranked author might show tighter visual clustering
but actually has worse mimicry performance.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, cdist
from tqdm import tqdm

from generation_config import EMBEDDINGS_DIR, CONSISTENCY_DIR


def load_author_embeddings(author_id, model_key, llm_key, full_run):
    """Load training + generated embeddings."""
    # Training
    train_path = EMBEDDINGS_DIR / model_key / f"{author_id}.npz"
    if not train_path.exists():
        return None
    
    train_data = np.load(train_path, allow_pickle=True)
    if 'selected_indices' in train_data:
        training_embs = train_data['embeddings'][train_data['selected_indices']]
    else:
        training_embs = train_data['embeddings'][:6]
    
    # Simple
    simple_path = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "simple" / f"fullrun{full_run}" / f"{author_id}.npz"
    if not simple_path.exists():
        return None
    simple_embs = np.load(simple_path, allow_pickle=True)['embeddings']
    
    # Complex
    complex_path = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "complex" / f"fullrun{full_run}" / f"{author_id}.npz"
    if not complex_path.exists():
        return None
    complex_embs = np.load(complex_path, allow_pickle=True)['embeddings']
    
    return {
        'training': training_embs,
        'simple': simple_embs,
        'complex': complex_embs,
    }


def analyze_clustering_vs_mimicry(author_data):
    """
    Compute two different metrics:
    1. Clustering: Average distance between generated docs (consistency)
    2. Mimicry: Average distance from generated docs to training centroid (accuracy)
    """
    training_centroid = np.mean(author_data['training'], axis=0, keepdims=True)
    
    results = {}
    
    for doc_type in ['simple', 'complex']:
        gen_docs = author_data[doc_type]
        
        # MIMICRY: Distance to training centroid (what ranking uses)
        mimicry_dists = cdist(gen_docs, training_centroid, metric='cosine').flatten()
        mimicry_score = np.mean(mimicry_dists)
        
        # CLUSTERING: Distance between generated docs (what looks good visually)
        if len(gen_docs) > 1:
            clustering_dists = pdist(gen_docs, metric='cosine')
            clustering_score = np.mean(clustering_dists)
        else:
            clustering_score = 0.0
        
        results[f'{doc_type}_mimicry'] = mimicry_score
        results[f'{doc_type}_clustering'] = clustering_score
    
    # Combined metrics
    results['avg_mimicry'] = (results['simple_mimicry'] + results['complex_mimicry']) / 2
    results['avg_clustering'] = (results['simple_clustering'] + results['complex_clustering']) / 2
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze clustering vs mimicry")
    parser.add_argument("--model-key", type=str, default="luar_crud_orig")
    parser.add_argument("--full-run", type=int, default=1)
    parser.add_argument("--llm-key", type=str, default="gpt-5.1")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--rank-by", type=str, default="average")
    
    args = parser.parse_args()
    
    # Load ranking from CSV
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{args.model_key}_fullrun{args.full_run}.csv"
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Sort by ranking method
    if args.rank_by == "simple":
        df_sorted = df.sort_values('dist_real_centroid_simple')
    elif args.rank_by == "complex":
        df_sorted = df.sort_values('dist_real_centroid_complex')
    elif args.rank_by == "best":
        df['best_mimicry_dist'] = df[['dist_real_centroid_simple', 'dist_real_centroid_complex']].min(axis=1)
        df_sorted = df.sort_values('best_mimicry_dist')
    else:  # average
        df['avg_mimicry_dist'] = (df['dist_real_centroid_simple'] + df['dist_real_centroid_complex']) / 2
        df_sorted = df.sort_values('avg_mimicry_dist')
    
    author_ids = df_sorted.head(args.top_n)['author_id'].tolist()
    
    print(f"\n{'='*100}")
    print(f"Clustering vs Mimicry Analysis: {args.model_key}, Run {args.full_run}")
    print(f"{'='*100}\n")
    
    results = []
    
    for idx, author_id in enumerate(tqdm(author_ids, desc="Analyzing authors")):
        author_data = load_author_embeddings(author_id, args.model_key, args.llm_key, args.full_run)
        
        if author_data is None:
            continue
        
        metrics = analyze_clustering_vs_mimicry(author_data)
        
        results.append({
            'rank': idx + 1,
            'author_id': author_id,
            **metrics
        })
    
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "="*100)
    print("KEY INSIGHT: Clustering (consistency) ≠ Mimicry (accuracy)")
    print("="*100)
    print("\nMIMICRY (Distance to Training Centroid) - WHAT RANKING USES:")
    print("  → Lower = Better (generated docs close to author's style)")
    print("  → This is what A1, A2, A3... labels represent\n")
    print("CLUSTERING (Distance Between Generated Docs) - WHAT LOOKS GOOD VISUALLY:")
    print("  → Lower = Tighter cluster (LLM is consistent)")
    print("  → But consistent ≠ accurate! Can be consistently wrong!\n")
    print("="*100)
    
    print(f"\n{'Rank':<6} {'Author ID':<20} {'Mimicry':<12} {'Clustering':<12} {'Insight'}")
    print("-"*100)
    
    for _, row in results_df.iterrows():
        # Determine if clustering is misleading
        if row['avg_clustering'] < 0.05:
            clustering_desc = "TIGHT"
        elif row['avg_clustering'] < 0.15:
            clustering_desc = "MODERATE"
        else:
            clustering_desc = "LOOSE"
        
        if row['avg_mimicry'] < 0.18:
            mimicry_desc = "EXCELLENT"
        elif row['avg_mimicry'] < 0.20:
            mimicry_desc = "GOOD"
        else:
            mimicry_desc = "FAIR"
        
        # Check for misleading cases
        if row['rank'] > 5 and row['avg_clustering'] < row['avg_mimicry'] * 0.5:
            insight = "⚠️  MISLEADING: Tight clustering but poor mimicry!"
        elif row['rank'] <= 3 and row['avg_clustering'] > 0.15:
            insight = "✓ Looks loose but mimics well (close to training)"
        else:
            insight = "✓ Visual matches ranking"
        
        print(f"{row['rank']:<6} {row['author_id']:<20} "
              f"{row['avg_mimicry']:.4f} ({mimicry_desc:<8})  "
              f"{row['avg_clustering']:.4f} ({clustering_desc:<8})  "
              f"{insight}")
    
    # Detailed breakdown
    print(f"\n{'='*100}")
    print("DETAILED BREAKDOWN:")
    print("="*100)
    
    for _, row in results_df.iterrows():
        print(f"\nRank {row['rank']}: {row['author_id']}")
        print(f"  Simple:  Mimicry={row['simple_mimicry']:.4f}  Clustering={row['simple_clustering']:.4f}")
        print(f"  Complex: Mimicry={row['complex_mimicry']:.4f}  Clustering={row['complex_clustering']:.4f}")
        
        # Ratio analysis
        if row['avg_clustering'] > 0:
            ratio = row['avg_mimicry'] / row['avg_clustering']
            if ratio < 2.0:
                print(f"  → Generated docs cluster tightly ({row['avg_clustering']:.4f}) "
                      f"but are far from training ({row['avg_mimicry']:.4f})")
            else:
                print(f"  → Generated docs are scattered ({row['avg_clustering']:.4f}) "
                      f"but some are close to training ({row['avg_mimicry']:.4f})")
    
    print(f"\n{'='*100}")
    print("CONCLUSION:")
    print("="*100)
    print("If a lower-ranked author shows 'better clustering' in the visualization,")
    print("it means the LLM is CONSISTENT (generates similar outputs) but NOT ACCURATE")
    print("(those outputs don't match the author's actual style).")
    print("")
    print("The ranking is based on ACCURACY (mimicry), not CONSISTENCY (clustering).")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
