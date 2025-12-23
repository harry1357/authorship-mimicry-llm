#!/usr/bin/env python3
"""
Diagnostic script to check embedding structures and detect issues.
"""
import numpy as np
from pathlib import Path
from generation_config import EMBEDDINGS_DIR

def check_embeddings():
    """Check if embeddings are properly structured."""
    
    model_key = "luar_mud_orig"
    llm_key = "gpt-5.2-2025-12-11"
    full_run = 1
    
    # Pick first author from directory
    real_dir = EMBEDDINGS_DIR / model_key
    simple_dir = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "simple" / f"fullrun{full_run}"
    complex_dir = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "complex" / f"fullrun{full_run}"
    
    # Get first author
    author_files = list(simple_dir.glob("*.npz"))
    if not author_files:
        print("[ERROR] No generated embeddings found")
        return
    
    author_id = author_files[0].stem
    print(f"[INFO] Checking author: {author_id}\n")
    
    # Load real embeddings
    real_path = real_dir / f"{author_id}.npz"
    real_data = np.load(real_path, allow_pickle=True)
    real_embs = real_data["embeddings"][:6]  # First 6
    
    # Load generated embeddings
    simple_path = simple_dir / f"{author_id}.npz"
    simple_data = np.load(simple_path, allow_pickle=True)
    simple_embs = simple_data["embeddings"]
    
    complex_path = complex_dir / f"{author_id}.npz"
    complex_data = np.load(complex_path, allow_pickle=True)
    complex_embs = complex_data["embeddings"]
    
    print("=" * 80)
    print("EMBEDDING SHAPES")
    print("=" * 80)
    print(f"Real (training):   {real_embs.shape}")
    print(f"Simple generated:  {simple_embs.shape}")
    print(f"Complex generated: {complex_embs.shape}")
    
    print("\n" + "=" * 80)
    print("EMBEDDING STATISTICS")
    print("=" * 80)
    print(f"\nReal (training):")
    print(f"  Mean: {real_embs.mean():.6f}")
    print(f"  Std:  {real_embs.std():.6f}")
    print(f"  Min:  {real_embs.min():.6f}")
    print(f"  Max:  {real_embs.max():.6f}")
    
    print(f"\nSimple generated:")
    print(f"  Mean: {simple_embs.mean():.6f}")
    print(f"  Std:  {simple_embs.std():.6f}")
    print(f"  Min:  {simple_embs.min():.6f}")
    print(f"  Max:  {simple_embs.max():.6f}")
    
    print(f"\nComplex generated:")
    print(f"  Mean: {complex_embs.mean():.6f}")
    print(f"  Std:  {complex_embs.std():.6f}")
    print(f"  Min:  {complex_embs.min():.6f}")
    print(f"  Max:  {complex_embs.max():.6f}")
    
    # Check cosine similarity between first training and first generated
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\n" + "=" * 80)
    print("COSINE SIMILARITIES (first doc of each type)")
    print("=" * 80)
    
    real_first = real_embs[0:1]
    simple_first = simple_embs[0:1]
    complex_first = complex_embs[0:1]
    
    sim_real_simple = cosine_similarity(real_first, simple_first)[0, 0]
    sim_real_complex = cosine_similarity(real_first, complex_first)[0, 0]
    sim_simple_complex = cosine_similarity(simple_first, complex_first)[0, 0]
    
    # Baseline: real to real
    if len(real_embs) > 1:
        sim_real_real = cosine_similarity(real_first, real_embs[1:2])[0, 0]
        print(f"Real doc 1 vs Real doc 2:      {sim_real_real:.6f}  (baseline)")
    
    print(f"Real doc 1 vs Simple doc 1:    {sim_real_simple:.6f}")
    print(f"Real doc 1 vs Complex doc 1:   {sim_real_complex:.6f}")
    print(f"Simple doc 1 vs Complex doc 1: {sim_simple_complex:.6f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if sim_real_simple < 0.5 and sim_real_complex < 0.5:
        print("⚠️  WARNING: Generated texts have LOW similarity to training docs!")
        print("    This explains why they cluster separately in t-SNE/UMAP.")
        print("    Possible causes:")
        print("    1. Generated texts are significantly different from real texts")
        print("    2. Embedding model issue")
        print("    3. Wrong texts were embedded")
    elif sim_real_simple > 0.8 and sim_real_complex > 0.8:
        print("✅ Good: Generated texts are similar to training docs")
        print("   The clustering issue might be a visualization artifact")
    else:
        print("⚠️  Moderate similarity - check if this is expected")
    
    # Check if embeddings are normalized
    real_norms = np.linalg.norm(real_embs, axis=1)
    simple_norms = np.linalg.norm(simple_embs, axis=1)
    complex_norms = np.linalg.norm(complex_embs, axis=1)
    
    print("\n" + "=" * 80)
    print("EMBEDDING NORMS (should be ~1.0 if normalized)")
    print("=" * 80)
    print(f"Real:    mean={real_norms.mean():.6f}, std={real_norms.std():.6f}")
    print(f"Simple:  mean={simple_norms.mean():.6f}, std={simple_norms.std():.6f}")
    print(f"Complex: mean={complex_norms.mean():.6f}, std={complex_norms.std():.6f}")
    
    if abs(real_norms.mean() - simple_norms.mean()) > 0.1:
        print("\n⚠️  WARNING: Different norm scales between real and generated!")
        print("   This could cause clustering artifacts")


if __name__ == "__main__":
    check_embeddings()
