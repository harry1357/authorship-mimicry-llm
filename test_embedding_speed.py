#!/usr/bin/env python
"""Quick test to verify embedding speed improvements."""

import sys
import time
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.embed_authors import load_author_ids, load_reviews_for_author
from src.model_configs import AUTHOR_LIST_FILE, MODEL_CONFIGS
from sentence_transformers import SentenceTransformer

# Test with first 10 authors
author_ids = load_author_ids(AUTHOR_LIST_FILE)[:10]

print("=" * 70)
print("PERFORMANCE TEST: Loading Model Once vs Loading Per Author")
print("=" * 70)

# Get device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"\nDevice: {device}")
print(f"Testing with {len(author_ids)} authors")

# Collect all texts
all_texts = []
for aid in author_ids:
    texts, _ = load_reviews_for_author(aid)
    if texts:
        all_texts.extend(texts[:6])  # First 6 reviews

print(f"Total texts to embed: {len(all_texts)}")

model_name = "AnnaWegmann/Style-Embedding"
batch_size = 32

print(f"\nModel: {model_name}")
print(f"Batch size: {batch_size}")
print("-" * 70)

# Test: Load model once (FAST)
print("\n✅ CORRECT WAY: Load model ONCE, reuse for all authors")
start = time.time()
model = SentenceTransformer(model_name, device=device)
load_time = time.time() - start
print(f"   Model load time: {load_time:.2f}s")

start = time.time()
embeddings = model.encode(all_texts, batch_size=batch_size, show_progress_bar=False)
encode_time = time.time() - start
total_fast = load_time + encode_time
print(f"   Encoding time: {encode_time:.2f}s")
print(f"   Total time: {total_fast:.2f}s")

# Test: Load model per author (SLOW)
print("\n❌ WRONG WAY: Load model for EACH author (old code)")
print("   (Simulating by loading model multiple times)")
start = time.time()
for i in range(len(author_ids)):
    # Reload model each time (what old code was doing)
    model_temp = SentenceTransformer(model_name, device=device)
    # Encode a small batch
    if i == 0:  # Only actually encode once to save time
        _ = model_temp.encode(all_texts[:10], batch_size=batch_size, show_progress_bar=False)
total_slow = time.time() - start

# Estimate full time
estimated_slow = total_slow * (len(all_texts) / 10)
print(f"   Estimated time: {estimated_slow:.2f}s")

print("\n" + "=" * 70)
print(f"SPEEDUP: {estimated_slow / total_fast:.1f}x FASTER with fix!")
print("=" * 70)
print(f"\nFor {len(author_ids)} authors:")
print(f"  Old (buggy):  ~{estimated_slow:.0f}s")
print(f"  New (fixed):  ~{total_fast:.0f}s")
print(f"  Time saved:   ~{estimated_slow - total_fast:.0f}s")
print("\nFor all 2144 authors:")
print(f"  Old (buggy):  ~{estimated_slow * 214:.0f}s ({estimated_slow * 214 / 60:.1f} minutes)")
print(f"  New (fixed):  ~{total_fast * 214:.0f}s ({total_fast * 214 / 60:.1f} minutes)")
print("=" * 70)
