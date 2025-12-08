# src/generation_config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


# Project root = one level above src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = PROJECT_ROOT / "amazon_product_data_corpus_mixed_topics_per_author_reformatted"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CONSISTENCY_DIR = DATA_DIR / "consistency"
PROMPTS_DIR = DATA_DIR / "prompts"
GENERATED_DIR = DATA_DIR / "generated"

PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Input files
AUTHOR_LIST_FILE = DATA_DIR / "author_ids_consensus_157.txt"
TOPICS_FILE = PROJECT_ROOT / "author_ids_three_training_topics_x_two_two_generation_topics.txt"
AUTHOR_CATEGORIES_FILE = PROJECT_ROOT / "author_ids_review_topics.txt"  # full category list per author

REFERENCE_MODEL_KEY = "luar_crud_orig"
REFERENCE_CONSISTENCY_CSV = CONSISTENCY_DIR / "luar_crud_orig_top100.csv"


@dataclass
class GenerationParams:
    max_tokens: int = 1200      # ~800 words
    temperature: float = 0.7


DEFAULT_GEN_PARAMS = GenerationParams()

# Default LLM key (we're using GPT-5.1 right now)
DEFAULT_LLM_KEY = "gpt-5.1"

# Style-embedding models we will later use for analysis
STYLE_MODEL_KEYS: List[str] = [
    "luar_crud_orig",
    "luar_mud_orig",
    "luar_crud_st",
    "luar_mud_st",
    "style_embedding",
    "star",
]