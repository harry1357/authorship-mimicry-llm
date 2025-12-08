# src/generation_config.py
"""
Configuration Module for Text Generation Experiments

This module defines all configuration parameters, file paths, and experimental settings
used throughout the authorship mimicry research project. Centralizing configuration
ensures consistency across different pipeline stages and facilitates reproducibility.

Key Configuration Areas:
- Directory structure and file paths
- Generation parameters (temperature, token limits)
- Model identifiers for LLMs and style embeddings
- Reference files for topic assignments and consistency metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


# Project root directory is one level above the src/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main data directories
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = PROJECT_ROOT / "amazon_product_data_corpus_mixed_topics_per_author_reformatted"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CONSISTENCY_DIR = DATA_DIR / "consistency"
PROMPTS_DIR = DATA_DIR / "prompts"
GENERATED_DIR = DATA_DIR / "generated"

# Ensure output directories exist
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Input configuration files
AUTHOR_LIST_FILE = DATA_DIR / "author_ids_consensus_157.txt"
TOPICS_FILE = PROJECT_ROOT / "author_ids_three_training_topics_x_two_two_generation_topics.txt"
AUTHOR_CATEGORIES_FILE = PROJECT_ROOT / "author_ids_review_topics.txt"

# Reference model and consistency metrics
REFERENCE_MODEL_KEY = "luar_crud_orig"
REFERENCE_CONSISTENCY_CSV = CONSISTENCY_DIR / "luar_crud_orig_top100.csv"


@dataclass
class GenerationParams:
    """
    Default parameters for text generation.
    
    Attributes:
        max_tokens: Maximum number of tokens to generate (approximately 800 words)
        temperature: Sampling temperature controlling generation randomness
    """
    max_tokens: int = 1200
    temperature: float = 0.7


# Default generation parameters instance
DEFAULT_GEN_PARAMS = GenerationParams()

# Default LLM identifier (currently using GPT-5.1)
DEFAULT_LLM_KEY = "gpt-5.1"

# Style embedding model identifiers for subsequent stylometric analysis
STYLE_MODEL_KEYS: List[str] = [
    "luar_crud_orig",
    "luar_mud_orig",
    "luar_crud_st",
    "luar_mud_st",
    "style_embedding",
    "star",
]