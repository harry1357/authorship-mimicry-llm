# src/model_configs.py
from pathlib import Path

# Project paths (assumes you run scripts from the repo root)
ROOT_DIR = Path(__file__).resolve().parents[1]

CORPUS_DIR = ROOT_DIR / "amazon_product_data_corpus_mixed_topics_per_author_reformatted"
AUTHOR_LIST_FILE = ROOT_DIR / "author_ids_three_training_topics_x_two_two_generation_topics.txt"

DATA_DIR = ROOT_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CONSISTENCY_DIR = DATA_DIR / "consistency"
PLOTS_DIR = DATA_DIR / "plots"

# Make sure folders exist
for d in [EMBEDDINGS_DIR, CONSISTENCY_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model definitions
# model_key values are used as folder names under data/embeddings and data/plots
MODEL_CONFIGS = {
    # Original LUAR checkpoints (episode-based, custom code)
    # Updated to use optimal max_length (512) for fair model comparison
    "luar_crud_orig": {
        "hf_name": "rrivera1849/LUAR-CRUD",
        "family": "luar_orig",      # handled via transformers + custom episode batching
        "max_length": 512,          # ✓ Using full capacity (was 256)
        "batch_size": 32,
    },
    "luar_mud_orig": {
        "hf_name": "rrivera1849/LUAR-MUD",
        "family": "luar_orig",
        "max_length": 512,          # ✓ Using full capacity (was 256)
        "batch_size": 32,
    },

    # Sentence-transformers versions of LUAR
    # Limited to 128 by fine-tuning
    "luar_crud_st": {
        "hf_name": "gabrielloiseau/LUAR-CRUD-sentence-transformers",  # sentence-transformers API
        "family": "sentence_transformers",
        "max_length": 128,          # ✓ Model's maximum
        "batch_size": 32,            # Normalized batch size
    },
    "luar_mud_st": {
        "hf_name": "gabrielloiseau/LUAR-MUD-sentence-transformers",   # sentence-transformers API 
        "family": "sentence_transformers",
        "max_length": 128,          # ✓ Model's maximum
        "batch_size": 32,            # Normalized batch size
    },

    # Style-Embedding (SentenceTransformer style model)
    "style_embedding": {
        "hf_name": "AnnaWegmann/Style-Embedding",
        "family": "sentence_transformers",
        "max_length": 514,          # ✓ Model's maximum (slightly odd number)
        "batch_size": 32,            # Normalized batch size
    },

    # STAR: Style Transformer for Authorship Representations (uses pooler_output)
    "star": {
        "hf_name": "AIDA-UPM/star",
        "tokenizer_name": "roberta-large",
        "family": "star",
        "max_length": 512,          # ✓ Already using full capacity
        "batch_size": 32,
    },
}

# TSNE + PCA hyperparameters (kept identical across models)
TSNE_RANDOM_STATE = 42
TSNE_PERPLEXITY = 30.0
TSNE_N_COMPONENTS = 2
PCA_N_COMPONENTS = 50