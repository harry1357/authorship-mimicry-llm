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
    "luar_crud_orig": {
        "hf_name": "rrivera1849/LUAR-CRUD",
        "family": "luar_orig",      # handled via transformers + custom episode batching
        "max_length": 256,
        "batch_size": 8,
    },
    "luar_mud_orig": {
        "hf_name": "rrivera1849/LUAR-MUD",
        "family": "luar_orig",
        "max_length": 256,
        "batch_size": 8,
    },

    # Sentence-transformers versions of LUAR
    "luar_crud_st": {
        "hf_name": "gabrielloiseau/LUAR-CRUD-sentence-transformers",  # sentence-transformers API [oai_citation:0‡Hugging Face](https://huggingface.co/gabrielloiseau/LUAR-CRUD-sentence-transformers)
        "family": "sentence_transformers",
        "batch_size": 64,  # Increased for better throughput on MPS/CUDA
    },
    "luar_mud_st": {
        "hf_name": "gabrielloiseau/LUAR-MUD-sentence-transformers",   # sentence-transformers API [oai_citation:1‡Hugging Face](https://huggingface.co/gabrielloiseau/LUAR-MUD-sentence-transformers)
        "family": "sentence_transformers",
        "batch_size": 64,  # Increased for better throughput on MPS/CUDA
    },

    # Style-Embedding (SentenceTransformer style model) [oai_citation:2‡GitHub](https://github.com/nlpsoc/Style-Embeddings)
    "style_embedding": {
        "hf_name": "AnnaWegmann/Style-Embedding",
        "family": "sentence_transformers",
        "batch_size": 64,  # Increased for better throughput on MPS/CUDA
    },

    # STAR: Style Transformer for Authorship Representations (uses pooler_output) [oai_citation:3‡Hugging Face](https://huggingface.co/AIDA-UPM/star?utm_source=chatgpt.com)
    "star": {
        "hf_name": "AIDA-UPM/star",
        "tokenizer_name": "roberta-large",
        "family": "star",
        "max_length": 512,
        "batch_size": 8,
    },
}

# TSNE + PCA hyperparameters (kept identical across models)
TSNE_RANDOM_STATE = 42
TSNE_PERPLEXITY = 30.0
TSNE_N_COMPONENTS = 2
PCA_N_COMPONENTS = 50