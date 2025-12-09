"""
Unified embedding utilities for job + interview pipelines + Streamlit app.

Supports:
- embed_texts(text_list)
- embed_query(single_string)

Uses BGE-Large (1024-dim) or whichever model is set in .env.
"""

from __future__ import annotations
import os
from functools import lru_cache
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

# Read model name from environment or default to BGE-Large
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")

# Will be automatically overwritten once model loads
EMBED_DIM = None  


# ---------------------------------------------------------
# Load Model (cached)
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    global EMBED_DIM

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    EMBED_DIM = model.get_sentence_embedding_dimension()

    print(f"✓ Model loaded ({EMBED_DIM} dimensions)")
    return model


# ---------------------------------------------------------
# Bulk Embeddings
# ---------------------------------------------------------

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using the BGE model.
    Returns ndarray shape: (N, EMBED_DIM)
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    model = load_embedding_model()

    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return emb.astype(np.float32)


# ---------------------------------------------------------
# Query Embedding (required by Streamlit app)
# ---------------------------------------------------------

def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string (used in semantic search in Streamlit).
    Returns ndarray shape: (EMBED_DIM,)
    """
    if not text:
        raise ValueError("embed_query() received empty text")

    model = load_embedding_model()

    emb = model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return emb[0].astype(np.float32)
