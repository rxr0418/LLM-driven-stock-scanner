"""
embeddings.py - Embedding generation for pgvector RAG.

Uses OpenAI text-embedding-3-small (1536-dim) via the openai client.
Supabase pgvector stores and retrieves via cosine distance (<->).

Setup:
  pip install openai
  Set OPENAI_API_KEY environment variable.

  In Supabase SQL editor, run once per table:
    ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS embedding vector(1536);
    ALTER TABLE swing_results ADD COLUMN IF NOT EXISTS embedding vector(1536);
    CREATE INDEX IF NOT EXISTS knowledge_embedding_idx
      ON knowledge USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

import os
from typing import Optional

from config import EMBEDDING_MODEL, EMBEDDING_DIM


def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for the given text.
    Returns a list of EMBEDDING_DIM floats.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client   = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model = EMBEDDING_MODEL,
        input = text.replace("\n", " "),
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts in a single API call (cheaper)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client   = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model = EMBEDDING_MODEL,
        input = [t.replace("\n", " ") for t in texts],
    )
    return [item.embedding for item in response.data]


def build_knowledge_query(
    ticker: str,
    signal_direction: str,
    regime: str,
    catalyst_type: Optional[str] = None,
) -> str:
    """
    Build a natural-language query string for semantic knowledge retrieval.
    The embedding of this string is compared against stored knowledge embeddings.
    """
    parts = [
        f"{regime} market regime",
        f"{signal_direction} signal",
        f"ticker {ticker}",
    ]
    if catalyst_type and catalyst_type not in ("UNKNOWN", "OTHER"):
        parts.append(f"catalyst: {catalyst_type.replace('_', ' ').lower()}")
    return " | ".join(parts)
