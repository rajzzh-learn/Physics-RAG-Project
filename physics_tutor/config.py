"""
Central configuration for the Physics Tutor RAG Agent.
Reads secrets from Streamlit Cloud (st.secrets) when available,
falls back to environment variables for local development.
"""
import os
from pathlib import Path


def get_config(key: str, default: str = "") -> str:
    """
    Resolve a config value dynamically at runtime:
    1. st.secrets  — when running on Streamlit Community Cloud
    2. os.environ  — local .env / shell exports
    3. default     — fallback
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # repo root
PDF_DIRS = [
    BASE_DIR / "Notes",
    BASE_DIR / "Exemplar",
    BASE_DIR / "Chapterwise imp questions",
    BASE_DIR / "PYQ",
    BASE_DIR / "book" / "leph1dd",
    BASE_DIR / "book" / "leph2dd",
    BASE_DIR / "Competency based Questions",
    BASE_DIR / "Secret Assignment",
]
VECTOR_STORE_DIR = BASE_DIR / "physics_tutor" / "vectorstore"

# ── Embedding & LLM ────────────────────────────────────────────────────────
EMBEDDING_MODEL = get_config("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVER_TOP_K = int(get_config("RETRIEVER_TOP_K", "6"))
CHUNK_SIZE = int(get_config("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(get_config("CHUNK_OVERLAP", "120"))
