"""
Central configuration for the Physics Tutor RAG Agent.
Reads secrets from Streamlit Cloud (st.secrets) when available,
falls back to environment variables for local development.
"""
import os
from pathlib import Path


def _get(key: str, default: str = "") -> str:
    """
    Resolve a config value:
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
]
VECTOR_STORE_DIR = BASE_DIR / "physics_tutor" / "vectorstore"

# ── Embedding & LLM ────────────────────────────────────────────────────────
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

LLM_PROVIDER = _get("LLM_PROVIDER", "openai")   # "openai" | "watsonx"

# OpenAI
OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o")

# IBM watsonx.ai
WATSONX_API_KEY = _get("WATSONX_API_KEY")
WATSONX_PROJECT_ID = _get("WATSONX_PROJECT_ID")
WATSONX_URL = _get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_MODEL = _get("WATSONX_MODEL", "ibm/granite-3-3-8b-instruct")

# ── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVER_TOP_K = int(_get("RETRIEVER_TOP_K", "6"))
CHUNK_SIZE = int(_get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "120"))
