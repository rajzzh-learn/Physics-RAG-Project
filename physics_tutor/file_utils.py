"""
Utility helpers for processing file attachments uploaded via the Streamlit UI.

Supported attachment types:
  - PDF  -> extracted plain text via pypdf
  - TXT  -> raw decoded text
  - Image (PNG / JPG / JPEG / WEBP / GIF) -> base64-encoded data URI for vision LLMs
"""
from __future__ import annotations

import base64
import io
from typing import Tuple

# -- PDF extraction ---------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes, max_chars: int = 6000) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(t.strip() for t in pages_text if t.strip())
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + f"\n\n[truncated at {max_chars} chars]"
    return full_text

# -- Plain-text extraction --------------------------------------------------

def extract_text_from_txt(file_bytes: bytes, max_chars: int = 6000) -> str:
    text = file_bytes.decode("utf-8", errors="replace").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated at {max_chars} chars]"
    return text

# -- Image -> base64 data URI -----------------------------------------------

_MIME_MAP = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
             "webp": "image/webp", "gif": "image/gif"}

SUPPORTED_IMAGE_EXTS = set(_MIME_MAP.keys())
SUPPORTED_DOC_EXTS   = {"pdf", "txt"}
SUPPORTED_EXTS       = SUPPORTED_IMAGE_EXTS | SUPPORTED_DOC_EXTS

def image_to_base64_uri(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    ext  = filename.rsplit(".", 1)[-1].lower()
    mime = _MIME_MAP.get(ext, "image/png")
    b64  = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}", mime

def is_image(filename: str) -> bool:
    return filename.rsplit(".", 1)[-1].lower() in SUPPORTED_IMAGE_EXTS
