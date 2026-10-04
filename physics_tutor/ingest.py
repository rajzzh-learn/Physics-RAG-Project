"""
PDF ingestion pipeline.
Loads all PDFs from configured directories, splits them into chunks,
and persists a ChromaDB vector store for retrieval.
"""
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # load .env for local runs

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from physics_tutor.config import (
    PDF_DIRS,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_pdfs(pdf_dirs: list[Path]) -> list:
    """Load every PDF found under the given directories."""
    docs = []
    for directory in pdf_dirs:
        if not directory.exists():
            logger.warning("Directory not found, skipping: %s", directory)
            continue
        for pdf_path in sorted(directory.glob("*.pdf")):
            logger.info("Loading: %s", pdf_path.name)
            loader = PyPDFLoader(str(pdf_path))
            docs.extend(loader.load())
    logger.info("Total pages loaded: %d", len(docs))
    return docs


def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Build (or load) the ChromaDB vector store.
    - On Streamlit Cloud: the pre-built vectorstore/ folder must be committed
      to the repo. Run `python -m physics_tutor.ingest` locally first, then
      commit the vectorstore/ directory before deploying.
    - Locally: set force_rebuild=True to re-ingest all PDFs from scratch.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if VECTOR_STORE_DIR.exists() and not force_rebuild:
        logger.info("Loading existing vector store from %s", VECTOR_STORE_DIR)
        return Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings,
        )

    logger.info("Building vector store — this may take a few minutes …")
    raw_docs = load_pdfs(PDF_DIRS)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info("Total chunks: %d", len(chunks))

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )
    logger.info("Vector store saved to %s", VECTOR_STORE_DIR)
    return vector_store


if __name__ == "__main__":
    build_vector_store(force_rebuild=True)
