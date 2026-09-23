"""
rag/vector_store.py
─────────────────────────────────────────────────────────────────────────────
RAG (Retrieval-Augmented Generation) layer for the Investment Analyst Agent.

Responsibilities:
  - Ingest PDF/text financial reports into a persistent ChromaDB vector store.
  - Provide a retriever that the LangChain agent can query at runtime.
  - Expose a @tool-decorated function so the agent can trigger document search.

Usage:
    from rag.vector_store import get_retriever_tool, ingest_documents
    ingest_documents(["./data/sample_reports/apple_10k.pdf"])
    tool = get_retriever_tool()
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton vector store
# ─────────────────────────────────────────────────────────────────────────────
_vector_store: Chroma | None = None
_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> Chroma:
    """Return (or create) the persistent Chroma vector store."""
    global _vector_store
    if _vector_store is None:
        persist_dir = settings.CHROMA_PERSIST_DIR
        os.makedirs(persist_dir, exist_ok=True)
        logger.info("Opening ChromaDB at: %s", persist_dir)
        _vector_store = Chroma(
            collection_name="financial_reports",
            embedding_function=_get_embeddings(),
            persist_directory=persist_dir,
        )
    return _vector_store


def ingest_documents(file_paths: List[str]) -> int:
    """
    Ingest one or more PDF or text files into the vector store.

    Args:
        file_paths: List of absolute or relative paths to documents.

    Returns:
        Number of document chunks added.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    store = get_vector_store()
    total_added = 0

    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning("File not found, skipping: %s", path)
            continue
        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8")

            raw_docs = loader.load()
            chunks = splitter.split_documents(raw_docs)
            # Tag each chunk with source metadata
            for chunk in chunks:
                chunk.metadata["source_file"] = path.name
            store.add_documents(chunks)
            total_added += len(chunks)
            logger.info("Ingested %d chunks from %s", len(chunks), path.name)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", path, exc)

    return total_added


def search_financial_documents(query: str, k: int | None = None) -> str:
    """
    Search the RAG knowledge base for passages relevant to the query.

    Args:
        query: Natural-language question or keyword string.
        k: Number of passages to return (default from settings).

    Returns:
        Concatenated passage texts with source attribution.
    """
    store = get_vector_store()
    k = k or settings.RAG_TOP_K
    try:
        docs = store.similarity_search(query, k=k)
        if not docs:
            return "No relevant documents found in the knowledge base."
        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"[{i}] Source: {src} (page {page})\n{doc.page_content}")
        return "\n\n".join(parts)
    except Exception as exc:
        logger.error("RAG search failed: %s", exc)
        return f"Document search error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# LangChain Tool wrapper
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_reports(query: str) -> str:
    """
    Search the internal knowledge base of ingested financial reports and
    company filings (10-K, 10-Q, annual reports, earnings transcripts).

    Use this tool when you need specific figures, management commentary,
    risk factors, or forward-looking statements from official documents.

    Args:
        query: The question or topic to search for, e.g.
               'Apple revenue breakdown by segment 2023'.

    Returns:
        Relevant excerpts from stored documents.
    """
    return search_financial_documents(query)
