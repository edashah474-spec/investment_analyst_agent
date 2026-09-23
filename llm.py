"""
llm.py — Local Granite LLM via Ollama (zero cost, no API key required).

Ollama runs IBM Granite models on your own machine.
Install Ollama: https://ollama.com/download
Pull the model: ollama pull granite3.3:8b

Provides:
  - get_granite_llm()    → LangChain-compatible ChatOllama instance
  - get_granite_direct() → raw OllamaLLM for non-chat prompt calls
  - generate_text()      → convenience wrapper for direct text generation
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_ollama import ChatOllama, OllamaLLM

from config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_granite_llm() -> ChatOllama:
    """
    Return a LangChain ChatOllama instance backed by a local IBM Granite model.

    Used by the ReAct agent for tool-calling and structured reasoning.
    Ollama must be running locally (default: http://localhost:11434).
    """
    logger.info(
        "Initialising ChatOllama — model: %s  base_url: %s",
        settings.OLLAMA_MODEL,
        settings.OLLAMA_BASE_URL,
    )
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        num_predict=settings.LLM_MAX_NEW_TOKENS,
    )


@lru_cache(maxsize=1)
def get_granite_direct() -> OllamaLLM:
    """
    Return a raw OllamaLLM for direct prompt → text generation calls
    (used by sentiment analysis and recommendation steps).
    """
    logger.info(
        "Initialising OllamaLLM — model: %s  base_url: %s",
        settings.OLLAMA_MODEL,
        settings.OLLAMA_BASE_URL,
    )
    return OllamaLLM(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        num_predict=settings.LLM_MAX_NEW_TOKENS,
    )


def generate_text(prompt: str, max_tokens: int | None = None) -> str:
    """
    Generate text from a plain prompt string using the local Ollama model.

    Args:
        prompt: Full prompt to send.
        max_tokens: Optional per-call token override.

    Returns:
        Generated text string.
    """
    if max_tokens is not None:
        # Create a one-off instance with the overridden token limit
        model = OllamaLLM(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            num_predict=max_tokens,
        )
    else:
        model = get_granite_direct()
    response = model.invoke(prompt)
    return response.strip() if isinstance(response, str) else str(response)
