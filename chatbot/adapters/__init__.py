"""Adapters package - Service adapters for external APIs"""
from chatbot.adapters.llm_adapter import LLMAdapter
from chatbot.adapters.embedding_adapter import EmbeddingAdapter

__all__ = ["LLMAdapter", "EmbeddingAdapter"]
