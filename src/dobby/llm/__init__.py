"""Factory that hands the agent an LLMClient based on config.

The agent never imports a provider SDK — it asks here.
"""
from __future__ import annotations

from dobby.config import LLM_PROVIDER
from dobby.llm.base import LLMClient


def get_client(provider: str | None = None) -> LLMClient:
    name = (provider or LLM_PROVIDER).lower()
    if name == "anthropic":
        from dobby.llm.anthropic_client import AnthropicClient

        return AnthropicClient()
    raise ValueError(f"unknown LLM provider: {name!r}")
