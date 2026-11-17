"""LLM protocols, deterministic mocks and (see ``openai_compat``) a real client."""

from .base import Embedder, LLMClient, LLMError, LLMResponse
from .mock import CallRecord, HashEmbedder, RuleLLM, ScriptedLLM, ScriptExhausted

__all__ = [
    "CallRecord",
    "Embedder",
    "HashEmbedder",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "RuleLLM",
    "ScriptExhausted",
    "ScriptedLLM",
]
