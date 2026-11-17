import pytest

from trellis_agent.llm import LLMError
from trellis_agent.llm.openai_compat import OpenAICompatClient


def test_from_env_names_every_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(LLMError) as info:
        OpenAICompatClient.from_env()
    assert "LLM_BASE_URL" in str(info.value) and "LLM_API_KEY" in str(info.value)
    assert "LLM_MODEL" not in str(info.value)


async def test_from_env_builds_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    client = OpenAICompatClient.from_env()
    try:
        assert client.model == "some-model" and client.base_url == "http://127.0.0.1:9/v1"
    finally:
        await client.aclose()
