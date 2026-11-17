import threading
from typing import Annotated, Any, Literal

import pytest

from trellis_agent.tools import ToolRegistry, tool
from trellis_agent.types import ToolCall


@tool
def add(a: int, b: int = 0) -> int:
    """Add two integers."""
    return a + b


def call(name: str, arguments: dict[str, Any], **kw: Any) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments, **kw)


# ----------------------------------------------------------------- schema generation
def test_schema_from_type_hints_and_docstring() -> None:
    schema = add.schema()
    assert schema.name == "add"
    assert schema.description == "Add two integers."
    assert schema.parameters == {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }


def test_schema_supports_common_types() -> None:
    @tool
    def f(
        s: str,
        x: float,
        flag: bool,
        items: list[int],
        mapping: dict[str, str],
        mode: Literal["fast", "slow"],
        maybe: str | None = None,
        note: Annotated[str, "free text"] = "",
    ) -> str:
        """Everything."""
        return s

    props = f.schema().parameters["properties"]
    assert props["s"] == {"type": "string"}
    assert props["x"] == {"type": "number"}
    assert props["flag"] == {"type": "boolean"}
    assert props["items"] == {"type": "array", "items": {"type": "integer"}}
    assert props["mapping"] == {"type": "object", "additionalProperties": {"type": "string"}}
    assert props["mode"] == {"enum": ["fast", "slow"]}
    assert props["maybe"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert props["note"] == {"type": "string", "description": "free text"}
    assert f.schema().parameters["required"] == ["s", "x", "flag", "items", "mapping", "mode"]


def test_name_and_description_overrides() -> None:
    @tool(name="plus", description="Adds.")
    def f(a: int) -> int:
        return a

    assert (f.name, f.description) == ("plus", "Adds.")


def test_unsupported_annotation_fails_at_definition_time() -> None:
    class Custom: ...

    with pytest.raises(TypeError, match="unsupported"):

        @tool
        def f(x: Custom) -> None: ...


def test_missing_annotation_fails_at_definition_time() -> None:
    with pytest.raises(TypeError, match="annotation"):

        @tool
        def f(x) -> None: ...  # type: ignore[no-untyped-def]


def test_varargs_rejected() -> None:
    with pytest.raises(TypeError, match=r"\*args|\*\*kwargs"):

        @tool
        def f(*args: int) -> None: ...


# ----------------------------------------------------------------- registry
def test_registry_rejects_duplicates_and_bad_names() -> None:
    registry = ToolRegistry([add])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(add)
    with pytest.raises(ValueError, match="name"):

        @tool(name="bad name!")
        def g(a: int) -> int:
            return a

    assert registry.names() == ["add"] and "add" in registry and len(registry) == 1
    assert [s.name for s in registry.schemas()] == ["add"]


# ----------------------------------------------------------------- calling
async def test_call_success_sync_and_async() -> None:
    @tool
    async def shout(text: str) -> str:
        """Upper-case."""
        return text.upper()

    registry = ToolRegistry([add, shout])
    r1 = await registry.call(call("add", {"a": 2, "b": 3}))
    assert (r1.content, r1.is_error, r1.error_type) == ("5", False, None)
    assert r1.latency_ms >= 0
    r2 = await registry.call(call("shout", {"text": "hi"}))
    assert r2.content == "HI"


async def test_non_string_results_are_json_and_keep_unicode() -> None:
    @tool
    def info(city: str) -> dict[str, Any]:
        """Info."""
        return {"city": city, "names": ["东京"]}

    result = await ToolRegistry([info]).call(call("info", {"city": "Tokyo"}))
    assert result.content == '{"city": "Tokyo", "names": ["东京"]}'


async def test_unknown_tool_error_lists_available_tools() -> None:
    result = await ToolRegistry([add]).call(call("nope", {}))
    assert result.is_error and result.error_type == "unknown_tool"
    assert "nope" in result.content and "add" in result.content


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({}, "'a' is a required property"),
        ({"a": "x"}, "is not of type 'integer'"),
        ({"a": True}, "is not of type 'integer'"),  # bool is not an int for the model
        ({"a": 1, "zzz": 2}, "zzz"),  # unexpected extra argument
    ],
)
async def test_invalid_arguments_are_reported_not_executed(
    arguments: dict[str, Any], fragment: str
) -> None:
    ran = []

    @tool(name="spy")
    def spy(a: int) -> int:
        """Spy."""
        ran.append(a)
        return a

    result = await ToolRegistry([spy]).call(call("spy", arguments))
    assert result.is_error and result.error_type == "invalid_arguments"
    assert "spy" in result.content and fragment in result.content
    assert ran == []


async def test_unparseable_arguments_from_model_are_reported() -> None:
    bad = call("add", {}, raw_arguments="{not json", arguments_error="Expecting property name")
    result = await ToolRegistry([add]).call(bad)
    assert result.is_error and result.error_type == "invalid_arguments"
    assert "not valid JSON" in result.content and "{not json" in result.content


async def test_tool_exception_becomes_error_result() -> None:
    @tool
    def boom(a: int) -> int:
        """Always fails."""
        raise ValueError("kaput")

    result = await ToolRegistry([boom]).call(call("boom", {"a": 1}))
    assert result.is_error and result.error_type == "ValueError"
    assert "boom" in result.content and "kaput" in result.content


async def test_long_results_are_truncated_with_marker() -> None:
    @tool
    def big(n: int) -> str:
        """Big."""
        return "x" * n

    result = await ToolRegistry([big], max_result_chars=10).call(call("big", {"n": 50}))
    assert result.content.startswith("x" * 10) and "truncated 40 chars" in result.content
    assert result.is_error is False


async def test_sync_tools_run_off_the_event_loop_thread() -> None:
    @tool
    def where() -> int:
        """Thread id."""
        return threading.get_ident()

    result = await ToolRegistry([where]).call(call("where", {}))
    assert int(result.content) != threading.get_ident()
