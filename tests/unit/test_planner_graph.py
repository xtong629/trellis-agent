import pytest

from trellis_agent.planner.graph import GraphError, TaskGraph, TaskNode, parse_tasks


def node(id_: str, *deps: str, kind: str = "default") -> TaskNode:
    return TaskNode(id=id_, goal=f"do {id_}", kind=kind, depends_on=tuple(deps))


# ----------------------------------------------------------------- structure
def test_a_valid_graph_validates_and_keeps_declaration_order() -> None:
    graph = TaskGraph([node("c", "a", "b"), node("a"), node("b", "a")])
    graph.validate()
    assert graph.ids() == ["c", "a", "b"]
    assert graph.get("b").depends_on == ("a",)
    assert len(graph) == 3 and "a" in graph


def test_ready_returns_nodes_whose_dependencies_are_all_done() -> None:
    graph = TaskGraph([node("a"), node("b", "a"), node("c", "a"), node("d", "b", "c")])
    assert [n.id for n in graph.ready(set())] == ["a"]
    assert [n.id for n in graph.ready({"a"})] == ["b", "c"]
    assert [n.id for n in graph.ready({"a", "b"})] == ["c"]  # d still waits for c; b is done
    assert [n.id for n in graph.ready({"a", "b", "c"})] == ["d"]
    assert graph.ready({"a", "b", "c", "d"}) == []


def test_dependents_are_transitive() -> None:
    graph = TaskGraph([node("a"), node("b", "a"), node("c", "b"), node("x")])
    assert graph.dependents("a") == {"b", "c"}
    assert graph.dependents("b") == {"c"}
    assert graph.dependents("x") == set()


# ----------------------------------------------------------------- rejections
def problems_of(graph: TaskGraph, **kw: object) -> list[str]:
    with pytest.raises(GraphError) as info:
        graph.validate(**kw)  # type: ignore[arg-type]
    return info.value.problems


def test_empty_graph_is_rejected() -> None:
    assert any("at least one" in p for p in problems_of(TaskGraph([])))


def test_duplicate_and_blank_ids_are_rejected() -> None:
    assert any(
        "duplicate" in p and "'a'" in p for p in problems_of(TaskGraph([node("a"), node("a")]))
    )
    assert any("id" in p for p in problems_of(TaskGraph([node(" ")])))


def test_unknown_and_self_dependencies_are_rejected() -> None:
    found = problems_of(TaskGraph([node("a", "ghost"), node("b", "b")]))
    assert any("'a'" in p and "unknown task 'ghost'" in p for p in found)
    assert any("'b'" in p and "itself" in p for p in found)


def test_cycles_are_rejected_and_the_message_names_the_tasks() -> None:
    found = problems_of(TaskGraph([node("a", "c"), node("b", "a"), node("c", "b"), node("ok")]))
    (cycle,) = [p for p in found if "cycle" in p]
    assert all(t in cycle for t in ("a", "b", "c")) and "ok" not in cycle


def test_two_node_cycle() -> None:
    assert any("cycle" in p for p in problems_of(TaskGraph([node("a", "b"), node("b", "a")])))


def test_size_limit() -> None:
    graph = TaskGraph([node(f"t{i}") for i in range(5)], max_nodes=4)
    assert any("at most 4" in p for p in problems_of(graph))


def test_kinds_must_be_registered_workers() -> None:
    graph = TaskGraph([node("a", kind="research"), node("b", kind="wizard")])
    graph.validate(known_kinds={"research", "wizard"})
    found = problems_of(graph, known_kinds={"research", "compute"})
    assert any(
        "'b'" in p and "kind 'wizard'" in p and "compute" in p and "research" in p for p in found
    )


def test_all_problems_are_reported_together_so_a_planner_can_fix_them_in_one_go() -> None:
    graph = TaskGraph([node("a", "ghost"), node("a"), node("b", "b", kind="nope")])
    with pytest.raises(GraphError) as info:
        graph.validate(known_kinds={"default"})
    assert len(info.value.problems) >= 4
    assert str(info.value) == "; ".join(info.value.problems)


# ----------------------------------------------------------------- parsing LLM output
def test_parse_tasks_builds_a_graph_and_applies_defaults() -> None:
    graph = parse_tasks(
        [
            {"id": "a", "goal": "find it", "kind": "research"},
            {"id": "b", "goal": "add", "depends_on": ["a"], "note": "extra keys are ignored"},
            {"id": "c", "goal": "tidy", "depends_on": None},
        ]
    )
    assert [(n.id, n.kind, n.depends_on) for n in graph.nodes] == [
        ("a", "research", ()),
        ("b", "default", ("a",)),
        ("c", "default", ()),
    ]


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ("not a list", "list"),
        ([], "at least one"),
        (["x"], "object"),
        ([{"goal": "g"}], "'id'"),
        ([{"id": "a"}], "'goal'"),
        ([{"id": "a", "goal": "g", "depends_on": "b"}], "depends_on"),
        ([{"id": "a", "goal": "g", "depends_on": [1]}], "depends_on"),
        ([{"id": "a", "goal": "g", "kind": 5}], "kind"),
    ],
)
def test_parse_tasks_reports_shape_problems(raw: object, fragment: str) -> None:
    with pytest.raises(GraphError, match=fragment):
        parse_tasks(raw)
