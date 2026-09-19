from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph.consts import ACT, AGENT_REASON, GRADE_RESPONSE
from graph.graph import app
from graph.nodes.agent import _resolve_dangling_tool_calls


def test_graph_has_expected_nodes():
    nodes = set(app.get_graph().nodes)
    assert {AGENT_REASON, ACT, GRADE_RESPONSE} <= nodes


def test_resolve_dangling_tool_calls_is_a_noop_on_healthy_history():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {}, "id": "call_1"}],
        ),
        ToolMessage(content="result", name="web_search", tool_call_id="call_1"),
        AIMessage(content="done"),
    ]
    assert _resolve_dangling_tool_calls(messages) == messages


def test_resolve_dangling_tool_calls_patches_an_interrupted_turn():
    # Simulates a checkpoint left behind by a process that died between the
    # agent deciding to call tools and the `act` node actually running.
    dangling_ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "web_search", "args": {}, "id": "call_1"},
            {"name": "generate_diagram", "args": {}, "id": "call_2"},
        ],
    )
    messages = [
        HumanMessage(content="explain ReAct and diagram it"),
        dangling_ai,
        HumanMessage(content="new message on the same corrupted thread"),
    ]

    patched = _resolve_dangling_tool_calls(messages)

    assert patched[0] is messages[0]
    assert patched[1] is dangling_ai
    assert isinstance(patched[2], ToolMessage) and patched[2].tool_call_id == "call_1"
    assert isinstance(patched[3], ToolMessage) and patched[3].tool_call_id == "call_2"
    assert patched[4] is messages[2]
    # Original list (and thus the checkpointed state) is left untouched.
    assert messages == [messages[0], dangling_ai, messages[2]]
