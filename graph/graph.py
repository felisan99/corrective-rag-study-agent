from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from graph.consts import ACT, AGENT_REASON, GRADE_RESPONSE
from graph.nodes.agent import run_agent_reasoning, tools
from graph.nodes.grade_response import grade_response
from graph.state import AgentState

LAST = -1


def should_continue(state: AgentState) -> str:
    if not state["messages"][LAST].tool_calls:
        return GRADE_RESPONSE
    return ACT


workflow = StateGraph(AgentState)

workflow.add_node(AGENT_REASON, run_agent_reasoning)

# handle_tool_errors=True: any tool exception becomes an error ToolMessage
# instead of aborting the node. Without this, a transient tool failure (e.g. a
# flaky retrieval call) leaves the just-written AIMessage's tool_calls without
# matching ToolMessages -- Anthropic then rejects every future call on that
# checkpointed thread with "tool_use ids were found without tool_result
# blocks", since the corrupted history is replayed on every turn.
workflow.add_node(ACT, ToolNode(tools, handle_tool_errors=True))
workflow.add_node(GRADE_RESPONSE, grade_response)

workflow.set_entry_point(AGENT_REASON)
workflow.add_conditional_edges(AGENT_REASON, should_continue, {GRADE_RESPONSE: GRADE_RESPONSE, ACT: ACT})
workflow.add_edge(ACT, AGENT_REASON)
# GRADE_RESPONSE routes onward itself via Command(goto=...), back to
# AGENT_REASON for a revision or to END once the answer passes.

# No checkpointer: each invoke() is a one-off with whatever state you pass in
# (fine for tests/scripts). main.py recompiles `workflow` with a persistent
# checkpointer -- see graph/persistence.py -- for the interactive CLI.
app = workflow.compile()
