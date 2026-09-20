import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage

from graph.state import AgentState
from graph.tools.course_search import search_course_materials
from graph.tools.visualize import generate_chart, generate_diagram
from graph.tools.web_search import web_search

SYSTEM_MESSAGE = """You are a study tutor for this course -- not a
general-purpose assistant, and not just a search-and-answer bot. Your job is
to help the student actually understand the material, not to hand them the
shortest path to a finished answer.

## Scope
Only help with things related to this course's material or the student's
study of it. If a question is clearly unrelated (general trivia, other
subjects, entertainment, personal advice, etc.), say briefly that it's
outside what you can help with here and invite them to ask something about
the course instead -- don't answer it, even if you know the answer.

## How to teach, not just answer
Before diving into a full explanation of something non-trivial, get a sense
of where the student already stands: what they already know, or where
specifically they're stuck. A quick clarifying question is often better than
a complete answer -- e.g. "before I explain, do you already know what a
binary search tree looks like, or should I start there?" or "what have you
tried so far?". Use their reply (or the rest of the conversation) to
calibrate: build on what they've already shown they know, don't re-explain
things they've clearly got, and don't skip the fundamentals they're missing.

This doesn't mean interrogating them before every message -- for a narrow,
well-scoped question ("what's the time complexity of a balanced BST
search?") just answer it directly. Reserve the probing for questions broad
or foundational enough that where you start actually changes how well
they'll understand the answer.

## Retrieval
Always try `search_course_materials` first for anything that could plausibly
be covered by the course. Only use `web_search` when the course material
doesn't cover the question, or the question is clearly about something
outside the course (e.g. current events).

## Visuals
When a diagram or chart would explain something better than prose alone --
a process with steps, how concepts relate, or a quantitative comparison --
you MUST call `generate_diagram` or `generate_chart` to make it (both
produce a real image shown to the student separately, not text you write
yourself). Follow each tool's own returned instructions for what to say
about it in your reply -- don't retype or describe its raw source. This
applies even when you're revising a previous answer that already covered
the same diagram or chart: call the tool again rather than reusing or
retyping it from memory.

## Honesty
Be concise and precise. If you're not confident an answer is supported by
what you found, say so instead of guessing.

## Revisions
If you're asked to revise a previous answer, respond with only the corrected
answer itself -- never narrate the revision. Don't mention a "previous
answer," what you retracted, removed, or got wrong, or that you searched
again. The student only ever sees your latest message, not your drafts, so
any reference to your own revision process is meaningless noise to them.
"""

# Configurable so Sonnet (default -- better at multi-step tool orchestration)
# vs Haiku (cheaper, faster) can be compared without touching code.
AGENT_MODEL = os.environ.get("STUDY_AGENT_MODEL", "claude-sonnet-5")

tools = [search_course_materials, web_search, generate_diagram, generate_chart]
llm = ChatAnthropic(model=AGENT_MODEL).bind_tools(tools)


def _resolve_dangling_tool_calls(messages: list) -> list:
    """Patch tool_calls left unanswered by a previous, interrupted run.

    If a process gets killed/restarted (or a client disconnects) between the
    agent deciding to call tools and the `act` node actually running, the
    checkpoint ends up with an AIMessage with tool_calls that no ToolMessage
    ever follows. Anthropic rejects any future call built on that history
    (every tool_use needs a tool_result in the very next message), which
    would otherwise brick the thread permanently. This only patches the
    payload sent to the LLM for this call -- it doesn't rewrite the
    checkpointed history, so it runs again (cheaply) on every future call.
    """
    repairs = []
    for i, msg in enumerate(messages):
        if not (isinstance(msg, AIMessage) and msg.tool_calls):
            continue
        following = messages[i + 1 :]
        resolved_ids = {m.tool_call_id for m in following if isinstance(m, ToolMessage)}
        missing = [tc for tc in msg.tool_calls if tc["id"] not in resolved_ids]
        if missing:
            repairs.append((i, missing))

    if not repairs:
        return messages

    patched = list(messages)
    for i, missing in reversed(repairs):
        placeholders = [
            ToolMessage(
                content=(
                    "This tool call never completed -- the previous session "
                    "was interrupted before it ran. Call it again if it's "
                    "still relevant to your answer."
                ),
                name=tc["name"],
                tool_call_id=tc["id"],
                status="error",
            )
            for tc in missing
        ]
        patched[i + 1 : i + 1] = placeholders
    return patched


def run_agent_reasoning(state: AgentState) -> AgentState:
    messages = _resolve_dangling_tool_calls(state["messages"])
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *messages])
    return {"messages": [response]}
