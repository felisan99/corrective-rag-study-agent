from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.types import Command

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import AGENT_REASON, MAX_REVISION_RETRIES
from graph.state import AgentState

_REVISION_REMINDER = (
    " Reply with only the corrected answer -- don't mention that this is a "
    "revision or refer to your last answer; the student never saw it."
)
NOT_GROUNDED_CRITIQUE = (
    "Your last answer included claims that weren't supported by anything "
    "found this conversation (course material, web search, or what the "
    "student already told you). Revise it so every claim is grounded, "
    "searching again if needed." + _REVISION_REMINDER
)
DOESNT_ANSWER_CRITIQUE = (
    "Your last answer didn't actually address the student's question. Try "
    "again, searching for more specific information if needed." + _REVISION_REMINDER
)


def _context(state: AgentState) -> str:
    # Everything the agent has legitimately seen so far -- earlier turns,
    # anything the student told it, and this turn's tool results -- counts as
    # grounding, not just this turn's tool calls. Excludes the draft answer
    # itself (the last message), which is what's being graded.
    context = "\n\n".join(f"{m.type}: {m.text}" for m in state["messages"][:-1] if m.text)
    return context or "(no prior conversation or tool results)"


def grade_response(state: AgentState) -> Command[Literal[AGENT_REASON, END]]:
    """Reflexion-style self-correction: check the agent's draft answer for
    groundedness and relevance before letting it reach the student, and send
    it back for revision (with a critique) if it fails either check.
    """
    response = state["messages"][-1].text
    question = state["current_question"]
    context = _context(state)

    grounded = hallucination_grader.invoke({"context": context, "response": response}).binary_score
    addresses_question = (
        answer_grader.invoke({"question": question, "response": response}).binary_score
        if grounded
        else False
    )

    if grounded and addresses_question:
        return Command(goto=END)

    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_REVISION_RETRIES:
        # Give up correcting and let the best-effort answer through rather
        # than looping forever.
        return Command(goto=END)

    critique = NOT_GROUNDED_CRITIQUE if not grounded else DOESNT_ANSWER_CRITIQUE
    return Command(
        goto=AGENT_REASON,
        update={
            "messages": [HumanMessage(content=critique)],
            "retry_count": retry_count + 1,
        },
    )
