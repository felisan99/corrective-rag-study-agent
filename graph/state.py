from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Conversation state for the study agent.

    Extends the prebuilt `MessagesState` (which already gives us an
    append-only `messages` list via the `add_messages` reducer) with a couple
    of fields specific to this graph:

    - `course_id`: which course's material the student is studying. Set once
      per session and never chosen by the LLM -- tools read it via
      `InjectedState` instead of accepting it as an argument, so the model
      can't "guess" a different course.
    - `retry_count`: how many times `grade_response` has sent the agent back
      to revise its answer, so we can cap the self-correction loop.
    - `current_question`: the student's question for this turn, set once by
      the caller. `grade_response` needs "what was actually asked" to check
      the draft answer against, and re-deriving that by scanning for the
      latest `HumanMessage` breaks once a revision cycle injects its own
      `HumanMessage` critique -- so it's carried explicitly instead.
    """

    course_id: str
    retry_count: int
    current_question: str
