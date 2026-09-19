from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class GradeHallucination(BaseModel):
    """Binary score for whether a response is grounded in the conversation."""

    reasoning: str = Field(
        description="One brief sentence tracing the response's claims to the context, before deciding."
    )
    binary_score: bool = Field(
        description="True if the response is grounded in the conversation context, else False."
    )


llm = ChatAnthropic(model="claude-haiku-4-5")
structured_llm_grader = llm.with_structured_output(GradeHallucination)

system = """You are a grader assessing whether an assistant's response is \
grounded in the conversation so far -- course material search results, web \
search results, and anything the student themselves said earlier in the \
conversation all count as legitimate grounding. Give a binary score: True if \
every factual claim in the response traces back to that context (or is \
trivial/conversational, e.g. acknowledging what the student just said), \
False if the response asserts something as fact that isn't supported by any \
of it. Fill in `reasoning` before `binary_score`."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Conversation context:\n\n{context}\n\nAssistant response:\n\n{response}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
