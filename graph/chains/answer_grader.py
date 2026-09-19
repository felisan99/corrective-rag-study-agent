from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class GradeAnswer(BaseModel):
    """Binary score for whether a response actually addresses the question."""

    reasoning: str = Field(
        description="One brief sentence on whether the response resolves the question, before deciding."
    )
    binary_score: bool = Field(
        description="True if the response addresses/resolves the question, else False."
    )


llm = ChatAnthropic(model="claude-haiku-4-5")
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an assistant's response \
addresses a student's question. Give a binary score: True if the response \
resolves the question (even a short or partial answer counts, as long as \
it's correct and on-topic), False if it dodges, is irrelevant, or fails to \
engage with what was asked. Fill in `reasoning` before `binary_score`."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Student question:\n\n{question}\n\nAssistant response:\n\n{response}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
