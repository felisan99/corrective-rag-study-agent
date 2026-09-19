from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class GradeDocument(BaseModel):
    """Binary score for relevance of a retrieved document to a question."""

    reasoning: str = Field(
        description="One brief sentence on the document's relevance to the question, before deciding."
    )
    binary_score: bool = Field(
        description="True if the document is relevant to the question, else False."
    )


llm = ChatAnthropic(model="claude-haiku-4-5")
structured_llm_grader = llm.with_structured_output(GradeDocument)

system = """You are a grader assessing relevance of a retrieved document to a \
user question. If the document contains keywords or semantic meaning related \
to the question, grade it as relevant. This is a lenient filter meant only to \
catch clearly unrelated documents -- if in doubt, grade it relevant. Fill in \
`reasoning` before `binary_score`."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
