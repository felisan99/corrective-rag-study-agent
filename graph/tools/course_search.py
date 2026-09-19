from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import AgentState
from ingestion.vectorstore import EMBEDDING_CALL_LOCK, get_retriever

NO_RESULTS_MESSAGE = (
    "No relevant course material was found for this query. "
    "Consider using web_search instead."
)


@tool
def search_course_materials(
    query: str, state: Annotated[AgentState, InjectedState]
) -> str:
    """Search the student's course material (documents the teacher uploaded)
    for content relevant to `query`. Always try this before web_search when
    the question could plausibly be covered by the course.
    """
    retriever = get_retriever(state["course_id"])
    # The agent can call this tool multiple times in parallel (e.g. two
    # search queries in one turn) and ToolNode runs those concurrently in
    # separate threads. The embedding model underneath is a single shared
    # instance, and concurrent inference calls into it from multiple threads
    # can segfault the whole process (no Python exception to catch) -- so
    # every embedding call is serialized through this lock.
    with EMBEDDING_CALL_LOCK:
        documents = retriever.invoke(query)

    # Corrective-RAG step: a nearest-neighbor search always returns *something*,
    # even when nothing in the corpus is actually relevant, so each chunk is
    # graded before being handed to the agent. This is what lets the agent
    # tell "the course doesn't cover this" apart from "I searched wrong".
    relevant_docs = [
        doc
        for doc in documents
        if retrieval_grader.invoke({"document": doc.page_content, "question": query}).binary_score
    ]

    if not relevant_docs:
        return NO_RESULTS_MESSAGE

    chunks = []
    for doc in relevant_docs:
        source = doc.metadata.get("source", "unknown source")
        chunks.append(f"[from {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)
