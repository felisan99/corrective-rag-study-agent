import threading
from pathlib import Path

import pytest

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.retrieval_grader import retrieval_grader
from graph.tools.course_search import search_course_materials
from ingestion.ingest import ingest
from ingestion.vectorstore import get_retriever, get_vectorstore

TEST_COURSE = "test-bst"
SAMPLE_DOC = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "cs101_binary_search_trees.md"


@pytest.fixture(scope="session", autouse=True)
def test_corpus():
    """Ingest the sample BST doc into an isolated test collection once per
    test run, clearing out whatever the collection had before so repeated
    runs don't pile up duplicate chunks.

    This deletes existing documents in place rather than calling
    `reset_collection()` (delete + recreate the collection): chromadb's
    persistent client doesn't clean up the old collection's on-disk segment
    directory when it's deleted, so every test run using `reset_collection`
    leaked one orphaned UUID folder under data/chroma/ -- 21 of them had
    piled up by the time this was noticed.
    """
    vectorstore = get_vectorstore(TEST_COURSE)
    existing_ids = vectorstore.get()["ids"]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)
    ingest(TEST_COURSE, SAMPLE_DOC, teacher=None)


def test_retrieval_grader_relevant():
    question = "What is the time complexity of BST search?"
    docs = get_retriever(TEST_COURSE).invoke(question)
    result = retrieval_grader.invoke({"question": question, "document": docs[0].page_content})
    assert result.binary_score is True


def test_retrieval_grader_irrelevant():
    docs = get_retriever(TEST_COURSE).invoke("BST time complexity")
    result = retrieval_grader.invoke(
        {"question": "How do I bake a chocolate cake?", "document": docs[0].page_content}
    )
    assert result.binary_score is False


def test_hallucination_grader_grounded():
    context = "BST search runs in O(log n) time when the tree is balanced."
    response = "Searching a balanced BST takes O(log n) time."
    result = hallucination_grader.invoke({"context": context, "response": response})
    assert result.binary_score is True


def test_hallucination_grader_not_grounded():
    context = "BST search runs in O(log n) time when the tree is balanced."
    response = "Searching a balanced BST takes O(1) constant time because of magic indexing."
    result = hallucination_grader.invoke({"context": context, "response": response})
    assert result.binary_score is False


def test_answer_grader_addresses_question():
    result = answer_grader.invoke(
        {
            "question": "What is the time complexity of BST search?",
            "response": "O(log n) for a balanced tree.",
        }
    )
    assert result.binary_score is True


def test_answer_grader_dodges_question():
    result = answer_grader.invoke(
        {
            "question": "What is the time complexity of BST search?",
            "response": "BSTs are a fundamental data structure used in computer science.",
        }
    )
    assert result.binary_score is False


def test_concurrent_searches_dont_crash():
    """The agent can call `search_course_materials` more than once in the same
    turn (e.g. two search queries), and LangGraph's ToolNode runs those
    concurrently in separate threads. The embedding model underneath is a
    single shared instance, and calling it from multiple threads at once used
    to segfault the whole process (no Python exception -- pytest would just
    die), rather than raise anything a `pytest.raises` could catch. This just
    has to survive without crashing the process.

    A `Barrier` forces every thread's embedding call to actually overlap --
    without it, thread startup jitter can make the race miss often enough for
    this to pass even with the lock removed, which would defeat the point.
    """
    queries = ["BST search", "binary tree insertion", "tree balance"]
    errors = []
    barrier = threading.Barrier(len(queries))
    # Warm the retriever/embedding-model cache outside the timed race so the
    # barrier lines up the actual concurrent embedding calls, not model load.
    get_retriever(TEST_COURSE)

    def worker(query: str) -> None:
        try:
            barrier.wait()
            state = {
                "course_id": TEST_COURSE,
                "messages": [],
                "retry_count": 0,
                "current_question": query,
            }
            search_course_materials.invoke({"query": query, "state": state})
        except Exception as e:  # noqa: BLE001 -- record it, don't swallow it
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(q,)) for q in queries]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
