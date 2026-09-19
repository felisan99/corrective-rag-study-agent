"""Chroma vector store access, scoped per course.

Each course gets its own Chroma *collection* (not a separate directory) inside
one persistent Chroma instance, so a student's retrieval never crosses into
another course's material, and a teacher's upload only ever touches their own
course's collection.
"""

import threading
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"

# all-MiniLM-L6-v2: small (~90MB), fast, runs on CPU, good enough for
# study-notes-scale semantic search. Loaded once and reused across courses.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# All courses share one Chroma persistent client (same persist_directory), and
# chromadb's client bootstrap has a known race when two threads construct a
# client for the same directory concurrently (Gradio/ToolNode run tools in
# worker threads). This lock serializes first-time construction so concurrent
# tool calls never hit chromadb's init race.
_vectorstore_init_lock = threading.Lock()

# The embedding model itself is a single shared instance (see get_embeddings
# below), and sentence-transformers/PyTorch inference is not safe to call
# concurrently from multiple threads -- ToolNode runs parallel tool calls in
# a thread pool, so two simultaneous `search_course_materials` calls (or a
# search racing an ingest) can segfault the whole process with no Python
# traceback. This lock serializes every embedding call, not just vectorstore
# construction.
EMBEDDING_CALL_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _collection_name(course_id: str) -> str:
    return f"course-{course_id}"


@lru_cache(maxsize=None)
def _build_vectorstore(course_id: str) -> Chroma:
    return Chroma(
        collection_name=_collection_name(course_id),
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIR),
    )


def get_vectorstore(course_id: str) -> Chroma:
    """Return the (cached) Chroma collection for a given course.

    Construction is serialized via a lock (see `_vectorstore_init_lock`);
    cache hits just acquire/release it, which is negligible.
    """
    with _vectorstore_init_lock:
        return _build_vectorstore(course_id)


def get_retriever(course_id: str, k: int = 4):
    return get_vectorstore(course_id).as_retriever(search_kwargs={"k": k})
