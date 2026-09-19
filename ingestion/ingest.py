"""Teacher-facing ingestion CLI.

Loads a single local file, splits it into chunks, and adds it to the Chroma
collection for a given course. This is the whole "teacher uploads material"
flow for now -- no auth, no web upload endpoint, just a script a teacher (or
us, standing in for one) runs locally.

Usage:
    uv run python -m ingestion.ingest --course bio101 --file notes.pdf
    uv run python -m ingestion.ingest --course bio101 --file notes.md --teacher "Dr. Ana"
"""

import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.vectorstore import EMBEDDING_CALL_LOCK, get_vectorstore

LOADERS_BY_SUFFIX = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_and_split(file_path: Path) -> list:
    loader_cls = LOADERS_BY_SUFFIX.get(file_path.suffix.lower())
    if loader_cls is None:
        supported = ", ".join(sorted(LOADERS_BY_SUFFIX))
        raise ValueError(f"Unsupported file type '{file_path.suffix}'. Supported: {supported}")

    documents = loader_cls(str(file_path)).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)


def ingest(course_id: str, file_path: Path, teacher: str | None) -> int:
    chunks = load_and_split(file_path)
    for chunk in chunks:
        chunk.metadata["course_id"] = course_id
        chunk.metadata["source"] = file_path.name
        if teacher:
            chunk.metadata["teacher"] = teacher

    vectorstore = get_vectorstore(course_id)
    # add_documents() calls the shared embedding model, which isn't safe to
    # call concurrently with another embedding call from another thread (see
    # EMBEDDING_CALL_LOCK) -- e.g. a teacher ingesting while a student's
    # question is mid-search.
    with EMBEDDING_CALL_LOCK:
        vectorstore.add_documents(chunks)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a document into a course's corpus.")
    parser.add_argument("--course", required=True, help="Course id, e.g. 'bio101'")
    parser.add_argument("--file", required=True, type=Path, help="Path to a .pdf, .txt or .md file")
    parser.add_argument("--teacher", default=None, help="Optional teacher name for metadata")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        raise SystemExit(1)

    num_chunks = ingest(args.course, args.file, args.teacher)
    print(f"Ingested {num_chunks} chunks from '{args.file.name}' into course '{args.course}'.")


if __name__ == "__main__":
    main()
