import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "checkpoints.sqlite"


def get_sqlite_checkpointer() -> SqliteSaver:
    """A LangGraph checkpointer that persists graph state to disk, keyed by
    `thread_id`.

    Without a checkpointer, `app.invoke()` only ever knows about the state
    you hand it in that call -- which is why every earlier test script had to
    keep re-passing the whole `messages` list by hand. With one, LangGraph
    loads whatever it already has for a given `thread_id` before running,
    and saves the result after every step, so a conversation survives
    restarting the process (and is what the LangGraph Studio dev server uses
    for its time-travel/replay UI).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: SqliteSaver may be called from a callback
    # thread; the connection itself is only ever used one call at a time.
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return checkpointer
