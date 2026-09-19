"""Gradio demo UI for the study agent.

A thin presentation layer over the same compiled graph `main.py` uses --
same `graph/graph.py` workflow, same `graph/persistence.py` SQLite
checkpointer, so a CLI session and a UI session on the same
`course_id`/`thread_id` share conversation history.

Run with:
    uv run python -m ui.app
"""

import re
import uuid
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from graph.graph import workflow  # noqa: E402
from graph.nodes.grade_response import DOESNT_ANSWER_CRITIQUE, NOT_GROUNDED_CRITIQUE  # noqa: E402
from graph.persistence import get_sqlite_checkpointer  # noqa: E402
from ingestion.ingest import ingest  # noqa: E402

app = workflow.compile(checkpointer=get_sqlite_checkpointer())

IMAGE_PATH_RE = re.compile(r"saved to (.+?\.png)")
_SELF_CORRECTION_CRITIQUES = {NOT_GROUNDED_CRITIQUE, DOESNT_ANSWER_CRITIQUE}


def _config(course_id: str, thread_id: str) -> dict:
    return {"configurable": {"thread_id": f"{course_id}:{thread_id}"}}


def _render_tool_output(msg: ToolMessage) -> dict | None:
    """Turn a generate_chart/generate_diagram ToolMessage into a chat bubble.

    Both tools save a PNG (matplotlib for charts, Graphviz for diagrams) and
    only mention the file path in text -- pull it out so it renders as an
    actual image instead of a path the student can't see.
    """
    if msg.name in ("generate_chart", "generate_diagram"):
        found = IMAGE_PATH_RE.search(msg.text)
        if found and Path(found.group(1)).exists():
            return {"role": "assistant", "content": gr.Image(found.group(1))}
    return None


def _messages_to_history(messages: list) -> list:
    """Reconstruct the student-facing chat log from the graph's full message
    history -- checkpoint state includes internal ReAct steps (tool-calling
    AIMessages, search/web-search ToolMessages, injected self-correction
    critiques) that were never meant to be shown, so this keeps only: the
    student's real questions, the final answer per turn, and any chart or
    diagram that turn produced. Used both to redraw an existing thread's
    history on load and to append the current turn's new messages.
    """
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            if msg.text in _SELF_CORRECTION_CRITIQUES:
                continue
            history.append({"role": "user", "content": msg.text})
        elif isinstance(msg, ToolMessage):
            rendered = _render_tool_output(msg)
            if rendered:
                history.append(rendered)
        elif isinstance(msg, AIMessage):
            # Intermediate ReAct steps are AIMessages that only carry
            # tool_calls (no text) -- skip those, keep final answers.
            if msg.tool_calls or not msg.text:
                continue
            history.append({"role": "assistant", "content": msg.text})
    return history


def load_thread(course_id: str, thread_id: str) -> list:
    """Redraw a thread's full history from the checkpoint -- called on page
    load and whenever the course/thread id changes. Without this, the
    checkpoint still gives the *agent* full memory of past turns (LangGraph
    loads it automatically on every invoke), but the chat window itself
    would stay empty until a new message was sent, hiding a conversation
    that's actually still there.
    """
    config = _config(course_id, thread_id)
    messages = app.get_state(config).values.get("messages", [])
    return _messages_to_history(messages)


def chat(message: str, history: list, course_id: str, thread_id: str):
    if not message.strip():
        return history, ""

    config = _config(course_id, thread_id)
    # Everything already saved for this thread_id -- new messages from this
    # turn are whatever `result["messages"]` has past that point.
    before = len(app.get_state(config).values.get("messages", []))

    result = app.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "course_id": course_id,
            "retry_count": 0,
            "current_question": message,
        },
        config=config,
    )

    history = history + _messages_to_history(result["messages"][before:])
    return history, ""


def new_conversation():
    return uuid.uuid4().hex[:8], []


def do_ingest(file_path: str | None, course_id: str, teacher: str):
    if not file_path:
        return "Please choose a file first."
    if not course_id.strip():
        return "Please provide a course id."
    num_chunks = ingest(course_id.strip(), Path(file_path), teacher.strip() or None)
    return f"Ingested {num_chunks} chunks from '{Path(file_path).name}' into course '{course_id.strip()}'."


with gr.Blocks(title="Study Platform Agent") as demo:
    gr.Markdown(
        "# Study Platform Agent\n"
        "Ask questions about your course material. The agent searches the course "
        "corpus first, falls back to web search when the corpus doesn't cover "
        "something, and can draw diagrams or charts to explain concepts."
    )

    with gr.Tab("Study Chat"):
        with gr.Row():
            course_box = gr.Textbox(label="Course ID", value="cs101", scale=1)
            thread_box = gr.Textbox(label="Thread ID", value="default", scale=1)
            new_conv_btn = gr.Button("New conversation", scale=1)

        chatbot = gr.Chatbot(height=480, label="Chat")
        msg_box = gr.Textbox(label="Your question", placeholder="Ask about the course material...")
        send_btn = gr.Button("Send", variant="primary")

        send_btn.click(chat, [msg_box, chatbot, course_box, thread_box], [chatbot, msg_box])
        msg_box.submit(chat, [msg_box, chatbot, course_box, thread_box], [chatbot, msg_box])
        new_conv_btn.click(new_conversation, None, [thread_box, chatbot])

        # The checkpoint gives the *agent* full memory of a thread already
        # (LangGraph loads it on every invoke) -- these redraw the *visible*
        # chat to match, on first page load and whenever you switch to an
        # existing course/thread id, so past turns don't just vanish from
        # view until you send a new message.
        demo.load(load_thread, [course_box, thread_box], chatbot)
        course_box.submit(load_thread, [course_box, thread_box], chatbot)
        thread_box.submit(load_thread, [course_box, thread_box], chatbot)

    with gr.Tab("Teacher: Ingest Material"):
        gr.Markdown("Upload a `.pdf`, `.txt`, or `.md` file to add it to a course's corpus.")
        file_box = gr.File(label="Course material", file_types=[".pdf", ".txt", ".md"], file_count="single")
        ingest_course_box = gr.Textbox(label="Course ID", value="cs101")
        teacher_box = gr.Textbox(label="Teacher name (optional)")
        ingest_btn = gr.Button("Ingest", variant="primary")
        ingest_status = gr.Textbox(label="Status", interactive=False)

        ingest_btn.click(do_ingest, [file_box, ingest_course_box, teacher_box], ingest_status)


if __name__ == "__main__":
    demo.launch()
