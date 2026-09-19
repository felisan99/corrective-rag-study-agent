import argparse

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402

from graph.graph import workflow  # noqa: E402
from graph.persistence import get_sqlite_checkpointer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the study agent.")
    parser.add_argument("--course", default="cs101", help="Course id to study (default: cs101)")
    parser.add_argument(
        "--thread",
        default="default",
        help="Conversation thread id -- reuse one to resume a past session (default: 'default')",
    )
    args = parser.parse_args()

    app = workflow.compile(checkpointer=get_sqlite_checkpointer())
    config = {"configurable": {"thread_id": f"{args.course}:{args.thread}"}}

    # `.values` is what's actually stored for this thread_id so far -- empty
    # on a first run, populated if we're resuming a previous conversation.
    existing_messages = app.get_state(config).values.get("messages", [])

    print(f"Studying course '{args.course}' (thread '{args.thread}'). Type 'exit' to quit.")
    if existing_messages:
        print(f"Resuming previous conversation ({len(existing_messages)} messages so far).")
    print()

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        # Only the new turn is passed in -- the checkpointer supplies
        # everything already known for this thread_id, and `add_messages`
        # appends this HumanMessage onto it rather than replacing it.
        result = app.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "course_id": args.course,
                "retry_count": 0,
                "current_question": user_input,
            },
            config=config,
        )

        # `.text` (not `.content`) because Claude's responses can come back as a
        # list of content blocks (e.g. thinking + text) rather than a plain string.
        # generate_chart/generate_diagram save PNGs and mention the file path
        # in the agent's reply -- the terminal can't show images inline, so
        # (like charts already did) the CLI just prints that path as text.
        print(f"agent> {result['messages'][-1].text}\n")


if __name__ == "__main__":
    main()
