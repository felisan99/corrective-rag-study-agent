import re
import uuid
from pathlib import Path
from typing import Literal

import graphviz
import matplotlib

matplotlib.use("Agg")  # headless: this runs in a CLI/server process, never a GUI

import matplotlib.pyplot as plt  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

DIAGRAMS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "diagrams"
CHARTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "charts"


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "untitled"
    return f"{slug}-{uuid.uuid4().hex[:8]}"


@tool
def generate_diagram(title: str, dot_source: str) -> str:
    """Generate a diagram (flowchart, decision tree, or a graph of how
    concepts/components relate to each other) -- write valid Graphviz DOT
    syntax yourself, e.g.:

        digraph {
            rankdir=TD;
            A [label="Start", shape=box];
            B [label="Need a tool?", shape=diamond];
            A -> B;
            B -> A [label="retry"];
        }

    Use `digraph` for directed flows (most cases) or `graph` for undirected
    relationships. Use this when a diagram would make a structural or
    procedural explanation clearer than prose alone.
    """
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAGRAMS_DIR / f"{_slug(title)}.png"
    # cleanup=True removes the intermediate .gv source file graphviz writes
    # alongside the render -- only the PNG is worth keeping.
    rendered_path = graphviz.Source(dot_source).render(
        outfile=str(out_path), format="png", cleanup=True
    )

    # ui/app.py reads this path back and shows it as an actual image (same
    # pattern as generate_chart below). An earlier version returned raw
    # Mermaid text for the agent to paste into its reply, but nothing ever
    # rendered it into a diagram: Gradio's Chatbot never triggered mermaid.js
    # for messages appended dynamically, and the agent itself was unreliable
    # at copying multi-line code verbatim into prose. Rendering server-side
    # to a real image sidesteps both problems.
    return f"Diagram '{title}' saved to {rendered_path}. Tell the student the file path so they can open it."


@tool
def generate_chart(
    title: str,
    chart_type: Literal["bar", "line"],
    labels: list[str],
    series: dict[str, list[float]],
    x_label: str = "",
    y_label: str = "",
) -> str:
    """Generate a bar or line chart to visually compare quantitative values
    (e.g. algorithmic growth rates, performance/complexity comparisons,
    numeric trends from the course material). `labels` are the x-axis
    categories/points; `series` maps a series name to one y-value per label
    (e.g. {"O(log n)": [...], "O(n)": [...]}) -- pass multiple series to
    compare them on the same chart.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, values in series.items():
        if chart_type == "bar":
            ax.bar(labels, values, label=name)
        else:
            ax.plot(labels, values, marker="o", label=name)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()

    path = CHARTS_DIR / f"{_slug(title)}.png"
    fig.savefig(path)
    plt.close(fig)

    return f"Chart '{title}' saved to {path}. Tell the student the file path so they can open it."
