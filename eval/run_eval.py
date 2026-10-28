"""End-to-end evaluation of the study agent.

Runs every case in eval/cases.yaml through the real graph (real Claude calls,
real Tavily search, real chart/diagram rendering) and checks, per case:

- tool routing: the tools the agent was expected to call were called, and the
  forbidden ones (e.g. web_search for questions the course covers) were not;
- answer content: regexes that must / must not appear in the final answer;
- self-correction: how many revisions grade_response asked for;
- latency and token cost of the whole turn, graders included.

Usage:
    uv run python -m eval.run_eval
    uv run python -m eval.run_eval --only in-balanced-search web-treap
    uv run python -m eval.run_eval --rescore eval/results/run-<timestamp>.json

Each run writes eval/results/run-<timestamp>.json with the raw per-case data.
"""

import argparse
import json
import os
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Eval runs shouldn't flood the LangSmith project with traces; load_dotenv()
# doesn't override variables that are already set.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
load_dotenv()

from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from graph.graph import app  # noqa: E402
from graph.nodes.agent import AGENT_MODEL  # noqa: E402
from ingestion.ingest import ingest  # noqa: E402
from ingestion.vectorstore import get_vectorstore  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CASES_PATH = EVAL_DIR / "cases.yaml"
RESULTS_DIR = EVAL_DIR / "results"
SAMPLE_DOC = EVAL_DIR.parent / "data" / "sample_docs" / "cs101_binary_search_trees.md"

# Isolated collection, so running the eval never touches a real course.
EVAL_COURSE = "eval-cs101"

# USD per million tokens (input, output), matched by model-name prefix.
PRICES = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
}


def ensure_corpus() -> None:
    if get_vectorstore(EVAL_COURSE)._collection.count() == 0:
        n = ingest(EVAL_COURSE, SAMPLE_DOC, teacher="Eval")
        print(f"Ingested {n} chunks into '{EVAL_COURSE}'.")


def cost_of(usage_by_model: dict) -> float:
    total = 0.0
    for model, usage in usage_by_model.items():
        price = next((p for prefix, p in PRICES.items() if model.startswith(prefix)), None)
        if price is None:
            raise KeyError(f"No price configured for model '{model}'")
        total += usage["input_tokens"] / 1e6 * price[0] + usage["output_tokens"] / 1e6 * price[1]
    return total


def run_case(case: dict) -> dict:
    question = case["question"]
    start = time.perf_counter()
    with get_usage_metadata_callback() as cb:
        result = app.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "course_id": EVAL_COURSE,
                "retry_count": 0,
                "current_question": question,
            }
        )
    latency = time.perf_counter() - start

    called = [tc["name"] for m in result["messages"] if isinstance(m, AIMessage) for tc in m.tool_calls]
    answer = result["messages"][-1].text
    failures = check(case, called, answer)

    usage = {model: dict(u) for model, u in cb.usage_metadata.items()}
    return {
        "id": case["id"],
        "category": case["category"],
        "question": question,
        "passed": not failures,
        "failures": failures,
        "tools_called": called,
        "revisions": result.get("retry_count", 0),
        "latency_s": round(latency, 2),
        "cost_usd": round(cost_of(usage), 5),
        "usage": usage,
        "answer": answer,
    }


def check(case: dict, called: list[str], answer: str) -> list[str]:
    # Before the tool was renamed (graph/tools/web_search.py), Tavily
    # registered itself as `tavily_search`; treat old runs the same way.
    called = ["web_search" if name == "tavily_search" else name for name in called]
    failures = []
    for tool in case.get("expect_tools", []):
        if tool not in called:
            failures.append(f"expected tool not called: {tool}")
    for tool in case.get("forbid_tools", []):
        if tool in called:
            failures.append(f"forbidden tool called: {tool}")
    for pattern in case.get("must_match", []):
        if not re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"answer missing /{pattern}/")
    for pattern in case.get("must_not_match", []):
        if re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"answer contains /{pattern}/")
    return failures


def rescore(run_path: Path, cases: list[dict]) -> None:
    """Re-apply the current checks to a saved run, without calling any API.

    Used when a check itself was wrong; the run file is updated in place and
    keeps the original failures under `failures_before_rescore`.
    """
    run = json.loads(run_path.read_text())
    by_id = {c["id"]: c for c in cases}
    for row in run["results"]:
        failures = check(by_id[row["id"]], row["tools_called"], row["answer"])
        if failures != row["failures"]:
            row.setdefault("failures_before_rescore", row["failures"])
        row["failures"], row["passed"] = failures, not failures
    run["summary"] = summarize(run["results"], run["summary"]["agent_model"])
    run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(json.dumps(run["summary"], indent=2))


def summarize(results: list[dict], agent_model: str = AGENT_MODEL) -> dict:
    def rate(rows):
        return f"{sum(r['passed'] for r in rows)}/{len(rows)}"

    categories = sorted({r["category"] for r in results})
    latencies = sorted(r["latency_s"] for r in results)
    costs = [r["cost_usd"] for r in results]
    p95_index = max(0, round(0.95 * len(latencies)) - 1)
    return {
        "agent_model": agent_model,
        "cases": len(results),
        "passed": rate(results),
        "by_category": {c: rate([r for r in results if r["category"] == c]) for c in categories},
        "first_draft_accepted": f"{sum(r['revisions'] == 0 for r in results)}/{len(results)}",
        "revisions_triggered": sum(r["revisions"] for r in results),
        "latency_median_s": round(statistics.median(latencies), 1),
        "latency_p95_s": round(latencies[p95_index], 1),
        "cost_per_query_usd": round(statistics.mean(costs), 4),
        "cost_total_usd": round(sum(costs), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the study agent's end-to-end eval.")
    parser.add_argument("--only", nargs="*", help="Run only these case ids")
    parser.add_argument("--rescore", type=Path, help="Re-check a saved run file instead of running")
    args = parser.parse_args()

    cases = yaml.safe_load(CASES_PATH.read_text())
    if args.rescore:
        rescore(args.rescore, cases)
        return
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]

    ensure_corpus()
    results = []
    for case in cases:
        row = run_case(case)
        results.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        print(f"[{status}] {row['id']:<24} {row['latency_s']:>5.1f}s  ${row['cost_usd']:.4f}  {row['failures'] or ''}")

    summary = summarize(results)
    print(json.dumps(summary, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False))
    print(f"Wrote {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
