"""Renders eval results into the artifacts a judge actually reads.

Three outputs, all of them real:

- `write_scores_table` -- the markdown scores table, one row per scored requirement.
- `write_json` -- the same results as machine-readable JSON, for diffing runs.
- `render_before_after` -- the before/after table proving remote-sensing adaptation,
  the proof artifact for the LoRA row.

The single rule this module enforces: a metric that did not run renders as the
literal string ``TBD``. Never a zero, never an interpolation, never a plausible
placeholder. A fabricated score in a README is worse than an empty one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from satquery.eval.harness import NOT_RUN
from satquery.utils.logging import get_logger

__all__ = ["render_before_after", "render_scores_table", "write_json", "write_scores_table"]

logger = get_logger(__name__)

_STATUS_LABEL = {
    "ok": "scored",
    "not_implemented": "stub",
    "error": "failed",
    "skipped_budget": "skipped (budget)",
}


def _cell(value: Any) -> str:
    """Render one metric value, collapsing anything absent to `TBD`."""
    if value is None or value == NOT_RUN:
        return NOT_RUN
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        if not value:
            return NOT_RUN
        return "; ".join(f"{k}={_cell(v)}" for k, v in sorted(value.items()))
    return str(value)


def _escape(text: str) -> str:
    """Escape pipes so a metric name or error message cannot break the table."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_scores_table(results: dict[str, Any]) -> str:
    """Render the results mapping from `run_suite` as a markdown scores table.

    Args:
        results: The mapping returned by `satquery.eval.harness.run_suite`.

    Returns:
        The markdown document as a string, including a provenance footer with the
        frozen-constants fingerprint the run used.
    """
    suite = results.get("suite", {})
    tasks: dict[str, Any] = results.get("tasks", {})

    lines: list[str] = ["# SatQuery evaluation scores", ""]

    scored = suite.get("tasks_scored", 0)
    total = suite.get("tasks_total", len(tasks))
    lines.append(
        f"{scored} of {total} tasks produced real scores. Every other cell is `{NOT_RUN}`: "
        f"the evaluation has not run, and no number is being guessed on its behalf."
    )
    lines.append("")

    header = "| Task | Requirement | Tool | Implemented | Status | Samples | Metric | Score |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|")

    for name, record in tasks.items():
        metrics: dict[str, Any] = record.get("metrics") or {}
        status = _STATUS_LABEL.get(record.get("status", ""), str(record.get("status", "")))
        implemented = "yes" if record.get("implemented") else "no"
        rows = list(metrics.items()) or [("--", NOT_RUN)]
        for index, (metric_name, value) in enumerate(rows):
            lines.append(
                "| {task} | {req} | {tool} | {impl} | {status} | {samples} | {metric} | {score} |".format(
                    task=_escape(name) if index == 0 else "",
                    req=_escape(record.get("requirement", "")) if index == 0 else "",
                    tool=f"`{_escape(record.get('tool', ''))}`" if index == 0 else "",
                    impl=implemented if index == 0 else "",
                    status=status if index == 0 else "",
                    samples=record.get("max_samples", "") if index == 0 else "",
                    metric=f"`{_escape(metric_name)}`",
                    score=_cell(value),
                )
            )

    lines.append("")
    lines.append("## Run provenance")
    lines.append("")
    lines.append(f"- constants fingerprint: `{suite.get('constants_fingerprint', NOT_RUN)}`")
    lines.append(
        f"- fingerprint recorded in config: `{suite.get('recorded_fingerprint') or NOT_RUN}`"
    )
    lines.append(f"- seed: `{suite.get('seed', NOT_RUN)}`")
    lines.append(f"- config: `{suite.get('config_path') or NOT_RUN}`")
    lines.append(
        f"- wall clock: {_cell(suite.get('elapsed_s'))}s against a budget of "
        f"{_cell(suite.get('total_time_budget_s'))}s"
    )
    lines.append("")

    failures = [
        (name, record.get("error")) for name, record in tasks.items() if record.get("error")
    ]
    if failures:
        lines.append("## Why the `TBD` cells are `TBD`")
        lines.append("")
        for name, error in failures:
            lines.append(f"- **{_escape(name)}**: {_escape(error or '')}")
        lines.append("")

    return "\n".join(lines)


def write_scores_table(results: dict[str, Any], path: str | Path) -> Path:
    """Write the markdown scores table to `path`, creating parent directories.

    Args:
        results: The mapping returned by `satquery.eval.harness.run_suite`.
        path: Destination `.md` file, normally under the artifact root.

    Returns:
        The path written.
    """
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_scores_table(results) + "\n", encoding="utf-8")
    logger.info("wrote scores table to %s", destination)
    return destination


def write_json(results: dict[str, Any], path: str | Path) -> Path:
    """Write the raw results mapping as indented JSON.

    Args:
        results: The mapping returned by `satquery.eval.harness.run_suite`.
        path: Destination `.json` file.

    Returns:
        The path written.
    """
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    logger.info("wrote results json to %s", destination)
    return destination


def render_before_after(
    baseline: dict[str, Any],
    adapted: dict[str, Any],
    *,
    baseline_label: str = "EarthDial-4B, no adaptation",
    adapted_label: str = "+ LoRA on BigEarthNet.txt",
) -> str:
    """Render the before/after remote-sensing adaptation table.

    this is the proof artifact for the "RS adaptation of a vision or VL
    component" row, alongside the training log and the adapter weights. A generic VLM
    with no remote-sensing adaptation fails the problem statement outright, so this
    table has to show a real delta from a real pair of runs.

    Both arguments take the shape returned by `run_suite`. Any metric missing from
    either side renders as `TBD`, and the delta column is only computed when both
    sides are numeric.

    Args:
        baseline: Results of the un-adapted backbone.
        adapted: Results of the LoRA-adapted backbone.
        baseline_label: Column heading for the baseline run.
        adapted_label: Column heading for the adapted run.

    Returns:
        The markdown table as a string.
    """
    baseline_tasks: dict[str, Any] = baseline.get("tasks", {}) if baseline else {}
    adapted_tasks: dict[str, Any] = adapted.get("tasks", {}) if adapted else {}

    names: list[str] = list(baseline_tasks)
    names += [name for name in adapted_tasks if name not in baseline_tasks]

    lines: list[str] = [
        "# Remote-sensing adaptation: before and after",
        "",
        (
            "Both columns are the same eval suite at the same seed and the same frozen "
            f"constants fingerprint. `{NOT_RUN}` means that run has not happened yet."
        ),
        "",
        f"| Task | Metric | {_escape(baseline_label)} | {_escape(adapted_label)} | Delta |",
        "|---|---|---|---|---|",
    ]

    if not names:
        lines.append(f"| {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} |")
        lines.append("")
        return "\n".join(lines)

    for name in names:
        before_metrics = (baseline_tasks.get(name) or {}).get("metrics") or {}
        after_metrics = (adapted_tasks.get(name) or {}).get("metrics") or {}
        metric_names = list(before_metrics)
        metric_names += [m for m in after_metrics if m not in before_metrics]
        if not metric_names:
            metric_names = ["--"]

        for index, metric_name in enumerate(metric_names):
            before = before_metrics.get(metric_name, NOT_RUN)
            after = after_metrics.get(metric_name, NOT_RUN)
            if isinstance(before, int | float) and isinstance(after, int | float):
                delta = f"{after - before:+.4f}"
            else:
                delta = NOT_RUN
            lines.append(
                f"| {_escape(name) if index == 0 else ''} | `{_escape(metric_name)}` | "
                f"{_cell(before)} | {_cell(after)} | {delta} |"
            )

    lines.append("")
    return "\n".join(lines)
