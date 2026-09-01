"""The evaluation harness: one entry point that runs every scored row.

Two hard requirements are fixed on this module.

1. **One command runs all evals** (`make eval`). That command must succeed today,
   while every learned tool is still an honest stub, and it must report those tools
   as not-yet-run rather than failing the whole suite. So `run_suite` isolates
   per-task failure and records the literal string ``TBD`` for any metric that could
   not be computed.
2. **The full suite must not exceed ten minutes on a single GPU.** The budget lives
   in `configs/eval/full_suite.yaml` as `total_time_budget_s`, is validated against
   the sum of the per-task budgets when the config loads, and is enforced while the
   suite runs.

The third job of this module is anti-drift. `assert_constants_match` compares the
frozen-constant fingerprint recorded in the config (and, later, in a checkpoint)
against `constants_fingerprint()` of the running process. Preprocessing drift
between the training path and the eval path is silent and close to undebuggable;
this turns it into a loud failure before a single sample is scored.

Only `run_task` is a stub. Everything else here is real.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from satquery.models.registry import ToolRegistry
from satquery.preprocess.constants import constants_fingerprint
from satquery.serve.contracts import ToolRequest
from satquery.utils.logging import get_logger
from satquery.utils.paths import artifact_root, repo_root
from satquery.utils.seed import DEFAULT_SEED

__all__ = [
    "NOT_RUN",
    "EvalSuiteConfig",
    "EvalTask",
    "assert_constants_match",
    "run_suite",
    "run_task",
]

logger = get_logger(__name__)

#: The only value a metric may take when its evaluation did not run. Never zero,
#: never a guess: fabricated scores are worse than no scores.
NOT_RUN = "TBD"

_STATUS_OK = "ok"
_STATUS_NOT_IMPLEMENTED = "not_implemented"
_STATUS_ERROR = "error"
_STATUS_SKIPPED_BUDGET = "skipped_budget"


@dataclass(slots=True)
class EvalTask:
    """One scored row of the judging table.

    Attributes:
        name: Stable identifier used as the results key and the table row label.
        dataset_config: Path to the dataset YAML, relative to `configs/` or absolute.
        split: Split name as the dataset loader understands it, e.g. `"test_1"`.
        task_type: `satquery.serve.contracts.TaskType` value as a string, e.g. `"vqa"`.
        tool_name: Registry name of the tool under evaluation, e.g. `"vlm.vqa"`.
        metrics: Metric identifiers this row reports, e.g. `["exact_match", "per_type"]`.
        max_samples: Cap on scored samples, chosen to keep the suite inside its budget.
        time_budget_s: Wall-clock budget for this task alone, in seconds.
        requirement: The judging-table row this task is the proof artifact for.
    """

    name: str
    dataset_config: str
    split: str
    task_type: str
    tool_name: str
    metrics: list[str]
    max_samples: int
    time_budget_s: float
    requirement: str = ""

    def resolved_dataset_config(self) -> Path:
        """Return `dataset_config` as an absolute path, resolved against `configs/`."""
        candidate = Path(self.dataset_config)
        if candidate.is_absolute():
            return candidate
        return (repo_root() / "configs" / candidate).resolve()


@dataclass(slots=True)
class EvalSuiteConfig:
    """The whole suite: what to run, where to write it, and how long it may take."""

    tasks: list[EvalTask]
    output_dir: Path = field(default_factory=lambda: artifact_root() / "eval")
    seed: int = DEFAULT_SEED
    total_time_budget_s: float = 600.0
    constants_fingerprint: str | None = None
    source_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalSuiteConfig:
        """Load a suite config from YAML with OmegaConf.

        Args:
            path: Path to the suite YAML, e.g. `configs/eval/full_suite.yaml`.

        Returns:
            The parsed config, with `output_dir` resolved under `artifact_root()`
            when the YAML gives a relative path.

        Raises:
            FileNotFoundError: The config file does not exist.
            ValueError: The config declares no tasks, or the per-task budgets sum to
                more than `total_time_budget_s`.
        """
        from omegaconf import OmegaConf

        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"eval suite config not found: {config_path}")

        raw = OmegaConf.load(config_path)
        container = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(container, dict):
            raise ValueError(f"{config_path} must parse to a mapping, got {type(container)!r}")

        raw_tasks = container.get("tasks") or []
        if not raw_tasks:
            raise ValueError(f"{config_path} declares no tasks; the suite would score nothing")

        tasks = [EvalTask(**dict(entry)) for entry in raw_tasks]

        raw_output = container.get("output_dir", "eval")
        output_dir = Path(str(raw_output)).expanduser()
        if not output_dir.is_absolute():
            output_dir = artifact_root() / output_dir

        cfg = cls(
            tasks=tasks,
            output_dir=output_dir,
            seed=int(container.get("seed", DEFAULT_SEED)),
            total_time_budget_s=float(container.get("total_time_budget_s", 600.0)),
            constants_fingerprint=container.get("constants_fingerprint") or None,
            source_path=config_path,
        )
        cfg.validate_budget()
        return cfg

    def validate_budget(self) -> None:
        """Check that the per-task budgets fit inside the suite budget.

        Raises:
            ValueError: The budgets are inconsistent, which would let `make eval`
                blow through the ten-minute ceiling.
        """
        if self.total_time_budget_s <= 0:
            raise ValueError("total_time_budget_s must be positive")
        task_total = sum(task.time_budget_s for task in self.tasks)
        if task_total > self.total_time_budget_s:
            raise ValueError(
                f"per-task time budgets sum to {task_total:.0f}s but total_time_budget_s is "
                f"{self.total_time_budget_s:.0f}s. Lower max_samples or drop a task; the suite "
                f"must stay inside ten minutes on a single GPU."
            )


def assert_constants_match(cfg: EvalSuiteConfig) -> None:
    """Fail loudly when frozen preprocessing constants have drifted since training.

    The GSD token format, the SAR rendering pipeline, the band order and the index
    definitions must be byte-identical on the training path and the eval path. If
    they are not, the scores are measuring two different preprocessing stacks and
    are not comparable to anything.

    A config that records no fingerprint is a warning, not a failure: the first eval
    run happens before any checkpoint exists, and refusing to run at all would mean
    `make eval` never works.

    Args:
        cfg: The suite config, whose `constants_fingerprint` is the value recorded
            when the adapter being evaluated was trained.

    Raises:
        RuntimeError: The recorded fingerprint and the running one differ.
    """
    running = constants_fingerprint()
    recorded = cfg.constants_fingerprint

    if recorded is None:
        logger.warning(
            "eval suite records no constants_fingerprint; drift cannot be detected. "
            "Running fingerprint is %s -- paste it into %s once the adapter it was "
            "trained with is the one being scored.",
            running,
            cfg.source_path or "the suite config",
        )
        return

    if recorded != running:
        raise RuntimeError(
            "FROZEN CONSTANT DRIFT.\n"
            f"  recorded at training time : {recorded}\n"
            f"  running in this process   : {running}\n"
            "Something in satquery/preprocess/constants.py changed after the weights "
            "being evaluated were trained. The GSD token, SAR rendering, band order or "
            "index definitions no longer match the training path, so these scores are "
            "not comparable and must not be reported. Either restore the constants to "
            "the recorded set, or retrain and record the new fingerprint."
        )

    logger.info("constants fingerprint verified: %s", running)


#: Which loader serves each dataset config, keyed by the config file's stem.
_DATASET_LOADERS: dict[str, str] = {
    "vrsbench": "satquery.data.datasets.vrsbench:VRSBenchDataset",
    "rsvqa": "satquery.data.datasets.rsvqa:RSVQADataset",
    "cdvqa": "satquery.data.datasets.cdvqa:CDVQADataset",
    "bigearthnet_txt": "satquery.data.datasets.bigearthnet_txt:BigEarthNetTxtDataset",
    # Deliberately the raster-backed loader, not PackedFusionDataset. Evaluation must
    # travel the same path a backend request does -- raw GeoTIFF in, tool renders and
    # stretches it -- rather than a packed shortcut that skips the ingest code entirely.
    "bigearthnet_fusion": "satquery.data.datasets.bigearthnet_fusion:BigEarthNetFusionDataset",
}

#: VRSBench splits its tasks into subsets rather than splits, so the task type selects it.
_VRSBENCH_SUBSET_FOR_TASK: dict[str, str] = {
    "vqa": "vqa",
    "caption": "caption",
    "grounding": "referring",
}


def _load_dataset(task: EvalTask) -> Any:
    """Instantiate the loader for this task's dataset config.

    Raises:
        FileNotFoundError: The corpus is absent.
        NotImplementedError: The loader is still a stub.
        KeyError: No loader is wired for this dataset config.
    """
    import importlib

    stem = Path(task.dataset_config).stem
    target = _DATASET_LOADERS.get(stem)
    if target is None:
        raise KeyError(f"no loader wired for dataset config {task.dataset_config!r}")

    module_name, class_name = target.split(":")
    loader = getattr(importlib.import_module(module_name), class_name)

    kwargs: dict[str, Any] = {"split": task.split}
    if stem == "vrsbench":
        subset = _VRSBENCH_SUBSET_FOR_TASK.get(task.task_type)
        if subset is None:
            raise KeyError(f"no VRSBench subset for task type {task.task_type!r}")
        # The config names splits like `test_referring` to keep row names unique; the
        # loader only knows train/val/test, and the subset carries the task.
        kwargs["split"] = task.split.split("_")[0]
        kwargs["subset"] = subset

    return _forced_index(loader(**kwargs))


def _forced_index(dataset: Any) -> Any:
    """Touch the index so a missing corpus fails here rather than mid-loop."""
    len(dataset)
    return dataset


def _request_for(sample: Any) -> ToolRequest:
    """Build the `ToolRequest` for one sample.

    Passes the RAW question, not `Sample.instruction`. The instruction already carries
    the frozen prompt template; handing that to a tool would apply the template twice and
    prompt the adapter off-distribution, quietly costing accuracy that would look like a
    model failure.
    """
    query = str(sample.metadata.get("question") or sample.instruction)
    return ToolRequest(query=query, images=list(sample.images))


#: Metrics served by pycocoevalcap. Corpus-level, so they cannot be averaged per sample.
_CAPTION_METRICS = ("bleu4", "meteor", "rouge_l", "cider")


def _caption_scores(predictions: list[Any], targets: list[Any]) -> dict[str, Any]:
    """Score captions once and memoise, so four metric names cost one scoring pass.

    Returns `NOT_RUN` for every metric when pycocoevalcap is absent -- an optional
    dependency being missing is "not measured", never zero.
    """
    key = (id(predictions), id(targets))
    cached = _CAPTION_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from satquery.eval.metrics.caption import caption_metrics

        references = [[t] if isinstance(t, str) else list(t) for t in targets]
        scores: dict[str, Any] = dict(caption_metrics([str(p) for p in predictions], references))
    except Exception as exc:
        logger.warning("caption metrics unavailable: %s", str(exc)[:160])
        scores = dict.fromkeys(_CAPTION_METRICS, NOT_RUN)

    _CAPTION_CACHE[key] = scores
    return scores


_CAPTION_CACHE: dict[tuple[int, int], dict[str, Any]] = {}


_MASK_METRICS = ("builtup_iou", "water_iou", "index_agreement")


def _score(
    task: EvalTask,
    predictions: list[Any],
    targets: list[Any],
    types: list[str],
    confidences: list[float | None] | None = None,
) -> dict:
    """Reduce predictions to the metrics this task names."""
    from satquery.eval.metrics.grounding import acc_at_tau, mean_iou
    from satquery.eval.metrics.vqa import (
        average_per_type_accuracy,
        exact_match_accuracy,
        per_type_accuracy,
    )

    dataset_key = Path(task.dataset_config).stem
    scores: dict[str, Any] = {}
    for metric in task.metrics:
        if metric == "exact_match":
            scores[metric] = round(exact_match_accuracy(predictions, targets, dataset_key), 4)
        elif metric == "per_type":
            by_type = per_type_accuracy(predictions, targets, types, dataset_key)
            scores[metric] = {k: round(v, 4) for k, v in by_type.items()}
            scores["average_per_type"] = round(average_per_type_accuracy(by_type), 4)
        elif metric.startswith("acc_at_"):
            scores[metric] = round(acc_at_tau(predictions, targets, float(metric[7:])), 4)
        elif metric == "mean_iou":
            scores[metric] = round(mean_iou(predictions, targets), 4)
        elif metric in _MASK_METRICS:
            from satquery.eval.metrics.segmentation import mask_iou, mean_agreement
            from satquery.preprocess.constants import FUSION_EXTRACTION_CLASSES

            if metric == "index_agreement":
                scores[metric] = round(mean_agreement(confidences or []), 4)
            else:
                name = metric.removesuffix("_iou")
                lookup = {"builtup": "built_up"}.get(name, name)
                index = FUSION_EXTRACTION_CLASSES.index(lookup) + 1
                scores[metric] = round(mask_iou(predictions, targets, index), 4)
        elif metric in _CAPTION_METRICS:
            caption_scores = _caption_scores(predictions, targets)
            scores[metric] = caption_scores.get(metric, NOT_RUN)
        else:
            scores[metric] = NOT_RUN
    return scores


def run_task(task: EvalTask, registry: ToolRegistry) -> dict[str, Any]:
    """Score one task by running its tool over its split.

    Loads the dataset, runs the tool sample by sample, and reduces with the metrics the
    task names. A sample whose tool call fails is counted as a miss rather than skipped:
    silently dropping failures lets a model inflate its score by refusing to answer.

    Stops early when `task.time_budget_s` is exhausted, and reports how many samples were
    actually scored so a truncated row is never mistaken for a full one.
    """
    dataset = _load_dataset(task)
    tool = registry.get(task.tool_name)
    limit = min(task.max_samples, len(dataset))
    is_grounding = task.task_type == "grounding"
    is_mask = task.task_type == "fusion_extraction"
    confidences: list[float | None] = []

    predictions: list[Any] = []
    targets: list[Any] = []
    types: list[str] = []
    failures = 0
    first_error: str | None = None
    started = time.perf_counter()

    for index in range(limit):
        if time.perf_counter() - started > task.time_budget_s:
            logger.warning(
                "task %s hit its %.0fs budget after %d/%d samples",
                task.name,
                task.time_budget_s,
                index,
                limit,
            )
            break

        sample = dataset[index]
        result = tool.run(_request_for(sample))
        if not result.ok:
            failures += 1
            if first_error is None:
                first_error = result.error

        if is_mask:
            # The prediction is a raster on disk, and the target is the reference mask
            # the loader materialised. A failed call appends None, which `mask_iou`
            # scores as an empty prediction rather than dropping the sample -- dropping
            # it would silently report only the calls that worked.
            predictions.append(result.evidence.mask_path if result.ok else None)
            targets.append(sample.mask_path)
            confidences.append(result.confidence if result.ok else None)
        elif is_grounding:
            boxes = result.evidence.boxes if result.ok else []
            predictions.append(boxes[0] if boxes else None)
            targets.append(list(sample.boxes))
        else:
            predictions.append(result.answer or "" if result.ok else "")
            targets.append(sample.answer_text or "")
        types.append(str(sample.metadata.get("question_type", task.task_type)))

    if not predictions:
        raise RuntimeError(f"task {task.name!r} scored no samples before its budget expired")

    # A task whose tool failed on EVERY sample has not been measured. Reducing those
    # failures gives 0.0, which is indistinguishable in a results table from a model that
    # genuinely scored zero -- and that is exactly the fabricated-number failure the
    # project forbids. Refuse to report a score instead.
    if failures == len(predictions):
        raise RuntimeError(
            f"task {task.name!r}: the tool failed on all {failures} samples, so nothing "
            f"was measured. Reporting 0.0 here would be a fabricated score. First error: "
            f"{first_error}"
        )
    if failures:
        logger.warning(
            "task %s: %d/%d tool calls failed and are counted as misses; the score is a "
            "lower bound. First error: %s",
            task.name,
            failures,
            len(predictions),
            first_error,
        )

    scores = _score(task, predictions, targets, types, confidences)
    scores["samples_scored"] = len(predictions)
    scores["samples_requested"] = task.max_samples
    scores["tool_failures"] = failures
    scores["elapsed_s"] = round(time.perf_counter() - started, 1)
    logger.info("task %s scored %d samples: %s", task.name, len(predictions), scores)
    return scores


def _blank_result(task: EvalTask, status: str, *, error: str | None = None) -> dict[str, Any]:
    """Build the honest not-run record for a task: every metric is `TBD`."""
    return {
        "tool": task.tool_name,
        "dataset": task.dataset_config,
        "split": task.split,
        "task_type": task.task_type,
        "requirement": task.requirement,
        "max_samples": task.max_samples,
        "implemented": False,
        "status": status,
        "metrics": {name: NOT_RUN for name in task.metrics},
        "elapsed_s": 0.0,
        "error": error,
    }


def _tool_is_implemented(task: EvalTask, registry: ToolRegistry) -> bool:
    """Whether the registry declares `task.tool_name` as implemented rather than a stub."""
    try:
        return registry.get_spec(task.tool_name).implemented
    except KeyError:
        return False


def run_suite(cfg: EvalSuiteConfig, registry: ToolRegistry | None = None) -> dict[str, Any]:
    """Run every task in `cfg`, isolating failures, and return the results mapping.

    One missing model must never abort the suite: each task is run inside its own
    try/except, and anything that could not run is recorded with `TBD` metrics and
    the reason. That is what lets `make eval` pass end to end today and still tell
    the truth about what has actually been measured.

    Args:
        cfg: The loaded suite config.
        registry: Registry to resolve tools against. Defaults to the package registry.

    Returns:
        A mapping with a `suite` block (fingerprint, seed, budget, wall clock) and a
        `tasks` block keyed by task name.
    """
    from satquery.models.registry import REGISTRY
    from satquery.utils.seed import seed_everything

    resolved_registry = registry if registry is not None else REGISTRY

    assert_constants_match(cfg)
    cfg.validate_budget()
    seed_everything(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "running %d eval tasks, total budget %.0fs, output %s",
        len(cfg.tasks),
        cfg.total_time_budget_s,
        cfg.output_dir,
    )

    tasks: dict[str, Any] = {}
    suite_start = time.perf_counter()

    for task in cfg.tasks:
        elapsed_suite = time.perf_counter() - suite_start
        if elapsed_suite >= cfg.total_time_budget_s:
            logger.warning(
                "suite budget of %.0fs exhausted after %.1fs; skipping task %s",
                cfg.total_time_budget_s,
                elapsed_suite,
                task.name,
            )
            tasks[task.name] = _blank_result(
                task,
                _STATUS_SKIPPED_BUDGET,
                error=f"suite time budget of {cfg.total_time_budget_s:.0f}s exhausted",
            )
            continue

        implemented = _tool_is_implemented(task, resolved_registry)
        task_start = time.perf_counter()
        try:
            metrics = run_task(task, resolved_registry)
        except NotImplementedError as exc:
            logger.warning("task %s not runnable yet: %s", task.name, exc)
            record = _blank_result(task, _STATUS_NOT_IMPLEMENTED, error=str(exc))
        except Exception as exc:
            logger.error("task %s failed: %s\n%s", task.name, exc, traceback.format_exc())
            record = _blank_result(task, _STATUS_ERROR, error=f"{type(exc).__name__}: {exc}")
        else:
            record = {
                "tool": task.tool_name,
                "dataset": task.dataset_config,
                "split": task.split,
                "task_type": task.task_type,
                "requirement": task.requirement,
                "max_samples": task.max_samples,
                "implemented": implemented,
                "status": _STATUS_OK,
                "metrics": {name: metrics.get(name, NOT_RUN) for name in task.metrics},
                "elapsed_s": 0.0,
                "error": None,
            }

        record["elapsed_s"] = round(time.perf_counter() - task_start, 3)
        record["implemented"] = record["implemented"] and record["status"] == _STATUS_OK
        if record["elapsed_s"] > task.time_budget_s:
            logger.warning(
                "task %s took %.1fs, over its %.0fs budget",
                task.name,
                record["elapsed_s"],
                task.time_budget_s,
            )
        tasks[task.name] = record

    total_elapsed = round(time.perf_counter() - suite_start, 3)
    ran = sum(1 for record in tasks.values() if record["status"] == _STATUS_OK)
    logger.info(
        "eval suite finished in %.1fs: %d/%d tasks produced real scores, the rest are %s",
        total_elapsed,
        ran,
        len(tasks),
        NOT_RUN,
    )
    if total_elapsed > cfg.total_time_budget_s:
        logger.warning(
            "suite took %.1fs, over the %.0fs budget set for `make eval`",
            total_elapsed,
            cfg.total_time_budget_s,
        )

    return {
        "suite": {
            "config_path": str(cfg.source_path) if cfg.source_path else None,
            "seed": cfg.seed,
            "constants_fingerprint": constants_fingerprint(),
            "recorded_fingerprint": cfg.constants_fingerprint,
            "total_time_budget_s": cfg.total_time_budget_s,
            "elapsed_s": total_elapsed,
            "tasks_total": len(tasks),
            "tasks_scored": ran,
            "output_dir": str(cfg.output_dir),
        },
        "tasks": tasks,
    }
