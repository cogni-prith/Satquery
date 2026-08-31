"""Answer normalisation and the scored metrics. Worth several accuracy points each."""

from __future__ import annotations

import pytest

from satquery.eval.answer_norm import (
    answers_match,
    normalize_answer,
    normalize_cdvqa_class,
    normalize_for_dataset,
    normalize_yes_no,
)
from satquery.eval.metrics.grounding import acc_at_tau, best_match, iou, mean_iou
from satquery.eval.metrics.vqa import (
    average_per_type_accuracy,
    exact_match_accuracy,
    per_type_accuracy,
)
from satquery.models.registry import REGISTRY
from satquery.preprocess.constants import CDVQA_ANSWERS
from satquery.serve.contracts import BoundingBox


def _box(x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
    return BoundingBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1, label="x")


# -- normalisation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Yes.", "yes"), ("  YES  ", "yes"), ("the water", "water"), ("A building!", "building")],
)
def test_normalisation_strips_formatting_not_content(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_yes_no_collapses_common_affirmatives_and_negatives() -> None:
    assert normalize_yes_no("Yeah") == "yes"
    assert normalize_yes_no("nope") == "no"


def test_yes_no_leaves_a_non_binary_answer_visibly_wrong() -> None:
    # Coercing this to "yes" or "no" would turn a wrong answer into a right one.
    assert normalize_yes_no("seventeen") == "seventeen"


def test_every_cdvqa_class_normalises_back_to_itself() -> None:
    for name in CDVQA_ANSWERS:
        assert normalize_cdvqa_class(name) == name


def test_cdvqa_synonyms_map_onto_the_closed_set() -> None:
    assert normalize_cdvqa_class("Building") == "buildings"
    # The dataset's own token is `NVG_surface`, not the prose CLAUDE.md quotes.
    assert normalize_cdvqa_class("non vegetated ground surface") == "NVG_surface"
    assert normalize_cdvqa_class("NVG surface") == "NVG_surface"


def test_change_ratio_buckets_tolerate_spacing_and_hyphens() -> None:
    """`0-10` and `0 to 10` mean the same bucket as `0_to_10` and must not score zero."""
    for written in ("0_to_10", "0 to 10", "0-10"):
        assert normalize_cdvqa_class(written) == "0_to_10"
    assert normalize_cdvqa_class("90-100") == "90_to_100"


def test_an_out_of_vocabulary_ratio_bucket_is_not_invented() -> None:
    # 15_to_25 is not a real bucket; normalising must not fabricate one.
    assert normalize_cdvqa_class("15 to 25") not in CDVQA_ANSWERS


def test_an_out_of_vocabulary_cdvqa_answer_is_not_snapped_to_a_class() -> None:
    assert normalize_cdvqa_class("spaceship") not in CDVQA_ANSWERS


def test_normalisation_is_applied_to_both_sides() -> None:
    assert answers_match("Yes.", "yes", "rsvqa")
    assert not answers_match("trees", "water", "cdvqa")


def test_unknown_dataset_falls_back_to_the_generic_normaliser() -> None:
    assert normalize_for_dataset("The Water.", "mystery") == "water"


# -- grounding --------------------------------------------------------------------------


def test_iou_of_identical_boxes_is_one() -> None:
    assert iou(_box(0, 0, 10, 10), _box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert iou(_box(0, 0, 10, 10), _box(20, 20, 30, 30)) == 0.0


def test_iou_of_a_half_overlap() -> None:
    assert iou(_box(0, 0, 10, 10), _box(5, 0, 15, 10)) == pytest.approx(1.0 / 3.0)


def test_best_match_picks_the_closest_reference() -> None:
    assert best_match(_box(0, 0, 10, 10), [_box(50, 50, 60, 60), _box(0, 0, 10, 10)]) == 1.0
    assert best_match(_box(0, 0, 10, 10), []) == 0.0


def test_acc_at_tau_counts_a_missing_prediction_as_a_miss() -> None:
    # Skipping a None would let a model inflate its score by declining to answer.
    accuracy = acc_at_tau([_box(0, 0, 10, 10), None], [[_box(0, 0, 10, 10)], [_box(0, 0, 10, 10)]])
    assert accuracy == pytest.approx(0.5)


def test_acc_at_tau_respects_the_threshold() -> None:
    preds = [_box(0, 0, 10, 10)]
    targets = [[_box(5, 0, 15, 10)]]  # IoU is 1/3
    assert acc_at_tau(preds, targets, tau=0.5) == 0.0
    assert acc_at_tau(preds, targets, tau=0.3) == 1.0


def test_grounding_metrics_reject_misaligned_sequences() -> None:
    with pytest.raises(ValueError, match="must align"):
        acc_at_tau([_box(0, 0, 1, 1)], [])
    with pytest.raises(ValueError, match="must align"):
        mean_iou([_box(0, 0, 1, 1)], [])


def test_mean_iou_averages_best_matches() -> None:
    value = mean_iou([_box(0, 0, 10, 10), None], [[_box(0, 0, 10, 10)], [_box(0, 0, 10, 10)]])
    assert value == pytest.approx(0.5)


def test_empty_evaluation_is_zero_not_a_crash() -> None:
    assert acc_at_tau([], []) == 0.0
    assert mean_iou([], []) == 0.0


# -- vqa --------------------------------------------------------------------------------


def test_exact_match_uses_normalisation() -> None:
    assert exact_match_accuracy(["Yes.", "No!"], ["yes", "no"], "rsvqa") == pytest.approx(1.0)


def test_exact_match_rejects_misaligned_sequences() -> None:
    with pytest.raises(ValueError, match="must align"):
        exact_match_accuracy(["yes"], [])


def test_per_type_accuracy_splits_by_question_family() -> None:
    result = per_type_accuracy(
        ["yes", "no", "3"], ["yes", "yes", "3"], ["presence", "presence", "count"], "rsvqa"
    )
    assert result == {"count": pytest.approx(1.0), "presence": pytest.approx(0.5)}


def test_per_type_result_is_sorted_for_a_stable_report() -> None:
    result = per_type_accuracy(["a", "b"], ["a", "b"], ["zeta", "alpha"])
    assert list(result) == ["alpha", "zeta"]


def test_average_per_type_is_unweighted() -> None:
    # Unweighted, because RSVQA's type distribution is heavily skewed.
    assert average_per_type_accuracy({"a": 1.0, "b": 0.0}) == pytest.approx(0.5)
    assert average_per_type_accuracy({}) == 0.0


# -- harness scoring --------------------------------------------------------------------
# These cover the reduction and request-building logic only. Nothing here touches a model:
# `run_task` itself needs weights and a GPU, so it is exercised by `make eval`, not here.


def _task(**overrides):
    from satquery.eval.harness import EvalTask

    base = {
        "name": "t",
        "dataset_config": "data/vrsbench.yaml",
        "split": "test",
        "task_type": "vqa",
        "tool_name": "vlm.vqa",
        "metrics": ["exact_match", "per_type"],
        "max_samples": 10,
        "time_budget_s": 60.0,
    }
    return EvalTask(**{**base, **overrides})


def test_score_reduces_exact_match_and_per_type() -> None:
    from satquery.eval.harness import _score

    scores = _score(_task(), ["yes", "no"], ["yes", "yes"], ["presence", "presence"])
    assert scores["exact_match"] == pytest.approx(0.5)
    assert scores["per_type"]["presence"] == pytest.approx(0.5)
    assert "average_per_type" in scores


def test_score_parses_the_tau_out_of_the_metric_name() -> None:
    from satquery.eval.harness import _score

    box = _box(0, 0, 10, 10)
    scores = _score(
        _task(task_type="grounding", metrics=["acc_at_0.5", "acc_at_0.7"]),
        [box],
        [[_box(0, 0, 10, 9)]],  # IoU 0.9 -> passes both thresholds
        ["ref"],
    )
    assert scores["acc_at_0.5"] == 1.0
    assert scores["acc_at_0.7"] == 1.0


def test_an_unknown_metric_reports_not_run_rather_than_zero() -> None:
    """ "Not measured" and "scored zero" must never look alike in a results table."""
    from satquery.eval.harness import NOT_RUN, _score

    scores = _score(_task(metrics=["some_metric_we_do_not_implement"]), ["a"], ["a"], ["x"])
    assert scores["some_metric_we_do_not_implement"] == NOT_RUN


def test_caption_metrics_are_dispatched_when_available() -> None:
    """pycocoevalcap was installed but `_score` never called it, so captions stayed TBD."""
    pytest.importorskip("pycocoevalcap")
    from satquery.eval.harness import _score

    scores = _score(
        _task(task_type="caption", metrics=["bleu4", "rouge_l"]),
        ["a road with cars", "a green field"],
        ["a road with cars", "a green field"],
        ["caption", "caption"],
    )
    assert isinstance(scores["bleu4"], float)
    assert scores["bleu4"] > 0.9  # identical captions


def test_request_uses_the_raw_question_not_the_templated_instruction() -> None:
    """Double-templating is silent: the model still answers, just off-distribution."""
    from satquery.data.schema import AnswerType, Sample
    from satquery.eval.harness import _request_for
    from satquery.serve.contracts import ImageRef, Modality, TaskType

    sample = Sample(
        sample_id="s",
        task=TaskType.VQA,
        answer_type=AnswerType.OPEN_TEXT,
        images=[ImageRef(path="/tmp/a.tif", modality=Modality.OPTICAL_RGB, width=8, height=8)],
        instruction="[vqa] What colour. A short answer to the question is",
        answer_text="red",
        source="vrsbench",
        metadata={"question": "What colour"},
    )
    assert _request_for(sample).query == "What colour"
    assert "[vqa]" not in _request_for(sample).query


def test_unknown_dataset_config_is_a_loud_key_error() -> None:
    from satquery.eval.harness import _load_dataset

    with pytest.raises(KeyError, match="no loader wired"):
        _load_dataset(_task(dataset_config="data/nonexistent.yaml"))


# -- caption metrics ----------------------------------------------------------------------

pycocoevalcap = pytest.importorskip("pycocoevalcap", reason="caption metrics are optional")


def test_identical_captions_score_a_perfect_bleu() -> None:
    from satquery.eval.metrics.caption import caption_metrics

    scores = caption_metrics(["a road with three vehicles"], [["a road with three vehicles"]])
    assert scores["bleu4"] == pytest.approx(1.0, abs=1e-6)
    assert scores["rouge_l"] == pytest.approx(1.0, abs=1e-6)


def test_unrelated_captions_score_near_zero() -> None:
    from satquery.eval.metrics.caption import caption_metrics

    scores = caption_metrics(["a road with three vehicles"], [["dense forest canopy"]])
    assert scores["bleu4"] < 0.05


def test_misaligned_inputs_are_rejected() -> None:
    from satquery.eval.metrics.caption import caption_metrics

    with pytest.raises(ValueError, match="must align"):
        caption_metrics(["a"], [])


def test_all_four_reported_metrics_are_produced() -> None:
    from satquery.eval.metrics.caption import CAPTION_METRICS, caption_metrics

    scores = caption_metrics(
        ["a road with vehicles", "a green field"],
        [["a road with cars"], ["a green meadow"]],
    )
    assert set(scores) == set(CAPTION_METRICS)


def test_a_task_whose_tool_always_fails_refuses_to_report_a_score() -> None:
    """100% tool failure must raise, not report 0.0.

    Reducing failures to 0.0 produces a results table that is indistinguishable from a
    model which genuinely scored zero. This actually happened: an unrelated image-loading
    bug made every tool call fail, and the suite reported `0.0000` across six rows while
    announcing "6/7 tasks produced real scores".
    """
    from satquery.eval.harness import run_task
    from satquery.models.registry import ToolRegistry
    from satquery.serve.contracts import Evidence, ToolResult

    class AlwaysFails:
        spec = REGISTRY.get_spec("vlm.vqa")

        def run(self, request):
            return ToolResult(
                request_id=request.request_id,
                tool_name="vlm.vqa",
                tool_version="0.1.0",
                evidence=Evidence(),
                error="synthetic failure",
            )

    registry = ToolRegistry()
    registry.register(REGISTRY.get_spec("vlm.vqa"), AlwaysFails)

    with pytest.raises(RuntimeError, match="failed on all"):
        run_task(_task(max_samples=3), registry)
