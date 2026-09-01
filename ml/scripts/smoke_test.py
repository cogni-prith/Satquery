#!/usr/bin/env python3
"""End-to-end smoke test over synthetic imagery with known ground truth.

Runs ingest, preprocessing, the symbolic layer and the template verbalizer, then asserts
the computed areas match the geometry that was drawn. No model weights are loaded and
none are needed -- which is the point. Every quantitative answer this system gives is
decided by code that runs here, so a break in the answer path fails on a laptop in under
a second rather than in an eval three weeks later.

Exits non-zero on any failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.io.raster import read_image_ref, write_raster
from satquery.preprocess.constants import constants_fingerprint
from satquery.preprocess.indices import ndwi
from satquery.preprocess.sar import render_sar
from satquery.symbolic import measures, predicates
from satquery.symbolic.record import AnswerRecord, AreaDelta, Fact
from satquery.symbolic.thresholds import classify_trend, confidence_band
from satquery.utils.logging import configure_logging, get_logger
from satquery.utils.seed import seed_everything
from satquery.verbalize.templates import verbalize

LOG = get_logger("smoke")

WATER, BUILT, VEG = 1, 2, 3
GSD_M = 10.0
SIDE = 100

#: Ground truth, drawn deliberately so the areas are known before anything computes them.
WATER_PX_T1 = 20 * 20  # a 20x20 lake
BUILT_PX_T1 = 10 * 10  # a 10x10 settlement
BUILT_PX_T2 = 30 * 30  # the settlement grows
PIXEL_AREA = GSD_M**2


def _fail(message: str) -> None:
    LOG.error("SMOKE FAILED: %s", message)
    raise SystemExit(1)


def _check(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def build_scene() -> tuple[np.ndarray, np.ndarray]:
    """Two dates of a synthetic land cover mask, with known class areas."""
    t1 = np.full((SIDE, SIDE), VEG, dtype=np.uint8)
    t1[10:30, 10:30] = WATER
    t1[70:80, 70:80] = BUILT

    t2 = t1.copy()
    t2[60:90, 60:90] = BUILT  # settlement expands over vegetation
    return t1, t2


def check_ingest(tmp: Path) -> None:
    """A written raster reads back with the metadata the contract promises."""
    rng = np.random.default_rng(0)
    optical = rng.integers(200, 3000, size=(4, SIDE, SIDE)).astype(np.uint16)
    path = write_raster(tmp / "optical.tif", optical, band_names=["B02", "B03", "B04", "B08"])

    ref = read_image_ref(path)
    _check(ref.band_names == ["B02", "B03", "B04", "B08"], f"band order lost: {ref.band_names}")
    # No affine transform was written, so the GSD is genuinely unknown and must say so
    # rather than reading 1.0 off the identity matrix.
    _check(ref.gsd_m is None, f"a non-georeferenced raster reported gsd_m={ref.gsd_m}")
    _check(ref.gsd_token == "<gsd:unknown>", f"gsd token was {ref.gsd_token}")
    LOG.info("ingest ok: %s, %s", ref.band_names, ref.gsd_token)


def check_sar() -> None:
    """The frozen SAR pipeline renders and flags a single-polarisation input."""
    rng = np.random.default_rng(1)
    vv = rng.uniform(0.01, 1.2, size=(SIDE, SIDE))
    vh = rng.uniform(0.005, 0.6, size=(SIDE, SIDE))

    rendered, warnings = render_sar(vv, vh)
    _check(rendered.shape == (SIDE, SIDE, 3), f"SAR render shape {rendered.shape}")
    _check(rendered.dtype == np.uint8, f"SAR render dtype {rendered.dtype}")
    _check(not warnings, f"dual-pol render should not warn, got {warnings}")

    _, single_warnings = render_sar(vv, None)
    _check(bool(single_warnings), "single-pol input must emit a warning")
    LOG.info("sar ok: dual-pol clean, single-pol warned")


def check_symbolic(t1: np.ndarray, t2: np.ndarray) -> AnswerRecord:
    """Areas and trends match the geometry that was drawn, not approximately."""
    water_area = measures.class_area_m2(t1, WATER, GSD_M)
    _check(
        water_area == WATER_PX_T1 * PIXEL_AREA,
        f"water area {water_area} != known {WATER_PX_T1 * PIXEL_AREA}",
    )

    delta = measures.area_delta(t1, t2, BUILT, GSD_M)
    expected_abs = (BUILT_PX_T2 - BUILT_PX_T1) * PIXEL_AREA
    _check(
        delta["absolute_m2"] == expected_abs,
        f"built-up delta {delta['absolute_m2']} != known {expected_abs}",
    )
    trend = classify_trend(delta["absolute_m2"], delta["relative"])
    _check(trend == "increased", f"a 8x growth classified as {trend}")

    matrix = measures.change_matrix(t1, t2, gsd_m=GSD_M)
    _check((VEG, BUILT) in matrix, "vegetation-to-built transition not detected")

    components = measures.connected_components(t1, WATER, min_area_px=4)
    _check(len(components) == 1, f"expected one lake, found {len(components)}")
    ranked = predicates.resolve_reference(components, "the largest water body", SIDE, SIDE)
    _check(bool(ranked) and "largest" in ranked[0]["matched"], "referring phrase unresolved")

    # Confidence is agreement between the learned mask and the deterministic index. Here
    # the "learned" mask is the ground truth, so agreement is exactly 1.0 and the band
    # must read high -- if this ever drifts, the confidence wording is lying.
    agreement = measures.agreement_iou(t1 == WATER, t1 == WATER)
    _check(agreement == 1.0, f"self-agreement was {agreement}")
    _check(confidence_band(agreement) == "high", "perfect agreement did not band high")

    record = AnswerRecord(
        intent="change_trend",
        facts=[
            Fact(
                key="built_up_area_t1",
                value=BUILT_PX_T1 * PIXEL_AREA,
                unit="m2",
                provenance="symbolic.measures.class_area_m2",
            ),
            Fact(
                key="built_up_area_t2",
                value=BUILT_PX_T2 * PIXEL_AREA,
                unit="m2",
                provenance="symbolic.measures.class_area_m2",
            ),
            Fact(
                key="built_up_delta",
                value=delta["absolute_m2"],
                unit="m2",
                provenance="symbolic.measures.area_delta",
                confidence=agreement,
            ),
        ],
        area_deltas={
            "built_up": AreaDelta(
                area_t1_m2=delta["area_t1_m2"],
                area_t2_m2=delta["area_t2_m2"],
                absolute_m2=delta["absolute_m2"],
                relative=delta["relative"],
                trend=trend,
            )
        },
        agreement_scores={"water": agreement},
    )
    record.validate_provenance()
    LOG.info("symbolic ok: areas exact, trend=%s, agreement=%.2f", trend, agreement)
    return record


def check_verbalizer(record: AnswerRecord) -> str:
    """Every claim in the text traces to a fact, and the firewall holds."""
    import inspect

    params = set(inspect.signature(verbalize).parameters)
    _check(params == {"record"}, f"verbalizer signature widened to {params}")

    text = verbalize(record)
    _check("increased" in text, f"trend missing from text: {text!r}")
    _check("Confidence is high" in text, f"confidence missing from text: {text!r}")

    from satquery.verbalize.llm import NumberGuardError, check_numbers

    check_numbers(record, text, text)  # the draft is trivially supported by itself
    try:
        check_numbers(record, text, "Built-up grew by 4321 hectares.")
    except NumberGuardError:
        pass
    else:
        _fail("the number guard let an invented figure through")

    LOG.info("verbalizer ok: firewall intact, number guard active")
    return text


def check_indices() -> None:
    """A synthetic water body is brighter in NDWI than a synthetic field."""
    green = np.full((10, 10), 0.30)
    nir = np.full((10, 10), 0.05)
    water_ndwi = ndwi(green, nir)
    field_ndwi = ndwi(np.full((10, 10), 0.10), np.full((10, 10), 0.40))
    _check(
        water_ndwi.mean() > field_ndwi.mean(),
        "NDWI did not separate water from vegetation",
    )
    LOG.info("indices ok: NDWI separates water from vegetation")


def main() -> int:
    configure_logging()
    seed_everything(1337)
    LOG.info("frozen constants fingerprint: %s", constants_fingerprint())

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        check_ingest(tmp)

    check_sar()
    check_indices()

    t1, t2 = build_scene()
    record = check_symbolic(t1, t2)
    text = check_verbalizer(record)

    print("\n--- AnswerRecord ---")
    print(record.model_dump_json(indent=2, exclude_defaults=True))
    print("\n--- verbalized ---")
    print(text)
    print("\nSMOKE PASSED: every computed area matched the geometry it was drawn from.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
