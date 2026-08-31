#!/usr/bin/env python3
"""End-to-end ingest and preprocessing check on synthetic rasters. No weights needed.

Builds a synthetic optical scene, a synthetic SAR scene and a synthetic bi-temporal
pair in a temporary directory, pushes each through the real ingest and preprocessing
path, and prints what came out. Exits non-zero on any failure.

This is the check that has to keep passing. It exercises every deterministic component
the learned models will sit on top of -- band mapping, modality inference, GSD from the
affine transform, the SAR rendering pipeline, the spectral indices, the compatibility
gate and the registry's candidate filter -- without needing a GPU or a single
downloaded byte.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from satquery.io.raster import read_image_ref, read_raster, write_raster  # noqa: E402
from satquery.io.validate import representative_gsd, validate_inputs  # noqa: E402
from satquery.models.registry import REGISTRY  # noqa: E402
from satquery.preprocess.constants import constants_fingerprint  # noqa: E402
from satquery.preprocess.indices import compute_indices  # noqa: E402
from satquery.preprocess.optical import reorder_bands  # noqa: E402
from satquery.preprocess.sar import render_sar  # noqa: E402
from satquery.serve.contracts import InputConfig, ToolRequest, Trace  # noqa: E402
from satquery.utils.logging import get_logger  # noqa: E402
from satquery.utils.seed import seed_everything  # noqa: E402

LOG = get_logger("smoke")

# A projected CRS in metres, so the GSD comes straight off the transform.
UTM_43N = "EPSG:32643"
TILE = 96


def _transform(gsd_m: float) -> tuple[float, float, float, float, float, float]:
    """North-up affine at the given ground sampling distance."""
    return (gsd_m, 0.0, 600000.0, 0.0, -gsd_m, 2000000.0)


def _synthetic_optical(path: Path, gsd_m: float, when: datetime, water_cols: int) -> Path:
    """A 10-band multispectral scene with a water strip on the left."""
    rng = np.random.default_rng(0)
    bands = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
    stack = rng.uniform(0.05, 0.35, size=(len(bands), TILE, TILE)).astype("float32")
    # Water: high green, very low NIR. That is what NDWI keys on.
    stack[bands.index("B03"), :, :water_cols] = 0.22
    stack[bands.index("B08"), :, :water_cols] = 0.02
    # Vegetation on the right: high NIR, low red.
    stack[bands.index("B08"), :, water_cols:] = 0.45
    stack[bands.index("B04"), :, water_cols:] = 0.06
    return write_raster(
        path, stack, crs=UTM_43N, transform=_transform(gsd_m), band_names=bands
    ) and _stamp(path, when)


def _synthetic_sar(path: Path, gsd_m: float, when: datetime) -> Path:
    """A dual-pol SAR scene in linear amplitude, speckled, with a dark water strip."""
    rng = np.random.default_rng(1)
    speckle = rng.gamma(shape=4.0, scale=0.25, size=(TILE, TILE))
    vv = np.full((TILE, TILE), 0.30)
    vv[:, : TILE // 3] = 0.004  # smooth water reflects away from the sensor
    vh = vv * 0.35
    stack = np.stack([vv * speckle, vh * speckle]).astype("float32")
    return write_raster(
        path, stack, crs=UTM_43N, transform=_transform(gsd_m), band_names=["VV", "VH"]
    ) and _stamp(path, when)


def _stamp(path: Path, when: datetime) -> Path:
    """Write an acquisition timestamp into the raster tags, as a real product would."""
    import rasterio

    with rasterio.open(path, "r+") as dataset:
        dataset.update_tags(ACQUISITION_DATETIME=when.strftime("%Y-%m-%dT%H:%M:%S"))
    return path


def _describe(label: str, ref) -> None:
    """Print an ImageRef the way the backend will see it."""
    LOG.info("%s", label)
    LOG.info("    path       %s", ref.path.name)
    LOG.info("    modality   %s", ref.modality.value)
    LOG.info("    gsd        %s  ->  token %s", ref.gsd_m, ref.gsd_token)
    LOG.info("    crs        %s", ref.crs)
    LOG.info("    size       %sx%s, bands %s", ref.width, ref.height, ref.band_names)
    LOG.info("    timestamp  %s", ref.timestamp)
    for warning in ref.warnings:
        LOG.info("    warning    %s", warning)


def main() -> int:
    """Run every stage. Returns 0 on success, 1 on the first failure."""
    seed_everything()
    LOG.info("frozen constants fingerprint: %s", constants_fingerprint())

    with tempfile.TemporaryDirectory(prefix="satquery-smoke-") as raw_tmp:
        tmp = Path(raw_tmp)
        t1 = datetime(2024, 1, 15, 5, 30, tzinfo=timezone.utc)
        t2 = datetime(2024, 7, 20, 5, 30, tzinfo=timezone.utc)

        optical_t1 = _synthetic_optical(tmp / "optical_t1.tif", 10.0, t1, TILE // 3)
        optical_t2 = _synthetic_optical(tmp / "optical_t2.tif", 10.0, t2, TILE // 6)
        sar_path = _synthetic_sar(tmp / "sar_t1.tif", 10.0, t1)

        # -- 1. optical ingest, band reordering, indices ------------------------------
        LOG.info("=" * 78)
        LOG.info("1. OPTICAL: ingest, canonical band order, spectral indices")
        array, optical_ref = read_raster(optical_t1)
        _describe("   ImageRef", optical_ref)

        reordered = reorder_bands(array, optical_ref.band_names)
        LOG.info("    reordered to canonical 10-band order: %s", reordered.shape)

        maps, index_warnings = compute_indices(array, optical_ref.band_names)
        for name, index_map in sorted(maps.items()):
            LOG.info("    %s range [%.3f, %.3f]", name.upper(), index_map.min(), index_map.max())
        for warning in index_warnings:
            LOG.info("    warning    %s", warning)
        if "ndwi" not in maps:
            raise AssertionError("NDWI was not computed from a stack that contains B03 and B08")

        # -- 2. SAR ingest and the frozen five-step rendering -------------------------
        LOG.info("=" * 78)
        LOG.info("2. SAR: ingest and the frozen five-step rendering pipeline")
        sar_array, sar_ref = read_raster(sar_path)
        _describe("   ImageRef", sar_ref)

        rgb, sar_warnings = render_sar(sar_array[0], sar_array[1])
        LOG.info(
            "    pseudo-RGB %s dtype=%s range=[%d, %d]", rgb.shape, rgb.dtype, rgb.min(), rgb.max()
        )
        for warning in sar_warnings:
            LOG.info("    warning    %s", warning)
        if rgb.shape != (TILE, TILE, 3) or rgb.dtype != np.uint8:
            raise AssertionError(f"unexpected SAR render output: {rgb.shape} {rgb.dtype}")

        single, single_warnings = render_sar(sar_array[0])
        if not (single[..., 2] == 0).all():
            raise AssertionError("single-pol blue channel must be zeros")
        LOG.info(
            "    single-pol path: R==G=%s, B all zero, warning emitted=%s",
            bool((single[..., 0] == single[..., 1]).all()),
            bool(single_warnings),
        )

        # -- 3. the compatibility gate on each input configuration --------------------
        LOG.info("=" * 78)
        LOG.info("3. GATE: InputConfig and registry candidates for each configuration")
        optical_t2_ref = read_image_ref(optical_t2)

        cases = [
            ("single optical", [optical_ref]),
            ("bi-temporal optical pair", [optical_ref, optical_t2_ref]),
            ("cross-modal optical + SAR pair", [optical_ref, sar_ref]),
        ]
        expected = [InputConfig.SINGLE, InputConfig.BI_TEMPORAL_PAIR, InputConfig.CROSS_MODAL_PAIR]

        for (label, refs), want in zip(cases, expected, strict=True):
            config, gate_warnings = validate_inputs(refs)
            gsd = representative_gsd(refs)
            candidates = REGISTRY.candidates(
                input_config=config,
                modalities=[ref.modality for ref in refs],
                gsd_m=gsd,
            )
            LOG.info("    %-32s -> %s", label, config.value)
            LOG.info("        representative gsd %s", gsd)
            LOG.info("        candidates: %s", [spec.name for spec in candidates] or "none")
            for warning in gate_warnings:
                LOG.info("        warning %s", warning)
            if config is not want:
                raise AssertionError(f"{label}: expected {want.value}, got {config.value}")

        # -- 4. a real tool call, end to end ------------------------------------------
        LOG.info("=" * 78)
        LOG.info("4. TOOL: indices.deterministic end to end (the one implemented tool)")
        request = ToolRequest(
            query="Use the optical and SAR images together to identify built-up and water-covered regions.",
            images=[optical_ref, sar_ref],
        )
        config, _ = validate_inputs(request.images)
        trace = Trace(request_id=request.request_id, input_config=config)
        result = REGISTRY.get("indices.deterministic").run(request)
        trace.record(result)

        LOG.info("    answer      %s", result.answer)
        LOG.info("    confidence  %s", result.confidence)
        LOG.info("    index maps  %s", sorted(result.evidence.index_maps))
        LOG.info("    latency     %.2f ms", result.latency_ms)
        for warning in result.warnings:
            LOG.info("    warning     %s", warning)
        if not result.ok:
            raise AssertionError(f"deterministic index tool failed: {result.error}")
        LOG.info("    trace:\n%s", trace.to_json())

        # -- 5. a stub tool must fail honestly ----------------------------------------
        LOG.info("=" * 78)
        LOG.info("5. STUBS: an unimplemented tool must report an error, never invent an answer")
        # Deliberately a tool with no weights behind it. The three VLM tools are now
        # really implemented, so calling one here would load an 8 GB checkpoint and make
        # this script slow and weights-dependent -- it must keep passing on a clean
        # machine with nothing downloaded.
        stub_name = "fusion.extraction"
        stub_result = REGISTRY.get(stub_name).run(
            ToolRequest(
                query="Identify built-up and water-covered regions.",
                images=[optical_ref, sar_ref],
            )
        )
        if stub_result.ok or stub_result.answer is not None:
            raise AssertionError(
                f"{stub_name} returned a successful result while unimplemented; a "
                "fabricated answer is exactly what the honest-stub rule exists to prevent"
            )
        LOG.info("    %s error: %s", stub_name, stub_result.error)

        implemented = [spec.name for spec in REGISTRY.list_specs(implemented_only=True)]
        LOG.info("    tools reporting implemented=True: %s", implemented)
        LOG.info(
            "    (the vlm.* tools are implemented and need the checkpoint; run "
            "`make infer` to exercise them)"
        )

    LOG.info("=" * 78)
    LOG.info("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None
