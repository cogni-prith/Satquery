"""The packed fusion cache must be a storage change only, never a preprocessing change.

`PackedFusionDataset` exists because the reBEN LMDB sits on a spinning USB drive where
random reads run about a thousand times slower than sequential ones. That is an I/O
decision, and it must not quietly become a *pixel* decision: a model trained on packed
bytes and served through the raster path would drift in exactly the way the frozen
constants exist to prevent, and nothing in the scores would show it.

So these tests build both a packed cache and the equivalent rasters from one synthetic
LMDB and assert the two collators agree bit for bit.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from satquery.preprocess.constants import FUSION_EXTRACTION_CLASSES, constants_fingerprint

pytest.importorskip("lmdb")
pytest.importorskip("torch")

PATCH_SIDE = 8
OPTICAL_BYTES = 4 * PATCH_SIDE * PATCH_SIDE
SAR_BYTES = 3 * PATCH_SIDE * PATCH_SIDE
MASK_BYTES = PATCH_SIDE * PATCH_SIDE


@pytest.fixture
def packed_cache(tmp_path):
    """A two-row packed cache with known contents."""
    import pandas as pd

    directory = tmp_path / "packed"
    directory.mkdir()
    rng = np.random.default_rng(7)

    rows = 2
    data = np.lib.format.open_memmap(
        directory / "patches.u8",
        mode="w+",
        dtype=np.uint8,
        shape=(rows, OPTICAL_BYTES + SAR_BYTES + MASK_BYTES),
    )
    data[:] = rng.integers(0, 256, size=data.shape, dtype=np.uint8)
    # Put known class codes in the mask region so the plane expansion is checkable.
    for row in range(rows):
        mask = np.zeros((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8)
        mask[:2, :2] = FUSION_EXTRACTION_CLASSES.index("built_up") + 1
        mask[4:6, 4:6] = FUSION_EXTRACTION_CLASSES.index("water") + 1
        data[row, OPTICAL_BYTES + SAR_BYTES :] = mask.ravel()
    data.flush()

    pd.DataFrame(
        {
            "patch_id": [f"S2_PATCH_{index}" for index in range(rows)],
            "s1_name": [f"S1_PATCH_{index}" for index in range(rows)],
            "labels": [["Inland waters"], ["Urban fabric"]],
            "country": ["Austria", "Austria"],
            "row": list(range(rows)),
        }
    ).to_parquet(directory / "index.parquet")

    (directory / "meta.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "rows": rows,
                "patch_side": PATCH_SIDE,
                "optical_bands": ["B02", "B03", "B04", "B08"],
                "patch_bytes": OPTICAL_BYTES + SAR_BYTES + MASK_BYTES,
                "constants_fingerprint": constants_fingerprint(),
                "selection": "stride",
                "source_rows": 100,
            }
        )
    )
    return directory


# -- dataset ------------------------------------------------------------------------------


def test_packed_dataset_emits_the_unified_schema(packed_cache):
    from satquery.data.datasets.bigearthnet_fusion import PackedFusionDataset
    from satquery.serve.contracts import Modality, TaskType

    dataset = PackedFusionDataset(packed_cache)
    assert len(dataset) == 2

    sample = dataset[0]
    assert sample.task is TaskType.FUSION_EXTRACTION
    assert [image.modality for image in sample.images] == [Modality.MULTISPECTRAL, Modality.SAR]
    assert sample.metadata["packed_index"] == 0
    assert sample.metadata["packed_cache"] == str(packed_cache)


def test_packed_sample_reports_the_computed_gsd(packed_cache):
    """GSD still comes from the patch extent over the array width, not from a constant."""
    from satquery.data.datasets.bigearthnet_fusion import PackedFusionDataset
    from satquery.io.reben import patch_gsd_m

    sample = PackedFusionDataset(packed_cache)[0]
    assert sample.images[0].gsd_m == pytest.approx(patch_gsd_m(PATCH_SIDE))
    assert sample.prompt().startswith("<gsd:")


def test_packed_refs_never_claim_georeferencing(packed_cache):
    from satquery.data.datasets.bigearthnet_fusion import PackedFusionDataset

    sample = PackedFusionDataset(packed_cache)[0]
    for image in sample.images:
        assert image.transform is None
        assert image.crs is None
        assert any("georeferencing is not preserved" in w for w in image.warnings)


def test_arrays_split_the_row_at_the_right_offsets(packed_cache):
    from satquery.data.datasets.bigearthnet_fusion import PackedFusionDataset

    optical, sar, mask = PackedFusionDataset(packed_cache).arrays(0)
    assert optical.shape == (4, PATCH_SIDE, PATCH_SIDE)
    assert sar.shape == (3, PATCH_SIDE, PATCH_SIDE)
    assert mask.shape == (PATCH_SIDE, PATCH_SIDE)
    assert set(np.unique(mask).tolist()) == {0, 1, 2}


def test_missing_cache_names_the_packer(tmp_path):
    from satquery.data.datasets.bigearthnet_fusion import PackedFusionDataset

    with pytest.raises(FileNotFoundError, match="pack_fusion_cache"):
        PackedFusionDataset(tmp_path / "absent")


def test_fingerprint_mismatch_warns_rather_than_silently_passing(packed_cache, caplog):
    """The cached pixels are self-consistent, but a comparison across a constants
    change is not, and that must be visible."""
    import logging

    meta = json.loads((packed_cache / "meta.json").read_text())
    meta["constants_fingerprint"] = "a-different-fingerprint"
    (packed_cache / "meta.json").write_text(json.dumps(meta))

    from satquery.data.datasets.bigearthnet_fusion import PackedFusionDataset

    with caplog.at_level(logging.WARNING):
        PackedFusionDataset(packed_cache)
    assert "constants fingerprint" in caplog.text


# -- the equivalence that matters ---------------------------------------------------------


def test_packed_and_raster_collators_agree_bit_for_bit(tmp_path):
    """Both routes must reach identical tensors from identical raw source.

    The packed cache applies `stretch_to_uint8` and `render_sar` at pack time; the
    raster path stores raw and applies them at load time. Those have to be the same
    arithmetic, or "packed" has silently become a second preprocessing path -- and the
    constants fingerprint would not catch it, because no constant changed.
    """
    import json

    import pandas as pd
    import torch

    from satquery.data.collate import FusionCollator, PackedFusionCollator
    from satquery.data.datasets.bigearthnet_fusion import PackedFusionDataset
    from satquery.data.schema import AnswerType, Sample
    from satquery.io.raster import write_raster
    from satquery.preprocess.optical import stretch_to_uint8
    from satquery.preprocess.sar import render_sar
    from satquery.serve.contracts import ImageRef, Modality, TaskType

    rng = np.random.default_rng(3)
    raw_optical = rng.integers(100, 4000, size=(4, PATCH_SIDE, PATCH_SIDE)).astype(np.uint16)
    raw_sar = rng.uniform(0.005, 1.2, size=(2, PATCH_SIDE, PATCH_SIDE)).astype(np.float32)
    mask = np.zeros((PATCH_SIDE, PATCH_SIDE), dtype=np.uint8)
    mask[:2, :2] = FUSION_EXTRACTION_CLASSES.index("built_up") + 1
    mask[4:6, 4:6] = FUSION_EXTRACTION_CLASSES.index("water") + 1

    # -- route A: raw rasters through FusionCollator
    rasters = tmp_path / "rasters"
    raster_sample = Sample(
        sample_id="S2_PATCH_0",
        task=TaskType.FUSION_EXTRACTION,
        answer_type=AnswerType.MASK,
        images=[
            ImageRef(
                path=write_raster(rasters / "optical.tif", raw_optical),
                modality=Modality.MULTISPECTRAL,
                gsd_m=10.0,
            ),
            ImageRef(
                path=write_raster(rasters / "sar.tif", raw_sar),
                modality=Modality.SAR,
                gsd_m=10.0,
                band_names=["VV", "VH"],
            ),
        ],
        instruction="extract",
        mask_path=write_raster(rasters / "mask.tif", mask),
        source="bigearthnet_fusion",
    )
    from_raster = FusionCollator(targets=FUSION_EXTRACTION_CLASSES).encode(raster_sample)

    # -- route B: the same raw source, processed at pack time
    directory = tmp_path / "packed"
    directory.mkdir()
    rendered, _ = render_sar(raw_sar[0], raw_sar[1])
    packed = np.lib.format.open_memmap(
        directory / "patches.u8",
        mode="w+",
        dtype=np.uint8,
        shape=(1, OPTICAL_BYTES + SAR_BYTES + MASK_BYTES),
    )
    packed[0, :OPTICAL_BYTES] = stretch_to_uint8(raw_optical.astype(np.float32)).ravel()
    packed[0, OPTICAL_BYTES : OPTICAL_BYTES + SAR_BYTES] = np.transpose(rendered, (2, 0, 1)).ravel()
    packed[0, OPTICAL_BYTES + SAR_BYTES :] = mask.ravel()
    packed.flush()

    pd.DataFrame(
        {
            "patch_id": ["S2_PATCH_0"],
            "s1_name": ["S1_PATCH_0"],
            "labels": [["Inland waters"]],
            "country": ["Austria"],
            "row": [0],
        }
    ).to_parquet(directory / "index.parquet")
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "rows": 1,
                "patch_side": PATCH_SIDE,
                "optical_bands": ["B02", "B03", "B04", "B08"],
                "patch_bytes": OPTICAL_BYTES + SAR_BYTES + MASK_BYTES,
                "constants_fingerprint": constants_fingerprint(),
                "selection": "stride",
                "source_rows": 1,
            }
        )
    )
    dataset = PackedFusionDataset(directory)
    from_packed = PackedFusionCollator(
        datasets={str(directory): dataset}, targets=FUSION_EXTRACTION_CLASSES
    ).encode(dataset[0])

    for key in ("pixel_values_optical", "pixel_values_sar", "labels"):
        torch.testing.assert_close(from_packed[key], from_raster[key], rtol=0, atol=0)


def test_collator_refuses_a_row_from_an_unregistered_cache(packed_cache):
    """A row number looked up in the wrong memmap returns a plausible wrong patch."""
    from satquery.data.collate import PackedFusionCollator
    from satquery.data.datasets.bigearthnet_fusion import PackedFusionDataset

    dataset = PackedFusionDataset(packed_cache)
    sample = dataset[0]
    sample.metadata["packed_cache"] = "/some/other/cache"

    collator = PackedFusionCollator(
        datasets={str(packed_cache): dataset}, targets=FUSION_EXTRACTION_CLASSES
    )
    with pytest.raises(KeyError, match="no packed dataset registered"):
        collator.encode(sample)


def test_collator_expands_the_mask_into_one_plane_per_target(packed_cache):
    from satquery.data.collate import PackedFusionCollator
    from satquery.data.datasets.bigearthnet_fusion import PackedFusionDataset

    dataset = PackedFusionDataset(packed_cache)
    encoded = PackedFusionCollator(
        datasets={str(packed_cache): dataset}, targets=FUSION_EXTRACTION_CLASSES
    ).encode(dataset[0])

    labels = encoded["labels"]
    assert labels.shape == (2, PATCH_SIDE, PATCH_SIDE)
    assert labels[0].sum().item() == 4  # the 2x2 built-up block
    assert labels[1].sum().item() == 4  # the 2x2 water block
    # The two classes are disjoint in CORINE, so no pixel may be in both planes.
    assert (labels[0] * labels[1]).sum().item() == 0


def test_target_subset_selects_the_right_plane(packed_cache):
    """Requesting only water must select water's plane, not plane 0."""
    from satquery.data.collate import PackedFusionCollator
    from satquery.data.datasets.bigearthnet_fusion import PackedFusionDataset

    dataset = PackedFusionDataset(packed_cache)
    encoded = PackedFusionCollator(
        datasets={str(packed_cache): dataset}, targets=("water",)
    ).encode(dataset[0])

    assert encoded["labels"].shape == (1, PATCH_SIDE, PATCH_SIDE)
    _, _, mask = dataset.arrays(0)
    expected = (mask == FUSION_EXTRACTION_CLASSES.index("water") + 1).sum()
    assert encoded["labels"].sum().item() == expected
