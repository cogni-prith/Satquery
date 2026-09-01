"""BigEarthNet fusion loader: co-registered optical and SAR pairs for extraction.

This is the only corpus in the mix that carries a Sentinel-1 and a Sentinel-2 view of
the *same* ground at the *same* time, which is exactly what the optical-and-SAR row of
the problem statement asks for. Each record pairs one S2 patch with its co-registered
S1 patch and targets a built-up / water extraction mask derived from the CORINE
reference map shipped with reBEN.

Three design decisions are worth stating, because each one is a place where a quiet
shortcut would have produced plausible-looking but wrong training data.

**Pairing comes from `metadata.parquet`, never from string surgery on patch ids.**
The S1 and S2 names share a tile and a row/column suffix but differ in platform and
timestamp, so reconstructing one from the other by pattern is guesswork that happens
to work often enough to hide its failures. The published table is authoritative and
also carries the official split, which keeps our splits identical to everyone else's.

**Masks are derived from the CORINE Level-1 digit, not from the patch label list.**
`metadata.parquet` gives scene-level multi-labels, which would only support a
presence/absence answer. The reference maps give per-pixel Level-3 codes, and CORINE
is hierarchical: the leading digit is the Level-1 class. Grouping on that digit yields
real extraction targets and stays correct across Level-3 code revisions.

**Rasters are materialised to a cache rather than passed as arrays.** `ImageRef` is
path-based, and a training sample and an inference request must travel the same code
path -- that identity is what keeps the GSD token honest on both sides. So a patch is
rendered once into the artifact cache and referenced from there, the same approach the
RSVQA loader takes for its embedded image bytes.

One honest limitation is recorded on every record it affects: the LMDB conversion does
not preserve the GeoTIFF affine transform, so the cached rasters carry a true GSD but
no CRS and no origin. `ImageRef.transform` is `None` and a warning says why. Inventing
a plausible origin would make the files look georeferenced when they are not.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from satquery.data.datasets import SampleDataset, SampleTransform
from satquery.data.schema import AnswerType, Sample
from satquery.io.reben import ReBENStore
from satquery.preprocess.constants import (
    CORINE_LEVEL1_BUILT_UP,
    CORINE_LEVEL1_WATER,
    FUSION_EXTRACTION_CLASSES,
    FUSION_MASK_BACKGROUND,
    INSTRUCTION_FUSION_EXTRACTION,
    OPTICAL_BAND_ORDER_4,
)
from satquery.serve.contracts import ImageRef, Modality, TaskType
from satquery.utils.logging import get_logger
from satquery.utils.paths import artifact_dir, dataset_dir

__all__ = [
    "BIGEARTHNET_FUSION_SPLITS",
    "BigEarthNetFusionDataset",
    "PackedFusionDataset",
    "corine_extraction_mask",
]

logger = get_logger(__name__)

#: Official reBEN splits, as published in `metadata.parquet`.
BIGEARTHNET_FUSION_SPLITS: tuple[str, ...] = ("train", "validation", "test")

#: Columns this loader requires from the metadata table.
_REQUIRED_COLUMNS = ("patch_id", "s1_name", "split", "labels")

#: Warning recorded on every ImageRef, because the fact is permanent, not incidental.
_NO_GEOREFERENCE_WARNING = (
    "georeferencing is not preserved by the reBEN LMDB conversion: GSD is computed "
    "from the patch extent, but CRS and origin are unknown and are not fabricated"
)


def corine_extraction_mask(reference_map: np.ndarray) -> np.ndarray:
    """Reduce a CORINE Level-3 reference map to the frozen extraction classes.

    CORINE codes are hierarchical three-digit integers whose leading digit is the
    Level-1 class: 1 artificial surfaces, 2 agricultural, 3 forest and semi-natural,
    4 wetlands, 5 water bodies. Built-up is therefore every ``1xx`` code and water
    every ``5xx`` code, which is both shorter and more durable than enumerating the
    forty-odd leaf codes.

    Wetlands (``4xx``) are deliberately *not* folded into water. They are seasonally
    and partially inundated, so an NDWI or SAR-backscatter water detector will
    legitimately disagree with them, and training the head to call them water would
    corrupt the index-agreement confidence signal that is supposed to flag exactly
    that kind of ambiguity.

    Args:
        reference_map: 2D array of CORINE Level-3 codes.

    Returns:
        uint8 array, same shape, encoding `FUSION_MASK_BACKGROUND` for background and
        ``index + 1`` for each class of `FUSION_EXTRACTION_CLASSES` in order.
    """
    codes = np.asarray(reference_map)
    if codes.ndim != 2:
        raise ValueError(f"expected a 2D reference map, got shape {codes.shape}")

    level1 = codes.astype(np.int32) // 100
    mask = np.full(codes.shape, FUSION_MASK_BACKGROUND, dtype=np.uint8)
    for position, name in enumerate(FUSION_EXTRACTION_CLASSES):
        level1_code = {"built_up": CORINE_LEVEL1_BUILT_UP, "water": CORINE_LEVEL1_WATER}[name]
        mask[level1 == level1_code] = position + 1
    return mask


class BigEarthNetFusionDataset(SampleDataset):
    """Map-style view over co-registered reBEN Sentinel-1 / Sentinel-2 pairs.

    Single responsibility: turn one row of `metadata.parquet` into one unified
    `Sample` carrying two `ImageRef`s and an extraction mask. Rendering happens once
    per patch into the artifact cache; no batching, no tokenisation, no tensors.
    """

    #: Value written to `Sample.source`. Never branched on downstream.
    SOURCE = "bigearthnet_fusion"

    def __init__(
        self,
        root: Path | None = None,
        split: str = "train",
        *,
        lmdb_path: Path | None = None,
        cache_dir: Path | None = None,
        bands: tuple[str, ...] = OPTICAL_BAND_ORDER_4,
        drop_cloud_and_snow: bool = True,
        transform: SampleTransform | None = None,
    ) -> None:
        """Configure the view without touching disk.

        Args:
            root: Corpus directory holding `metadata.parquet`. Defaults to
                `dataset_dir("bigearthnet_txt")`, where the tarballs are unpacked.
            split: One of `BIGEARTHNET_FUSION_SPLITS`.
            lmdb_path: Converted LMDB directory. Defaults to `<root>/../bigearthnet_lmdb`.
            cache_dir: Where rendered rasters are written. Defaults to an artifact cache.
            bands: Optical bands to stack, in order. Must share one resolution; the
                default is the four native 10 m bands.
            drop_cloud_and_snow: Exclude patches flagged for cloud, shadow or seasonal
                snow. On by default: a water extractor trained on cloud shadow learns
                to call shadows water, which is the single most common failure mode of
                optical water mapping and the one SAR is in the mix to avoid.

                Note that with the standard `metadata.parquet` this is already a
                no-op -- that table is the *recommended* set of 480,038 patches, from
                which the 69,450 cloud, shadow and snow patches have been removed
                upstream and published separately as
                `metadata_for_patches_with_snow_cloud_or_shadow.parquet`. The filter
                is kept because it becomes load-bearing the moment someone
                concatenates that second table in to enlarge the corpus, which is a
                plausible thing to try and a silent way to poison water extraction.
            transform: Optional `Sample`-to-`Sample` rewrite applied last.

        Raises:
            ValueError: If `split` is not an official reBEN split.
        """
        if split not in BIGEARTHNET_FUSION_SPLITS:
            raise ValueError(
                f"unknown reBEN split {split!r}; expected one of {BIGEARTHNET_FUSION_SPLITS!r}"
            )

        self.root = Path(root) if root is not None else dataset_dir("bigearthnet_txt")
        self.split = split
        self.lmdb_path = (
            Path(lmdb_path) if lmdb_path is not None else self.root.parent / "bigearthnet_lmdb"
        )
        self.bands = tuple(bands)
        self.drop_cloud_and_snow = drop_cloud_and_snow
        self.transform = transform

        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else artifact_dir("cache", self.SOURCE, create=False)
        )
        self._store = ReBENStore(self.lmdb_path)
        self._index: list[dict[str, Any]] | None = None

        logger.debug(
            "BigEarthNetFusionDataset configured: root=%s split=%s lmdb=%s bands=%s",
            self.root,
            self.split,
            self.lmdb_path,
            self.bands,
        )

    # -- index -------------------------------------------------------------------------

    def metadata_path(self) -> Path:
        """Path of the pairing table. Pure, safe to call without data present."""
        return self.root / "metadata.parquet"

    def _load_index(self) -> list[dict[str, Any]]:
        """Read `metadata.parquet`, filtered to this split.

        Rows whose S1 or S2 patch is absent from the LMDB are dropped with a count
        logged rather than silently skipped at access time, so a partial conversion
        shows up as a smaller dataset at construction instead of as sporadic
        `KeyError`s hours into a run.
        """
        if self._index is not None:
            return self._index

        import pandas as pd

        path = self.metadata_path()
        if not path.exists():
            raise FileNotFoundError(
                f"reBEN metadata table not found at {path}. It ships with BigEarthNet v2.0 "
                "from Zenodo record 10891137; scripts/download_datasets.py prints the "
                "command and verifies the checksum. Nothing is downloaded automatically."
            )

        frame = pd.read_parquet(path)
        missing_columns = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"{path} is missing column(s) {missing_columns!r}; found {list(frame.columns)!r}"
            )

        frame = frame[frame["split"] == self.split]
        if self.drop_cloud_and_snow:
            for flag in ("contains_cloud_or_shadow", "contains_seasonal_snow"):
                if flag in frame.columns:
                    frame = frame[~frame[flag].astype(bool)]

        rows = [
            {
                "patch_id": str(row.patch_id),
                "s1_name": str(row.s1_name),
                "labels": [str(label) for label in row.labels],
                "country": str(getattr(row, "country", "")),
            }
            for row in frame.itertuples(index=False)
        ]

        self._index = rows
        logger.info(
            "BigEarthNetFusionDataset: %d rows in split %r after filtering", len(rows), self.split
        )
        return rows

    # -- materialisation ---------------------------------------------------------------

    def _patch_cache(self, patch_id: str) -> Path:
        """Directory holding the rendered rasters for one patch."""
        # Sharded two levels deep on a hash of the id: a single directory holding half
        # a million entries is slow to stat on every filesystem we might land on, and
        # hashing spreads them evenly where slicing the id would not -- reBEN ids end in
        # a grid row and column, so their tails cluster hard.
        digest = hashlib.sha1(patch_id.encode("utf-8")).hexdigest()
        return self._cache_dir / digest[:2] / digest[2:4] / patch_id

    def _materialise(self, patch_id: str, s1_name: str) -> tuple[Path, Path, Path, list[str]]:
        """Render one patch into the cache if it is not already there.

        Returns:
            `(optical_path, sar_path, mask_path, warnings)`.
        """
        import warnings as warnings_module

        from rasterio.errors import NotGeoreferencedWarning

        from satquery.io.raster import write_raster

        cache = self._patch_cache(patch_id)
        optical_path = cache / "optical.tif"
        sar_path = cache / "sar.tif"
        mask_path = cache / "extraction_mask.tif"

        if optical_path.exists() and sar_path.exists() and mask_path.exists():
            return optical_path, sar_path, mask_path, []

        cache.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []

        # These rasters are deliberately written without a transform (see the module
        # docstring), so rasterio's NotGeoreferencedWarning is the expected outcome
        # rather than a problem. Silencing it here keeps it meaningful everywhere else;
        # the fact itself is reported on every ImageRef via _NO_GEOREFERENCE_WARNING.
        warnings_module.filterwarnings(
            "ignore", category=NotGeoreferencedWarning, module="rasterio"
        )

        # Both rasters are written RAW -- unstretched reflectance and unrendered linear
        # backscatter -- so a cached patch is byte-for-byte the kind of input a real
        # backend request carries, and reading one exercises the production path rather
        # than a shortcut around it.
        #
        # Writing them processed was a mistake worth recording. A pre-stretched optical
        # raster silently breaks the spectral indices, because the frozen per-band
        # percentile stretch rescales each band independently and destroys the
        # cross-band ratio NDWI measures: 0.850 IoU on raw values against 0.211 on
        # stretched ones. A pre-rendered SAR raster gets rendered a second time by any
        # caller that treats it as raw, applying a speckle filter and a dB conversion to
        # what are already display values. Both failures produce a confident-looking
        # mask, and neither is visible downstream.
        stack, _ = self._store.optical(patch_id, self.bands)
        write_raster(optical_path, stack.astype(np.uint16), band_names=list(self.bands))

        vv, vh, _, sar_warnings = self._store.sar(s1_name)
        warnings.extend(sar_warnings)
        write_raster(sar_path, np.stack([vv, vh]).astype(np.float32), band_names=["VV", "VH"])

        reference, _ = self._store.reference_map(patch_id)
        write_raster(mask_path, corine_extraction_mask(reference), band_names=["extraction"])

        return optical_path, sar_path, mask_path, warnings

    def _image_refs(
        self, patch_id: str, optical_path: Path, sar_path: Path, warnings: list[str]
    ) -> list[ImageRef]:
        """Build the optical and SAR references, both with a computed GSD."""
        from satquery.io.reben import patch_gsd_m

        optical_stack, optical_gsd = self._store.optical(patch_id, self.bands)
        height, width = optical_stack.shape[-2:]

        shared = {
            "crs": None,
            "transform": None,
            "width": int(width),
            "height": int(height),
        }
        return [
            ImageRef(
                path=optical_path,
                modality=Modality.MULTISPECTRAL,
                gsd_m=optical_gsd,
                band_names=list(self.bands),
                warnings=[_NO_GEOREFERENCE_WARNING, *warnings],
                **shared,
            ),
            ImageRef(
                path=sar_path,
                modality=Modality.SAR,
                gsd_m=patch_gsd_m(width),
                band_names=["VV", "VH"],
                warnings=[_NO_GEOREFERENCE_WARNING],
                **shared,
            ),
        ]

    # -- SampleDataset -----------------------------------------------------------------

    def __len__(self) -> int:
        """Number of co-registered pairs in this split after filtering."""
        return len(self._load_index())

    def __getitem__(self, index: int) -> Sample:
        """Return pair `index` as a unified `Sample`.

        The optical image comes first and the SAR image second, matching
        `FUSION_CLASS_INDEX`'s expectation that the optical view backs the spectral
        indices while SAR supplies the all-weather second opinion.
        """
        rows = self._load_index()
        row = rows[index]
        patch_id, s1_name = row["patch_id"], row["s1_name"]

        optical_path, sar_path, mask_path, warnings = self._materialise(patch_id, s1_name)
        images = self._image_refs(patch_id, optical_path, sar_path, warnings)

        sample = Sample(
            sample_id=patch_id,
            task=TaskType.FUSION_EXTRACTION,
            answer_type=AnswerType.MASK,
            images=images,
            instruction=INSTRUCTION_FUSION_EXTRACTION,
            mask_path=mask_path,
            source=self.SOURCE,
            split=self.split,
            metadata={
                "s1_name": s1_name,
                "labels": row["labels"],
                "country": row["country"],
                "classes": list(FUSION_EXTRACTION_CLASSES),
            },
        )
        return self.transform(sample) if self.transform is not None else sample


class PackedFusionDataset(SampleDataset):
    """Fusion pairs served from the packed `uint8` memmap built by `pack_fusion_cache.py`.

    Same records as `BigEarthNetFusionDataset`, different storage. It exists because the
    reBEN LMDB sits on a spinning USB drive where random reads run at roughly 1 patch
    per second against 1,009 keys per second sequentially -- a seek limit that makes a
    shuffled training epoch against it take days. The packer streams the corpus once and
    writes it flat; this reads that file at local-disk latency.

    Pixels are identical to the raster path: the packer applies the same
    `stretch_to_uint8` and `render_sar` before writing, so nothing about preprocessing
    changes with the storage backend. `tests/test_fusion_packed.py` asserts that rather
    than relying on this paragraph.

    `Sample.images` still carries real `ImageRef`s pointing at the raster cache, so a
    record from here and a record from the LMDB-backed loader are interchangeable
    downstream. The packed arrays are exposed separately through `arrays()` for the
    collator, which is the only thing that should use them.
    """

    SOURCE = "bigearthnet_fusion"

    def __init__(
        self,
        cache_dir: Path,
        *,
        transform: SampleTransform | None = None,
    ) -> None:
        """Open a packed cache directory written by `scripts/pack_fusion_cache.py`."""
        import json

        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"packed fusion cache not found at {self.cache_dir}. Build it with "
                "scripts/pack_fusion_cache.py; it streams the reBEN LMDB once."
            )
        self.meta = json.loads(meta_path.read_text())
        self.split = self.meta["split"]
        self.transform = transform

        self._side = int(self.meta["patch_side"])
        self._optical_bands = tuple(self.meta["optical_bands"])
        self._optical_bytes = len(self._optical_bands) * self._side * self._side
        self._sar_bytes = 3 * self._side * self._side

        self._index: Any | None = None
        self._data: Any | None = None

        fingerprint = self.meta.get("constants_fingerprint")
        from satquery.preprocess.constants import constants_fingerprint

        running = constants_fingerprint()
        if fingerprint != running:
            # A warning, not a refusal: the packed pixels were produced by the recorded
            # constants and are still self-consistent. But a model trained on them and
            # served under different constants would drift, so this must be visible.
            logger.warning(
                "packed cache %s was written under constants fingerprint %s but this "
                "process runs %s; the cached pixels reflect the recorded set. Repack "
                "before trusting a comparison across the change.",
                self.cache_dir,
                fingerprint,
                running,
            )

    # -- storage ------------------------------------------------------------------------

    def _table(self) -> Any:
        if self._index is None:
            import pandas as pd

            self._index = pd.read_parquet(self.cache_dir / "index.parquet").reset_index(drop=True)
        return self._index

    def _memmap(self) -> Any:
        # Re-opened per process: a numpy memmap does not survive a DataLoader fork
        # cleanly, and the handle is cheap to rebuild.
        import os

        import numpy as np

        pid = os.getpid()
        if self._data is None or getattr(self, "_data_pid", None) != pid:
            self._data = np.load(self.cache_dir / "patches.u8", mmap_mode="r")
            self._data_pid = pid
        return self._data

    def arrays(self, index: int) -> tuple[Any, Any, Any]:
        """Return `(optical, sar, mask)` uint8 arrays for record `index`.

        Optical is `(bands, side, side)`, SAR `(3, side, side)`, mask `(side, side)`.
        """
        row = int(self._table().iloc[index]["row"])
        raw = self._memmap()[row]
        side = self._side
        optical = raw[: self._optical_bytes].reshape(len(self._optical_bands), side, side)
        sar = raw[self._optical_bytes : self._optical_bytes + self._sar_bytes].reshape(
            3, side, side
        )
        mask = raw[self._optical_bytes + self._sar_bytes :].reshape(side, side)
        return optical, sar, mask

    # -- SampleDataset --------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._table())

    def __getitem__(self, index: int) -> Sample:
        """Return record `index`, with `ImageRef`s pointing at the raster cache."""
        from satquery.io.reben import patch_gsd_m

        row = self._table().iloc[index]
        patch_id = str(row["patch_id"])
        gsd = patch_gsd_m(self._side)

        shared = {
            "crs": None,
            "transform": None,
            "width": self._side,
            "height": self._side,
            "warnings": [_NO_GEOREFERENCE_WARNING],
        }
        images = [
            ImageRef(
                path=self.cache_dir / "patches.u8",
                modality=Modality.MULTISPECTRAL,
                gsd_m=gsd,
                band_names=list(self._optical_bands),
                **shared,
            ),
            ImageRef(
                path=self.cache_dir / "patches.u8",
                modality=Modality.SAR,
                gsd_m=gsd,
                band_names=["VV", "VH", "ratio"],
                **shared,
            ),
        ]
        sample = Sample(
            sample_id=patch_id,
            task=TaskType.FUSION_EXTRACTION,
            answer_type=AnswerType.MASK,
            images=images,
            instruction=INSTRUCTION_FUSION_EXTRACTION,
            mask_path=self.cache_dir / "patches.u8",
            source=self.SOURCE,
            split=self.split,
            metadata={
                "s1_name": str(row["s1_name"]),
                "labels": list(row["labels"]),
                "country": str(row["country"]),
                "classes": list(FUSION_EXTRACTION_CLASSES),
                "packed_index": index,
                # Names which packed cache this row belongs to. Trainer uses a single
                # collator for both train and eval, so without this an eval sample
                # would be looked up in the training memmap -- same row number, wrong
                # patch, and eval scores that look plausible and mean nothing.
                "packed_cache": str(self.cache_dir),
            },
        )
        return self.transform(sample) if self.transform is not None else sample
