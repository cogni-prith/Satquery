"""Reader for the reBEN (BigEarthNet v2.0) LMDB store.

BigEarthNet ships as two zstd tarballs holding roughly 8.2 million small GeoTIFFs.
Reading those directly is impractical: on a 128 KB-cluster filesystem the unpacked
tree occupies about a terabyte for 113 GB of data, and a training epoch becomes
millions of individual file opens. `scripts/` converts them once with `rico-hdl` into
a single memory-mapped LMDB whose keys are patch identifiers and whose values are
safetensors dictionaries of band arrays. This module is the only thing that reads it.

Two properties of that store shape everything below.

**No affine transform survives the conversion.** safetensors holds arrays, not
georeferencing. `preprocess/gsd.py` forbids inferring a GSD from a sensor name or a
filename, so instead it is *computed* from the one invariant reBEN guarantees: every
patch covers the same 1200 m square, whatever the band's pixel count. A 120-pixel band
is therefore 10 m, a 60-pixel band 20 m, a 20-pixel band 60 m -- derived from the array
in hand rather than assumed. The georeferencing itself is genuinely absent and is
reported as such, never fabricated.

**Sentinel-1 arrives already in decibels.** Raw Sentinel-1 GRD and RISAT do not. Since
the frozen SAR pipeline starts from linear backscatter, this reader inverts the dB
before handing anything on, so `preprocess/sar.render_sar` runs byte-identically for
reBEN at training time and RISAT at evaluation time. See `preprocess.sar.db_to_linear`
for why inverting beats skipping the step.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from satquery.preprocess.constants import (
    REBEN_PATCH_EXTENT_M,
    REBEN_SAR_DB_RANGE,
)
from satquery.preprocess.sar import db_to_linear
from satquery.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from types import TracebackType

__all__ = [
    "REFERENCE_MAP_SUFFIX",
    "ReBENStore",
    "patch_gsd_m",
]

logger = get_logger(__name__)

#: Reference-map keys are the optical patch id plus this suffix, as written by rico-hdl.
REFERENCE_MAP_SUFFIX = "_reference_map"

#: safetensors key holding the single band of a reference map.
_REFERENCE_MAP_BAND = "Data"

#: Sentinel-1 polarisation keys, in the co-polarised, cross-polarised order that
#: `render_sar` expects.
_SAR_BANDS = ("VV", "VH")


def patch_gsd_m(width: int, *, extent_m: float = REBEN_PATCH_EXTENT_M) -> float:
    """Return the ground sampling distance of a reBEN band from its pixel width.

    Every reBEN patch covers `extent_m` metres on a side regardless of which band is
    being read, so the GSD follows from the array's own shape. This is the substitute
    for the affine transform the LMDB conversion drops -- a computation over the data,
    not a guess from the sensor name.

    Args:
        width: Band width in pixels.
        extent_m: Ground extent of one patch edge, in metres.

    Returns:
        Ground sampling distance in metres.

    Raises:
        ValueError: If `width` is not positive.
    """
    if width <= 0:
        raise ValueError(f"band width must be positive, got {width}")
    return extent_m / float(width)


class ReBENStore:
    """Read-only view over the converted reBEN LMDB.

    Single responsibility: turn a patch identifier into band arrays with honest units
    and an honest GSD. It performs no rendering, no resampling and no sample
    construction; `preprocess/` and the dataset loaders own those.

    The environment is opened lazily on first read so that constructing a store -- as
    every dataset loader does at import time -- never touches disk. It is safe to share
    one instance across DataLoader workers: the environment is opened with
    ``readonly=True, lock=False``, and each worker re-opens it after the fork because
    the handle is keyed on the process id.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        map_size: int | None = None,
        readahead: bool = False,
    ) -> None:
        """Configure the store without opening it.

        Args:
            path: Directory holding `data.mdb`, as produced by `rico-hdl`.
            map_size: Optional LMDB map size override. The default lets LMDB use the
                size recorded in the existing database, which is what a reader wants.
            readahead: Whether to let the OS prefetch ahead of each read. Off by
                default, which is right for the scattered single-patch reads a serving
                path makes -- prefetching neighbours that will never be touched only
                evicts useful pages.

                Turn it **on** for a full cursor scan. Measured on the spinning USB
                drive holding this store, a sequential scan runs at roughly 1,009
                keys/s with readahead and about 54 keys/s without it, because without
                OS prefetch every page becomes its own seek.
                `scripts/pack_fusion_cache.py` is the caller that needs this.
        """
        self.path = Path(path)
        self._map_size = map_size
        self._readahead = readahead
        self._env: Any = None
        self._env_pid: int | None = None

    # -- lifecycle ---------------------------------------------------------------------

    def _environment(self) -> Any:
        """Return the open LMDB environment, opening it on first use.

        Re-opens after a fork. LMDB environments do not survive `fork` safely, and a
        DataLoader with ``num_workers > 0`` forks after the parent has already read a
        sample, so the handle is rebound whenever the process id changes.
        """
        import os

        pid = os.getpid()
        if self._env is not None and self._env_pid == pid:
            return self._env

        import lmdb

        if not self.path.exists():
            raise FileNotFoundError(
                f"reBEN LMDB not found at {self.path}. Build it from the BigEarthNet "
                "tarballs with scripts/convert_reben.py; nothing is downloaded or "
                "converted automatically."
            )
        kwargs: dict[str, Any] = {
            "readonly": True,
            "lock": False,
            "readahead": self._readahead,
        }
        if self._map_size is not None:
            kwargs["map_size"] = self._map_size
        self._env = lmdb.open(str(self.path), **kwargs)
        self._env_pid = pid
        logger.debug("opened reBEN LMDB at %s", self.path)
        return self._env

    def close(self) -> None:
        """Close the environment if it is open. Idempotent."""
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_pid = None

    def __enter__(self) -> ReBENStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __len__(self) -> int:
        """Total number of keys, counting S1, S2 and reference-map entries separately."""
        return int(self._environment().stat()["entries"])

    # -- raw access --------------------------------------------------------------------

    def read(self, key: str) -> dict[str, np.ndarray]:
        """Return every band stored under `key`.

        Args:
            key: A patch identifier, e.g. an S2 `patch_id`, an S1 `s1_name`, or a
                reference-map key ending in `REFERENCE_MAP_SUFFIX`.

        Returns:
            Mapping of band name to 2D array, in the store's own dtype and units.

        Raises:
            KeyError: If `key` is not present.
        """
        from safetensors.numpy import load

        with self._environment().begin(write=False) as transaction:
            payload = transaction.get(key.encode("utf-8"))
        if payload is None:
            raise KeyError(f"patch {key!r} is not in the reBEN LMDB at {self.path}")
        return dict(load(bytes(payload)))

    def has(self, key: str) -> bool:
        """True when `key` is present in the store."""
        with self._environment().begin(write=False) as transaction:
            return transaction.get(key.encode("utf-8")) is not None

    # -- typed access ------------------------------------------------------------------

    def optical(self, patch_id: str, bands: tuple[str, ...]) -> tuple[np.ndarray, float]:
        """Return a Sentinel-2 band stack and its GSD.

        Args:
            patch_id: The S2 patch identifier.
            bands: Canonical band names in the order they should be stacked, e.g.
                `OPTICAL_BAND_ORDER_4`.

        Returns:
            `(stack, gsd_m)` with `stack` channels-first, shape `(len(bands), h, w)`,
            dtype float32.

        Raises:
            KeyError: If the patch or one of `bands` is absent.
            ValueError: If the requested bands do not share a pixel grid. Sentinel-2
                stores 10 m, 20 m and 60 m bands at their native sizes, so mixing
                resolutions in one stack is a caller error, not something to paper over
                with a silent resample.
        """
        stored = self.read(patch_id)
        missing = [band for band in bands if band not in stored]
        if missing:
            raise KeyError(
                f"patch {patch_id!r} is missing band(s) {missing!r}; it holds {sorted(stored)!r}"
            )

        arrays = [np.asarray(stored[band], dtype=np.float32) for band in bands]
        shapes = {array.shape for array in arrays}
        if len(shapes) > 1:
            detail = ", ".join(f"{band}={stored[band].shape}" for band in bands)
            raise ValueError(
                f"patch {patch_id!r}: requested bands span more than one resolution "
                f"({detail}). Sentinel-2 keeps 10 m, 20 m and 60 m bands at native size; "
                "resample explicitly before stacking rather than relying on this reader."
            )
        stack = np.stack(arrays)
        return stack, patch_gsd_m(stack.shape[-1])

    def sar(self, s1_name: str) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
        """Return Sentinel-1 VV and VH in **linear** backscatter, plus the GSD.

        reBEN stores sigma-nought in decibels. The frozen SAR pipeline begins at linear
        amplitude, so the dB is inverted here -- once, at the edge of the system -- and
        `render_sar` then runs unbranched for both reBEN and RISAT.

        Returns:
            `(vv, vh, gsd_m, warnings)`, arrays float64 in linear units.

        Raises:
            KeyError: If the patch or a polarisation is absent.
        """
        stored = self.read(s1_name)
        missing = [band for band in _SAR_BANDS if band not in stored]
        if missing:
            raise KeyError(
                f"S1 patch {s1_name!r} is missing polarisation(s) {missing!r}; it holds "
                f"{sorted(stored)!r}"
            )

        warnings: list[str] = []
        low, high = REBEN_SAR_DB_RANGE
        linear = []
        for band in _SAR_BANDS:
            values = np.asarray(stored[band], dtype=np.float64)
            observed = (float(np.nanmin(values)), float(np.nanmax(values)))

            # Two separate checks, because they catch two different failures and the
            # range check alone cannot catch the important one: linear sigma0 lives in
            # roughly [0, 2], which sits comfortably *inside* the plausible dB range,
            # so a units flip would slide straight through it.
            #
            # The discriminator that does work is the sign. Sigma0 in dB is a ratio
            # below unity almost everywhere on Earth, so a real dB patch essentially
            # always contains negative values; linear sigma0 is non-negative by
            # construction. An all-non-negative patch therefore means the source has
            # changed units, and inverting a dB that is not there would produce a
            # plausible-looking but meaningless image that nothing downstream notices.
            if observed[0] >= 0.0:
                warnings.append(
                    f"{s1_name} {band}: no negative values (range {observed}), so this "
                    "is very likely linear sigma0 rather than the expected dB; the dB "
                    "inversion would corrupt it"
                )
            elif not (low <= observed[0] and observed[1] <= high):
                warnings.append(
                    f"{s1_name} {band}: values {observed} fall outside the expected "
                    f"sigma0 dB range {REBEN_SAR_DB_RANGE}; the source units may have "
                    "changed and the dB inversion may be wrong"
                )
            if warnings:
                logger.warning("%s", warnings[-1])
            linear.append(db_to_linear(values))

        vv, vh = linear
        return vv, vh, patch_gsd_m(vv.shape[-1]), warnings

    def reference_map(self, patch_id: str) -> tuple[np.ndarray, float]:
        """Return the CORINE reference map for `patch_id` and its GSD.

        Values are CORINE Land Cover Level-3 codes, not contiguous class indices.

        Raises:
            KeyError: If no reference map is stored for this patch.
        """
        key = f"{patch_id}{REFERENCE_MAP_SUFFIX}"
        stored = self.read(key)
        if _REFERENCE_MAP_BAND not in stored:
            raise KeyError(
                f"reference map {key!r} has no {_REFERENCE_MAP_BAND!r} band; it holds "
                f"{sorted(stored)!r}"
            )
        array = np.asarray(stored[_REFERENCE_MAP_BAND])
        return array, patch_gsd_m(array.shape[-1])
