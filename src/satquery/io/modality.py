"""Map sensor-specific band names to the frozen canonical order, and infer modality.

Band order is never assumed by position. A sensor that happens to store red first
would otherwise be read as blue and every index computed from it would be quietly
wrong -- the kind of bug that survives a smoke test and ruins an eval run.

Everything a sensor might call a band is aliased here, and here only.
"""

from __future__ import annotations

from satquery.preprocess.constants import (
    OPTICAL_BAND_ORDER_4,
    OPTICAL_BAND_ORDER_10,
)
from satquery.serve.contracts import Modality

__all__ = [
    "BAND_ALIASES",
    "CO_POL_BANDS",
    "CROSS_POL_BANDS",
    "SAR_BANDS",
    "canonical_band_name",
    "infer_modality",
    "map_band_names",
    "polarisation_slots",
]

#: Every alias we accept, lowercased, mapped to its canonical name. Canonical optical
#: names follow the Sentinel-2 convention because that is what the training data uses.
BAND_ALIASES: dict[str, str] = {}


def _register(canonical: str, *aliases: str) -> None:
    """Record a canonical band and all the names sensors give it."""
    BAND_ALIASES[canonical.lower()] = canonical
    for alias in aliases:
        BAND_ALIASES[alias.lower()] = canonical


# Sentinel-2 / generic optical. Cartosat-2S multispectral is a four-band
# blue/green/red/NIR sensor, so its MS1..MS4 map straight onto the 4-band order.
_register("B01", "b1", "coastal", "aerosol", "coastal_aerosol")
_register("B02", "b2", "blue", "ms1")
_register("B03", "b3", "green", "ms2")
_register("B04", "b4", "red", "ms3")
_register("B05", "b5", "rededge1", "red_edge_1", "vre1")
_register("B06", "b6", "rededge2", "red_edge_2", "vre2")
_register("B07", "b7", "rededge3", "red_edge_3", "vre3")
_register("B08", "b8", "nir", "ms4", "nir_broad")
_register("B8A", "b8a", "nir_narrow", "rededge4", "vre4")
_register("B09", "b9", "water_vapour", "water_vapor")
_register("B11", "b11", "swir1", "swir_1", "swir")
_register("B12", "b12", "swir2", "swir_2")

# Panchromatic.
_register("PAN", "pan", "panchromatic", "p", "grey", "gray")

# SAR polarisations. RISAT commonly delivers HH/HV where Sentinel-1 delivers VV/VH.
_register("VV", "sigma0_vv", "gamma0_vv", "vv_db", "vv_amplitude")
_register("VH", "sigma0_vh", "gamma0_vh", "vh_db", "vh_amplitude")
_register("HH", "sigma0_hh", "gamma0_hh", "hh_db", "hh_amplitude")
_register("HV", "sigma0_hv", "gamma0_hv", "hv_db", "hv_amplitude")

#: Co-polarised channels occupy the red slot of the SAR pseudo-RGB.
CO_POL_BANDS: tuple[str, ...] = ("VV", "HH")
#: Cross-polarised channels occupy the green slot.
CROSS_POL_BANDS: tuple[str, ...] = ("VH", "HV")
SAR_BANDS: tuple[str, ...] = CO_POL_BANDS + CROSS_POL_BANDS

_RGB_BANDS: frozenset[str] = frozenset({"B02", "B03", "B04"})


def canonical_band_name(raw: str | None) -> str | None:
    """Return the canonical name for a raw band name, or None if it is unrecognised.

    Matching is case-insensitive and tolerates the surrounding whitespace that GDAL
    band descriptions often carry.
    """
    if raw is None:
        return None
    key = raw.strip().lower()
    if not key:
        return None
    return BAND_ALIASES.get(key)


def map_band_names(
    raw_names: list[str | None] | tuple[str | None, ...],
) -> tuple[list[str], list[str]]:
    """Map a raster's band descriptions onto canonical names.

    Args:
        raw_names: Band descriptions in stored order. Entries may be None, which is
            what rasterio returns for a band with no description.

    Returns:
        `(canonical, warnings)`. An unrecognised band keeps a stable placeholder name
        `band_<index>` so the stack stays the right width and the mismatch is visible,
        rather than being silently dropped and shifting every later band by one.
    """
    canonical: list[str] = []
    warnings: list[str] = []

    for index, raw in enumerate(raw_names):
        mapped = canonical_band_name(raw)
        if mapped is None:
            placeholder = f"band_{index + 1}"
            canonical.append(placeholder)
            warnings.append(
                f"band {index + 1} has an unrecognised name {raw!r}; kept as "
                f"{placeholder!r}. Add an alias in io/modality.py if this sensor matters."
            )
        else:
            canonical.append(mapped)

    duplicates = {
        name for name in canonical if canonical.count(name) > 1 and not name.startswith("band_")
    }
    if duplicates:
        warnings.append(
            f"duplicate canonical band names {sorted(duplicates)}; the first occurrence wins "
            "when reordering"
        )
    return canonical, warnings


def infer_modality(
    band_names: list[str] | tuple[str, ...],
    band_count: int | None = None,
) -> tuple[Modality, list[str]]:
    """Infer the sensor modality from canonical band names.

    Decision order, most specific first:

    1. Any SAR polarisation present -> SAR.
    2. A single band named PAN -> PANCHROMATIC.
    3. NIR or SWIR present, or the full 10-band set -> MULTISPECTRAL.
    4. Exactly the three visible bands -> OPTICAL_RGB.
    5. Anything else -> a best guess plus a warning.

    Returns:
        `(modality, warnings)`.
    """
    warnings: list[str] = []
    names = list(band_names)
    count = band_count if band_count is not None else len(names)
    present = set(names)

    if present & set(SAR_BANDS):
        mixed = present - set(SAR_BANDS)
        if mixed:
            warnings.append(
                f"raster mixes SAR polarisations with non-SAR bands {sorted(mixed)}; treated as SAR"
            )
        return Modality.SAR, warnings

    if "PAN" in present:
        if count > 1:
            warnings.append(
                f"raster contains PAN alongside {count - 1} other band(s); treated as "
                "panchromatic, pan-sharpen it before use"
            )
        return Modality.PANCHROMATIC, warnings

    if present & {"B08", "B8A", "B11", "B12", "B05", "B06", "B07"}:
        return Modality.MULTISPECTRAL, warnings

    if set(OPTICAL_BAND_ORDER_10).issubset(present) or set(OPTICAL_BAND_ORDER_4).issubset(present):
        return Modality.MULTISPECTRAL, warnings

    if _RGB_BANDS.issubset(present):
        return Modality.OPTICAL_RGB, warnings

    if count == 1:
        warnings.append(
            f"single unnamed band {names}; assumed panchromatic. Set a band description "
            "or the GSD-dependent tools will be routed wrongly."
        )
        return Modality.PANCHROMATIC, warnings

    if count >= 5:
        warnings.append(f"{count} bands with unrecognised names {names}; assumed multispectral")
        return Modality.MULTISPECTRAL, warnings

    warnings.append(f"cannot identify bands {names}; assumed 3-band optical RGB")
    return Modality.OPTICAL_RGB, warnings


def polarisation_slots(
    band_names: list[str] | tuple[str, ...],
) -> tuple[str | None, str | None]:
    """Return `(co_pol, cross_pol)` band names for the SAR pseudo-RGB layout.

    VV and HH both take the red slot; VH and HV both take the green slot. That is what
    lets a Sentinel-1 VV/VH training tile and a RISAT HH/HV evaluation tile go through
    one identical rendering function.
    """
    present = list(band_names)
    co_pol = next((name for name in CO_POL_BANDS if name in present), None)
    cross_pol = next((name for name in CROSS_POL_BANDS if name in present), None)
    return co_pol, cross_pol
