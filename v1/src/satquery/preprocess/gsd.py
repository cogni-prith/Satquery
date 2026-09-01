"""Ground sampling distance: compute it from an affine transform, format it, snap it.

The GSD token is the single strongest signal the model has for the resolution gap
between 10 m Sentinel training tiles and sub-metre Cartosat-2S evaluation imagery. It
is therefore computed here and nowhere else, from geometry rather than from filenames.

This module is deliberately free of rasterio and of any GPU dependency: it takes the
transform and two facts about the CRS as plain arguments, so it is testable in
isolation and behaves identically on the training and inference paths.
"""

from __future__ import annotations

import math
import re

from satquery.preprocess.constants import (
    CANONICAL_GSD_SCALES_M,
    GSD_ANISOTROPY_TOLERANCE,
    GSD_TOKEN_FORMAT,
    GSD_TOKEN_PATTERN,
    GSD_TOKEN_UNKNOWN,
    METRES_PER_DEGREE_LAT,
    METRES_PER_DEGREE_LON_EQUATOR,
)

__all__ = [
    "format_gsd_token",
    "gsd_from_transform",
    "nearest_canonical_gsd",
    "prefix_instruction",
    "resample_scale_factor",
    "strip_gsd_token",
]

Transform = tuple[float, float, float, float, float, float]

_TOKEN_RE = re.compile(GSD_TOKEN_PATTERN)

# Smallest GSD that survives `.1f` formatting as something other than "0.0".
_MIN_REPRESENTABLE_GSD_M = 0.05


def gsd_from_transform(
    transform: Transform | None,
    *,
    is_geographic: bool = False,
    centre_latitude: float | None = None,
) -> tuple[float | None, list[str]]:
    """Derive the ground sampling distance in metres from an affine transform.

    The transform is `(a, b, c, d, e, f)` with `x = a*col + b*row + c` and
    `y = d*col + e*row + f`. Pixel edge lengths are the norms of the two column
    vectors `(a, d)` and `(b, e)`, which stays correct for rotated transforms where
    reading `a` and `e` alone would not.

    A raster in a geographic CRS has a transform in degrees. Interpreting those as
    metres yields a GSD near zero and a confidently wrong token, so the caller must
    say whether the CRS is geographic and supply the footprint's centre latitude for
    the conversion.

    Args:
        transform: Affine coefficients, or None when the raster carries no transform.
        is_geographic: True when the CRS units are degrees rather than metres.
        centre_latitude: Latitude at the centre of the raster, required when
            `is_geographic` is True.

    Returns:
        `(gsd_m, warnings)`. `gsd_m` is the geometric mean of the two pixel edge
        lengths, or None when it cannot be established. Never a guess.
    """
    warnings: list[str] = []

    if transform is None:
        return None, ["raster carries no affine transform; GSD cannot be computed"]

    a, b, c, d, e, f = transform
    del c, f  # translation does not affect pixel size

    if not all(math.isfinite(v) for v in (a, b, d, e)):
        return None, ["affine transform contains non-finite coefficients; GSD cannot be computed"]

    size_x = math.hypot(a, d)
    size_y = math.hypot(b, e)

    if size_x <= 0.0 or size_y <= 0.0:
        return None, ["affine transform has zero pixel extent; GSD cannot be computed"]

    if is_geographic:
        if centre_latitude is None:
            return None, [
                "geographic CRS without a centre latitude; degrees cannot be converted to "
                "metres, so GSD is reported as unknown rather than guessed"
            ]
        if not -90.0 <= centre_latitude <= 90.0:
            return None, [f"centre latitude {centre_latitude} is out of range; GSD unknown"]
        size_x = size_x * METRES_PER_DEGREE_LON_EQUATOR * math.cos(math.radians(centre_latitude))
        size_y = size_y * METRES_PER_DEGREE_LAT
        warnings.append(
            f"GSD converted from a geographic CRS at latitude {centre_latitude:.4f}; "
            "reproject to a projected CRS for an exact value"
        )
        if size_x <= 0.0:
            return None, [*warnings, "degenerate longitude scaling at this latitude; GSD unknown"]

    largest = max(size_x, size_y)
    if abs(size_x - size_y) / largest > GSD_ANISOTROPY_TOLERANCE:
        warnings.append(
            f"anisotropic pixels: {size_x:.4f} m by {size_y:.4f} m; "
            "GSD reported as their geometric mean"
        )

    return math.sqrt(size_x * size_y), warnings


def format_gsd_token(gsd_m: float | None) -> str:
    """Format the frozen GSD token that prefixes every instruction string.

    Returns `<gsd:unknown>` for a missing, non-finite, non-positive, or
    sub-representable value. A GSD of 0.04 m would format as `<gsd:0.0m>`, which
    reads as a valid measurement and is worse than admitting ignorance.
    """
    if gsd_m is None or not math.isfinite(gsd_m) or gsd_m < _MIN_REPRESENTABLE_GSD_M:
        return GSD_TOKEN_UNKNOWN
    return GSD_TOKEN_FORMAT.format(value=gsd_m)


def nearest_canonical_gsd(gsd_m: float | None) -> float | None:
    """Snap a GSD to the nearest canonical scale, or None if the input is unknown.

    Nearest is measured in log space because resolution is a ratio, not a difference:
    3 m is much closer to 2 m than to 5 m in resampling terms, even though the linear
    distances are equal.
    """
    if gsd_m is None or not math.isfinite(gsd_m) or gsd_m <= 0.0:
        return None
    return min(CANONICAL_GSD_SCALES_M, key=lambda scale: abs(math.log(gsd_m / scale)))


def resample_scale_factor(source_gsd_m: float, target_gsd_m: float) -> float:
    """Return the linear factor to multiply raster dimensions by when resampling.

    Going from 10 m to 2 m means each pixel becomes five, so the factor is 5.0.
    """
    if source_gsd_m <= 0.0 or target_gsd_m <= 0.0:
        raise ValueError(
            f"GSD values must be positive, got source={source_gsd_m}, target={target_gsd_m}"
        )
    return source_gsd_m / target_gsd_m


def prefix_instruction(instruction: str, gsd_m: float | None) -> str:
    """Prefix an instruction with its GSD token, idempotently.

    Prefixing twice would put a second token in the string and break the pattern the
    model was trained on, so an instruction that already carries a token is returned
    unchanged.
    """
    stripped = instruction.lstrip()
    if _TOKEN_RE.match(stripped):
        return stripped
    return f"{format_gsd_token(gsd_m)} {stripped}"


def strip_gsd_token(instruction: str) -> str:
    """Remove a leading GSD token. Used by the eval harness when comparing raw text."""
    return _TOKEN_RE.sub("", instruction, count=1).lstrip()
