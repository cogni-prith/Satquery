"""The compatibility gate: turn a set of parsed rasters into an `InputConfig`.

This is stage one of the router, and it is deliberately deterministic. Parsed raster
metadata goes in; one of three input configurations plus a warning list comes out. No
model is consulted and nothing is inferred from the query text.

The rules, from the project's router note:

- one image                              -> SINGLE
- two images, different sensors          -> CROSS_MODAL_PAIR
- two images, same sensor, different date -> BI_TEMPORAL_PAIR

Sensor difference is checked before date difference. An optical and SAR pair taken
weeks apart is still a fusion request: fusion is the only thing that can consume two
modalities, whereas the change tools need a common sensor to compare against.
"""

from __future__ import annotations

from satquery.io.pairing import PairCheck, check_pair
from satquery.preprocess.constants import BI_TEMPORAL_MIN_DELTA_DAYS
from satquery.serve.contracts import ImageRef, InputConfig, Modality

__all__ = ["modalities_of", "representative_gsd", "validate_inputs"]


def modalities_of(images: list[ImageRef]) -> list[Modality]:
    """Modality of every input, in order."""
    return [image.modality for image in images]


def representative_gsd(images: list[ImageRef]) -> float | None:
    """The GSD to gate on: the coarsest known value across the inputs.

    The coarsest, not the finest, because it bounds what is actually resolvable once
    a pair is put on a common grid. Returns None when no input has a computable GSD,
    which never excludes a tool -- see `ToolSpec.accepts_gsd`.
    """
    known = [image.gsd_m for image in images if image.gsd_m is not None]
    return max(known) if known else None


def validate_inputs(images: list[ImageRef]) -> tuple[InputConfig, list[str]]:
    """Decide the input configuration and report every problem found on the way.

    Args:
        images: One or two parsed rasters, in the order the caller supplied them.

    Returns:
        `(input_config, warnings)`. Warnings are advisory: an imperfect pair still
        gets routed, because on the hidden evaluation set refusing to answer scores
        worse than answering with a stated caveat.

    Raises:
        ValueError: Zero inputs, or more than two. There is no `InputConfig` that can
            represent a longer series, so this fails loudly at the boundary rather
            than silently dropping an image.
    """
    if not images:
        raise ValueError("validate_inputs needs at least one image, got none")
    if len(images) > 2:
        raise ValueError(
            f"validate_inputs accepts at most two images, got {len(images)}. Time series "
            "beyond a pair have no InputConfig; split the request."
        )

    warnings: list[str] = []
    for index, image in enumerate(images):
        warnings.extend(f"image {index}: {message}" for message in image.warnings)

    if len(images) == 1:
        return InputConfig.SINGLE, warnings

    first, second = images
    check: PairCheck = check_pair(first, second)
    warnings.extend(check.warnings)

    if not check.same_modality:
        if (
            check.timestamp_delta_days is not None
            and check.timestamp_delta_days >= BI_TEMPORAL_MIN_DELTA_DAYS
        ):
            warnings.append(
                f"inputs differ in both sensor ({first.modality.value} vs "
                f"{second.modality.value}) and date ({check.timestamp_delta_days:.1f} days); "
                "routed as a cross-modal pair, since only fusion can consume two modalities"
            )
        return InputConfig.CROSS_MODAL_PAIR, warnings

    delta = check.timestamp_delta_days
    if delta is None:
        warnings.append(
            "two same-sensor inputs with no comparable timestamps; assumed bi-temporal. "
            "Set acquisition timestamps to make this deterministic."
        )
    elif delta < BI_TEMPORAL_MIN_DELTA_DAYS:
        warnings.append(
            f"two same-sensor inputs only {delta:.3f} days apart, below the "
            f"{BI_TEMPORAL_MIN_DELTA_DAYS} day threshold; still routed as bi-temporal, but "
            "there is unlikely to be real change to detect"
        )

    return InputConfig.BI_TEMPORAL_PAIR, warnings
