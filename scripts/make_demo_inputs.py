#!/usr/bin/env python3
"""Generate the two demo rasters you can upload to the UI. Downloads nothing.

Synthetic on purpose, and labelled as such: the point of a demo input is that you already
know the right answer, so you can tell whether the system computed it or guessed. A real
Sentinel tile looks better in a screenshot and tells you nothing about whether the number
is correct.

Two dates of the same 512x512 scene at 10 m GSD, EPSG:32643 (UTM 43N, northern India):

  - a circular reservoir that grows from a 60 px radius to a 92 px radius
  - a rectangular built-up block that does NOT change
  - vegetation everywhere else, with speckle so the masks are not trivially clean

Ground truth is printed when this runs, so you can check the answer against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

SIDE = 512
GSD_M = 10.0
#: Reservoir radius in pixels at each date.
LAKE_R_T1, LAKE_R_T2 = 60, 92
#: The built-up block, identical at both dates: (row slice, col slice).
BUILT = (slice(60, 150), slice(330, 470))

#: Reflectances chosen so each index lands on the right side of its literature threshold.
#: Water: bright green, dark NIR -> NDWI > 0. Built-up: bright SWIR -> NDBI > 0.
#: Vegetation: bright NIR, dark red -> NDVI > 0.2.
SURFACES = {
    #                 red   green   nir    swir
    "vegetation": (0.04, 0.08, 0.42, 0.20),
    "water": (0.05, 0.09, 0.02, 0.03),
    "built_up": (0.22, 0.24, 0.26, 0.40),
}
BAND_NAMES = ("B04", "B03", "B08", "B11")


def build(lake_radius: int, seed: int) -> np.ndarray:
    """A four-band float32 stack for one date."""
    rng = np.random.default_rng(seed)
    stack = np.stack(
        [np.full((SIDE, SIDE), value, dtype=np.float32) for value in SURFACES["vegetation"]]
    )

    rows, cols = np.mgrid[0:SIDE, 0:SIDE]
    lake = (rows - 300) ** 2 + (cols - 200) ** 2 < lake_radius**2
    for band, value in enumerate(SURFACES["water"]):
        stack[band][lake] = value
    for band, value in enumerate(SURFACES["built_up"]):
        stack[band][BUILT] = value

    # Sensor noise, so the thresholds do the work rather than a perfectly flat field.
    # Small enough that no pixel crosses a threshold it should not, which keeps the
    # printed ground truth exact.
    stack += rng.normal(0.0, 0.004, stack.shape).astype(np.float32)
    return np.clip(stack, 0.0, 1.0)


def write(path: Path, stack: np.ndarray) -> Path:
    """Write a georeferenced GeoTIFF with named bands.

    The affine transform is what makes the GSD real. Without it every area in the answer
    would be a pixel count and the tool would say `<gsd:unknown>` rather than guess 1.0.
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIDE,
        width=SIDE,
        count=len(BAND_NAMES),
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000.0, 3000000.0, GSD_M, GSD_M),
    ) as dst:
        for index, name in enumerate(BAND_NAMES, start=1):
            dst.write(stack[index - 1], index)
            dst.set_band_description(index, name)
    return path


def main() -> int:
    out = Path(__file__).resolve().parents[1] / "demo_inputs"
    out.mkdir(parents=True, exist_ok=True)

    first = build(LAKE_R_T1, seed=11)
    second = build(LAKE_R_T2, seed=12)
    write(out / "demo_input_tif_1.tif", first)
    write(out / "demo_input_tif_2.tif", second)

    rows, cols = np.mgrid[0:SIDE, 0:SIDE]
    area = lambda mask: int(np.count_nonzero(mask)) * GSD_M**2  # noqa: E731
    lake_t1 = area((rows - 300) ** 2 + (cols - 200) ** 2 < LAKE_R_T1**2)
    lake_t2 = area((rows - 300) ** 2 + (cols - 200) ** 2 < LAKE_R_T2**2)
    built = (BUILT[0].stop - BUILT[0].start) * (BUILT[1].stop - BUILT[1].start) * GSD_M**2

    print(f"wrote {out}/demo_input_tif_1.tif and demo_input_tif_2.tif")
    print(f"  {SIDE}x{SIDE}, 4 bands {list(BAND_NAMES)}, EPSG:32643, {GSD_M:.0f} m GSD\n")
    print("ground truth -- check the answer against these:")
    print(f"  water  date 1 : {lake_t1:>12,.0f} m2  ({lake_t1 / 10_000:.2f} ha)")
    print(f"  water  date 2 : {lake_t2:>12,.0f} m2  ({lake_t2 / 10_000:.2f} ha)")
    print(
        f"  water  change : {lake_t2 - lake_t1:>12,.0f} m2  "
        f"({(lake_t2 - lake_t1) / 10_000:.2f} ha, "
        f"+{(lake_t2 - lake_t1) / lake_t1 * 100:.1f}%)"
    )
    print(f"  built-up      : {built:>12,.0f} m2  at both dates -- change must be 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
