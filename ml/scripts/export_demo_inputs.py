#!/usr/bin/env python3
"""Export two real Sentinel-2 patches from the reBEN LMDB as GeoTIFFs. Downloads nothing.

Reads a dataset you already have on disk and writes two files. If the LMDB is not mounted
it says so and exits; it never fetches anything.

The pair is one BigEarthNet cell imaged on two dates:

    T29SNB, cell 27_09, Alentejo, Portugal
      2017-10-02  the end of the 2017 Iberian drought
      2018-03-26  after the winter rains returned

The reservoir in this cell refills between the two dates. That is a real hydrological
event in real Sentinel-2 L2A reflectance, not a shape drawn into an array, which is the
whole reason to prefer it as a demo input.

Georeferencing, stated plainly: the **pixel size is exactly 10 m** and the CRS is the
tile's true UTM zone, so every area the system reports is correct. The **origin is
nominal** -- BigEarthNet ships no per-patch affine transform, and deriving one from the
MGRS tile identifier means reimplementing the 100 km square lettering, which is easy to
get subtly and invisibly wrong. Rather than fabricate a position, the origin is the tile's
approximate corner and this docstring says so. Nothing the tool computes depends on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: The cell, and the two acquisitions of it.
PATCHES = {
    "date_1_2017-10-02": "S2A_MSIL2A_20171002T112111_N9999_R037_T29SNB_27_09",
    "date_2_2018-03-26": "S2B_MSIL2A_20180326T112109_N9999_R037_T29SNB_27_09",
}
LMDB = Path("/media/cyborg-prithwish/Expansion/satquery-data/bigearthnet_lmdb")

#: Written in this order, named so the index code finds them by name rather than position.
BANDS = ("B02", "B03", "B04", "B08", "B11")
GSD_M = 10.0
#: UTM zone 29N, the true CRS of tile T29SNB. See the note above about the origin.
CRS = "EPSG:32629"
NOMINAL_ORIGIN = (500000.0, 4209800.0)
#: Sentinel-2 L2A is distributed as reflectance scaled by 10000.
REFLECTANCE_SCALE = 10000.0


def read_patch(store, patch_id: str) -> np.ndarray:
    """One patch as a (4, 120, 120) float32 reflectance stack.

    B11 is a 20 m band and arrives at 60x60. It is upsampled by pixel replication to the
    10 m grid, which is the honest operation: it adds no detail and pretends to none. The
    alternative -- refusing the stack because the bands differ in resolution -- would mean
    no NDBI, and NDBI is what keeps the built-up answer from being water.
    """
    stored = store.read(patch_id)
    bands = []
    for name in BANDS:
        array = np.asarray(stored[name], dtype=np.float32) / REFLECTANCE_SCALE
        if array.shape != (120, 120):
            factor = 120 // array.shape[0]
            array = np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)
        bands.append(array)
    return np.stack(bands)


def write_quicklook(path: Path, stack: np.ndarray) -> Path:
    """Write a human-viewable PNG: true colour on the left, detected water on the right.

    Left is B04/B03/B02 as red/green/blue -- what the scene looks like. Right paints every
    pixel the tool counted as water.

    The GeoTIFFs are four-band float32 and most image viewers will not open them, so
    "check it yourself" needs a picture. The right panel paints every pixel the tool
    counted as water, which is the actual claim being made -- a true-colour thumbnail
    alone would show you the scene without showing you the answer.
    """
    from PIL import Image

    from satquery.preprocess.indices import ndwi, water_mask
    from satquery.preprocess.optical import stretch_to_uint8

    blue, green, red, nir = stack[0], stack[1], stack[2], stack[3]
    rgb = np.ascontiguousarray(stretch_to_uint8(np.stack([red, green, blue])).transpose(1, 2, 0))

    overlay = rgb.copy()
    water = water_mask(ndwi(green, nir))
    overlay[water] = (0, 140, 255)  # the pixels the answer is counting

    gap = np.full((rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    panel = np.concatenate([rgb, gap, overlay], axis=1)
    Image.fromarray(panel).resize((panel.shape[1] * 3, panel.shape[0] * 3), Image.NEAREST).save(
        path
    )
    return path


def write_tif(path: Path, stack: np.ndarray) -> Path:
    """Write one patch as a GeoTIFF with named bands. Shared with find_change_pairs.py."""
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=stack.shape[1],
        width=stack.shape[2],
        count=len(BANDS),
        dtype="float32",
        crs=CRS,
        transform=from_origin(*NOMINAL_ORIGIN, GSD_M, GSD_M),
    ) as dst:
        for index, name in enumerate(BANDS, start=1):
            dst.write(stack[index - 1], index)
            dst.set_band_description(index, name)
    return path


def main() -> int:
    if not (LMDB / "data.mdb").exists():
        print(f"reBEN LMDB not found at {LMDB}.", file=sys.stderr)
        print("Mount the drive and re-run. This script downloads nothing.", file=sys.stderr)
        return 1

    import rasterio
    from rasterio.transform import from_origin

    from satquery.io.reben import ReBENStore

    out = Path(__file__).resolve().parents[1] / "demo_inputs"
    out.mkdir(parents=True, exist_ok=True)
    store = ReBENStore(LMDB, readahead=True)

    for stem, patch_id in PATCHES.items():
        stack = read_patch(store, patch_id)
        path = out / f"{stem}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=stack.shape[1],
            width=stack.shape[2],
            count=len(BANDS),
            dtype="float32",
            crs=CRS,
            transform=from_origin(*NOMINAL_ORIGIN, GSD_M, GSD_M),
        ) as dst:
            for index, name in enumerate(BANDS, start=1):
                dst.write(stack[index - 1], index)
                dst.set_band_description(index, name)
            dst.update_tags(SOURCE_PATCH_ID=patch_id, SOURCE_DATASET="BigEarthNet-V2 (reBEN)")

        write_quicklook(out / f"{stem}_preview.png", stack)

        green, nir = stack[1], stack[3]
        ndwi = (green - nir) / (green + nir + 1e-10)
        water_px = int(np.count_nonzero(ndwi > 0.0))
        print(
            f"{path.name}: {patch_id}\n"
            f"   water {water_px:>6} px = {water_px * GSD_M**2:>10,.0f} m2 "
            f"({water_px / ndwi.size * 100:.1f}% of the patch)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
