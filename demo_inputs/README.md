# Demo inputs — real Sentinel-2

Two acquisitions of **one BigEarthNet cell**, exported from the reBEN LMDB. Real
Sentinel-2 L2A surface reflectance, not drawn into an array.

| file | acquisition | source patch |
|---|---|---|
| `demo_input_tif_1.tif` | 2017-10-02 | `S2A_MSIL2A_20171002T112111_N9999_R037_T29SNB_27_09` |
| `demo_input_tif_2.tif` | 2018-03-26 | `S2B_MSIL2A_20180326T112109_N9999_R037_T29SNB_27_09` |

120×120 px, 4 bands (`B04` `B03` `B08` `B11`), EPSG:32629, **10 m GSD**. Tile T29SNB,
cell 27_09 — Alentejo, Portugal. The two dates straddle the end of the 2017 Iberian
drought: the reservoir in this cell refills over the winter. CORINE labels the cell
*Agro-forestry areas, Broad-leaved forest, Inland waters, Transitional woodland/shrub*.

Regenerate with `PYTHONPATH=src python scripts/export_demo_inputs.py` (reads the mounted
LMDB, downloads nothing).

## What it should say

| question | answer | check |
|---|---|---|
| "How has the water body changed between these two dates?" | water increased by **31.87 ha** (118.1%) | 269,900 → 588,600 m², matching NDWI computed directly on the bands |
| "Describe the land cover" (date 1) | built up 39.7%, vegetation 39.4%, water 18.7% | water 18.7% is right; see the caveat below |
| "Describe the land cover" (date 2) | water 40.9% | the reservoir after the rains |

**Water is the trustworthy number here.** NDWI is a genuinely reliable water detector and
the measured change matches the imagery.

## Two honest caveats, both surfaced as warnings

**Built-up is an upper bound.** NDBI separates "bright in SWIR, dark in NIR" from
everything else, and dry bare soil and harvested cropland look exactly like concrete under
that test. This cell has no urban CORINE class at all, and the tool still reports 39.7%
built-up. Open water and vegetation are masked out — both are definitionally not built-up
— but no arithmetic can tell a car park from a ploughed field in the Alentejo in October.
Read it as "built-up or bare ground". This is precisely the gap a trained
`LandCoverSegmenter` closes, and the reason it is next.

The threshold was deliberately **not** retuned to make this scene look better. Fitting a
frozen constant to one demo patch is how train/eval drift gets introduced.

**The percentages overlap.** NDWI, NDBI and NDVI are independent thresholds, not a
partition, so a pixel can satisfy two and the numbers need not sum to 100.

## Georeferencing

Pixel size is exactly 10 m and the CRS is the tile's true UTM zone, so **every area the
system reports is correct**. The origin is nominal: BigEarthNet ships no per-patch affine
transform, and deriving one from the MGRS tile identifier means reimplementing the 100 km
square lettering, which is easy to get subtly and invisibly wrong. Nothing the tool
computes depends on the origin.

## What will not work

Any question needing a learned model — VQA, captioning, counting, grounding, semantic
change. Those tools are unbound because no weights exist yet; the UI says so rather than
answering. An RGB screenshot is refused: no near-infrared band, so no index applies.
