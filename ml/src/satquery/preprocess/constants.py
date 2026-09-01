"""Frozen preprocessing constants. The single place any of these values may live.

Every value here must be byte-identical at training time and at inference time.
Drift between the two paths silently destroys accuracy and is close to
undebuggable, so:

- never inline any of these values at a call site,
- never branch on train-vs-eval,
- never add a sensor-specific override.

Sentinel-1 at train time and RISAT at test time go through the same numbers.
Sentinel-2 at train time and Cartosat-2S at test time go through the same numbers.

`constants_fingerprint()` hashes the whole frozen set. Entry scripts log it, and
the eval harness asserts the fingerprint recorded in a checkpoint matches the one
in the running process. That turns silent drift into a loud failure.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

__all__ = [
    "BAND_ROLE_BLUE",
    "BAND_ROLE_GREEN",
    "BAND_ROLE_NIR",
    "BAND_ROLE_RED",
    "BAND_ROLE_SWIR",
    "BI_TEMPORAL_MIN_DELTA_DAYS",
    "BOX_COORDINATE_SCALE",
    "CANONICAL_GSD_SCALES_M",
    "CDVQA_ANSWERS",
    "CDVQA_LAND_COVER_ANSWERS",
    "CDVQA_NOMINAL_GSD_M",
    "CDVQA_QUESTION_TYPES",
    "CHANGE_ABSOLUTE_FLOOR_M2",
    "CHANGE_RELATIVE_FLOOR",
    "CONFIDENCE_BANDS",
    "CORINE_LEVEL1_BUILT_UP",
    "CORINE_LEVEL1_TO_CLASS",
    "CORINE_LEVEL1_WATER",
    "DB_EPS",
    "DB_SCALE",
    "EXCESS_GREEN_THRESHOLD",
    "FINGERPRINT_EXCLUDED",
    "FINGERPRINT_SCOPES",
    "FUSION_CLASS_INDEX",
    "FUSION_EXTRACTION_CLASSES",
    "FUSION_MASK_BACKGROUND",
    "GSD_ANISOTROPY_TOLERANCE",
    "GSD_TOKEN_FORMAT",
    "GSD_TOKEN_PATTERN",
    "GSD_TOKEN_UNKNOWN",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INDEX_EPS",
    "INSTRUCTION_CAPTION",
    "INSTRUCTION_CHANGE_DESCRIPTION",
    "INSTRUCTION_CHANGE_QUESTION_TEMPLATE",
    "INSTRUCTION_FUSION_EXTRACTION",
    "INSTRUCTION_REFER_TEMPLATE",
    "INSTRUCTION_VQA_TEMPLATE",
    "LANDCOVER_CLASSES",
    "LANDCOVER_IGNORE_INDEX",
    "METRES_PER_DEGREE_LAT",
    "METRES_PER_DEGREE_LON_EQUATOR",
    "NDBI_BUILTUP_THRESHOLD",
    "NDVI_VEGETATION_THRESHOLD",
    "NDWI_WATER_THRESHOLD",
    "OPTICAL_BAND_ORDER_4",
    "OPTICAL_BAND_ORDER_10",
    "PAIR_MIN_EXTENT_OVERLAP",
    "PANSHARPEN_METHOD_DEFAULT",
    "PANSHARPEN_METHOD_FALLBACK",
    "PANSHARPEN_VARIANCE_EPS",
    "REBEN_PATCH_EXTENT_M",
    "REBEN_SAR_DB_RANGE",
    "REBEN_SAR_UNITS",
    "RGB_CHANGE_DISTANCE_THRESHOLD",
    "RGB_CHANGE_MIN_COMPONENT_PX",
    "SAR_PSEUDO_RGB_LAYOUT",
    "SAR_SINGLE_POL_WARNING",
    "SAR_WATER_DB_THRESHOLD",
    "SPECKLE_FILTER_NUM_LOOKS",
    "SPECKLE_FILTER_WINDOW",
    "SPECKLE_SUBWINDOW",
    "STRETCH_OUTPUT_RANGE",
    "STRETCH_PERCENTILES",
    "constants_fingerprint",
]

# --------------------------------------------------------------------------------------
# GSD token
# --------------------------------------------------------------------------------------
# Every instruction string handed to the model is prefixed with the ground sampling
# distance of its input. The GSD is computed from the GeoTIFF affine transform --
# never parsed from a filename, never assumed from the sensor name. When it genuinely
# cannot be computed we emit the unknown token rather than a plausible guess, because a
# wrong GSD token is worse than an absent one: the model has been trained to trust it.

GSD_TOKEN_FORMAT: Final[str] = "<gsd:{value:.1f}m>"
GSD_TOKEN_UNKNOWN: Final[str] = "<gsd:unknown>"
GSD_TOKEN_PATTERN: Final[str] = r"<gsd:(?:unknown|\d+\.\d+m)>"

# Canonical scales the resampler snaps to. Chosen to span every sensor in play:
# Cartosat-2S PAN (~0.65 m), Cartosat-2S MS (~1.6-2 m), RISAT FRS/MRS (~1-25 m),
# Sentinel-2 (10/20/60 m), Sentinel-1 GRD (10 m).
CANONICAL_GSD_SCALES_M: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)

# Fractional difference between x and y pixel size above which a raster is flagged
# as anisotropic and a warning is attached to its ImageRef.
GSD_ANISOTROPY_TOLERANCE: Final[float] = 0.02

# Used only to convert a degrees-based affine transform (geographic CRS) into metres.
# A raster in EPSG:4326 has a transform in degrees; reading it as metres yields a GSD
# near zero and a silently wrong token, so the conversion is mandatory, not optional.
METRES_PER_DEGREE_LAT: Final[float] = 110574.0
METRES_PER_DEGREE_LON_EQUATOR: Final[float] = 111320.0

# --------------------------------------------------------------------------------------
# SAR rendering
# --------------------------------------------------------------------------------------
# Raw SAR is never fed to a model. The fixed five-step pipeline is implemented in
# `preprocess/sar.py::render_sar` and these are its only tunables.

# Step 1 -- Refined Lee speckle filter.
SPECKLE_FILTER_WINDOW: Final[int] = 7
# Sub-block size used for directional edge detection inside the 7x7 window. The
# window is covered by a 3x3 grid of 3x3 sub-block means whose gradients give the
# dominant edge direction (one of eight).
SPECKLE_SUBWINDOW: Final[int] = 3
# Equivalent number of looks, used to derive the speckle coefficient of variation
# Cu = 1/sqrt(L), which is the threshold separating "this variation is speckle, smooth
# it" from "this variation is an edge, preserve it".
#
# Frozen at 4.0 (Cu = 0.5), a representative multi-look value that covers both
# Sentinel-1 GRD IW at train time (ENL around 4.4) and RISAT multi-look products at
# test time. One number for both is the point: a per-sensor ENL would be marginally
# more accurate per sensor and would reintroduce exactly the train/test drift this
# module exists to prevent.
#
# The single-look value 1.0 gives Cu = 1.0, under which even a 1-to-100 step edge has
# a coefficient of variation below threshold and gets smoothed away. Re-estimate this
# from a homogeneous patch of real RISAT data before the final submission.
SPECKLE_FILTER_NUM_LOOKS: Final[float] = 4.0

# Step 2 -- linear amplitude to decibels: DB_SCALE * log10(x + DB_EPS).
DB_EPS: Final[float] = 1e-6
DB_SCALE: Final[float] = 10.0

# Step 3 -- per-image percentile stretch, then clip. Shared by SAR and optical so the
# two paths cannot drift apart.
STRETCH_PERCENTILES: Final[tuple[float, float]] = (2.0, 98.0)
STRETCH_OUTPUT_RANGE: Final[tuple[float, float]] = (0.0, 255.0)

# Step 4 -- pseudo-RGB channel assignment. The ratio channel is computed in the dB
# domain, where a ratio becomes a difference: VV_dB - VH_dB.
SAR_PSEUDO_RGB_LAYOUT: Final[tuple[str, str, str]] = ("VV", "VH", "VV/VH_dB_ratio")

# Step 5 -- single-polarisation input.
SAR_SINGLE_POL_WARNING: Final[str] = (
    "single-polarisation SAR input: R=G=available polarisation, B=zeros; "
    "the VV/VH ratio channel carries no information for this scene"
)

# Backscatter threshold for the deterministic water mask, in dB on the VV channel.
# Smooth open water is a specular reflector and returns very little energy to the
# sensor; -18 dB is the widely used operational cut for Sentinel-1 GRD flood mapping.
SAR_WATER_DB_THRESHOLD: Final[float] = -18.0

# --------------------------------------------------------------------------------------
# Optical band order
# --------------------------------------------------------------------------------------
# Internal order is fixed and positional access is only ever legal AFTER a raster has
# been reordered into it. Sensor-specific band names are mapped in `io/modality.py`.

OPTICAL_BAND_ORDER_4: Final[tuple[str, ...]] = ("B02", "B03", "B04", "B08")
OPTICAL_BAND_ORDER_10: Final[tuple[str, ...]] = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
)

# Canonical band carrying each physical role, so index maths never indexes by position.
BAND_ROLE_BLUE: Final[str] = "B02"
BAND_ROLE_GREEN: Final[str] = "B03"
BAND_ROLE_RED: Final[str] = "B04"
BAND_ROLE_NIR: Final[str] = "B08"
BAND_ROLE_SWIR: Final[str] = "B11"

# --------------------------------------------------------------------------------------
# Spectral indices
# --------------------------------------------------------------------------------------
# NDWI = (G - NIR) / (G + NIR)        McFeeters 1996
# NDBI = (SWIR - NIR) / (SWIR + NIR)  Zha et al. 2003
# NDVI = (NIR - R) / (NIR + R)        Rouse et al. 1974
# Every denominator is guarded with INDEX_EPS.
INDEX_EPS: Final[float] = 1e-6

# Decision thresholds for the deterministic masks. These are the literature defaults;
# they are the safety net that still produces a defensible, visible answer when the
# learned model is uncertain on an unseen sensor.
NDWI_WATER_THRESHOLD: Final[float] = 0.0
NDBI_BUILTUP_THRESHOLD: Final[float] = 0.0
NDVI_VEGETATION_THRESHOLD: Final[float] = 0.2

# ── RGB-only change detection ─────────────────────────────────────────────────────
#
# For imagery with no near-infrared band -- a screenshot, a JPEG, an aerial RGB tile --
# no spectral index applies and no class can be named. What remains is radiometric: a
# pixel whose colour moved a lot between two dates changed, whatever it changed into.
#
# This is a genuinely weaker instrument and is reported as such. It says WHERE, never
# WHAT, and it cannot tell a new building from a ploughed field from a cloud shadow.

#: FLOOR on the change distance, not the operating threshold.
#:
#: The threshold itself is found per pair by Otsu, because no fixed distance transfers.
#: Measured across real SECOND pairs the median per-pixel distance ran from 0.8 to 2.2 and
#: the 99th percentile from 5.5 to 18: two acquisitions disagree by an amount set by how
#: far apart they are in time, the season, the sun angle and the sensor. This constant was
#: originally the threshold, reasoned from the geometry of the normalised space rather
#: than checked against imagery, and it sat *below the median* on real data -- it flagged
#: 90% of every scene as changed.
#:
#: It survives as a floor, and the floor does real work: Otsu always finds a split, even
#: in a distribution that is entirely noise, so a genuinely unchanged pair would otherwise
#: come back with a confident partition of its own sensor noise.
RGB_CHANGE_DISTANCE_THRESHOLD: Final[float] = 0.35

#: Connected components smaller than this are dropped from the RGB change mask.
#:
#: Radiometric differencing is noisy in a way index thresholding is not: compression
#: artefacts, resampling and a one-pixel registration error all fire it. Requiring a
#: contiguous patch removes the speckle that would otherwise dominate the count.
RGB_CHANGE_MIN_COMPONENT_PX: Final[int] = 12

#: Excess Green: 2G - R - B, a vegetation proxy for imagery with no NIR.
#:
#: Woebbecke et al. 1995. Far weaker than NDVI -- it separates green things from
#: non-green things, which is not the same as separating live vegetation from anything
#: painted green -- so it is used to describe an RGB change, never to measure vegetation
#: area on its own.
EXCESS_GREEN_THRESHOLD: Final[float] = 0.08

# --------------------------------------------------------------------------------------
# Pan-sharpening
# --------------------------------------------------------------------------------------
# Cartosat-2S delivers a high-resolution panchromatic band plus lower-resolution
# multispectral bands. Sharpening happens before anything else sees the stack, so the
# fusion tool always receives one consistent resolution.
PANSHARPEN_METHOD_DEFAULT: Final[str] = "gram_schmidt"
PANSHARPEN_METHOD_FALLBACK: Final[str] = "brovey"
# Below this variance the Gram-Schmidt first component is degenerate and the transform
# is not invertible; we fall back to Brovey rather than dividing by ~0.
PANSHARPEN_VARIANCE_EPS: Final[float] = 1e-8

# --------------------------------------------------------------------------------------
# Pair validation
# --------------------------------------------------------------------------------------
# Two acquisitions closer together than this are treated as the same date, so a pair
# that differs only by minutes is not mistaken for a change-detection request.
BI_TEMPORAL_MIN_DELTA_DAYS: Final[float] = 1.0
# Minimum intersection-over-union of the two footprints for a pair to be usable.
PAIR_MIN_EXTENT_OVERLAP: Final[float] = 0.5

# --------------------------------------------------------------------------------------
# Vision-language backbone image normalisation
# --------------------------------------------------------------------------------------
# EarthDial is InternVL-derived and was trained with ImageNet normalisation on 448 px
# tiles. These are the backbone's own numbers, not ours to choose -- but they are frozen
# here for the same reason as everything else in this file: the tiling and normalisation
# applied when we fine-tune must be byte-identical to the tiling and normalisation
# applied when we evaluate, or the adapter is being scored on a distribution it never
# saw. Read `force_image_size` and `max_dynamic_patch` from the checkpoint's own
# config.json rather than hardcoding them; only the normalisation lives here.
IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------------------
# Instruction templates
# --------------------------------------------------------------------------------------
# The exact phrasings the model is instruction-tuned on, taken verbatim from
# VRSBench_train.json (142,390 records: 85,813 [vqa], 36,313 [refer], 20,264 [caption]).
#
# These are frozen for the same reason as the numeric constants: the prompt used at
# training time and the prompt used at inference time must be byte-identical. Prompting
# an adapter with a paraphrase is prompting it off-distribution, and the failure is
# quiet -- the model still answers, just worse, and no test catches it.
#
# The `<image>` placeholder and the leading task tag are part of the format.
INSTRUCTION_CAPTION: Final[str] = "[caption] Could you describe the contents of this image for me?"
INSTRUCTION_VQA_TEMPLATE: Final[str] = "[vqa] {question}. A short answer to the question is"
#: Fusion extraction. Named in the same imperative register as the other frozen
#: instructions so the adapted model sees one consistent prompt style across tasks.
INSTRUCTION_FUSION_EXTRACTION: Final[str] = (
    "[fusion] Using the optical and SAR images together, extract the built-up areas "
    "and the water bodies in this scene."
)

#: Pixel encoding of the fusion extraction mask. Background is 0, then the classes of
#: `FUSION_EXTRACTION_CLASSES` in order, offset by one. Single-band rather than
#: one-hot so the mask opens as a readable categorical raster in QGIS.
FUSION_MASK_BACKGROUND: Final[int] = 0

#: Free-form bi-temporal change description.
#:
#: The two-image framing is explicit -- "Image 1" and "Image 2" -- because an InternVL-based
#: backbone receives multiple images as a flat sequence of tiles with nothing marking where
#: one ends and the next begins. Without a positional cue the model has no way to know which
#: acquisition is earlier, and a change description with before and after transposed is
#: worse than no answer: it is confidently backwards.
#:
#: The `[change]` tag follows the pattern of the other frozen instructions. Unlike them it
#: is OUR convention rather than a verified upstream EarthDial tag, because the published
#: prompt format for its bi-temporal checkpoint is not documented. Recorded here so nobody
#: later assumes it was copied from the model card.
INSTRUCTION_CHANGE_DESCRIPTION: Final[str] = (
    "[change] Image 1 is the earlier acquisition and Image 2 is the later one. "
    "Describe what changed between them."
)

#: The same, when the caller asked a specific question rather than wanting a summary.
INSTRUCTION_CHANGE_QUESTION_TEMPLATE: Final[str] = (
    "[change] Image 1 is the earlier acquisition and Image 2 is the later one. {question}"
)

INSTRUCTION_REFER_TEMPLATE: Final[str] = (
    "[refer] could you tell me the location for <p>{phrase}</p>?"
)

# Box coordinate scale used by the referring targets, e.g. `{<45><45><59><59>}`.
# VRSBench normalises to 0-100, NOT the 0-1000 grid InternVL uses by default. Getting
# this wrong yields boxes off by 10x, which still parse and still render, so it would
# survive review and simply score near zero.
#
# Observed values exceed 100 in a minority of records (max 154 in a 400-record sample),
# so boxes are clamped rather than assumed in range.
BOX_COORDINATE_SCALE: Final[float] = 100.0

# --------------------------------------------------------------------------------------
# CDVQA closed answer set
# --------------------------------------------------------------------------------------
# VERIFIED against the published annotations (github.com/YZHJessica/CDVQA), all four
# splits: the answer vocabulary is exactly 19 values, not the six the architecture describes.
#
# The six land-cover classes are the answer set for only three of the eight question
# types, and cover 23.4% of answers. The other 76.6% are yes/no (52.2%) and change-ratio
# buckets (24.4%). A six-way head would be structurally unable to answer three quarters
# of the benchmark -- it would train happily and cap out around 23%.
#
# This tuple's ORDER is the classification head's output-layer ordering. Permuting it
# silently scrambles a reloaded checkpoint, so it is frozen and grouped by answer kind.
CDVQA_ANSWERS: Final[tuple[str, ...]] = (
    # binary, 52.2% of answers
    "no",
    "yes",
    # land cover, 23.4%
    "NVG_surface",
    "buildings",
    "low_vegetation",
    "trees",
    "water",
    "playgrounds",
    # change-ratio buckets, 24.4%
    "0",
    "0_to_10",
    "10_to_20",
    "20_to_30",
    "30_to_40",
    "40_to_50",
    "50_to_60",
    "60_to_70",
    "70_to_80",
    "80_to_90",
    "90_to_100",
)

#: The land-cover subset, in the dataset's own token spelling. Reported as a breakdown so
#: the six-class framing still has a number, but it is NOT the head's output.
#: Note the tokens are `NVG_surface` etc., not the prose the architecture quotes.
CDVQA_LAND_COVER_ANSWERS: Final[tuple[str, ...]] = (
    "NVG_surface",
    "buildings",
    "low_vegetation",
    "trees",
    "water",
    "playgrounds",
)

#: The eight question types, used for per-type accuracy. Not mentioned in the architecture.
CDVQA_QUESTION_TYPES: Final[tuple[str, ...]] = (
    "change_or_not",
    "change_ratio_types",
    "increase_or_not",
    "decrease_or_not",
    "change_to_what",
    "smallest_change",
    "largest_change",
    "change_ratio",
)

#: SECOND (the imagery CDVQA is built on) is documented at 0.53 m/pixel. The CDVQA
#: annotations carry `res_x: .1524m`, which is exactly 6 inches and looks like an
#: inherited template value rather than a measurement. The documented figure is used and
#: the conflict is recorded, because a wrong GSD token is worse than a coarse one.
CDVQA_NOMINAL_GSD_M: Final[float] = 0.53

# -- reBEN / BigEarthNet v2.0 -----------------------------------------------------------

#: Every reBEN patch, whatever its band resolution, covers the same 1200 m x 1200 m
#: footprint on the ground. That invariant is what lets the GSD be *computed* rather
#: than assumed: a 120-pixel band is 10 m, a 60-pixel band 20 m, a 20-pixel band 60 m.
#:
#: This matters because the LMDB conversion (rico-hdl) stores band arrays as
#: safetensors and does **not** carry the GeoTIFF affine transform. Without this
#: constant the only remaining route to a GSD would be guessing from the sensor name,
#: which `preprocess/gsd.py` exists to forbid. Dividing the documented extent by the
#: observed array width keeps the value derived from the data in hand.
REBEN_PATCH_EXTENT_M: Final[float] = 1200.0

#: reBEN Sentinel-1 patches ship as sigma-nought **already in decibels**, unlike raw
#: Sentinel-1 GRD or RISAT, which arrive as linear amplitude. The frozen SAR pipeline
#: starts from linear, so a reBEN reader must invert this before entering it -- see
#: `preprocess.sar.db_to_linear`. Speckle is multiplicative in the linear domain and
#: additive in the log domain, so filtering dB values directly would apply the wrong
#: noise model; converting back is the correct operation, not a workaround.
REBEN_SAR_UNITS: Final[str] = "db"

#: Plausibility bounds for Sentinel-1 sigma0 in dB, checked on load so a future change
#: of source units fails loudly instead of silently producing a black image.
#:
#: Measured over 400 randomly sampled reBEN patches (800 polarisation bands): observed
#: minimum -51.9 dB, 99th-percentile maximum 16.7 dB, absolute maximum 24.3 dB. The
#: upper bound is 30 rather than 20 because bright urban corner reflectors legitimately
#: exceed 20 dB in about 0.4% of patches -- a bound that fires on real data trains
#: everyone to ignore the warning, which is worse than not having one.
REBEN_SAR_DB_RANGE: Final[tuple[float, float]] = (-60.0, 30.0)

#: CORINE Land Cover is hierarchical: the leading digit of a Level-3 code is the
#: Level-1 class. Grouping on that digit rather than enumerating ~44 leaf codes keeps
#: the mapping short and immune to leaf-code revisions.
#:
#: 1 artificial surfaces, 2 agricultural, 3 forest and semi-natural, 4 wetlands,
#: 5 water bodies.
CORINE_LEVEL1_BUILT_UP: Final[int] = 1
CORINE_LEVEL1_WATER: Final[int] = 5

#: The two extraction targets the fusion tool is scored on. Ordered; the fusion head's
#: output channels follow this order, and so does the eval report.
#:
#: Deliberately paired with the spectral indices that back them up: built-up with NDBI,
#: water with NDWI. When the learned output and the index disagree on unseen sensor
#: data, that disagreement is the confidence signal.
FUSION_EXTRACTION_CLASSES: Final[tuple[str, ...]] = ("built_up", "water")

#: Index that provides the deterministic second opinion for each extraction class.
FUSION_CLASS_INDEX: Final[dict[str, str]] = {"built_up": "ndbi", "water": "ndwi"}

# ── land-cover segmentation classes ───────────────────────────────────────────────
#
# CORINE Level-1 collapsed to five classes, in the order the segmentation head emits.
# Level-1 rather than Level-3 because reBEN's per-patch label distribution is long-tailed:
# many of the 44 Level-3 classes appear in a handful of patches, and a head trained on
# them would learn the frequent few and guess the rest. Five classes it can actually see
# beats forty-four it cannot.
#
# Index 0 is reserved for "unlabelled" so a pixel with no reference value is ignored by
# the loss rather than silently taught as a class.
LANDCOVER_CLASSES: Final[tuple[str, ...]] = (
    "unlabelled",
    "built_up",
    "agriculture",
    "vegetation",
    "wetland",
    "water",
)

#: CORINE Level-1 digit -> index into `LANDCOVER_CLASSES`.
#:
#: The leading digit of a Level-3 code is its Level-1 group, which is what makes this a
#: lookup rather than a table of 44 entries: 512 (water bodies) and 511 (water courses)
#: both start with 5 and both mean water.
CORINE_LEVEL1_TO_CLASS: Final[dict[int, int]] = {
    1: 1,  # artificial surfaces  -> built_up
    2: 2,  # agricultural areas   -> agriculture
    3: 3,  # forest and semi-natural -> vegetation
    4: 4,  # wetlands             -> wetland
    5: 5,  # water bodies         -> water
}

#: Ignored by the segmentation loss. Matches `LANDCOVER_CLASSES[0]`.
LANDCOVER_IGNORE_INDEX: Final[int] = 0

# -- change verdicts --------------------------------------------------------------------

#: A trend is reported only when BOTH floors are exceeded. A large relative change on a
#: tiny area is noise; a large absolute change on a huge area can sit inside measurement
#: error. Requiring both is what keeps "increased" a defensible claim.
#:
#: 10,000 m2 is one hectare, and 0.05 is five percent of the class's own area at T1.
CHANGE_ABSOLUTE_FLOOR_M2: Final[float] = 10_000.0
CHANGE_RELATIVE_FLOOR: Final[float] = 0.05

#: Agreement IoU to confidence band, highest floor first. Bands rather than a raw number
#: because the number is only meaningful as a comparison between two independent signals,
#: and a reader should not be invited to over-read a second decimal place.
CONFIDENCE_BANDS: Final[tuple[tuple[str, float], ...]] = (
    ("high", 0.75),
    ("medium", 0.45),
    ("low", 0.0),
)


#: Names EXCLUDED from the fingerprint.
#:
#: The fingerprint answers one question: "would a model trained earlier see different
#: pixels or different tokens if it ran in this process?" Constants that cannot change
#: that answer must be excluded, or the guard fires on work that did not touch
#: preprocessing at all.
#:
#: That is not hypothetical. Hashing the whole module meant adding land-cover class names
#: and RGB-change thresholds -- for capabilities that did not exist when the fusion
#: encoder was trained, and which its preprocessing never consults -- invalidated its
#: checkpoint. Verified at the time: no existing constant changed value and none was
#: removed; nine were added. A guard that cries drift on every addition is one people
#: learn to re-record their way past, which is exactly how real drift then slips through.
#:
#: Anything genuinely upstream of a model -- band names and order, index thresholds, the
#: SAR pipeline, normalisation, tiling, instruction templates -- must NOT be listed here.
FINGERPRINT_EXCLUDED: Final[frozenset[str]] = frozenset(
    {
        # Reporting thresholds: applied to measurements after a model has run.
        "CHANGE_ABSOLUTE_FLOOR_M2",
        "CHANGE_RELATIVE_FLOOR",
        "CONFIDENCE_BANDS",
        # Land-cover label vocabulary: the segmenter's own output space, and it carries
        # its own config. Nothing trained before it consumes these.
        "CORINE_LEVEL1_TO_CLASS",
        "LANDCOVER_CLASSES",
        "LANDCOVER_IGNORE_INDEX",
        # RGB radiometric change: a separate tool with no learned weights at all.
        "EXCESS_GREEN_THRESHOLD",
        "FINGERPRINT_EXCLUDED",
        "FINGERPRINT_SCOPES",
        "RGB_CHANGE_DISTANCE_THRESHOLD",
        "RGB_CHANGE_MIN_COMPONENT_PX",
    }
)


def _frozen_values() -> dict[str, object]:
    """Collect the preprocessing-critical constants into a JSON-serialisable mapping."""
    module = globals()
    return {
        name: module[name]
        for name in sorted(__all__)
        if name != "constants_fingerprint"
        and name not in FINGERPRINT_EXCLUDED
        and not callable(module[name])
    }


#: Named subsets a model can pin itself to.
#:
#: One global fingerprint cannot serve models with disjoint inputs. The optical+SAR fusion
#: encoder consumes band names, index thresholds and the SAR pipeline; it never sees an
#: instruction template. Yet adding two VLM instruction constants invalidated its
#: checkpoint, because the hash covered both. A guard that fires on changes a model cannot
#: possibly observe is one people route around, and routing around it is how real drift
#: gets through.
#:
#: A model pins the scope it actually depends on. `None` still hashes everything, which is
#: the right default for anything that reads both pixels and prompts.
FINGERPRINT_SCOPES: Final[dict[str, tuple[str, ...]]] = {
    # Everything upstream of pixels reaching a model: which bands, in what order, scaled
    # and stretched how, and every threshold applied to them.
    "pixels": (
        "BAND_ROLE_BLUE",
        "BAND_ROLE_GREEN",
        "BAND_ROLE_NIR",
        "BAND_ROLE_RED",
        "BAND_ROLE_SWIR",
        "DB_EPS",
        "DB_SCALE",
        "IMAGENET_MEAN",
        "IMAGENET_STD",
        "INDEX_EPS",
        "NDBI_BUILTUP_THRESHOLD",
        "NDVI_VEGETATION_THRESHOLD",
        "NDWI_WATER_THRESHOLD",
        "SAR_PSEUDO_RGB_LAYOUT",
        "SAR_WATER_DB_THRESHOLD",
        "SPECKLE_FILTER_NUM_LOOKS",
        "SPECKLE_FILTER_WINDOW",
        "SPECKLE_SUBWINDOW",
        "STRETCH_OUTPUT_RANGE",
        "STRETCH_PERCENTILES",
    ),
    # What a language model is told, which decides its tokens as surely as bands decide
    # a segmenter's pixels.
    "instructions": (
        "GSD_TOKEN_FORMAT",
        "GSD_TOKEN_UNKNOWN",
        "INSTRUCTION_CAPTION",
        "INSTRUCTION_CHANGE_DESCRIPTION",
        "INSTRUCTION_CHANGE_QUESTION_TEMPLATE",
        "INSTRUCTION_REFER_TEMPLATE",
        "INSTRUCTION_VQA_TEMPLATE",
    ),
}


def constants_fingerprint(scope: str | None = None) -> str:
    """Return a stable SHA-256 over the preprocessing-critical constants.

    Log this at the top of every training run and assert it in the eval harness. If the
    fingerprint in a checkpoint does not match the fingerprint of the process loading it,
    preprocessing has drifted and the scores are not comparable.

    Scoped by `FINGERPRINT_EXCLUDED` to the constants that decide what a model actually
    sees. Everything downstream of inference -- reporting thresholds, a newer model's own
    label vocabulary -- is excluded, so adding a capability does not invalidate every
    checkpoint that predates it.

    Args:
        scope: A key of `FINGERPRINT_SCOPES`, hashing only that subset, or `None` for
            every included constant. A model should pin the narrowest scope that covers
            what it reads: a wider one makes the guard fire on changes it cannot observe,
            and a narrower one lets real drift through.

    Raises:
        KeyError: Unknown scope. Better than silently hashing everything and reporting a
            mismatch the caller cannot explain.
    """
    values = _frozen_values()
    if scope is not None:
        if scope not in FINGERPRINT_SCOPES:
            raise KeyError(
                f"unknown fingerprint scope {scope!r}; known: {sorted(FINGERPRINT_SCOPES)}"
            )
        names = FINGERPRINT_SCOPES[scope]
        missing = [name for name in names if name not in values]
        if missing:
            raise KeyError(
                f"scope {scope!r} names constants that do not exist: {missing}. A scope "
                "that silently skips a constant would stop guarding it."
            )
        values = {name: values[name] for name in names}
    payload = json.dumps(values, sort_keys=True, default=list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
