"""Answer normalisation. On closed-set benchmarks this is worth several accuracy points.

RSVQA and CDVQA are scored by exact match. A model that answers "Yes." when the
reference is "yes", or "buildings." when the reference is "buildings", is correct and
would score zero without this module. Equally, normalisation must never be so
aggressive that it maps a wrong answer onto a right one -- so it only ever removes
formatting, never content.

Normalisation is applied identically to the prediction and the reference. Applying it
to only one side would inflate the score.
"""

from __future__ import annotations

import re
import string
import unicodedata

from satquery.preprocess.constants import CDVQA_ANSWERS

__all__ = [
    "answers_match",
    "normalize_answer",
    "normalize_cdvqa_class",
    "normalize_for_dataset",
    "normalize_yes_no",
]

_ARTICLES = re.compile(r"\b(?:a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = str.maketrans("", "", string.punctuation)

_YES_FORMS: frozenset[str] = frozenset(
    {"yes", "y", "true", "yeah", "yep", "correct", "affirmative"}
)
_NO_FORMS: frozenset[str] = frozenset({"no", "n", "false", "nope", "negative", "incorrect"})

# Wordings that mean the same CDVQA answer. Keys are already normalised by
# `normalize_answer` (lowercased, punctuation and underscores stripped), values are the
# dataset's own tokens from `CDVQA_ANSWERS`.
#
# Note the dataset spells the land-cover classes `NVG_surface`, `low_vegetation` and so
# on -- NOT the prose quoted elsewhere. A model that answers in prose, or a reference
# quoted from the paper rather than the data, has to land on the dataset token or it
# scores zero.
_CDVQA_SYNONYMS: dict[str, str] = {
    # land cover
    "nvg surface": "NVG_surface",
    "nvgsurface": "NVG_surface",
    "nonvegetated ground surface": "NVG_surface",
    "non vegetated ground surface": "NVG_surface",
    "bare land": "NVG_surface",
    "bare ground": "NVG_surface",
    "ground": "NVG_surface",
    "building": "buildings",
    "house": "buildings",
    "houses": "buildings",
    "tree": "trees",
    "woodland": "trees",
    "forest": "trees",
    "playground": "playgrounds",
    "sports field": "playgrounds",
    "grass": "low_vegetation",
    "grassland": "low_vegetation",
    "lowvegetation": "low_vegetation",
    "low vegetation": "low_vegetation",
    "water body": "water",
    "waterbody": "water",
    "lake": "water",
    "river": "water",
}

# Change-ratio buckets are written `0_to_10` in the data. A model that answers
# "0 to 10" or "0-10" means the same bucket and must not be marked wrong for spacing.
_RATIO_BUCKET_RE = re.compile(r"^(\d+)\s*(?:to|-)\s*(\d+)$")


def normalize_answer(text: str) -> str:
    """Lowercase, strip accents and punctuation, drop articles, collapse whitespace.

    The SQuAD normalisation recipe, which is what the VQA literature uses. Note that
    the hyphen in "non-vegetated ground surface" is punctuation and is therefore
    removed here; `normalize_cdvqa_class` puts the canonical form back.
    """
    if text is None:
        return ""

    lowered = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    lowered = lowered.lower().strip()
    lowered = lowered.translate(_PUNCTUATION)
    lowered = _ARTICLES.sub(" ", lowered)
    return _WHITESPACE.sub(" ", lowered).strip()


def normalize_yes_no(text: str) -> str:
    """Collapse every affirmative to `"yes"` and every negative to `"no"`.

    Returns the plainly normalised text when it is neither, so a non-binary answer to
    a binary question stays visibly wrong rather than being coerced to one of the two.
    """
    normalised = normalize_answer(text)
    first = normalised.split(" ")[0] if normalised else ""
    if normalised in _YES_FORMS or first in _YES_FORMS:
        return "yes"
    if normalised in _NO_FORMS or first in _NO_FORMS:
        return "no"
    return normalised


def normalize_cdvqa_class(text: str) -> str:
    """Map a free-form answer onto one of the six frozen CDVQA classes.

    Returns the normalised input unchanged when it matches no class, so an
    out-of-vocabulary answer scores as wrong instead of being silently snapped onto
    the nearest class.
    """
    normalised = normalize_answer(text)

    canonical_by_normalised = {normalize_answer(name): name for name in CDVQA_ANSWERS}
    if normalised in canonical_by_normalised:
        return canonical_by_normalised[normalised]
    if normalised in _CDVQA_SYNONYMS:
        return _CDVQA_SYNONYMS[normalised]

    # Matched against the RAW text as well as the normalised form: `normalize_answer`
    # strips the hyphen, so "0-10" would otherwise arrive here as "010".
    bucket = _RATIO_BUCKET_RE.match(str(text).strip().lower()) or _RATIO_BUCKET_RE.match(normalised)
    if bucket is not None:
        candidate = f"{bucket.group(1)}_to_{bucket.group(2)}"
        if candidate in CDVQA_ANSWERS:
            return candidate
    return normalised


def normalize_for_dataset(text: str, dataset: str) -> str:
    """Apply the normalisation appropriate to a benchmark's answer conventions.

    Args:
        text: Raw predicted or reference answer.
        dataset: Benchmark key. `"rsvqa"`, `"cdvqa"`, `"bigearthnet_txt"` and
            `"vrsbench"` are recognised; anything else falls back to `normalize_answer`.
    """
    key = dataset.strip().lower()
    if key == "cdvqa":
        return normalize_cdvqa_class(text)
    if key in {"rsvqa", "bigearthnet_txt"}:
        # These benchmarks are dominated by presence and comparison questions whose
        # references are literally "yes" or "no"; counts and area classes fall through
        # to the plain normaliser unchanged.
        collapsed = normalize_yes_no(text)
        return collapsed
    return normalize_answer(text)


def answers_match(prediction: str, reference: str, dataset: str = "") -> bool:
    """Whether a prediction matches a reference after identical normalisation."""
    if dataset:
        return normalize_for_dataset(prediction, dataset) == normalize_for_dataset(
            reference, dataset
        )
    return normalize_answer(prediction) == normalize_answer(reference)
