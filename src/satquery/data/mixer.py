"""Weighted multi-dataset sampler and its scale-resampling policy.

One training stream is drawn from several corpora at fixed proportions. The default
mix (`configs/data/mix_v1.yaml`) is 40 percent BigEarthNet.txt, 40 percent VRSBench
and 20 percent CDVQA: the first supplies remote-sensing adaptation and the only
co-registered SAR/optical pairs, the second supplies single-image VQA, captioning
and grounding at aerial resolution, the third supplies the mandatory bi-temporal
change task. RSVQA is evaluation-only and is not a mix component.

**Scale-resampling policy.** Training imagery is Sentinel at 10 m; the hidden ISRO
evaluation set is Cartosat-2S at sub-metre and RISAT SAR. That is more than a 20x
resolution gap, and it is the single largest risk to the whole submission. The
defence is aggressive random resampling: each drawn sample is resampled to a GSD
picked at random from the canonical ladder in
`satquery.preprocess.constants.CANONICAL_GSD_SCALES_M`, with per-scale probabilities
from the mix config, so the model sees the same scene across the full 0.5 m to 60 m
range instead of memorising 10 m. Because the resampler rewrites `ImageRef.gsd_m`
and the GSD token is derived from it by `Sample.prompt()`, the prompt tracks the
augmentation automatically and the model learns to read the token rather than to
assume a resolution.

The resampling itself is **not implemented here**; `MixedDataset.__getitem__` names
it as missing along with the component loaders.

Only `load_mix_spec` and its validation are real. Everything that needs pixels or a
built component dataset raises `NotImplementedError`.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from satquery.data.datasets import SampleDataset, SampleTransform
from satquery.data.schema import Sample
from satquery.preprocess.constants import CANONICAL_GSD_SCALES_M
from satquery.utils.logging import get_logger

__all__ = [
    "WEIGHT_TOLERANCE",
    "ConcatSampleDataset",
    "MixComponent",
    "MixSpec",
    "MixedDataset",
    "ScalePolicy",
    "SubsetSampleDataset",
    "load_mix_spec",
]

logger = get_logger(__name__)

#: Absolute tolerance on the component weight sum. Tight enough to catch a real
#: mistake (a dropped or duplicated component), loose enough for decimal YAML
#: literals that do not sum exactly in binary floating point.
WEIGHT_TOLERANCE: float = 1e-6


@dataclass(frozen=True)
class ScalePolicy:
    """How aggressively a mix resamples across the canonical GSD ladder.

    `scales_m` must be a subset of `CANONICAL_GSD_SCALES_M`; snapping to the frozen
    ladder rather than to arbitrary floats keeps the set of GSD tokens the model ever
    sees finite and identical between training and inference.
    """

    enabled: bool = True
    scales_m: tuple[float, ...] = CANONICAL_GSD_SCALES_M
    probabilities: tuple[float, ...] = ()
    apply_probability: float = 1.0

    def __post_init__(self) -> None:
        """Validate the ladder subset and the probability vector. Pure, runs on load."""
        off_ladder = tuple(s for s in self.scales_m if s not in CANONICAL_GSD_SCALES_M)
        if off_ladder:
            raise ValueError(
                f"scale policy names GSD scales {off_ladder!r} that are not on the frozen ladder "
                f"CANONICAL_GSD_SCALES_M={CANONICAL_GSD_SCALES_M!r}"
            )
        if not 0.0 <= self.apply_probability <= 1.0:
            raise ValueError(
                f"scale policy apply_probability must be in [0, 1], got {self.apply_probability}"
            )
        if self.probabilities:
            if len(self.probabilities) != len(self.scales_m):
                raise ValueError(
                    f"scale policy has {len(self.scales_m)} scales but "
                    f"{len(self.probabilities)} probabilities; they must be the same length"
                )
            total = sum(self.probabilities)
            if abs(total - 1.0) > WEIGHT_TOLERANCE:
                raise ValueError(
                    f"scale policy probabilities must sum to 1.0 within {WEIGHT_TOLERANCE}, "
                    f"got {total!r} for {dict(zip(self.scales_m, self.probabilities, strict=True))!r}"
                )


@dataclass(frozen=True)
class MixComponent:
    """One corpus in a mix, and the share of the stream it owns."""

    name: str
    """Component key, matching the dataset config stem, e.g. `bigearthnet_txt`."""

    weight: float
    """Share of drawn samples, in [0, 1]. All weights in a `MixSpec` sum to 1.0."""

    config: Path
    """Path to the dataset YAML this component is built from."""

    splits: tuple[str, ...] = ()
    """Splits of that corpus to draw from, concatenated before weighting."""

    options: dict[str, Any] = field(default_factory=dict)
    """Per-component loader overrides, e.g. `tasks` or `subset`."""

    def __post_init__(self) -> None:
        """Validate the weight range and that at least one split is named. Pure."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(
                f"component {self.name!r}: weight must be in [0, 1], got {self.weight}"
            )
        if not self.splits:
            raise ValueError(f"component {self.name!r}: at least one split must be named")


@dataclass(frozen=True)
class MixSpec:
    """A fully validated training mix: components, their weights and the scale policy."""

    name: str
    components: tuple[MixComponent, ...]
    scale_policy: ScalePolicy = field(default_factory=ScalePolicy)
    seed: int = 0
    source_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate that components are non-empty, uniquely named and sum to 1.0. Pure."""
        if not self.components:
            raise ValueError(f"mix {self.name!r}: at least one component is required")

        names = [component.name for component in self.components]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"mix {self.name!r}: duplicate component name(s) {duplicates!r}")

        total = sum(component.weight for component in self.components)
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(
                f"mix {self.name!r}: component weights must sum to 1.0 within {WEIGHT_TOLERANCE}, "
                f"got {total!r} from {self.weights!r}"
            )

    @property
    def weights(self) -> dict[str, float]:
        """Component weights keyed by name, in declaration order."""
        return {component.name: component.weight for component in self.components}

    def component(self, name: str) -> MixComponent:
        """Return the component called `name`.

        Raises:
            KeyError: If no component has that name.
        """
        for candidate in self.components:
            if candidate.name == name:
                return candidate
        raise KeyError(f"mix {self.name!r} has no component {name!r}; have {sorted(self.weights)}")


def _require(node: DictConfig, key: str, context: str) -> Any:
    """Fetch `key` from `node`, raising a located ValueError when it is absent."""
    if key not in node:
        raise ValueError(f"{context}: required key {key!r} is missing")
    return node[key]


def _as_tuple(value: Any) -> tuple[Any, ...]:
    """Normalise an OmegaConf list (or a bare scalar) into a plain tuple."""
    if value is None:
        return ()
    if isinstance(value, ListConfig | list | tuple):
        return tuple(OmegaConf.to_object(OmegaConf.create(list(value))))
    return (value,)


def load_mix_spec(path: str | Path) -> MixSpec:
    """Load and validate a mix YAML into a `MixSpec`.

    Fully implemented. Weight validation is the point of this function: a mix whose
    weights do not sum to 1.0 trains on a silently different distribution from the one
    written down, which is undetectable from the loss curve, so it is a hard error.

    Component config paths are resolved relative to the mix file's own directory when
    they are not absolute, so a mix and the dataset configs it references move together.

    Args:
        path: Path to a mix YAML, e.g. `configs/data/mix_v1.yaml`.

    Returns:
        The validated spec.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If a required key is missing, a weight is out of range, component
            names collide, the weights do not sum to 1.0, or the scale policy is
            inconsistent.
    """
    mix_path = Path(path).expanduser()
    if not mix_path.is_file():
        raise FileNotFoundError(f"mix config not found: {mix_path}")

    config = OmegaConf.load(mix_path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"{mix_path}: expected a mapping at the top level")

    context = str(mix_path)
    name = str(_require(config, "name", context))

    raw_components = _require(config, "components", context)
    if not isinstance(raw_components, ListConfig) or len(raw_components) == 0:
        raise ValueError(f"{context}: 'components' must be a non-empty list")

    components: list[MixComponent] = []
    for position, raw in enumerate(raw_components):
        item_context = f"{context}: components[{position}]"
        if not isinstance(raw, DictConfig):
            raise ValueError(f"{item_context}: expected a mapping")
        component_name = str(_require(raw, "name", item_context))
        component_config = Path(str(_require(raw, "config", item_context))).expanduser()
        if not component_config.is_absolute():
            component_config = (mix_path.parent / component_config).resolve()
        components.append(
            MixComponent(
                name=component_name,
                weight=float(_require(raw, "weight", item_context)),
                config=component_config,
                splits=tuple(str(split) for split in _as_tuple(raw.get("splits"))),
                options=dict(OmegaConf.to_object(raw.get("options", OmegaConf.create({})))),  # type: ignore[arg-type]
            )
        )

    raw_policy = config.get("scale_policy")
    if raw_policy is None:
        policy = ScalePolicy()
    else:
        if not isinstance(raw_policy, DictConfig):
            raise ValueError(f"{context}: 'scale_policy' must be a mapping")
        scales = tuple(float(value) for value in _as_tuple(raw_policy.get("scales_m")))
        policy = ScalePolicy(
            enabled=bool(raw_policy.get("enabled", True)),
            scales_m=scales or CANONICAL_GSD_SCALES_M,
            probabilities=tuple(float(v) for v in _as_tuple(raw_policy.get("probabilities"))),
            apply_probability=float(raw_policy.get("apply_probability", 1.0)),
        )

    spec = MixSpec(
        name=name,
        components=tuple(components),
        scale_policy=policy,
        seed=int(config.get("seed", 0)),
        source_path=mix_path,
    )
    logger.info(
        "loaded mix %r from %s: %s, scale_policy.enabled=%s",
        spec.name,
        mix_path,
        spec.weights,
        spec.scale_policy.enabled,
    )
    return spec


class MixedDataset(SampleDataset):
    """Weighted draw across several `SampleDataset` components.

    Single responsibility: decide *which* component a given index comes from and
    delegate, then apply the scale-resampling transform. It never parses an on-disk
    annotation format itself -- that belongs to the component loaders -- and it never
    tokenises, which belongs to the collator.

    Sampling is index-deterministic rather than stateful: a given epoch index always
    maps to the same component and the same underlying record for a given
    `MixSpec.seed`, so a run is reproducible and resumable without carrying RNG state.
    """

    def __init__(
        self,
        spec: MixSpec,
        components: dict[str, SampleDataset] | None = None,
        transform: SampleTransform | None = None,
        epoch_length: int | None = None,
    ) -> None:
        """Configure the mix without touching disk.

        Args:
            spec: A validated `MixSpec`, normally from `load_mix_spec`.
            components: Already-built component datasets keyed by `MixComponent.name`.
                `None` means they have not been built yet, which is why the map-style
                methods raise.
            transform: Optional `Sample`-to-`Sample` rewrite applied after the scale
                resampling, e.g. a prompt-template rewrite.
            epoch_length: Number of draws that constitute one epoch. `None` means
                derive it from the component lengths and weights.

        Raises:
            ValueError: If `components` is given but its keys do not match the spec.
        """
        if components is not None:
            expected = {component.name for component in spec.components}
            got = set(components)
            if got != expected:
                raise ValueError(
                    f"mix {spec.name!r}: component datasets {sorted(got)} do not match the spec's "
                    f"components {sorted(expected)}"
                )
        self.spec = spec
        self.components = components
        self.transform = transform
        self.epoch_length = epoch_length

    def _missing_components(self) -> tuple[str, ...]:
        """Names of components with no built dataset behind them. Pure."""
        if self.components is None:
            return tuple(component.name for component in self.spec.components)
        return tuple(
            component.name
            for component in self.spec.components
            if component.name not in self.components
        )

    def __len__(self) -> int:
        """Number of draws in one epoch.

        Raises:
            NotImplementedError: Always, until the component loaders are implemented.
        """
        raise NotImplementedError(
            f"MixedDataset.__len__ is not implemented for mix {self.spec.name!r}. Missing: "
            f"loaders for component(s) {self._missing_components()!r} -- every dataset in "
            "satquery.data.datasets still raises NotImplementedError from __len__, so an epoch "
            "length cannot be derived."
        )

    def __getitem__(self, index: int) -> Sample:
        """Draw `index` from the weighted mix and return it as a unified `Sample`.

        Raises:
            NotImplementedError: Always, until the component loaders and the scale
                resampler are implemented.
        """
        raise NotImplementedError(
            f"MixedDataset.__getitem__ is not implemented for mix {self.spec.name!r}. Missing: "
            f"(1) loaders for component(s) {self._missing_components()!r}, whose __getitem__ "
            "still raises; (2) the scale-resampling transform over "
            f"CANONICAL_GSD_SCALES_M={self.spec.scale_policy.scales_m!r}, which must resample the "
            "raster and rewrite ImageRef.gsd_m so Sample.prompt() emits the resampled GSD token "
            "(satquery.preprocess is the intended home for the raster resample itself)."
        )


class ConcatSampleDataset(SampleDataset):
    """Flat concatenation of several `SampleDataset` views.

    Used when the full weighted mix cannot be honoured because some corpora are not on
    disk. Deliberately not weighted: pretending to apply mix weights while a component
    is missing would misrepresent the trained distribution. `build_datasets` logs which
    components were actually included.
    """

    def __init__(self, parts: list[SampleDataset]) -> None:
        if not parts:
            raise ValueError("ConcatSampleDataset needs at least one component")
        self.parts = parts
        self._offsets: list[int] = []
        total = 0
        for part in parts:
            self._offsets.append(total)
            total += len(part)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Sample:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        position = bisect.bisect_right(self._offsets, index) - 1
        return self.parts[position][index - self._offsets[position]]


class SubsetSampleDataset(SampleDataset):
    """A fixed index subset of another dataset. Used for small held-out eval slices."""

    def __init__(self, base: SampleDataset, indices: Iterable[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        return self.base[self.indices[index]]
