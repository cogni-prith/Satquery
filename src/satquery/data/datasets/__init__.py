"""Dataset loaders, one module per corpus, all emitting the same record.

Every loader in this package returns `satquery.data.schema.Sample` and nothing else.
That is the whole point of the package: the mixer, the collator, the model and the
eval harness never learn which corpus a record came from, so a dataset can be added,
reweighted or dropped without touching anything downstream.

`SampleDataset` is the shared base. It is deliberately a tiny local ABC rather than
`torch.utils.data.Dataset`, because torch is an optional `gpu` extra and this package
must import on a CPU-only laptop with no torch installed. It is duck-compatible with
`torch.utils.data.Dataset`: a map-style dataset needs only `__len__` and
`__getitem__`, so any subclass can be handed straight to `torch.utils.data.DataLoader`
(alongside `satquery.data.collate.collate_samples`) without an adapter or a
`register` call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

from satquery.data.schema import Sample

__all__ = ["SampleDataset", "SampleTransform"]

#: A sample-to-sample rewrite applied at the end of `__getitem__`.
#:
#: Transforms operate on the typed record, not on raw pixels, so a scale-resampling
#: transform can rewrite `ImageRef.gsd_m` and the GSD token follows automatically via
#: `Sample.prompt()`. That coupling is what keeps the training prompt and the
#: inference prompt identical after augmentation.
SampleTransform = Callable[[Sample], Sample]


class SampleDataset(ABC):
    """Map-style dataset of unified `Sample` records.

    Single responsibility: expose an indexable, sized sequence of `Sample`. Subclasses
    own the on-disk parsing and the mapping into the unified schema; they own nothing
    else -- no batching, no tensor conversion, no tokenisation.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of records in this dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> Sample:
        """Return the record at `index`, already normalised to the unified schema."""

    def __iter__(self) -> Iterator[Sample]:
        """Iterate records in index order. Convenience for eval loops and smoke tests."""
        for index in range(len(self)):
            yield self[index]
