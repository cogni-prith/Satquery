"""Bind implemented tools to their specs. Imported for its side effects.

`ToolRegistry` loads this module the first time a tool instance is requested, so a spec
without a binding here stays a spec: asking for it raises `NotImplementedError` naming
what is missing rather than returning something that answers.

Two tools are bound: the spectral index path and the radiometric fallback. Every learned tool needs weights that do not
exist yet, and binding a stub would make the router offer a tool that cannot answer.
"""

from __future__ import annotations

from satquery.models.indices.deterministic import DeterministicIndexTool
from satquery.models.indices.radiometric import RadiometricChangeTool
from satquery.models.registry import REGISTRY

REGISTRY.bind(
    "indices.deterministic",
    lambda: DeterministicIndexTool(REGISTRY.get_spec("indices.deterministic")),
)
REGISTRY.bind(
    "change.radiometric",
    lambda: RadiometricChangeTool(REGISTRY.get_spec("change.radiometric")),
)
