"""The only process that touches the GPU.

8188 MiB of VRAM and EarthDial-4B needs about 4 GB of it loaded 4-bit. That fits once, so
exactly one component may hold models and it must load them at startup and never reload.
Everything else in this service is arranged around that fact.

Execution is serialised deliberately. Two concurrent inference calls on an 8 GB card is how
a live demo turns into an OOM in front of judges; a queue is how it becomes a short wait.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from satquery.serve.contracts import ImageRef, TaskType, ToolRequest, ToolResult
from satquery.utils.logging import get_logger

_LOG = get_logger("worker")


class GpuRuntime:
    """Loads the models once, then runs one request at a time.

    `load()` is slow (about 45 seconds, dominated by EarthDial) and is called from the app's
    lifespan on a background thread, so the API answers `/api/health` immediately and the
    frontend can show a loading state rather than appearing broken.
    """

    def __init__(self, models_root: Path | None = None) -> None:
        self.models_root = models_root
        self._lock = threading.Lock()
        self._loaded = False
        self._loading = False
        self._error: str | None = None
        self._backbone: Any = None

    # -- state ------------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def loading(self) -> bool:
        return self._loading

    @property
    def error(self) -> str | None:
        return self._error

    def vram_used_mb(self) -> float | None:
        """Current allocation, or None when torch is absent or has no CUDA device."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            return torch.cuda.memory_allocated() / 1024**2
        except Exception:
            return None

    # -- lifecycle --------------------------------------------------------------------

    def load(self) -> None:
        """Load every model. Idempotent; safe to call once per process.

        Failures are recorded rather than raised: the API must keep answering /api/health
        so the UI can say what went wrong, instead of the whole service dying at boot.
        """
        if self._loaded or self._loading:
            return
        self._loading = True
        try:
            from satquery.models.registry import REGISTRY
            from satquery.models.vlm.backbone import BackboneConfig, EarthDialBackbone
            from satquery.models.vlm.tasks import CaptionTool, GroundingTool, VqaTool

            from satquery.utils.paths import configs_dir

            # Resolve through the ML package, not this service's working directory: the
            # config belongs to satquery and moves with it. A relative path here would
            # break the moment the backend is started from anywhere else.
            _LOG.info("loading EarthDial-4B; this takes about 45 seconds")
            config = BackboneConfig.from_yaml(configs_dir() / "model" / "earthdial_4b_rgb.yaml")
            backbone = EarthDialBackbone(config)
            backbone.load()
            self._backbone = backbone

            # One backbone instance shared by all three VLM tools. Binding a factory per
            # tool would load EarthDial three times and exhaust the card immediately.
            REGISTRY.bind("vlm.vqa", lambda: VqaTool(backbone=backbone))
            REGISTRY.bind("vlm.caption", lambda: CaptionTool(backbone=backbone))
            REGISTRY.bind("vlm.grounding", lambda: GroundingTool(backbone=backbone))

            self._loaded = True
            _LOG.info("models resident, %.0f MB VRAM", self.vram_used_mb() or 0.0)
        except Exception as exc:  # noqa: BLE001 - recorded and surfaced through /api/health
            self._error = f"{type(exc).__name__}: {exc}"
            _LOG.exception("model load failed")
        finally:
            self._loading = False

    # -- inference --------------------------------------------------------------------

    def run(self, tool_name: str, request: ToolRequest) -> ToolResult:
        """Run one tool, serialised against every other call.

        A tool that raises still returns a well-formed `ToolResult` with `error` set --
        `BaseTool.run` guarantees that, and it is what keeps the trace well formed on a
        failure path. This method does not add its own error handling on top.
        """
        from satquery.models.registry import REGISTRY

        with self._lock:
            tool = REGISTRY.get(tool_name)
            return tool.run(request)


def build_image_refs(paths: list[Path]) -> list[ImageRef]:
    """Parse rasters into `ImageRef`s, so GSD comes from the transform, never a guess."""
    from satquery.io.raster import read_image_ref

    return [read_image_ref(path) for path in paths]


def build_request(query: str, images: list[ImageRef], task: TaskType | None) -> ToolRequest:
    """Assemble the request the tools accept. Nothing here is backend-specific."""
    return ToolRequest(query=query, images=images, task=task)
