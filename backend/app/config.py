"""Settings, from the environment. No hardcoded paths.

Mirrors the ML package's discipline: `satquery.utils.paths` reads roots from the
environment, and this service must agree with it or the tools will look for models
somewhere the trainer never wrote them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Everything this service needs to start."""

    upload_root: Path
    """Where uploaded rasters land. The BlobStore's backing directory."""

    job_ttl_seconds: float = 3600.0
    """How long a finished job stays readable before eviction."""

    max_upload_bytes: int = 512 * 1024 * 1024
    """Reject larger uploads rather than filling the disk mid-demo."""

    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    """Vite's dev server. The frontend is served separately."""

    demo_dir: Path | None = None
    """Directory of bundled demo rasters, or None when none is installed.

    Optional on purpose: the service must start and work with no demo scene present, so
    a missing directory is a 404 on one endpoint rather than a failure at boot."""

    @classmethod
    def from_env(cls) -> Settings:
        root = os.environ.get("SATQUERY_UPLOAD_ROOT")
        upload_root = Path(root).expanduser() if root else Path.home() / "satquery-uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        demo = os.environ.get("SATQUERY_DEMO_DIR")
        return cls(
            upload_root=upload_root,
            demo_dir=Path(demo).expanduser() if demo else None,
        )


SETTINGS = Settings.from_env()
