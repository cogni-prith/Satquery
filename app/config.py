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

    @classmethod
    def from_env(cls) -> Settings:
        root = os.environ.get("SATQUERY_UPLOAD_ROOT")
        upload_root = Path(root).expanduser() if root else Path.home() / "satquery-uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        return cls(upload_root=upload_root)


SETTINGS = Settings.from_env()
