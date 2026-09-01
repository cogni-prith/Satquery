"""Single source of truth for every filesystem location the package touches.

No module may hardcode an absolute path. Roots are read from the environment
(`SATQUERY_DATA_ROOT`, `SATQUERY_ARTIFACT_ROOT`) and fall back to repo-relative
directories so that CPU-only smoke tests run on a fresh clone with no setup.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "artifact_dir",
    "artifact_root",
    "configs_dir",
    "data_root",
    "dataset_dir",
    "repo_root",
]

_DATA_ROOT_ENV = "SATQUERY_DATA_ROOT"
_ARTIFACT_ROOT_ENV = "SATQUERY_ARTIFACT_ROOT"


def repo_root() -> Path:
    """Return the repository root, resolved from this file's location.

    `src/satquery/utils/paths.py` -> parents[3] is the repo root.
    """
    return Path(__file__).resolve().parents[3]


def _root_from_env(env_var: str, fallback_name: str) -> Path:
    """Resolve a root directory from `env_var`, falling back to `repo_root()/fallback_name`."""
    raw = os.environ.get(env_var)
    if raw:
        return Path(raw).expanduser().resolve()
    return repo_root() / fallback_name


def data_root() -> Path:
    """Root of the read-only dataset tree (`$SATQUERY_DATA_ROOT`, else `<repo>/data`)."""
    return _root_from_env(_DATA_ROOT_ENV, "data")


def artifact_root() -> Path:
    """Root for checkpoints, caches and eval output (`$SATQUERY_ARTIFACT_ROOT`, else `<repo>/artifacts`)."""
    return _root_from_env(_ARTIFACT_ROOT_ENV, "artifacts")


def configs_dir() -> Path:
    """Directory holding the YAML configuration tree."""
    return repo_root() / "configs"


def dataset_dir(name: str) -> Path:
    """Return the directory for dataset `name` under the data root.

    Does not create the directory and does not check that it exists; datasets are
    never downloaded automatically, so a missing directory is the caller's problem
    to report.
    """
    return data_root() / name


def artifact_dir(*parts: str, create: bool = True) -> Path:
    """Return `<artifact_root>/<parts...>`, creating it by default."""
    path = artifact_root().joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
