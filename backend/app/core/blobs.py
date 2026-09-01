"""Where uploaded rasters live.

A seam. The demo writes to a local directory; hosting later swaps in object storage
without touching callers.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Protocol


class BlobStore(Protocol):
    """Stores an uploaded file and hands back a stable id and a readable path."""

    def put(self, stream: BinaryIO, suffix: str) -> tuple[str, Path]:
        """Store `stream`, returning `(blob_id, path)`."""
        ...

    def path_for(self, blob_id: str) -> Path:
        """Return the path for `blob_id`.

        Raises:
            KeyError: No such blob.
        """
        ...


class LocalBlobStore:
    """Files under one directory, named by a generated id.

    The id is generated rather than taken from the upload's filename: a user-supplied
    name can contain path separators, and joining it to a root is a directory-traversal
    bug. The original name is not needed for anything downstream.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, stream: BinaryIO, suffix: str) -> tuple[str, Path]:
        blob_id = uuid.uuid4().hex
        # Keep the extension only: rasterio dispatches on it, and it is the one part of
        # a user-supplied filename that is safe to carry forward.
        safe_suffix = "".join(c for c in suffix if c.isalnum() or c == ".")[:16]
        path = self.root / f"{blob_id}{safe_suffix}"
        with path.open("wb") as handle:
            shutil.copyfileobj(stream, handle)
        return blob_id, path

    def path_for(self, blob_id: str) -> Path:
        matches = list(self.root.glob(f"{blob_id}*"))
        if not matches:
            raise KeyError(f"no uploaded image with id {blob_id!r}")
        return matches[0]
