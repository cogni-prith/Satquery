"""Upload endpoint: raster in, parsed `ImageRef` metadata out.

Parsing happens at upload rather than at query time so ingest warnings -- no CRS, an
unrecognised band, a missing timestamp -- reach the user while they are still looking at
the file they chose, instead of surfacing three clicks later attached to an answer.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile

from app.core.schemas import UploadedImage
from app.state import STATE

router = APIRouter(prefix="/api/images", tags=["images"])


@router.post("", response_model=UploadedImage)
async def upload(file: UploadFile) -> UploadedImage:
    """Store one raster and return what the ingest layer made of it."""
    from satquery.io.raster import read_image_ref

    suffix = "." + file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else ".tif"
    blob_id, path = STATE.blobs.put(file.file, suffix)

    try:
        ref = read_image_ref(path)
    except Exception as exc:  # noqa: BLE001 - a bad upload is a client error, not a 500
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"could not read {file.filename!r} as a raster: {type(exc).__name__}: {exc}",
        ) from exc

    return UploadedImage(
        image_id=blob_id,
        filename=file.filename or path.name,
        modality=ref.modality.value,
        gsd_m=ref.gsd_m,
        gsd_token=ref.gsd_token,
        width=ref.width,
        height=ref.height,
        band_names=ref.band_names,
        crs=ref.crs,
        warnings=ref.warnings,
    )
