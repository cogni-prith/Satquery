"""Upload endpoint: raster in, parsed `ImageRef` metadata out.

Parsing happens at upload rather than at query time so ingest warnings -- no CRS, an
unrecognised band, a missing timestamp -- reach the user while they are still looking at
the file they chose, instead of surfacing three clicks later attached to an answer.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import Response

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


@router.get("/{image_id}/preview")
async def preview(image_id: str) -> Response:
    """Return the image as a PNG, rendered exactly as the model sees it.

    Deliberately routed through `load_model_input` -- the same function the tools call --
    rather than a separate thumbnailer. A SAR raster therefore previews as the frozen
    pseudo-RGB (refined Lee, dB, percentile stretch, VV/VH/ratio), and a multispectral
    stack previews as its stretched RGB bands. A prettier preview produced some other way
    would be showing the user something the model never saw, which is exactly the kind of
    quiet divergence between display and reality this project avoids elsewhere.
    """
    import io

    from PIL import Image

    from satquery.io.raster import read_image_ref
    from satquery.models.base import load_model_input

    try:
        path = STATE.blobs.path_for(image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        rgb, _ = load_model_input(read_image_ref(path))
    except Exception as exc:  # noqa: BLE001 - a preview failure must not be a 500
        raise HTTPException(status_code=422, detail=f"cannot render a preview: {exc}") from exc

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
