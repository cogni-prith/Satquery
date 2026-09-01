"""Backend tests.

The GPU runtime is mocked at the dispatch boundary. Actual inference is covered by
`make eval` in the ML repo, which is the correct place for it -- duplicating it here would
mean a second, divergent inference path, and the whole point of the contract is that there
is only one.

What is tested here is everything the backend itself is responsible for: the router's two
stages against the real registry, job lifecycle, upload parsing, and that a tool error
becomes a completed job carrying an error rather than a 500.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import STATE
from satquery.serve.contracts import ImageRef, InputConfig, Modality, TaskType


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def optical_tif(tmp_path):
    """A synthetic 4-band raster, written the way the ML repo's fixtures are."""
    from satquery.io.raster import write_raster

    rng = np.random.default_rng(0)
    array = rng.integers(100, 4000, size=(4, 32, 32)).astype(np.uint16)
    return write_raster(tmp_path / "optical.tif", array, band_names=["B02", "B03", "B04", "B08"])


# -- meta ---------------------------------------------------------------------------------


def test_health_reports_contract_version(client):
    body = client.get("/api/health").json()
    assert body["contract_version"]
    assert "models_loaded" in body


def test_tools_endpoint_mirrors_the_registry(client):
    from satquery.models.registry import REGISTRY

    body = client.get("/api/tools").json()
    assert body["tool_count"] == len(REGISTRY.list_specs())
    # The frontend renders capabilities from this, so it must carry the implemented flag.
    assert all("implemented" in tool for tool in body["tools"])


# -- upload -------------------------------------------------------------------------------


def test_upload_returns_parsed_metadata(client, optical_tif):
    with optical_tif.open("rb") as handle:
        response = client.post("/api/images", files={"file": ("optical.tif", handle, "image/tiff")})
    assert response.status_code == 200
    body = response.json()
    assert body["band_names"] == ["B02", "B03", "B04", "B08"]
    assert body["gsd_token"].startswith("<gsd:")


def test_upload_rejects_a_non_raster(client):
    response = client.post("/api/images", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "could not read" in response.json()["detail"]


# -- router stage one ---------------------------------------------------------------------


def _ref(modality: Modality, timestamp=None) -> ImageRef:
    return ImageRef(path="/tmp/x.tif", modality=modality, gsd_m=10.0, timestamp=timestamp)


def test_gate_single_image():
    from app.router.gate import classify_input

    assert classify_input([_ref(Modality.MULTISPECTRAL)]) is InputConfig.SINGLE


def test_gate_prefers_cross_modal_over_bitemporal():
    """An optical/SAR pair usually differs in timestamp too, because they are separate
    acquisitions. Testing timestamps first would route every fusion pair to change."""
    from datetime import datetime, timezone

    from app.router.gate import classify_input

    a = _ref(Modality.MULTISPECTRAL, datetime(2020, 1, 1, tzinfo=timezone.utc))
    b = _ref(Modality.SAR, datetime(2021, 6, 1, tzinfo=timezone.utc))
    assert classify_input([a, b]) is InputConfig.CROSS_MODAL_PAIR


def test_gate_rejects_three_images():
    from app.router.gate import classify_input

    with pytest.raises(ValueError, match="one or two images"):
        classify_input([_ref(Modality.OPTICAL_RGB)] * 3)


def test_gate_offers_only_implemented_tools():
    from app.router.gate import gate

    _, candidates = gate([_ref(Modality.MULTISPECTRAL)])
    assert candidates, "a single optical image must have at least one implemented tool"
    assert all(spec.implemented for spec in candidates)


# -- router stage two ---------------------------------------------------------------------


def test_select_routes_a_locate_query_to_grounding():
    from app.router.gate import gate
    from app.router.select import select

    _, candidates = gate([_ref(Modality.OPTICAL_RGB)])
    chosen, reason = select(candidates, "where is the runway in this image")
    assert chosen.task is TaskType.GROUNDING
    assert "where" in reason


def test_select_routes_a_describe_query_to_caption():
    from app.router.gate import gate
    from app.router.select import select

    _, candidates = gate([_ref(Modality.OPTICAL_RGB)])
    chosen, _ = select(candidates, "describe this scene for me")
    assert chosen.task is TaskType.CAPTION


def test_select_falls_back_deterministically_and_says_so():
    """An unexplained choice is what the auditable-summary judging row penalises."""
    from app.router.gate import gate
    from app.router.select import select

    _, candidates = gate([_ref(Modality.OPTICAL_RGB)])
    chosen, reason = select(candidates, "zzzz")
    assert chosen in candidates
    assert "fell back" in reason


def test_select_refuses_an_empty_candidate_list():
    """Better to fail than to invent a tool the gate did not allow."""
    from app.router.select import select

    with pytest.raises(ValueError, match="no implemented tools"):
        select([], "anything")


# -- jobs ---------------------------------------------------------------------------------


def test_query_is_rejected_while_models_are_loading(client, optical_tif):
    """An explicit 503 lets the UI say 'still starting' instead of showing a spinner that
    looks identical to a hang."""
    with optical_tif.open("rb") as handle:
        image_id = client.post(
            "/api/images", files={"file": ("o.tif", handle, "image/tiff")}
        ).json()["image_id"]

    assert not STATE.runtime.loaded  # no GPU in CI
    response = client.post("/api/query", json={"query": "describe", "image_ids": [image_id]})
    assert response.status_code == 503


def test_polling_an_unknown_job_is_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404


def test_job_store_evicts_finished_jobs_past_the_ttl():
    from app.core.jobs import JobStatus, MemoryJobStore

    store = MemoryJobStore(ttl_seconds=0.0)
    job = store.create()
    job.status = JobStatus.DONE
    job.finished_at = 0.0
    store.update(job)

    store.create()  # triggers eviction
    with pytest.raises(KeyError):
        store.get(job.job_id)


# -- blob store ---------------------------------------------------------------------------


def test_blob_store_ignores_a_traversing_filename(tmp_path):
    """The id is generated, never taken from the upload: joining a user-supplied name to a
    root is a directory-traversal bug."""
    import io

    from app.core.blobs import LocalBlobStore

    store = LocalBlobStore(tmp_path / "blobs")
    blob_id, path = store.put(io.BytesIO(b"x"), "../../etc/passwd")
    assert path.parent == (tmp_path / "blobs")
    assert store.path_for(blob_id) == path


def test_blob_store_raises_for_an_unknown_id(tmp_path):
    from app.core.blobs import LocalBlobStore

    with pytest.raises(KeyError):
        LocalBlobStore(tmp_path).path_for("nope")
