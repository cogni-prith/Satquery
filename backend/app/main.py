"""SatQuery backend.

FastAPI holds no models. One worker thread owns the GPU and runs inference serialised,
because EarthDial-4B needs about 4 GB of an 8 GB card and two concurrent calls is how a
live demo becomes an OOM.

Models load in the background at startup so the API answers `/api/health` immediately and
the frontend can show a loading state rather than appearing broken for 45 seconds.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import images, meta, query
from app.config import SETTINGS
from app.state import STATE


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.dispatcher.start()
    # Loading is slow and blocking; a thread keeps the event loop answering health checks.
    asyncio.create_task(asyncio.to_thread(STATE.runtime.load))
    yield
    await STATE.dispatcher.stop()


app = FastAPI(
    title="SatQuery AI",
    version="0.1.0",
    description="Vision-language analysis of remote sensing imagery, queried in natural language.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(images.router)
app.include_router(query.router)
app.include_router(meta.router)
