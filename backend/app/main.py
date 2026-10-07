"""FastAPI application: JSON API under /api, built frontend (frontend/dist) at /."""
from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import OperationalError

from . import __version__
from .api import clips, exports, jobs, maintenance, sources, speakers, system, uploads
from .config import PROJECT_ROOT, configure_model_cache, get_settings
from .db import create_schema, init_engine
from .logging_setup import setup_logging
from .paths import DataPaths

log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    paths = DataPaths.from_settings(settings)
    paths.ensure()
    setup_logging("api", paths.logs)
    configure_model_cache(settings)
    init_engine(paths.db_file)
    create_schema()
    log.info("DATA_ROOT=%s  device=%s", paths.root, settings.device)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker = None
        if settings.worker.embedded:
            from .worker.runner import start_embedded

            worker = start_embedded(settings)
            log.info("embedded worker started")
        yield
        if worker is not None:
            worker.stop_event.set()

    app = FastAPI(title="Orpheus TTS Dataset Portal", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.server.cors_origins, allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(OperationalError)
    async def _db_busy(_request: Request, exc: OperationalError):
        if isinstance(exc.orig, sqlite3.OperationalError) and "locked" in str(exc.orig):
            return JSONResponse({"detail": "database busy, please retry"}, status_code=503)
        raise exc

    for module in (system, speakers, uploads, sources, clips, jobs, exports, maintenance):
        app.include_router(module.router, prefix="/api")

    dist = PROJECT_ROOT / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
