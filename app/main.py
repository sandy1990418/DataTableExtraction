from __future__ import annotations

from fastapi import FastAPI

from app.routes.data_table import router as data_table_router
from app.routes.download import router as download_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Data Table Extraction API",
        description="Generate a grounded data table and export it as PPTX.",
        version="0.2.0",
    )

    app.include_router(data_table_router)
    app.include_router(download_router)

    return app


app = create_app()
