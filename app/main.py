from __future__ import annotations

from fastapi import FastAPI

from app.routes.analyze import router as analyze_router
from app.routes.chat import router as chat_router
from app.routes.download import router as download_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Table PPTX API",
        description="Send text → LLM decides → generate table as PPTX download.",
        version="0.1.0",
    )

    app.include_router(chat_router)
    app.include_router(download_router)
    app.include_router(analyze_router)

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
