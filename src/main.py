"""App factory for the Revenue Engine AI service."""

from fastapi import FastAPI

from api.router import router
from core.errors import install_error_handlers
from core.observability import configure_logging, install_request_id


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="Revenue Engine AI service",
        version="0.1.0",
        description="Extraction, semantic search, drafting, client queries and review (proposal builder).",
    )
    install_error_handlers(app)
    install_request_id(app)
    app.include_router(router)

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    return app


app = create_app()
