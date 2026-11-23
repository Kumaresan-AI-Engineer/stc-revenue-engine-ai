# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RE_AI_EMBEDDING_CACHE_DIR=/app/models \
    RE_AI_QDRANT_PATH=/app/data/qdrant
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
# The embedding model is baked into the image, so a container starts without downloading it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models')"

COPY alembic.ini ./
COPY migrations ./migrations

RUN useradd --create-home app && mkdir -p /app/data/qdrant && chown -R app /app/models /app/data
USER app
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/healthz', timeout=4)"
CMD ["uvicorn", "revenue_engine_ai.main:app", "--host", "0.0.0.0", "--port", "8001"]
