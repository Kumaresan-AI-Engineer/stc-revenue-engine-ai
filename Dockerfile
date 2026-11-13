FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
# Bake the free embedding model into the image so the first request does not download it.
ENV RE_AI_EMBEDDING_CACHE_DIR=/app/models
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models')"

COPY alembic.ini ./
COPY migrations ./migrations

RUN useradd --create-home app && chown -R app /app/models
USER app
WORKDIR /app/src
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
