# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/newslens/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 newslens && useradd --uid 10001 --gid 10001 --create-home newslens \
    && mkdir -p /models && chown newslens:newslens /models
COPY infra/docker/constraints.txt /tmp/constraints.txt
# Use CPU wheels explicitly; do not pull CUDA packages into this local CPU image.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
COPY pyproject.toml README.md alembic.ini ./
COPY backend ./backend
COPY ingestion ./ingestion
COPY nlp ./nlp
COPY infra/docker ./infra/docker
RUN python -m pip install --constraint /tmp/constraints.txt . && python -m pip check
USER newslens
EXPOSE 8000
ENTRYPOINT ["python", "infra/docker/entrypoint.py"]
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--no-proxy-headers"]
