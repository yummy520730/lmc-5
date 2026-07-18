FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LMC5_DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install .

RUN useradd --create-home --uid 10001 lmc5 \
    && mkdir -p /data \
    && chown -R lmc5:lmc5 /app /data

USER lmc5

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD curl --fail "http://127.0.0.1:${PORT:-8080}/healthz" || exit 1

CMD ["python", "-m", "lmc5_web"]

