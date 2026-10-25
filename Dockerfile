# One image for every service (worker, UI, publisher, sandbox server); each service picks its command.
# python:3.11-slim pinned by digest (pulled 2026-10-09) for reproducible builds.
FROM python:3.11-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
# --no-deps: requirements.txt is the full pinned closure, so any missing pin fails the build loudly.
RUN pip install --no-deps -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
# Installed (not PYTHONPATH): MCP stdio subprocesses get a filtered environment without PYTHONPATH.
RUN pip install --no-deps .
COPY ui ./ui
COPY migrations ./migrations
COPY alembic.ini ./
COPY benchmark/cases ./benchmark/cases

RUN useradd --create-home --uid 10001 app && mkdir -p /app/.workspaces && chown app /app/.workspaces
USER app
