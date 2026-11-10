# =============================================================================
# Dockerfile for the prediction API
# =============================================================================
# A Dockerfile is a recipe for building an IMAGE: a self-contained package of
# the operating system, Python, libraries and our code. The same image runs on
# your laptop (docker compose) and in the cloud (Render), which removes
# "works on my machine" problems.
#
# Build and run locally:
#   docker build -t diabetes-api .
#   docker run -p 8000:8000 diabetes-api
# =============================================================================

# Start from an official, minimal Python image. The version matches
# .python-version and requires-python in pyproject.toml (environment parity).
FROM python:3.13-slim

# Copy the uv executable from its official image. Pinned to an exact version
# (not "latest") so builds are reproducible; keep it in sync with ci.yml.
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# Settings for uv and Python inside the container:
#   UV_COMPILE_BYTECODE  pre-compile .py files -> faster container start-up
#   UV_LINK_MODE=copy    copy files instead of linking (avoids warnings in Docker)
#   UV_PYTHON_DOWNLOADS  use the image's Python; never download another one
#   PYTHONUNBUFFERED     print logs immediately (so they show up in Render's logs)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Install dependencies FIRST, before copying our code ---------------------
# Docker caches each step ("layer"). Dependencies change rarely and code changes
# often, so installing dependencies first means a code change doesn't trigger a
# slow reinstall of every package.
COPY pyproject.toml uv.lock ./
# --locked             fail if uv.lock is out of date with pyproject.toml
# --no-default-groups  install ONLY the runtime dependencies (no MLflow, Prefect,
#                      pytest, Streamlit...), keeping the image small and secure
RUN uv sync --locked --no-default-groups

# --- Then copy only what the API needs ----------------------------------------
COPY src/ src/
COPY api/ api/
COPY models/ models/

# Put the project's virtual environment first on the PATH, so `uvicorn` and
# `python` refer to the versions installed from uv.lock.
ENV PATH="/app/.venv/bin:$PATH"

# Security best practice: don't run as the all-powerful root user.
# The app only needs to write prediction logs, so it gets just that folder.
RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser /app/logs
USER appuser

# Render (and most hosts) tell the app which port to use via $PORT.
# Locally it defaults to 8000.
ENV PORT=8000
EXPOSE 8000

# `exec` makes uvicorn the main process, so it receives stop signals from
# Docker/Render and shuts down cleanly.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
