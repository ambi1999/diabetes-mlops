# =============================================================================
# Dockerfile for the Streamlit web app (used by docker compose)
# =============================================================================
# The UI is a separate service from the API: it only shows a form and calls
# the API over HTTP. Keeping them separate means each can be deployed, scaled
# and updated independently.
# =============================================================================

FROM python:3.13-slim

# Same pinned uv version as the API image and CI
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# requirements.txt is EXPORTED from uv.lock (see README), so the UI gets the
# exact same locked versions as everywhere else.
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY app.py .

# Don't run as root (security best practice)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
# --server.address=0.0.0.0 makes Streamlit reachable from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
