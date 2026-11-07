# Base image: official Python 3.12, "slim" variant (Debian without extras), same version as the venv
FROM python:3.12-slim

# Don't write .pyc files inside the container (they're useless in an image)
ENV PYTHONDONTWRITEBYTECODE=1
# Print logs immediately instead of buffering them, so `docker logs` shows lines in real time
ENV PYTHONUNBUFFERED=1
# Don't keep pip's download cache in the image (saves space)
ENV PIP_NO_CACHE_DIR=1
# Skip the "new pip available" notice during builds
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
# Give pip 120 s per network request instead of 15 s, for slow connections
ENV PIP_DEFAULT_TIMEOUT=120
# Retry a failed download up to 10 times instead of 5 before giving up
ENV PIP_RETRIES=10
# Default port; deploy platforms can override it with their own PORT value
ENV PORT=8000

# All following paths are inside /app; this becomes the project root (ROOT in predictor.py and store.py)
WORKDIR /app

# Install OpenMP (libgomp1), which LightGBM needs at runtime, then delete apt's package lists to keep the layer small
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy ONLY the requirements first, so the pip layer below stays cached until this file changes
COPY requirements-api.txt .
# Install the pinned runtime packages (the slow step; cached on later builds)
RUN pip install -r requirements-api.txt

# Create an unprivileged user (fixed ID 10001) to run the API instead of root
RUN useradd --create-home --uid 10001 appuser

# Copy the API code into /app/src/api (same layout as the project, so the relative paths still work)
COPY src/api ./src/api
# Copy the trained model, metadata and risk cutoffs into /app/models
COPY models ./models
# Copy the serving features and grid into /app/data/serving
COPY data/serving ./data/serving

# From here on, everything runs as appuser, not root
USER appuser

# Document that the app listens on port 8000 (the real mapping happens with -p at docker run)
EXPOSE 8000

# Every 30 s, call /health; wait 20 s after start for the model to load; 3 failures in a row = "unhealthy"
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health', timeout=4)"

# Start uvicorn on all network interfaces at $PORT; exec makes it the main process; no duplicate access log
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --no-access-log"]