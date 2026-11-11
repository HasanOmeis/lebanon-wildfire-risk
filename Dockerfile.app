# Base image: official Python 3.12, "slim" variant, same version as the venv and the API image
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
# Retry a failed download up to 10 times before giving up
ENV PIP_RETRIES=10
# Default port; deploy platforms can override it with their own PORT value
ENV PORT=8501
# Where the API lives; inside Docker Compose the API service is reachable by its name "api"
ENV WILDFIRE_API_URL=http://api:8000

# All following paths are inside /app; Streamlit finds .streamlit/config.toml here
WORKDIR /app

# Copy ONLY the requirements first, so the pip layer below stays cached until this file changes
COPY requirements-app.txt .
# Install the pinned runtime packages (the slow step; cached on later builds)
RUN pip install -r requirements-app.txt

# Create an unprivileged user (same fixed ID as the API image) to run the app instead of root
RUN useradd --create-home --uid 10001 appuser

# Copy the app code into /app/app (same layout as the project, so "from app.api_client import ..." works)
COPY app ./app
# Copy the theme and settings (colors, minimal toolbar, no usage stats)
COPY .streamlit/config.toml ./.streamlit/config.toml

# From here on, everything runs as appuser, not root
USER appuser

# Document that the app listens on port 8501 (the real mapping happens in docker-compose.yml)
EXPOSE 8501

# Every 30 s, call Streamlit's own health endpoint; allow 20 s for startup; 3 failures in a row = "unhealthy"
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:' + os.environ.get('PORT', '8501') + '/_stcore/health', timeout=4)"

# Start Streamlit on all network interfaces at $PORT; headless = no browser; no file watching inside the image
CMD ["sh", "-c", "exec streamlit run app/streamlit_app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true --server.fileWatcherType none --server.runOnSave false"]