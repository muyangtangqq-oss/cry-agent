FROM python:3.12-slim

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 libsndfile1-dev \
    libportaudio2 portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev --no-install-project

# Copy source
COPY . .

# Model files (mounted or copied)
VOLUME ["/root/.config/cry-agent/models"]
VOLUME ["/root/.config/cry-agent/data"]

# Expose API
EXPOSE 8000

# Start Tachyone server + API server
CMD .venv/bin/python tachyone_server.py 8901 & \
    .venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1
