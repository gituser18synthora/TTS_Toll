# Orpheus TTS dataset portal: API + worker image (same image, different command).
#   CPU:  docker compose up -d --build
#   GPU:  TORCH_VARIANT=cu128 docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# ---- 1. build the web UI -----------------------------------------------------------
FROM node:22-bookworm-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

# ---- 2. runtime --------------------------------------------------------------------
FROM python:3.11-slim-bookworm
# cpu | cu126 | cu128   (CUDA 12 builds: ctranslate2 / faster-whisper need CUDA 12 libraries)
ARG TORCH_VARIANT=cpu
# Set to true only if the build fails with "Release file ... is not valid yet": the machine's
# clock is behind (common on WSL2 after sleep; fix with `sudo hwclock -s`).
ARG APT_CLOCK_SKEW_OK=false
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    PORTAL_DATA_ROOT=/data \
    PORTAL_MODELS_DIR=/models
RUN if [ "$APT_CLOCK_SKEW_OK" = "true" ]; then \
        echo 'Acquire::Check-Valid-Until "false"; Acquire::Check-Date "false";' > /etc/apt/apt.conf.d/99clock-skew; fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/99clock-skew
WORKDIR /app
COPY requirements.txt requirements-torch.txt requirements-nodeps.txt constraints.txt ./
RUN pip install -r requirements-torch.txt --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
 && pip install -r requirements.txt -c constraints.txt \
 && pip install --no-deps -r requirements-nodeps.txt
# no usage telemetry from pyannote.audio / huggingface_hub (the app sets the same defaults)
ENV PYANNOTE_METRICS_ENABLED=false \
    HF_HUB_DISABLE_TELEMETRY=1
COPY config.yaml ./
COPY docker/entrypoint.sh /usr/local/bin/portal-entrypoint
COPY backend/ backend/
COPY --from=ui /ui/dist frontend/dist
WORKDIR /app/backend
EXPOSE 8000
ENTRYPOINT ["portal-entrypoint"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
