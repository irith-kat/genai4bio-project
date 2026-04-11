# Use the official Debian-hosted Python image
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1


# Install OS dependencies + headers needed for pybigwig, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        ca-certificates \
        build-essential \
        zlib1g-dev \
        libcurl4-openssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install uv
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

# Create non-root user
RUN useradd -m -s /bin/bash app
WORKDIR /app

# Copy project files FIRST
COPY . .

# Ensure folder belongs to app user
RUN chown -R app:app /app

# Switch to app user BEFORE uv sync (critical)
USER app

# Install Python dependencies into .venv as app user (including dev group)
RUN uv sync --group dev

# Default command: start Jupyter Lab
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]


