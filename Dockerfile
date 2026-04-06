FROM python:3.11.9-slim-bookworm

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    # playwright needs these browser dependencies
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install playwright browser binaries (separate step from pip install)
RUN playwright install chromium

# Copy source — overridden by volume mount in dev, used as-is in production
COPY src/ ./src/

WORKDIR /app/src