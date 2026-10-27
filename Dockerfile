FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (excluding data and warehouse — mount as volumes)
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY dbt_m5/ ./dbt_m5/
COPY sql/ ./sql/
COPY reports/ ./reports/
COPY Makefile .

# Create necessary dirs
RUN mkdir -p data/raw data/features data/forecasts warehouse models

# Expose ports
EXPOSE 8000 8501

# Default: run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
