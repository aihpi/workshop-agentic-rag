# Multi-target Dockerfile
# Build with:  docker build --target app -t workshop-app .
#              docker build --target ingestion -t workshop-ingestion .

# ---- shared base ----
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip

# ---- Chat app ----
FROM base AS app

WORKDIR /app
COPY app/ /app/
RUN pip install -e .
EXPOSE 8000
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]

# ---- Ingestion app ----
FROM base AS ingestion

WORKDIR /app
COPY ingestion/ /app/
RUN pip install -e .
EXPOSE 8001
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8001"]
