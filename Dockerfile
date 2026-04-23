# TORCH_FLAVOR controls whether torch pulls CPU-only wheels or the default
# (CUDA) wheels. Defaults to cpu → main workshop-app image is small (~2 GB).
# CI builds a second image with --build-arg TORCH_FLAVOR=cuda for the
# docling-service Deployment so it actually runs on GPU.
ARG TORCH_FLAVOR=cpu

FROM python:3.12-slim

ARG TORCH_FLAVOR
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Docling -> docling_ibm_models -> cv2 (OpenCV) needs these at runtime.
# Without them any PDF upload fails with ImportError: libxcb.so.1.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
        libxext6 \
        libsm6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY app/ /app/
COPY system.md /app/system.md
COPY data/data_docling_json_ocr/ /data/data_docling_json_ocr/
COPY data/data_raw/ /data/data_raw/

# Install CPU torch wheels first (when TORCH_FLAVOR=cpu) so Docling's
# transitive torch dep doesn't pull ~4 GB of CUDA nvidia-* wheels.
# For TORCH_FLAVOR=cuda this is a no-op and pip picks the default
# (CUDA) torch build during `pip install -e .`.
RUN if [ "$TORCH_FLAVOR" = "cpu" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
    fi

RUN pip install -e .

EXPOSE 8000
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
