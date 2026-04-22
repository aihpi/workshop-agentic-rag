FROM python:3.12-slim

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
RUN pip install -e .

EXPOSE 8000
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
