FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip \
    && pip install -e .

EXPOSE 8001

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8001"]
