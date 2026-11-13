# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-serve.txt .
RUN python -m pip install --no-cache-dir --retries 10 --timeout 180 -r requirements-serve.txt

COPY app ./app
COPY frontend ./frontend
COPY static ./static
COPY app.py ./app.py
COPY model ./model
COPY data/synthetic_evidence_outcomes.csv ./data/synthetic_evidence_outcomes.csv

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]