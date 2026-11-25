FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app

COPY requirements.txt .
COPY wheels /wheels
RUN if ls /wheels/*.whl >/dev/null 2>&1; then \
      pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt; \
    else \
      pip install --no-cache-dir --default-timeout=300 --retries 10 -r requirements.txt; \
    fi && rm -rf /wheels

COPY src ./src
COPY models ./models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]