# Claim-extraction API + GUI. Works on any container host (e-infra, Hugging Face
# Spaces, Fly, Render, a plain VM). Required env at runtime:
#   OPENAI_API_KEY, OPENAI_API_MODEL, FACTICLI_API_KEY
# Recommended:
#   OPENAI_API_BASE_URL, FACTICLI_CORS_ORIGINS
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FACTICLI_WEB_HOST=0.0.0.0 \
    FACTICLI_WEB_PORT=8000

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[web]"

# Hugging Face Spaces runs as a non-root user; harmless elsewhere.
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

CMD ["python", "-m", "facticli.web"]
