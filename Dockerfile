FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PHARMAPULSE_DB=/app/data/pharmapulse.duckdb
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
COPY pyproject.toml README.md LICENSE ./
COPY pharmapulse ./pharmapulse
RUN uv pip install --system --no-cache ".[app]"

COPY app ./app
# bake the synthetic database into the image so the container starts instantly
RUN python -m pharmapulse.cli build-data

EXPOSE 8000 8501
# default: API. For the dashboard: docker run -p 8501:8501 pharmapulse streamlit run app/streamlit_app.py
CMD ["uvicorn", "pharmapulse.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
