PY ?= .venv/bin/python
UV ?= uv

.PHONY: setup data test lint eval eval-claude api app demo docker notebooks clean

setup:            ## create venv and install everything
	$(UV) venv .venv
	$(UV) pip install --python $(PY) -e ".[app,dev]"

data:             ## generate the synthetic DuckDB database
	$(PY) -m pharmapulse.cli build-data

test:             ## unit + integration tests (offline, mock LLM)
	$(PY) -m pytest

lint:             ## ruff + format check + mypy
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

eval:             ## agent eval on golden questions with the offline mock provider
	$(PY) -m pharmapulse.evals.run_evals --provider mock --min-pass-rate 0.95

eval-claude:      ## agent eval against Claude (needs ANTHROPIC_API_KEY)
	$(PY) -m pharmapulse.evals.run_evals --provider anthropic

api:              ## REST API on :8000
	$(PY) -m uvicorn pharmapulse.api.main:app --reload --port 8000

app:              ## Streamlit dashboard on :8501
	$(PY) -m streamlit run app/streamlit_app.py

demo: data app

notebooks:        ## execute notebooks in place
	$(PY) scripts/build_notebooks.py

docker:
	docker build -t pharmapulse .

clean:
	rm -rf data/*.duckdb .pytest_cache .mypy_cache .ruff_cache
