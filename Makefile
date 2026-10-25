PY := .venv/bin/python
export UV_CACHE_DIR := $(CURDIR)/.uv-cache

.PHONY: install test test-all bench bench-validate up down review

install:  ## create .venv with the pinned dependencies
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt -e .

test:  ## unit tests: no Docker, no network, no API key
	$(PY) -m pytest -q tests/unit

test-all:  ## everything: needs Docker and network (GitHub, PyPI, OSV)
	$(PY) -m pytest -q -m "docker or network or not (docker or network)"

bench-validate:  ## check benchmark labels and print the labelled lines
	$(PY) -m prreview.bench validate

bench:  ## run the benchmark against the real model (paid API calls); TRIALS=3 for repeats
	$(PY) -m prreview.bench run --trials $(or $(TRIALS),1) --yes

review:  ## make review URL=https://github.com/owner/repo/pull/1
	$(PY) -m prreview review $(URL)

up:  ## Docker Compose stack (UI on http://127.0.0.1:8501)
	docker compose up --build -d

down:
	docker compose down
