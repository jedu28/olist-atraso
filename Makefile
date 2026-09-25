.PHONY: setup train test api

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

train:
	. .venv/bin/activate && python -m src.train

test:
	. .venv/bin/activate && pytest

api:
	. .venv/bin/activate && uvicorn src.api.main:app --reload --port 8000
