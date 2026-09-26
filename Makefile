PYTHON := .venv/bin/python

.PHONY: help setup data train tune evaluate figures notebooks test lint format all

help:  ## lista os comandos
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-10s %s\n", $$1, $$2}'

setup:  ## cria o .venv e instala dependências (runtime + dev)
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

data:  ## monta o dataset de features em data/processed/dataset.parquet
	$(PYTHON) -m src.features

train:  ## treina o modelo final e salva pipeline + threshold
	$(PYTHON) -m src.train --compare-baselines

tune:  ## refaz a busca de hiperparâmetros do Optuna
	$(PYTHON) -m src.tuning

evaluate:  ## reavalia o artefato salvo no teste
	$(PYTHON) -m src.evaluate

figures:  ## gera os gráficos da documentação (docs/figures)
	$(PYTHON) scripts/make_figures.py

notebooks:  ## re-executa os notebooks do zero
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

test:  ## roda os testes
	$(PYTHON) -m pytest

lint:  ## checa estilo e imports
	$(PYTHON) -m ruff check .

format:  ## formata o código
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format src tests scripts

all: lint test train figures  ## pipeline completo
