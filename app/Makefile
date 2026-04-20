.PHONY: help init-env up logs down ps reingest local-install local-run

help:
	@echo "Targets:"
	@echo "  make init-env      # create .env from .env.example if missing"
	@echo "  make up            # docker compose up -d --build"
	@echo "  make logs          # follow chainlit logs"
	@echo "  make down          # stop stack"
	@echo "  make ps            # list compose services"
	@echo "  make reingest      # force recreate qdrant collection + restart chainlit"
	@echo "  make local-install # create .venv and install app"
	@echo "  make local-run     # run chainlit locally"

init-env:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; else echo ".env already exists"; fi

up: init-env
	docker compose up -d --build

logs:
	docker compose logs -f chainlit

down:
	docker compose down

ps:
	docker compose ps

reingest:
	INGEST_RECREATE=true docker compose run --rm ingest
	docker compose up -d chainlit

local-install:
	python -m venv .venv
	. .venv/bin/activate && pip install -e .

local-run:
	. .venv/bin/activate && chainlit run app.py -w
