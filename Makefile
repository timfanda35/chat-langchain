.PHONY: start
start:
	uvicorn main:app --reload --port 9000

.PHONY: ingest
ingest:
	python3 ingest.py

.PHONY: format
format:
	black .
	isort .