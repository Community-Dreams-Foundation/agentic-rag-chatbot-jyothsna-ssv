.PHONY: sanity web

PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

sanity:
	$(PY) -m app.sanity

web:
	$(PY) -m app.web

