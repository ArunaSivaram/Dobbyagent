PTH_FILES := .venv/lib/python3.12/site-packages/*.pth

.PHONY: setup unhide test lint check eval run

setup:
	uv sync
	@$(MAKE) --no-print-directory unhide

unhide:
	@chflags nohidden $(PTH_FILES) 2>/dev/null || true

test: unhide
	uv run --no-sync pytest tests/ -v

lint: unhide
	uv run --no-sync ruff check

check: lint test

MODEL ?= claude-haiku-4-5-20251001
PROVIDER ?= anthropic
RUNS ?= 3

eval: unhide
	uv run --no-sync python -m evals.run_evals --model $(MODEL) --provider $(PROVIDER) --runs $(RUNS)

run: unhide
	uv run --no-sync python $(ARGS)
