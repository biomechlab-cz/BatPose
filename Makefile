.PHONY: install lint test e2e run-gui clean

install:
	uv venv .venv --python 3.12
	uv pip install -e ".[dev]" --constraint <(echo "PySide6<6.8")

install-extras:
	uv pip install -e ".[dev,rtmpose]" --constraint <(echo "PySide6<6.8")

lint:
	uv run ruff check app/ tests/ --fix
	uv run ruff format app/ tests/

lint-check:
	uv run ruff check app/ tests/
	uv run ruff format --check app/ tests/

test:
	uv run pytest tests/unit -v

e2e:
	uv run pytest tests/e2e -v

run-gui:
	uv run python -m app.gui

run-calib:
	uv run python -m app.calib --help

run-pose2d:
	uv run python -m app.pose2d --help

run-recon3d:
	uv run python -m app.recon3d --help

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist build *.egg-info

.DEFAULT_GOAL := help
help:
	@echo "Available targets:"
	@echo "  install       - Install package with dev dependencies (uv)"
	@echo "  install-extras- Install with rtmpose extras too"
	@echo "  lint          - Run ruff linter + formatter (auto-fix)"
	@echo "  lint-check    - Check lint without fixing"
	@echo "  test          - Run unit tests"
	@echo "  e2e           - Run end-to-end tests"
	@echo "  run-gui       - Launch the GUI"
	@echo "  clean         - Remove build artifacts and caches"
