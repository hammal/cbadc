# cbadc developer entry points.
#
# Thin wrappers over scripts/, which hold the actual command definitions and are
# what both CI systems call. Keeping the commands in scripts/ (not here) means
# GitLab CI, GitHub Actions and a developer's shell all run byte-identical
# commands -- see docs/development-workflow.md.

PODMAN ?= podman

.PHONY: help install lint test test-unit test-doctest test-performance test-ngspice test-ngspice-docker test-ngspice-podman test-all docs clean

help:
	@echo "cbadc development targets"
	@echo
	@echo "  make install           uv sync --extra dev --extra delsig"
	@echo "  make lint              ruff check + format --check (read-only)"
	@echo "  make format            ruff format (rewrites files)"
	@echo "  make test              lint + doctests + unit tests (the merge gate)"
	@echo "  make test-unit         full unit suite"
	@echo "  make test-doctest      doctests in src/"
	@echo "  make test-performance  the ngspice bench (alias of test-ngspice)"
	@echo "  make test-ngspice      cbadc against ngspice circuits (needs XSPICE)"
	@echo "  make test-ngspice-docker  the same bench inside the Docker image CI uses"
	@echo "  make test-ngspice-podman  the same bench inside a Podman container"
	@echo "  make docs              build the Sphinx HTML documentation"
	@echo "  make clean             remove build artefacts and JUnit reports"

install:
	uv sync --extra dev --extra delsig

lint:
	./scripts/lint

format:
	uv run --extra dev --extra delsig ruff format src tests

test test-all:
	./scripts/test-all

test-unit:
	./scripts/test-unit

test-doctest:
	./scripts/test-doctest

test-performance:
	./scripts/test-performance

test-ngspice:
	./scripts/test-performance ngspice

test-ngspice-docker:
	docker build -t cbadc-ngspice .
	docker run --rm cbadc-ngspice

test-ngspice-podman:
	$(PODMAN) build -t cbadc-ngspice .
	$(PODMAN) run --rm localhost/cbadc-ngspice:latest

docs:
	uv run --all-extras sphinx-build -b html docs/source docs/build/html

clean:
	rm -rf docs/build dist build .pytest_cache .ruff_cache
	rm -f pytest-*.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
