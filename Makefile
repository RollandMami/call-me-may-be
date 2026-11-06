RM = rm -rf
MAIN = src
MYPY_FLAGS = --warn-return-any \
             --warn-unused-ignores \
             --ignore-missing-imports \
             --disallow-untyped-defs \
             --check-untyped-defs

GOINFRE = $(HOME)/sgoinfre/$(USER)

HF_HOME := $(GOINFRE)/.llm
UV_CACHE_DIR := $(GOINFRE)/.uv-cache
UV_PYTHON_INSTALL_DIR := $(GOINFRE)/.uv-python
UV_PROJECT_ENVIRONMENT := $(GOINFRE)/cmb-venv
UV_HTTP_TIMEOUT := 120

export HF_HOME
export UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR
export UV_PROJECT_ENVIRONMENT
export UV_HTTP_TIMEOUT

ENV = $(UV_PROJECT_ENVIRONMENT)

BIN = $(ENV)/bin
PYTHON = $(BIN)/python3

install:
	@mkdir -p $(UV_CACHE_DIR) $(HF_HOME)
	uv sync -q

run: install
	uv run -q python -m $(MAIN)

debug: install
	@if [ -f $(MAIN) ]; then \
		uv run -q python3 -m pdb $(MAIN); \
	else \
		echo "\nPas de fichier main\n"; \
	fi

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find ./data/ -type d -name "output" -exec rm -rf {} +

fclean: clean
	$(RM) $(ENV)

lint: install
	uv run -q mypy . $(MYPY_FLAGS) --exclude="^\$(ENV)"
	uv run -q flake8 . --exclude=$(ENV)

lint-strict: install
	uv run -q mypy . --strict --exclude="^\$(ENV)"
	uv run -q flake8 . --exclude=$(ENV)

.PHONY: install run debug clean fclean lint lint-strict