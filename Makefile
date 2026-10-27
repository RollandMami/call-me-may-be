RM = rm -rf
MAIN = main.py
MYPY_FLAGS = --warn-return-any \
			 --warn-unused-ignores \
			 --ignore-missing-imports \
			 --disallow-untyped-defs \
			 --check-untyped-defs

GOINFRE = $(HOME)/sgoinfre/$(USER)

export HF_HOME = $(GOINFRE)/.llm
export UV_CACHE_DIR = $(GOINFRE)/.uv-cache
export UV_PYTHON_INSTALL_DIR = $(GOINFRE)/.uv-python
export UV_PROJECT_ENVIRONMENT = $(GOINFRE)/cmb-venv

ENV = $(UV_PROJECT_ENVIRONMENT)

BIN = $(ENV)/bin
PYTHON = $(BIN)/python3

install:
	@mkdir -p $(UV_CACHE_DIR) $(HF_HOME)
	@uv sync -q

run: install
	@if [ -f $(MAIN) ]; then \
		uv run -q $(MAIN); \
	else \
		echo "\nPas de fichier main\n"; \
	fi

debug: install
	@if [ -f $(MAIN) ]; then \
		uv run -q python3 -m pdb $(MAIN); \
	else \
		echo "\nPas de fichier main\n"; \
	fi

clean:
	@find . -type d -name "__pycache__" -exec $(RM) {} +
	@find . -type d -name ".mypy_cache" -exec $(RM) {} +
	@find . -type d -name ".pytest_cache" -exec $(RM) {} +

fclean: clean
	$(RM) $(ENV)

lint: install
	uv run -q mypy . $(MYPY_FLAGS) --exclude="^\$(ENV)"
	uv run -q flake8 . --exclude=$(ENV)

lint-strict: install
	uv run -q mypy . --strict --exclude="^\$(ENV)"
	uv run -q flake8 . --exclude=$(ENV)

.PHONY: install run debug clean fclean lint lint-strict
