ENV = .venv
BIN = $(ENV)/bin
PYTHON = $(BIN)/python3

RM = rm -rf
MAIN = main.py
MYPY_FLAGS = --warn-return-any \
			 --warn-unused-ignores \
			 --ignore-missing-imports \
			 --disallow-untyped-defs \
			 --check-untyped-defs

install:
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
	@$(RM) $(ENV)

lint: install
	@uv run -q mypy . $(MYPY_FLAGS) --exclude="^\$(ENV)"
	@uv run -q flake8 . --exclude=$(ENV)

lint-strict: install
	@uv run -q mypy . --strict --exclude="^\$(ENV)"
	@uv run -q flake8 . --exclude=$(ENV)

.PHONY: install run debug clean fclean lint lint-strict
