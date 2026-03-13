PROJETNAME=polars_compare
SRCPATH := $(CURDIR)/pl_compare

help: ## Print help for make commands
	@grep -h -E '^[a-zA-Z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


.PHONY: format
format: ## Formats the files with ruff
	ruff check --fix $(SRCPATH)
	ruff format $(SRCPATH)
	ruff check $(SRCPATH)

.PHONY: check
check: ## Runs checks using ruff, black, mypy and pytest
	ruff check $(SRCPATH)
	mypy --strict pl_compare/compare.py
	pytest --doctest-glob="README.md" -v

.PHONY: test
test: ## Runs tests
	pytest --doctest-glob="README.md" -v
