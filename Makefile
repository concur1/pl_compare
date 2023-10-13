PROJETNAME=polars_compare
SRCPATH := $(CURDIR)/pl_compare

.PHONY: format
format:
	-ruff --fix $(SRCPATH)
	-black $(SRCPATH)
	-ruff $(SRCPATH)
	-black --check $(SRCPATH)

.PHONY: check
check:
	-ruff $(SRCPATH)
	-black --check $(SRCPATH)
	-mypy --strict pl_compare/compare.py
	-pytest $(SRCPATH)

.PHONY: test
test:
	pytest .