PROJETNAME=polars_compare
SRCPATH := $(CURDIR)

.PHONY: format
format:
	-ruff --fix $(SRCPATH)
	-black $(SRCPATH)
	-ruff $(SRCPATH)
	-black --check $(SRCPATH)
	-mypy --strict $(SRCPATH)

.PHONY: check
check:
	-ruff $(SRCPATH)
	-black --check $(SRCPATH)
	-mypy --strict compare.py
	-pytest $(SRCPATH)

.PHONY: test
test:
	pytest .