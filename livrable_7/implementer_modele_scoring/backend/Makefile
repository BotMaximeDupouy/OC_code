.PHONY: install
install:  ## Install python dependencies
    @pip install --upgrade pip
    @pip install -r requirements-dev.txt
    @pre-commit install

.PHONY: tests
tests: ## Launch test suite
    @echo "Launching test suite"
    @PYTHONPATH=`pwd` python -m pytest -v --cov api --md-report --md-report-zeros empty tests/*
