# Makefile for managing project tasks
ARGS ?= ""

.PHONY: test lint format clean train train-cbm

# Run tests
test:
	PYTHONPATH=src pytest -v --tb=short test/

# Lint the code (optional)
lint:
	flake8 src/ test/

# Autoformat the code (optional)
format:
	black src/ test/

# Clean Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 
	find . -type d -name "MagicMock" -exec rm -r {} + 
	find . -type d -name ".pytest_cache" -exec rm -r {} + 
	find . -type f -name "*.pyc" -delete

train:
	python src/train.py $(ARGS)

train-cbm:
	python src/train_cbm.py $(ARGS)