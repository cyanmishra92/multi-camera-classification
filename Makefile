.PHONY: install test clean lint format run-example

install:
	pip install -e .[dev,viz]

test:
	pytest tests/ -v --cov=src

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/

run-example:
	python src/main.py --config configs/default_config.yaml
