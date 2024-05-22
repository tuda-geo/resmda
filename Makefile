help:
	@echo "Commands:"
	@echo ""
	@echo "  install        install in editable mode"
	@echo "  dev-install    install in editable mode with dev requirements"
	@echo "  pytest         run the test suite and report coverage"
	@echo "  flake8         style check with flake8"
	@echo "  clean          clean up all generated files"
	@echo ""

install:
	python -m pip install --no-build-isolation --no-deps -e .

dev-install:
	python -m pip install -r requirements-dev.txt && python -m pip install --no-build-isolation --no-deps -e .

.ONESHELL:
pytest:
	rm -rf .coverage htmlcov/ .pytest_cache/
	pytest --cov=resmda
	coverage html

flake8:
	flake8 setup.py resmda/ tests/

clean:
	python -m pip uninstall resmda -y
	rm -rf build/ dist/ .eggs/ resmda.egg-info/ resmda/version.py  # build
	rm -rf */__pycache__/ */*/__pycache__/      # python cache
	rm -rf .coverage htmlcov/ .pytest_cache/    # tests and coverage
