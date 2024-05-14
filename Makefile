help:
	@echo "Commands:"
	@echo ""
	@echo "  install        install in editable mode"
	@echo "  flake8         style check with flake8"
	@echo "  clean          clean up all generated files"
	@echo ""

install:
	python -m pip install --no-build-isolation --no-deps -e .

flake8:
	flake8 setup.py resmda/

clean:
	python -m pip uninstall resmda -y
	rm -rf build/ dist/ .eggs/ resmda.egg-info/ resmda/version.py  # build
	rm -rf */__pycache__/ */*/__pycache__/      # python cache
