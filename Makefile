.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	mypy .

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build
