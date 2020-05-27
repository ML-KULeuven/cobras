.PHONY: docs
docs:
	export PYTHONPATH=..; cd docs; make html

.PHONY: prepare_dist
prepare_dist:
	rm -rf dist/*
	python3 setup.py sdist bdist_wheel

.PHONY: deploy
deploy: prepare_dist
	@echo "Check whether repo is clean"
	git diff-index --quiet HEAD
	@echo "Add tag"
	git tag "v$$(python3 setup.py --version)"
	@echo "Start uploading"
	twine upload dist/*

