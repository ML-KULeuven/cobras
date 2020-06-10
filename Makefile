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
	@echo "Check version"
	$(eval IV := $(shell sed -n "s/^__version__ = '\(.*\)'$$/\1/p" cobras_ts/__init__.py))
	$(eval SV := $(shell sed -n "s/^    version='\(.*\)',.*$$/\1/p" setup.py))
	@if [[ ! "${IV}" == "${SV}" ]]; then \
		echo "Versions in __init__.py (${IV}) and setup.py (${SV}) do not match"sh ; \
		exit 1 ;\
	fi
	@echo "Add tag"
	git tag "v$$(python3 setup.py --version)"
	@echo "Start uploading"
	twine upload --repository cobrasts dist/*

