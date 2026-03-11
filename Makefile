.PHONY: update-core

update-core:
	git submodule update --init --remote --recursive core
