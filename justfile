build:
    poetry build

publish: build
    poetry publish

format:
    black MODULE_NAME

lint:
    pylint MODULE_NAME

test:
    pytest tests/

version:
    poetry version