build:
    poetry build

publish: build
    poetry publish

format:
    black pose_db_io

lint:
    pylint pose_db_io

test:
    pytest tests/

version:
    poetry version