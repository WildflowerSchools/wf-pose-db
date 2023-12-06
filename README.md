# MODULE_NAME

SHORT_DESCRIPTION

## Installation

`pip install PYPI_PACKAGE_NAME`

## Development

### Requirements

* [Poetry](https://python-poetry.org/)
* [just](https://github.com/casey/just)

### Install

`poetry install`


#### Install w/ Python Version from PyEnv

```
# Specify pyenv python version
pyenv shell --unset
pyenv local <<VERSION>>

# Set poetry python to pyenv version
poetry env use $(pyenv which python)
poetry cache clear . --all
poetry install
```

## Task list
* TBD
