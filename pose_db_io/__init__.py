import importlib.metadata
from pathlib import Path

import toml

from .core import *
from .handle import *


def get_version() -> str:
    try:
        version: str = importlib.metadata.version(__package__ or __name__)
    except importlib.metadata.PackageNotFoundError:
        path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        pyproject = toml.load(str(path))
        version: str = pyproject["tool"]["poetry"]["version"]
    return version


__version__ = get_version()
