"""Products on disk: their layout (:mod:`~pollux.io.layout`), the code version of runs
(:mod:`~pollux.io.code_version`), their index (:mod:`~pollux.io.index`) and loader (:mod:`~pollux.io.loader`).

Only light dependencies, so that products can be read without the full pollux package.
"""
from .loader import PolluxLoader

__all__ = ['PolluxLoader']
