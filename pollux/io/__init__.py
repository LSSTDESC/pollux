"""Products on disk: their layout (:mod:`~pollux.io.layout`), the code version of runs
(:mod:`~pollux.io.code_version`), their index (:mod:`~pollux.io.index`) and loader (:mod:`~pollux.io.loader`).

Without the dependencies of the builders, so that products can be read without the full pollux package: numpy, healpy, healsparse, pyccl, sacc, astropy and PyYAML, but not NaMaster, TreeCorr, BaryonForge or Parsl.
"""
from .loader import PolluxLoader

__all__ = ['PolluxLoader']
