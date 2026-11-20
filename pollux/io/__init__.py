"""Products on disk: their files (:class:`~pollux.io.layout.PolluxIO`, :mod:`~pollux.io.layout`), configurations (:mod:`~pollux.io.config`), the code version of runs
(:mod:`~pollux.io.code_version`), their index (:mod:`~pollux.io.index`) and loader (:mod:`~pollux.io.loader`).

Without the dependencies of the builders, so that products can be read without the full pollux package: numpy, healpy, healsparse, pyccl, sacc, astropy and PyYAML, but not NaMaster, TreeCorr, BaryonForge or Parsl.
"""
from .layout import PolluxIO
from .loader import PolluxLoader

__all__ = ['PolluxIO', 'PolluxLoader']
