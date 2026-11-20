"""pollux: weak lensing and large-scale structure simulations from N-body simulations.

The main classes are available from the package: readers, shell, lensing and baryonified builders, catalog
generators and augmentors, and :class:`~pollux.io.PolluxLoader` to read products, which only needs the I/O
dependencies (:mod:`pollux.io`: no NaMaster, TreeCorr, BaryonForge or Parsl).
"""

import warnings

from .io import PolluxIO, PolluxLoader

__all__ = ['PolluxIO', 'PolluxLoader']

try:
    from .lensing import BornLensingBuilder, SnapshotBornLensingBuilder
    from .reader import SLICSReader, HACCReader, HACCHaloReader, DummyReader
    from .shells import SnapshotShellBuilder, SnapshotInterpShellBuilder, LightconeShellBuilder
    from .catalog import CatalogUniform, CatalogClusteredMultinomial, CatalogClusteredPoisson, CatalogAugmentorShear
    from .ia import CatalogAugmentorIA
    from .baryons import BaryonifiedShellBuilder
    __all__ += [
        'BornLensingBuilder', 'SnapshotBornLensingBuilder', 'SLICSReader', 'HACCReader', 'HACCHaloReader', 'DummyReader', 'SnapshotShellBuilder',
        'SnapshotInterpShellBuilder', 'LightconeShellBuilder', 'CatalogUniform', 'CatalogClusteredMultinomial', 'CatalogClusteredPoisson', 'CatalogAugmentorIA',
        'CatalogAugmentorShear', 'BaryonifiedShellBuilder'
    ]
except ImportError as e:
    # e.g. on a machine with only the I/O dependencies, to read products
    warnings.warn(f"Could not import the full pollux package ({e}): only pollux.io (PolluxLoader) is available.")
