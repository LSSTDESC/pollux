"""pollux: weak lensing and large-scale structure simulations from N-body simulations.

The main classes are available from the package: readers, shell, lensing and baryonified builders, catalog
generators and augmentors, and :class:`~pollux.io.PolluxLoader` to read products, which only needs the I/O
dependencies (:mod:`pollux.io`).
"""

import warnings

from .io import PolluxLoader

__all__ = ['PolluxLoader']

try:
    from .lensing import BornLensingBuilder, SnapshotBornLensingBuilder
    from .reader import SLICSReader, HACCReader, HACCHaloReader, DummyReader
    from .shells import SnapshotShellBuilder, SnapshotInterpBuilder, LightconeSlicingBuilder
    from .catalog import CatalogSourceUniform, CatalogSourceClusteredMultinomial, CatalogSourceClusteredPoisson
    from .ia import IACalculator, CatalogAugmentorShear
    from .baryons import BaryonifiedShellBuilder
    __all__ += [
        'BornLensingBuilder', 'SnapshotBornLensingBuilder', 'SLICSReader', 'HACCReader', 'HACCHaloReader', 'DummyReader', 'SnapshotShellBuilder',
        'SnapshotInterpBuilder', 'LightconeSlicingBuilder', 'CatalogSourceUniform', 'CatalogSourceClusteredMultinomial', 'CatalogSourceClusteredPoisson',
        'IACalculator', 'CatalogAugmentorShear', 'BaryonifiedShellBuilder'
    ]
except ImportError as e:
    # e.g. on a machine with only the I/O dependencies, to read products
    warnings.warn(f"Could not import the full pollux package ({e}): only pollux.io (PolluxLoader) is available.")
