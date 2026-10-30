API Reference
=============

The Pollux package is organized into the following core modules:

.. toctree::
   :maxdepth: 2

   modules/reader
   modules/map_builder
   modules/shells
   modules/lensing
   modules/baryons
   modules/catalog
   modules/ia
   modules/cosmo
   modules/plotting
   modules/utils
   modules/io
   modules/pipeline

Module Descriptions
-------------------

**Reader** (:mod:`pollux.reader`)
   Interfaces with N-body simulation snapshots (HACC, SLICS) and provides particle data for lightcone construction.

**Map Builder** (:mod:`pollux.map_builder`)
   Abstract base classes for building HealSparse maps from simulation data, with I/O and spherical harmonic transform support.

**Shells** (:mod:`pollux.shells`)
   Builds particle and halo lightcone shells with replicated boxes and random isometries (translations, rotations, flips).

**Lensing** (:mod:`pollux.lensing`)
   Generates weak lensing maps in the Born approximation with tomographic support.

**Baryons** (:mod:`pollux.baryons`)
   Baryonifies shells with BaryonForge, and computes the matching correction of the matter power spectrum.

**Catalog** (:mod:`pollux.catalog`)
   Generates tomographic galaxy catalogs from lightcone shells using uniform or clustered sampling methods, measures their statistics and predicts them.

**Intrinsic alignments** (:mod:`pollux.ia`)
   Catalog augmentors adding intrinsic alignments, reduced shear, magnification and observed ellipticities.

**Utils** (:mod:`pollux.utils`)
   Helper functions for map operations, binning, sampling and files, and the record of the code version of a run.

**Cosmology** (:mod:`pollux.cosmo`)
   Fixed cosmologies, tracers of observable fields and their theory predictions.

**Plotting** (:mod:`pollux.plotting`)
   Plots of two-point statistics of bin pairs against theory.

**I/O** (:mod:`pollux.io`)
   Loads and manages Pollux simulation data.

**Pipeline** (:mod:`pollux.pipeline`)
   Parsl tasks and execution configuration, see :doc:`pipeline`.


