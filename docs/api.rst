API Reference
=============

The Pollux package is organized into the following core modules:

.. toctree::
   :maxdepth: 2

   modules/reader
   modules/map_builder
   modules/shells
   modules/lensing
   modules/catalog
   modules/utils
   modules/cosmo
   modules/io

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

**Catalog** (:mod:`pollux.catalog`)
   Generates tomographic galaxy catalogs from lightcone shells using uniform or clustered sampling methods.

**Utils** (:mod:`pollux.utils`)
   Helper functions for map operations, power spectrum computations, and array manipulations.

**Cosmology** (:mod:`pollux.cosmo`)
   Fixed cosmologies and cosmological utility functions.

**I/O** (:mod:`pollux.io`)
   Loads and manages Pollux simulation data.


