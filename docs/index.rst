Pollux Documentation
====================

DESC package to create weak lensing and large-scale structure simulations from N-body simulations for the Higher-Order Statistics Topical Team.

Authors: Cyrille Doux and Juan Mena-Fernández

.. image:: ../pollux_maps.png
   :alt: Example pollux map images
   :width: 600px

Features
--------

- Interfaces with HACC and SLICS N-body simulations
- Construction of particle and halo lightcones with replicated boxes and random isometries, on the full sky or any footprint (optionally padded and apodized for harmonic transforms)
- Lensing in the Born approximation with tomography support, shear and tidal fields from spherical harmonics, with optional Gaussian smoothing
- Baryonification of the shells with `BaryonForge <https://github.com/DhayaaAnbajagane/BaryonForge>`_, with the matching correction of the matter power spectrum
- Generation of tomographic galaxy catalogs with multiple sampling methods, intrinsic alignments and observed shapes
- Catalog statistics (Cls with NaMaster, 2pt functions with TreeCorr) and theory predictions with CCL
- Efficient parallel computation using `Parsl <https://parsl.readthedocs.io/>`_, with a record of the code and environment of each run
- Integration with DESC tools: `CCL <https://github.com/LSSTDESC/CCL>`_, `healsparse <https://github.com/LSSTDESC/healsparse>`_, and `NaMaster <https://github.com/LSSTDESC/NaMaster>`_

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   pipeline
   api
   changelog

Getting Started
---------------

Runs go through the parsl pipeline described in :doc:`pipeline`. For tutorial notebooks, see the ``notebooks`` folder of the `GitHub repository <https://github.com/LSSTDESC/pollux>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
