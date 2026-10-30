Changelog
=========

Development
-----------

- Maps on padded and apodized footprints, for all harmonic transforms.
- Density contrast relative to the cosmic mean; means removed only for harmonic transforms, and added back after smoothing.
- Smoothing per field in catalogs (``smooth_fields``, ``smooth_Mpc``), accounted for in predictions.
- :class:`~pollux.cosmo.Tracer` and :class:`~pollux.cosmo.Theory`, shared by map builders and catalogs: spectra, Gaussian errors and covariance, 2pt functions.
- Catalog Cls and 2pt functions, generated per shell with reproducible draws.
- Baryon-corrected matter power spectrum with the profiles and mass cut of the baryonified shells.
- Parsl tasks and configuration in :mod:`pollux.pipeline`, for runs by several users from a shared space.
- Record of the code version and environment of runs, and commit in catalog headers.
- Plotting module (:mod:`pollux.plotting`) shared by map builders and catalogs.
- Type hints and docstrings throughout.

Version 0.1
-----------

Initial release of Pollux package.
