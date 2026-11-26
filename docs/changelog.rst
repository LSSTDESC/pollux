Changelog
=========

Version 0.2
-----------

Production of the HACC-HOS-Y1 lightcones (`PR #34 <https://github.com/LSSTDESC/pollux/pull/34>`_).


- Maps on padded and apodized footprints, for all harmonic transforms.
- Density contrast relative to the cosmic mean; means removed only for harmonic transforms, and added back after smoothing.
- Smoothing per field in catalogs (``smooth_fields``, ``smooth_Mpc``), accounted for in predictions.
- :class:`~pollux.cosmo.Tracer` and :class:`~pollux.cosmo.Theory`, shared by map builders and catalogs: spectra, Gaussian errors and covariance, 2pt functions.
- Catalog Cls and 2pt functions, generated per shell with reproducible draws.
- Baryon-corrected matter power spectrum with the profiles and mass cut of the baryonified shells.
- Parsl tasks and configuration in :mod:`pollux.pipeline`, for runs by several users from a shared space.
- Record of the code version and environment of runs, and commit in catalog headers.
- Plotting module (:mod:`pollux.statistics.plotting`) shared by map builders and catalogs.
- Type hints and docstrings throughout.
- Products in a root directory per simulation and footprint: :class:`~pollux.footprint.Footprint` (mask, padding,
  weights of the harmonic transforms, its ``config.yaml``) in ``footprint/``, one directory per seed;
  :class:`~pollux.io.files.PolluxIO` holds the root and the options of the objects, and reads and writes the files
  by format.
- Pipeline tasks decide what to skip; builders and catalogs only compute and write, and read their inputs.
- Safeguards of the runs (:mod:`pollux.pipeline.safeguards`): no retries of tasks that cannot fit in a block, a canary
  of each code and configuration before a run, and a guard stopping runs whose tasks take too long or keep failing.
- :class:`~pollux.reader.SyntheticReader` (particles drawn in a box, for tests) replaces ``MockReader`` and
  ``DummyReader``; first tests in ``tests/`` (pytest).
- Catalog 2pt functions by groups of bin pairs sharing their randoms (:func:`~pollux.pipeline.tasks.pair_groups_2pt`),
  on a ``twopt`` executor, two tasks per node; thread caps of the ``highmem`` workers.
- Non-Limber correction of the spectra of density tracers, e.g. thin shells (``non_limber=True`` in ``predict_cls``,
  off by default: about a minute per pair).

Version 0.1
-----------

Initial release of Pollux package.
