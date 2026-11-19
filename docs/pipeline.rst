Running the pipeline
====================

Large runs chain the steps of :doc:`getting_started` as `Parsl <https://parsl.readthedocs.io/>`_ tasks
(:mod:`pollux.pipeline.tasks`), on Slurm jobs set by :mod:`pollux.pipeline.config`. A run is a Python script
chaining the tasks, with a YAML configuration file.

Steps
-----

1. **Shells**: particles of each sub-volume are binned in count maps (``construct_shell_sub``), then summed per shell (``combine_shell_subs``).
2. **Haloes and baryonification** (optional): halo catalogs per shell (``construct_halo_catalog``), used to baryonify the shells (``construct_baryonified_shell``, with a baryon model).
3. **Lensing**: convergence maps summing weighted shells, per tomographic bin and per shell (``construct_kappa_map``), and their spherical harmonics (``compute_kappa_alms``).
4. **Fields of the catalogs**: shear (``compute_shear_maps``), tidal (``compute_tidal_maps``) and smoothed density and convergence maps, as needed by the catalog (``catalog_map_tasks``).
5. **Catalogs**: tomographic source catalogs (``construct_catalog``, or per shell with ``construct_catalog_shell`` then ``merge_catalog``).
6. **Statistics**: Cls of the shells, convergence maps and catalogs (``compute_shell_cls``, ``compute_kappa_cls``, ``compute_catalog_cls``) and 2pt functions of the catalogs (``compute_catalog_2pt``), with the lensing shear, the intrinsic alignments and their sum; plots compare them to their predictions (``plot_catalog_cls``, ``plot_catalog_2pt``).

The tasks use :class:`~pollux.reader.HACCReader`, :class:`~pollux.shells.SnapshotShellBuilder` and
:class:`~pollux.catalog.CatalogClusteredPoisson`; the other readers, builders and catalogs are used from
Python directly.

Tasks
-----

Each task makes one product (or part of one), and takes the options of the objects it needs as dictionaries:
``reader_dict`` (:class:`~pollux.reader.HACCReader`), ``shell_dict``
(:class:`~pollux.shells.SnapshotShellBuilder`), ``lensing_dict`` (:class:`~pollux.lensing.BornLensingBuilder`, or
None for one source plane per shell), ``ia_dict`` (:class:`~pollux.ia.CatalogAugmentorIA`), ``catalog_dict``
(:class:`~pollux.catalog.CatalogClusteredPoisson`), ``baryon_dict`` (the tag, halo mass and profile
parameters of :class:`~pollux.baryons.BaryonifiedShellBuilder`, or None for dark matter only) and ``stats_dict``.
Workers rebuild the objects from these dictionaries, so tasks only pass small arguments, and the products go
through the disk. Dependencies are the ``inputs`` of each task: the futures of the tasks making its inputs.

Writing a run
-------------

A run script:

1. reads its configuration (:func:`~pollux.pipeline.config.load_run_config`, which expands environment variables
   such as ``$PSCRATCH`` in its paths), and builds the dictionaries of the tasks from it;
2. creates a reader and a shell builder in the main process, for what the tasks need to know in advance (number of
   sub-volumes, of shells, smoothing scales per shell from :meth:`~pollux.cosmo.Theory.lognormal_smoothing`);
3. records its code version (:func:`~pollux.io.code_version.save_code_version`, see below);
4. loads the Parsl configuration (:func:`~pollux.pipeline.config.get_config`) and creates the futures of all
   tasks, chained by their ``inputs``;
5. waits for the futures. A failed task makes the tasks depending on it fail, but not the others: catching the
   exception of each future lets the run go on and list the failures at the end.

Run scripts are launched from a login node, in the background (``nohup``), since they wait for the Slurm jobs of
their tasks.

Execution
---------

:func:`~pollux.pipeline.config.get_config` defines three executors, by the memory of their tasks, each submitting
one-node Slurm jobs (blocks) on demand (``EXECUTORS``):

- ``'default'``: many tasks of up to about 10 GB per node (sub-volume shells, assembly of shells, convergence);
- ``'highmem'``: four tasks of up to about 120 GB per node (power spectra, halo catalogs, baryonification);
- ``'heavy'``: one task per node, for full-sky harmonic transforms and catalogs.

Blocks stop taking tasks before the end of their walltime, so that running tasks finish. ``debug=True`` uses the
debug queue. The NERSC account is ``$POLLUX_ACCOUNT``, and Slurm notifications go to ``$POLLUX_MAIL`` if set.

Parsl records the tasks in a monitoring database, from which
``python -m pollux.pipeline.report {monitoring.db}`` reports the time, memory and failures of each kind of task.

Resuming
--------

Tasks skip the products already on disk, and every file is written to a temporary name then renamed, so that an
interrupted task leaves no partial file. Statistics are computed per bin pair, each pair saved as an intermediate
file (in a folder named by an identifier of the binning and options of the statistic), then combined: tasks skip
the pairs already saved, and the statistics already combined, but always redraw the plots. Launching a run again therefore resumes it. For the same reason, a product
must be moved away (or the run given a new output directory) to be recomputed. A product whose ``config.yaml``
differs from the options of the run raises an error, instead of mixing the two (see :doc:`outputs`); it is checked
before skipping, so a skipped product has the options of the run. Likewise, the combined statistics record the
identifier of their binning and options (``options_id`` in the metadata of their SACC file), and one computed
with other options raises an error.

Code version
------------

:func:`pollux.io.code_version.save_code_version` records each run in a new folder ``code_version/{date}`` of its
output directory: git commit and state of pollux, date, host and command, lists of installed Python and conda
packages, and copies of the run script and configuration. With ``require_clean=True``, it refuses to run if pollux
has uncommitted changes.

Catalogs also store the commit in their FITS headers: ``POLLUXV`` (first 12 characters) and ``DIRTY`` (uncommitted
changes).
