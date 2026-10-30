Running the pipeline
====================

The pipeline turns N-body snapshots into maps and mock catalogs. Each step is a parsl task (:mod:`pollux.pipeline.tasks`),
run on Slurm jobs set by :mod:`pollux.pipeline.config`. A run is described by a Python script chaining the tasks and a YAML
configuration file.

Steps
-----

1. **Shells**: particles of each sub-volume are binned in count maps (``construct_shell``), then summed per shell (``construct_full_shells``).
2. **Haloes and baryonification** (optional): halo catalogs per shell (``construct_halo_catalog``), used to baryonify the shells (``construct_maps``).
3. **Lensing**: convergence maps summing weighted shells, per tomographic bin and per shell (``construct_kappas``), and their spherical harmonics (``construct_kappas_alms``).
4. **Fields of the catalogs**: shear (``compute_shear_maps``), tidal (``compute_tidal_maps``) and smoothed density and convergence maps, as needed by the catalog (``catalog_map_tasks``).
5. **Catalogs**: tomographic source catalogs (``construct_source_catalog``, or per shell with ``construct_source_catalog_shell`` then ``merge_source_catalog``).
6. **Statistics**: map and catalog Cls and 2pt functions, compared to predictions in plots (``plot_catalog_stats``).

Tasks skip the outputs already on disk, so a run can be resumed by launching it again. For the same reason, a run that must recompute a product needs a new output directory.

Smoothing
---------

The density used to place galaxies and the fields interpolated at galaxies can be smoothed by a Gaussian of comoving scale ``smooth_Mpc``, set by two options of the configuration:

- ``catalog.smooth_fields``: the smoothed fields, among ``kappa``, ``gamma1``, ``gamma2``, ``density``, ``t11``, ``t22``, ``t12`` and ``positions`` (the density placing galaxies);
- ``smooth_Mpc``: the scale, or a dictionary of scales per field.

Fields made together must share their scale (the tidal tensor components, and the shear components). Predictions account for the smoothing of ``positions`` (see :class:`pollux.cosmo.Tracer`).

Linear galaxy bias
------------------

Clustered catalogs place galaxies with a density proportional to ``1 + b * delta``. With ``b > 1`` this is negative in underdense pixels: it is clipped to zero, with a warning giving the fraction of pixels affected.
Clipping changes the effective bias, and the mean density of Poisson-sampled catalogs.

Production in a shared space
----------------------------

Several users can run the same code from a shared space::

    HACC-HOS-Y1/
    |-- pollux/    clone of this repository at a fixed commit
    |-- scripts/   run scripts, their configurations, and setup.sh
    |-- data/      masks and redshift distributions
    `-- deps/      pinned dependencies

``setup.sh`` activates the shared Python environment, in which this clone of pollux is installed, and removes anything from the user's environment taking precedence over it. The ``.sh`` launchers do it themselves, so a run only takes::

    /path/to/HACC-HOS-Y1/scripts/run_xxx.sh [configuration]

Each user's outputs go to their own scratch (``save_dir_root`` of the configuration, where environment variables such as ``$PSCRATCH`` are expanded), and logs to ``$PSCRATCH/HACC-HOS-Y1/logs``.

Two environment variables set the use of the NERSC allocation:

- ``POLLUX_ACCOUNT``: account charged (default ``m1727``);
- ``POLLUX_MAIL``: address of the Slurm notifications (default: none).

Code version
------------

:func:`pollux.utils.save_code_version` records each run in a new folder ``code_version/{date}`` of its output directory: git commit and state of pollux, date, host and command, lists of installed Python and conda packages, and copies of the run script and configuration.
Run scripts refuse to start if pollux has uncommitted changes, unless the configuration sets ``allow_uncommitted: true``.

Catalogs also store the commit in their FITS headers: ``POLLUXV`` (first 12 characters) and ``DIRTY`` (uncommitted changes).

Output structure
----------------

Under ``{save_dir_root}/{simulation}/nside{nside}/seed{seed}/``:

- ``shells/`` (``shells_pad{padding}/`` with padding): shell maps, density and tidal maps, and their spherical harmonics and Cls, with sub-volume maps in ``subs/``;
- ``shells/kappa_{tag}/``: convergence and shear maps, for the tomographic bins and for the snapshots (``kappa_zsnapshots``);
- ``shells/kappa_{tag}/catalogs_{tag}/``: source catalogs, their statistics (``cat_2pt.fits``, ``cat_cls.fits``) and ``plots/``;
- ``shells/baryons_{tag}/``: the same products from baryonified shells;
- ``code_version/``: the records of the runs.

:class:`pollux.io.PolluxIO` loads these products.
