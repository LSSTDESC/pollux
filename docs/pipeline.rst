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

.. figure:: figures/workflow.*
   :width: 100%
   :alt: The Parsl tasks of a realization, from particles to 2pt functions

   The tasks of one realization of HACC091_1024 at nside 2048, coloured by the executor they run on, with their numbers.
   With baryons, the tasks after the shells run on the dark-matter-only shells and again on the baryonified ones.

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

1. reads its configuration (:func:`~pollux.pipeline.config.read_run_config`, which expands environment variables
   such as ``$PSCRATCH`` in its paths), and builds the dictionaries of the tasks from it;
2. creates a reader and a shell builder in the main process, for what the tasks need to know in advance (number of
   sub-volumes, of shells, smoothing scales per shell from :meth:`~pollux.cosmo.Theory.lognormal_smoothing`);
3. records its code version (:func:`~pollux.io.code_version.write_code_version`, see below);
4. loads a Parsl configuration (see `Execution`_) and creates the futures of all tasks, chained by their ``inputs``;
5. waits for the futures. A failed task makes the tasks depending on it fail, but not the others: catching the
   exception of each future lets the run go on and list the failures at the end.

Run scripts are launched from a login node, in the background (``nohup``), since they wait for the Slurm jobs of
their tasks.

A minimal run, from the particles of a HACC simulation to the tomographic convergence and shear on the full sky, on
the local machine (the paths and the redshift distributions are placeholders; on Perlmutter, use the commented line
instead):

.. code-block:: python

    import os

    import numpy as np
    import parsl

    import pollux
    from pollux.pipeline.config import local_config, perlmutter_config
    from pollux.pipeline.safeguards import run_tasks
    from pollux.pipeline.tasks import (combine_shell_subs, compute_kappa_alms, compute_shear_maps, construct_kappa_map,
                                       construct_shell_sub)

    out = os.path.expandvars('$PSCRATCH/pollux_example')
    nside, seed = 2048, 0

    # options of the objects, from which the workers rebuild them
    reader_dict = dict(path='/path/to/HACC/simulation', run_params='run/indat.params', snap_sub_every_n=2)
    footprint = pollux.Footprint(nside, pollux.PolluxIO(os.path.join(out, 'simulation', f'nside{nside}')))
    shell_dict = dict(footprint=footprint, seed=seed, apply_grid_isometry=True, group_isometries=True)
    z = np.linspace(0., 3., 300)
    nzs = [(z, np.exp(-(z - zm)**2 / (2 * 0.1**2))) for zm in (0.5, 1.)]  # two source bins
    lensing_dict = dict(nzs=nzs, tag='example')

    # in the main process, only what the tasks need to know in advance: sub-volumes and shells
    reader = pollux.reader.HACCReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)

    if __name__ == '__main__':
        monitoring_db = os.path.join(out, 'monitoring.db')
        parsl.load(local_config(os.path.join(out, 'runinfo'), workflow_name='example', monitoring_db=monitoring_db))
        # parsl.load(perlmutter_config(os.path.join(out, 'runinfo'), debug=True, workflow_name='example', monitoring_db=monitoring_db))

        # shells: one task per sub-volume of the snapshots, then one per shell summing its sub-volumes
        subs = {ijob: construct_shell_sub(ijob, reader_dict, shell_dict) for ijob in range(reader.njobs)}
        shells = [combine_shell_subs(ishell, reader_dict, shell_dict, inputs=[subs[ijob] for ijob in reader.ijobs_for_snapshot(ishell)])
                  for ishell in range(builder.nmaps)]
        # convergence of each bin from all the shells, then its spherical harmonics and shear
        kappas = [construct_kappa_map(i, reader_dict, shell_dict, lensing_dict, inputs=shells) for i in range(len(nzs))]
        alms = [compute_kappa_alms(i, reader_dict, shell_dict, lensing_dict, inputs=[kappas[i]]) for i in range(len(nzs))]
        shears = [compute_shear_maps(i, reader_dict, shell_dict, lensing_dict, inputs=[alms[i]]) for i in range(len(nzs))]

        failed = run_tasks(list(subs.values()) + shells + kappas + alms + shears, monitoring_db)
        print('failed tasks:', failed)
        parsl.dfk().cleanup()

Each task skips its product if it is already on disk, so running the script again resumes it. The scripts of the
HACC-HOS-Y1 runs add the catalogs and statistics, the configuration files, the code version and the safeguards
below.

Execution
---------

The tasks run on four Parsl executors, named by the memory and CPU use of their tasks
(:data:`~pollux.pipeline.config.EXECUTOR_LABELS`): any Parsl configuration of a run must define executors with these
labels, which :func:`~pollux.pipeline.config.check_labels` checks.

- ``'default'``: light tasks (sub-volume shells, assembly of shells, convergence), many per node;
- ``'highmem'``: tasks of up to about 120 GB at nside 8192 (power spectra, halo catalogs, baryonification, catalog shells);
- ``'heavy'``: full-sky harmonic transforms, which use all the cores of a node;
- ``'twopt'``: the catalog 2pt functions, up to about 200 GB each.

**On NERSC's Perlmutter.** :func:`~pollux.pipeline.config.perlmutter_config` submits one-node Slurm jobs (blocks) on
the CPU nodes on demand, with the number of tasks per node, threads and walltime of each executor in
:data:`~pollux.pipeline.config.EXECUTORS` (32, 4, 1 and 2 tasks per node). Blocks stop taking tasks before the end of
their walltime, so that running tasks finish. ``debug=True`` uses the debug queue. The NERSC account must be set in
``$POLLUX_ACCOUNT`` (there is no default), and Slurm notifications go to ``$POLLUX_MAIL`` if set. This configuration
is specific to NERSC: the queues, node size and Slurm options are those of Perlmutter.

**On one machine.** :func:`~pollux.pipeline.config.local_config` runs all the executors on the local machine (a
laptop, a login or interactive node), one task each by default, for tests and small runs (low nside, small
footprints).

**Elsewhere.** Write a :class:`parsl.config.Config` with executors of the same labels and the provider of the site's
scheduler (Slurm, PBS, ...), following the `Parsl documentation
<https://parsl.readthedocs.io/en/stable/userguide/configuring.html>`_, and size each executor from the memory of its
tasks (:data:`~pollux.pipeline.config.EXECUTORS` gives them for nside 8192). Cap the threads of each task
(``OMP_NUM_THREADS``, ``NUMBA_NUM_THREADS``) when several share a node.

Parsl records the tasks in a monitoring database, from which
``python -m pollux.pipeline.report {monitoring.db}`` reports the time, memory and failures of each kind of task.

Safeguards
----------

:mod:`pollux.pipeline.safeguards` keeps tasks that do not work from using the allocation:

- **Retries.** Blocks stop starting tasks a drain margin before their walltime, and a task still running at the
  walltime is lost with its block. If it had run almost a whole block, it cannot fit in one and is not retried;
  otherwise it is retried once, in case it fits when started earlier. Tasks lost earlier (node failures) are
  retried as usual, tasks whose worker was lost (memory) four times, file system errors twice, other errors not.
- **Canary.** Each code and configuration needs a canary before a run: one task of each kind, on the debug queue,
  on a mirror of an existing realization (links to its files; the canary remakes the products of its tasks only).
  It prints the time and memory of each kind of task next to those of the last accepted canary, and the projected
  node-hours of a realization. A change beyond the tolerance, a failure or a first canary must be accepted.
- **Guard.** During a run, a thread stops it (cancelling its blocks) when a kind of task takes much longer than in
  its canary, or keeps failing.

With the run scripts::

    run_validation.sh config.yaml --canary          # one debug job, then the comparison
    run_validation.sh config.yaml --accept-canary   # accept the last canary of this code and configuration
    run_validation.sh config.yaml                   # the run, refused without an accepted canary
    run_validation.sh config.yaml --no-canary       # the run without canary (the guard then checks failures only)

The realization mirrored by the canary is set by ``canary: {seed: 0, save_dir_root: ...}`` in the configuration.

Resuming
--------

Tasks skip the products already on disk, so launching a run again resumes it. Every file is written to a temporary
name then renamed, so an interrupted task leaves no partial file.

Only tasks skip. The methods of the builders and catalogs compute and write their products when called, and read
their inputs, raising if one is missing (e.g. the a_lm of :meth:`~pollux.map_builder.MapBuilder.read_alms`, made by
:meth:`~pollux.map_builder.MapBuilder.compute_alms`).

Statistics are computed per bin pair, or per group of bin pairs for the catalog 2pt functions (sharing the randoms,
see :func:`~pollux.pipeline.tasks.pair_groups_2pt`), each saved as an intermediate file in a folder named by an
identifier of the binning and options of the statistic, then combined. Tasks skip the pairs already saved and the
statistics already combined, but always redraw the plots.

To recompute a product, move it away (or give the run a new output directory). A product whose ``config.yaml``
differs from the options of the run raises an error instead of mixing the two (see :doc:`outputs`); it is checked
before skipping, so a skipped product has the options of the run. Likewise, the combined statistics record the
identifier of their binning and options (``options_id`` in the metadata of their SACC file), and one computed with
other options raises an error.

Code version
------------

:func:`pollux.io.code_version.write_code_version` records each run in a new folder ``code_version/{date}`` of its
output directory: git commit and state of pollux, date, host and command, lists of installed Python and conda
packages, and copies of the run script and configuration. With ``require_clean=True``, it refuses to run if pollux
has uncommitted changes.

Catalogs also store the commit in their metadata (HDF5 attributes or FITS headers): ``POLLUXV`` (first 12
characters) and ``DIRTY`` (uncommitted changes).
