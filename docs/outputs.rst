Outputs
=======

Layout
------

A root directory holds the products of one simulation on one footprint (e.g. ``{outputs}/{simulation}/nside2048``), for
any number of seeds: its ``footprint/`` directory, shared by the seeds, and one directory per seed. Each object adds
its part of the paths: :class:`~pollux.io.files.PolluxIO` knows the root directory only, the shell builder adds its
seed, the baryon builder its tag, the lensing builder its n(z), the catalogs their tag. Every file follows one rule,
``{product directory}[/{folder}]/{name}.{ext}``, with intermediate files at the same relative path under
``intermediate/`` of their seed:

- ``footprint/``: the footprint (its ``config.yaml``), its padded mask and weights of the harmonic transforms
  (``padded_mask.fits``, ``alm_weights.fits``, with padding), and the NaMaster workspaces of the maps
  (``wsp_b{binning}.wsp``), see :class:`~pollux.footprint.Footprint`;
- ``seed{seed}/``: the products of one realization:

  - ``dmo/``: the dark-matter-only products:

    - ``shells/``: shell maps (``shell_{iii}.fits``), their Cls (``cls.fits``), the table of the shells (``shells.fits``:
      redshift and distance edges, mean densities), and their smoothed density and tidal maps
      (``smoothed/{s}Mpc/density_{iii}``, ``t11_{iii}``, ...);
    - ``kappa/{nz}/``: convergence and shear maps of the tomographic bins (``kappa_bin{b}``, ``gamma1_bin{b}``, ...), and
      their redshift distributions (``nzs.txt``);
    - ``kappa/snapshots/``: convergence and shear maps with one source plane per shell (``kappa_{iii}``, ...);
    - ``catalogs/{nz}/{tag}/``: source catalogs (``cat_bin{b}.fits``) and their statistics (``cat_2pt.fits``,
      ``cat_cls.fits``, and ``cat_2pt_{shapes}.fits``, ``cat_cls_{shapes}.fits`` with the shapes ``'ia'``,
      ``'shear_plus_ia'`` and ``'observed'``, see :doc:`statistics`);

  - ``dmb/{tag}/``: the same products for the baryon model ``tag``;
  - ``halos/``: halo catalogs, shared by all baryon models;
  - ``intermediate/``: files needed only to make or resume the products (sub-volume maps, spherical harmonics,
    catalogs of each shell, catalog fields and workspaces of the catalog Cls, statistics of each bin pair), at the
    same relative paths;
  - ``code_version/``: the records of the runs (see :doc:`pipeline`).

The root directory of a run is up to the user: its footprint is checked like the other products (below), so a root
directory never mixes footprints.

Configuration of products
-------------------------

Each product directory, the footprint's included, has a ``config.yaml`` with its parameters (for shells and halos, also
the simulation and its cosmology), its inputs and the code version that made it, and its plots in ``plots/``. Products are resumed in
place, and a directory made with other parameters raises an error instead of being mixed.

Index and loader
----------------

``python -m pollux.io.index {root_dir}`` lists the products under a directory, with their main parameters.

:class:`pollux.io.PolluxLoader <pollux.io.loader.PolluxLoader>` reads the products of one realization (a root directory and a seed) from their files only, without creating
builders, so that it only needs the I/O dependencies (numpy, healpy, healsparse, pyccl, sacc, astropy and PyYAML, but not NaMaster, TreeCorr, BaryonForge or Parsl):

.. code-block:: python

    from pollux import PolluxLoader

    dmo = PolluxLoader('/path/to/outputs/simulation/nside2048', seed=0, catalog_tag='b2')
    dmb = PolluxLoader('/path/to/outputs/simulation/nside2048', seed=0, baryon_tag='S19', catalog_tag='b2')

    density = dmo.load_density(30)              # density contrast of shell 30
    kappa = dmb.load_kappa(4)                   # convergence of tomographic bin 4
    gamma1, gamma2 = dmb.load_shear(4)
    cls = dmb.load_kappa_cls()                  # SACC
    xi = dmb.load_catalog_2pt(shapes='ia')      # SACC of the 2pt functions of the intrinsic alignments

The ``nz`` of the loader names the redshift distributions of the convergence maps and catalogs (``'SRDv1_nz'`` by
default), and ``loader.footprint`` holds the parameters of the footprint (its ``config.yaml``). The loader is a
:class:`~pollux.io.files.PolluxIO`: :meth:`~pollux.io.files.PolluxIO.path` gives any file by product and name, and the
readers by format (``read_map``, ``read_alm``, ``read_table``, ``read_sacc``, ``read_npy``) read it:

.. code-block:: python

    gamma1 = dmb.read_map(dmb.path('kappa', 'gamma1', 'bin4'))                     # seed0/dmb/S19/kappa/SRDv1_nz/gamma1_bin4.fits
    density = dmo.read_map(dmo.path('shells', 'density', '030', smooth_Mpc=2.))    # seed0/dmo/shells/smoothed/2Mpc/density_030.fits
    alm = dmo.read_alm(dmo.path('shells', 'alms', '030', intermediate=True))       # seed0/intermediate/dmo/shells/alms_030.fits
    config = dmb.read_config('catalogs')                                          # config.yaml of the catalogs
