Shells
======

A shell is a range of comoving distance from the observer. Shell builders (:mod:`pollux.shells`) bin the
particles of a simulation into one HEALPix map per shell, from which the other products follow.

Maps are stored as HealSparse maps (NESTED, the default), or as dense full-sky HEALPix maps in RING ordering
(``map_format='healpix'``, ``.npy`` files, faster to read and write). They are computed as dense RING maps, as the
spherical harmonic transforms need them, and converted only when written or read in the other format. The maps of
the sub-volumes have their own format (``sub_format``): HealSparse maps of the non-empty pixels, or their RING pixels
and counts (``.npz``); the particles are binned in the matching ordering, so that they are written without
conversion.

Readers
-------

Readers (:mod:`pollux.reader`) give access to a simulation: particles of each snapshot by sub-volume
(``read_sub``), its cosmology (a CCL cosmology from its parameter file), box size and particle mass, and the
snapshots kept (``snap_sub_every_n`` keeps one out of ``n``, which sets the thickness of the shells).

- :class:`~pollux.reader.HACCReader`: snapshots of HACC (GenericIO files);
- :class:`~pollux.reader.HACCLightconeReader`: lightcone outputs of HACC;
- :class:`~pollux.reader.HACCHaloReader`: halo catalogs of HACC (halo properties files);
- :class:`~pollux.reader.SLICSReader`: snapshots of SLICS;
- :class:`~pollux.reader.SkySimReader`;
- :class:`~pollux.reader.SyntheticReader`: particles drawn uniformly in a box, without files, for tests and examples
  (none with ``nparticles_per_sub=0``, for a builder of predictions only).

Shell builders
--------------

Three builders make the shells from particles in different ways:

- :class:`~pollux.shells.SnapshotShellBuilder` fills each shell with the snapshot of the closest time, replicating
  its periodic box as many times as needed. Shell edges are placed between snapshots (``shell_edges_match``: at
  the middle in redshift, scale factor, comoving distance or volume). This is the builder of the pipeline.
- :class:`~pollux.shells.LightconeShellBuilder` slices the lightcone outputs of a simulation
  (:class:`~pollux.reader.HACCLightconeReader`), stacking the outputs that overlap each shell.
- :class:`~pollux.shells.SnapshotInterpShellBuilder` interpolates the position of each particle between consecutive
  snapshots to where it crosses the lightcone, with orphans for particles that move between sub-volumes.

All share the options of :class:`~pollux.shells.ShellBuilder`. Each particle sub-volume is binned on its own
(``construct_maps_sub``), then the sub-volumes are summed (``combine_maps_subs``); ``construct_maps`` does both.

Replicas and isometries
-----------------------

Replicating the same box along the line of sight repeats structures at the same positions. With
``apply_grid_isometry=True``, each shell gets a random isometry of the periodic box: a random translation, a
random rotation and a random flip of the axes. ``group_isometries=True`` uses the same isometry for consecutive
shells within one box size, so that structures are not cut at each shell edge. The ``seed`` sets the isometries:
each seed is a realization of the lightcone from the same simulation. An isometry moves the box and its replicas as
:math:`\mathbf{x}' = (\mathbf{x} + \mathbf{b}) A`, with :math:`A` a random rotation times random flips and
:math:`\mathbf{b}` a random translation (see :meth:`~pollux.shells.ShellBuilder.random_isometries`).

The alternative ``apply_replicate_rotflip`` applies instead one of the 48 rotations and flips of the cubic grid to
each replica, as in HACC.

Subsampling
-----------

``subsample_box`` keeps a fraction of the particles, to save time and memory: a number, the same for all shells, or
one fraction per shell:

- ``{'densitymax_z': z}``: shells beyond redshift ``z``, which have more particles per pixel, are subsampled down
  to the particle density per pixel of the shell at ``z``;
- ``{'clmax_r': r}``: the shot noise of each shell is ``r`` times below its theory power spectrum at the largest
  multipole.

The fractions are given by :meth:`~pollux.shells.ShellBuilder.subsample_ratio`. The shot noise of a shell is
``1 / (r n)``, with ``n`` its expected number of particles per steradian and ``r`` its fraction; that of a
convergence map follows from its weights (see :doc:`lensing`).

Footprint
---------

The :class:`~pollux.footprint.Footprint` given to the shell builder is that of all maps, shared by the builders made
from it, with the files of the root directory of the products (:class:`~pollux.io.files.PolluxIO`, see :doc:`outputs`).
Its ``mask`` is ``None`` for the full sky, ``'octant'``, or a HEALPix mask (array or FITS file, NESTED
ordering). Only the replicas overlapping a shell and the footprint are read. With ``padding_deg``, maps are built on
the footprint dilated by that angle, and harmonic transforms use the padded footprint apodized over the padding
(``apotype``), which limits the leakage at the edges. Measurements and catalogs use the footprint itself.

Derived maps
------------

From the count maps, builders derive:

- the density contrast relative to the cosmic mean (``field_map``);
- spherical harmonics (``compute_alms``), and maps smoothed by a Gaussian of fixed comoving scale
  (``compute_maps_from_alms(smooth_Mpc=...)``);
- the tidal field ``t11``, ``t22``, ``t12`` of each shell, transverse to the line of sight (``t11`` and ``t22``
  traceless in 3D, and ``t12`` twice the off-diagonal component, :math:`t_{12} = 2 s_{12}`, so that the tidal
  alignment is :math:`e_1 = C_1 (t_{11} - t_{22})`, :math:`e_2 = C_1 t_{12}`)
  (``compute_tidal_tensor_maps``), possibly smoothed;
- halo catalogs, with a halo reader (``construct_catalogs``);
- a lensing builder with one source plane at the far edge of each shell (``snapshot_lensing_builder``,
  see :doc:`lensing`).

Their power spectra, predictions and plots are described in :doc:`statistics`.
