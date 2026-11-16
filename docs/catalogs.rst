Catalogs
========

Catalog generators (:mod:`pollux.catalog`) draw tomographic source catalogs from the shells and lensing maps: for
each tomographic bin of a lensing builder, galaxies get a redshift, a position, the fields interpolated there, and
the columns added by augmentors (intrinsic alignments, observed shapes). Each bin is saved as a FITS catalog
``cat_bin{i}.fits``.

Sampling
--------

Redshifts are drawn from the redshift distribution of the bin, which assigns each galaxy to a shell. Positions are
then drawn by one of:

- :class:`~pollux.catalog.CatalogSourceUniform`: uniformly over the footprint;
- :class:`~pollux.catalog.CatalogSourceClusteredMultinomial`: from the density of the galaxy's shell, with a fixed
  number of galaxies per bin;
- :class:`~pollux.catalog.CatalogSourceClusteredPoisson`: a Poisson number of galaxies per pixel and shell, of mean
  set by the density, so that the total number fluctuates. This one can be generated shell by shell
  (``generate(ishell=...)``), then merged (``combine_shells``), to split the work.

``nbar_input`` is the number of galaxies per arcmin² (one value, or one per bin). Within a pixel, positions are
uniform. Random numbers come from separate streams per realization seed, bin and shell, so that a catalog does not
depend on how its generation is split.

Galaxy bias
-----------

Clustered catalogs place galaxies with a relative density set by ``bias`` (one value, or one per bin) and
``bias_model``:

- ``'linear'`` (default): ``1 + b delta``. With ``b > 1``, this is negative in underdense pixels, where it is
  clipped to zero, with a warning giving the fraction of pixels affected. Clipping lowers the effective bias on large
  scales (and the mean density of Poisson catalogs), more so at high resolution where the density contrast varies
  most;
- ``'power_law'``: ``(1 + delta)^b``, normalized over the shell. It is positive everywhere and has bias ``b`` on
  large scales, but is not linear on small scales.

Smoothing
---------

Fields can be smoothed by a Gaussian of comoving scale ``smooth_Mpc``, before the galaxies are placed and the fields
interpolated:

- ``smooth_fields``: the smoothed fields, among ``positions`` (the density placing galaxies), ``density``, ``t11``,
  ``t22``, ``t12``, ``kappa``, ``gamma1`` and ``gamma2``;
- ``smooth_Mpc``: one scale, one scale per shell, or a dictionary of either per field. Fields made together (the
  tidal field components, the shear components) share their scale.

Smoothing the density reduces clipping, so that the galaxies keep the bias they are given.
:meth:`~pollux.cosmo.Theory.lognormal_smoothing` chooses one scale per shell: the smallest round scale, at least
the pixel size, for which a lognormal density field predicts a large-scale bias deficit below ``max_bias_error``
(or a clipped fraction below ``max_clip``). The lognormal model overestimates this deficit, so the scales are
conservative. Predictions include the smoothing of each tracer (see :doc:`statistics`).

Interpolated fields
-------------------

``interp_fields`` lists the fields interpolated at the galaxies (fields needed by the augmentors are added):

- ``kappa``, ``gamma1``, ``gamma2`` come from the convergence and shear of the source planes at the far edges of
  the shells (:meth:`~pollux.shells.ShellBuilder.snapshot_lensing_builder`), interpolated linearly in comoving
  distance between the two planes around each galaxy;
- ``density``, ``t11``, ``t22``, ``t12`` are averages over a shell, and take the value of the galaxy's own shell.

``interp_method`` interpolates the maps across pixels (``'bilinear'``, default) or takes the value of the pixel
(``'nearest'``).

Augmentors
----------

Augmentors add columns computed from the others:

- :class:`~pollux.ia.IACalculator`: intrinsic alignments from the tidal field (columns ``e1_ia_ta``, ``e2_ia_ta``,
  and ``e1_ia_tt``, ``e2_ia_tt``), see :doc:`intrinsic_alignments`;
- :class:`~pollux.ia.CatalogAugmentorShear`: reduced shear ``g1``, ``g2``, magnification ``mu``, intrinsic shapes
  ``e1_int``, ``e2_int`` (Gaussian of dispersion ``sigma_e``), and observed shapes ``e1_obs``, ``e2_obs``: the
  intrinsic shape plus the intrinsic alignments, sheared by the reduced shear.

Columns and headers
-------------------

Catalogs have ``ra``, ``dec``, ``z``, the comoving distance ``chi``, the interpolated fields and the augmentor
columns. Their headers record the options of the generator and the code version (``POLLUXV``, the git
commit, and ``DIRTY`` for uncommitted changes).
