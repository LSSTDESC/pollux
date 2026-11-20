Statistics and predictions
==========================

Every product is measured and compared to a prediction made with CCL for the same tracers, binning and smoothing.
Measurements are saved as `SACC <https://github.com/LSSTDESC/sacc>`_ files next to the product, and plots against
the predictions in its ``plots/`` folder.

Power spectra of maps
---------------------

Map builders (shells, convergence) measure angular power spectra with NaMaster:

- :meth:`~pollux.map_builder.MapBuilder.compute_alms` computes the spherical harmonics of the maps, on the padded
  and apodized footprint if any (see :doc:`shells`), and caches them;
- :meth:`~pollux.statistics.maps.MapStatistics.compute_cls` computes the decoupled bandpowers of pairs of maps
  (``pairs``, or all with ``auto_only``), with the binning ``b`` (by default ``nlb`` = 20 bandpowers log-spaced from ℓ = 8 to 2 ``nside``) and a NaMaster
  workspace cached per footprint;
- :meth:`~pollux.statistics.maps.MapStatistics.predict_cls` predicts them, convolved with the bandpower windows of the
  workspace (``binning='wsp'``) or averaged in each bin, with the pixel window (``f_ell``), and optionally Gaussian
  error bars with the noise of the maps;
- :meth:`~pollux.statistics.maps.MapStatistics.plot_cls` plots both.

Builders also compute moments of smoothed maps (:meth:`~pollux.statistics.maps.MapStatistics.compute_map_moments`), and
Gaussian maps with the predicted spectra (:meth:`~pollux.statistics.maps.MapStatistics.generate_gaussian_maps`).

Statistics of catalogs
----------------------

Catalogs measure, for all pairs of tomographic bins:

- ``'clustering'``: the angular correlation of galaxies ``w(θ)``, against randoms (``nran_factor`` per galaxy,
  shared by all bins with ``shared_randoms``);
- ``'shear'``: the shear correlations ``ξ±(θ)``;
- ``'shear_clustering'``: galaxy-galaxy lensing ``γt(θ)``, for all ordered pairs of lens and source bins.

:meth:`~pollux.statistics.catalogs.CatalogStatistics.compute_2pt` measures them with TreeCorr between ``theta_min`` and
``theta_max`` (``ntheta`` logarithmic bins, ``bin_slop``), and :meth:`~pollux.statistics.catalogs.CatalogStatistics.compute_cls`
measures the corresponding power spectra with NaMaster, from maps of the catalogs. ``subsample`` uses a random
fraction of the galaxies, and ``remove_mean`` subtracts the mean shear of each bin. Error bars are shot noise only.

``shapes`` sets the shapes of the shear legs, of the 2pt functions and of the Cls alike (files ``cat_cls[_{shapes}].fits``
for the Cls):

- ``'shear'`` (default): the lensing shear (``shear_cols``, ``gamma1`` and ``gamma2`` by default), in
  ``cat_2pt.fits``;
- ``'ia'``: the intrinsic alignments alone, II for ``ξ±`` and gI for ``γt``, in ``cat_2pt_ia.fits``;
- ``'shear_plus_ia'``: their sum, without shape noise, in ``cat_2pt_shear_plus_ia.fits``;
- ``'observed'``: the observed ellipticities ``e1_obs``, ``e2_obs`` of the catalogs (see :doc:`catalogs`), with
  shape noise, in ``cat_2pt_observed.fits``. Their prediction is that of ``'shear_plus_ia'`` (of ``'shear'`` if
  they do not include the intrinsic alignments), the shape noise having zero mean and the reduced shear being
  approximated by the shear; the error bars include the shape noise.

The first three are measured on the same galaxies: with :math:`\gamma` the shear and :math:`e^I` the intrinsic
alignments, :math:`\xi^{\gamma + e^I} = \xi^{GG} + \xi^{GI} + \xi^{IG} + \xi^{II}`, so that GI (with IG) is the
difference ``shear_plus_ia - shear - ia``, which the
plots of ``'ia'`` also show. Plots are named after the shapes: ``xi_shear``, ``xi_ia`` (II) and
``xi_shear_plus_ia`` for ``ξ±``, with ``_x_density`` for ``γt`` (``xi_ia_x_density`` is gI), ``xi_shear_x_ia`` for GI,
``xi_observed`` for the observed ellipticities, and ``xi_density`` for ``w(θ)``; those of the Cls are ``cl_{shapes}_ee``,
``cl_{shapes}_x_density`` and ``cl_density``.

Predictions
-----------

Predictions use two objects of :mod:`pollux.cosmo`, shared by builders and catalogs:

- :class:`~pollux.cosmo.Tracer` describes an observable: a density (with its linear bias), a shear, intrinsic
  alignments (with their amplitudes) or a convergence, with its redshift distribution and smoothing. Builders and
  catalogs give theirs (``tracers``, ``density_tracers``, ``lensing_tracers``, ``ia_tracers``);
- :class:`~pollux.cosmo.TracerSum` sums tracers of one field, e.g. the lensing shear and intrinsic alignments of a
  bin for the shapes ``'shear_plus_ia'``: predictions being bilinear, their spectra are the sums of those of the terms
  (GG + GI + IG + II), and so are the bandpowers, correlation functions and Gaussian errors of the summed field;
- :class:`~pollux.cosmo.Theory` predicts their power spectra (:meth:`~pollux.cosmo.Theory.angular_cl`, in the
  Limber approximation), bandpowers (:meth:`~pollux.cosmo.Theory.predict_cls`) and correlation functions
  (:meth:`~pollux.cosmo.Theory.predict_2pt`, by Hankel transform), with a matter power spectrum: CCL's nonlinear
  one, or that of the baryon model (see :doc:`baryons`).

Bandpowers are predicted as measured: with the bandpower windows :math:`W_{b\ell}` of the NaMaster workspace and
the pixel window :math:`p_\ell`, :math:`C_b = \sum_\ell W_{b\ell}\, p_\ell^2\, C_\ell`
(:meth:`~pollux.cosmo.Theory.binned_cl`). The pixel window is also in the correlation functions of the catalogs,
from :math:`p_\ell^2 C_\ell` (``pixel_window`` of :meth:`~pollux.statistics.catalogs.CatalogStatistics.predict_cls` and
:meth:`~pollux.statistics.catalogs.CatalogStatistics.predict_2pt`), since galaxies are placed in the pixels of the density maps and
the fields are interpolated from pixelized maps. The bilinear interpolation probably adds a window of its own, which the
predictions don't include: ``ξ-`` remains low below a few pixels.

A tracer smoothed by a Gaussian of comoving scale ``s`` multiplies the matter power spectrum by
``exp(-k² s² / 2)``, which is exact in Limber for a transverse smoothing. Scales that change from shell to shell
are applied shell by shell, and a smoothing cannot be smaller than the pixel
(:meth:`~pollux.cosmo.Tracer.effective_smooth_Mpc`). Intrinsic alignments beyond NLA use perturbation theory, with
their B modes included in ``ξ±`` and in the bandpowers of two spin-2 fields (see :doc:`intrinsic_alignments`).

Plots
-----

Plots (:func:`pollux.statistics.plotting.plot_pairs`) show one panel per pair of bins, in modes:

- ``'values'``: data and prediction;
- ``'ratio'``: data over prediction, minus 1;
- ``'ratio_to_reference'``: the ratios of the data and of the predictions to those of another product, e.g. a
  baryonified catalog to the dark-matter-only one.

Vertical lines mark the reference scales of each pair (:meth:`~pollux.statistics.maps.MapStatistics.reference_scales`):
the smoothing, the pixel size and the size of the simulation box at the distance of the tracers. Measurements are
not expected to follow predictions below the pixel or smoothing scales, nor at angles approaching the box size.

Validation
----------

Test runs at NERSC compare every product to its prediction, above the pixel and smoothing scales and below the
box size. On the full sky at nside 8192:

- convergence power spectra agree to 1–2% for 100 < ℓ < 5000, in every pair of tomographic bins;
- the cosmic shear ``ξ+`` of the catalogs agrees to a few percent from 0.5 to 100 arcmin;
- with the lognormal smoothing (``max_bias_error`` of 10%), the galaxy clustering ``w(θ)`` agrees to 2–8%,
  and galaxy-galaxy lensing ``γt`` to a few percent.

With baryons (at nside 2048), the suppression of the convergence power spectra and of ``ξ±`` follows the prediction
of the same baryon model. The intrinsic alignments of NLA catalogs agree with their predictions (II, GI, gI) to
better than about 10%.
