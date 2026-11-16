Intrinsic alignments
====================

:class:`~pollux.ia.IACalculator` adds intrinsic alignments to the catalogs, from the tidal field of each galaxy's
shell interpolated at its position. The tidal field ``t11``, ``t22``, ``t12`` of a shell
(:meth:`~pollux.shells.ShellBuilder.compute_tidal_tensor_maps`) is that of its density contrast, transverse to the
line of sight, and can be smoothed (see :doc:`catalogs`).

Models
------

``model`` is one of:

- ``'TA'``, tidal alignment: ``e = C1 s``, with ``s`` the traceless tidal field, ``e1 = C1 (t11 - t22)`` and
  ``e2 = C1 t12`` (columns ``e1_ia_ta``, ``e2_ia_ta``);
- ``'TT'``, tidal torquing: ``e = C2 (s s)``, quadratic in the tidal field (columns ``e1_ia_tt``, ``e2_ia_tt``);
- ``'TATT'``: both.

The coefficients ``C1`` and ``C2`` are those of CCL (``pyccl.nl_pt.translate_IA_norm``), with the amplitudes
``A_TA`` and ``A_TT`` of ``model_params``, so that predictions use the same amplitudes. Since the tidal field is
that of the nonlinear density at the redshift of the shell, ``A_TA`` alone gives the nonlinear alignment model
(NLA).

Density weighting
-----------------

Galaxies placed with ``1 + b delta`` (see :doc:`catalogs`) already weight the alignment by the density: summed over
galaxies, the shapes are those of ``C1 (1 + b delta) s``, the density-weighted tidal alignment (δ-NLA) of TATT with
``b_TA = b``. ``delta_b_TA`` changes this weighting: the alignment of each galaxy is multiplied by
``(1 + (b + delta_b_TA) delta) / (1 + b delta)``, so that ``b_TA = b + delta_b_TA`` (``delta_b_TT`` likewise for
the tidal torquing term). Uniform catalogs have no density weighting unless ``delta_b_TA`` is set.

Predictions
-----------

:attr:`~pollux.catalog.CatalogSourceGenerator.ia_tracers` gives the tracers of the alignments of each bin, with
amplitudes ``a1 = A_TA``, ``a1delta = b_TA a1`` and ``a2 = A_TT``
(:meth:`~pollux.ia.IACalculator.tracer_amplitudes`). NLA is computed with the matter power spectrum of the
predictions, and the other terms with perturbation theory (FAST-PT through CCL), which also gives their B modes.
Their 2pt functions (II, GI, gI) are measured and predicted with the ``shapes`` options of :doc:`statistics`.
