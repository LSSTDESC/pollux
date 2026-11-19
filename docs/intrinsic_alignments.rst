Intrinsic alignments
====================

:class:`~pollux.ia.CatalogAugmentorIA` adds intrinsic alignments to the catalogs, from the tidal field of each galaxy's
shell interpolated at its position. The tidal field ``t11``, ``t22``, ``t12`` of a shell
(:meth:`~pollux.shells.ShellBuilder.compute_tidal_tensor_maps`) is that of its density contrast, transverse to the
line of sight, and can be smoothed (see :doc:`catalogs`). A shell has no line-of-sight component, so that
:math:`t_{11} + t_{22} = \delta / 3`.

Models
------

``model`` is one of:

- ``'TA'``, tidal alignment, linear in the tidal field (columns ``e1_ia_ta``, ``e2_ia_ta``);
- ``'TT'``, tidal torquing, quadratic in the tidal field (columns ``e1_ia_tt``, ``e2_ia_tt``);
- ``'TATT'``: both.

With :math:`t_{11}`, :math:`t_{22}`, :math:`t_{12}` the transverse tidal field of the galaxy's shell,

.. math::

    e^{\rm TA} = C_1 \left(t_{11} - t_{22},\; t_{12}\right), \qquad
    e^{\rm TT} = C_2 \left(t_{11}^2 - t_{22}^2,\; t_{12} (t_{11} + t_{22})\right)

(see :meth:`~pollux.ia.CatalogAugmentorIA.augment_catalog`).

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

:attr:`~pollux.catalog.CatalogGenerator.ia_tracers` gives the tracers of the alignments of each bin, with
amplitudes ``a1 = A_TA``, ``a1delta = b_TA a1`` and ``a2 = A_TT``
(:meth:`~pollux.ia.CatalogAugmentorIA.tracer_amplitudes`). NLA is computed with the matter power spectrum of the
predictions, and the other terms with perturbation theory (FAST-PT through CCL), which also gives their B modes.
Their 2pt functions (II, GI, gI) are measured and predicted with the ``shapes`` options of :doc:`statistics`.
