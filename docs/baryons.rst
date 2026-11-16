Baryons
=======

:class:`~pollux.baryons.BaryonifiedShellBuilder` applies a baryonification model of
`BaryonForge <https://github.com/DhayaaAnbajagane/BaryonForge>`_ to the shells: matter is displaced around each
halo so that its profile goes from dark matter only to that of dark matter, gas and stars. The result is a shell
builder like the others: lensing and catalogs use it in place of the dark-matter-only shells.

Inputs
------

- the dark-matter-only shell builder, whose maps are displaced, and whose options (isometries, footprint, ...) are
  kept;
- a halo builder giving the halos of each shell: a :class:`~pollux.shells.SnapshotShellBuilder` with a
  :class:`~pollux.reader.HACCHaloReader` and the same options, whose ``construct_catalogs`` writes one halo catalog
  per shell;
- the dark-matter-only and baryonified halo profiles, e.g. those of Schneider et al. (2019) from
  :func:`~pollux.baryons.s19_profiles`, and the halo mass definition and column (``mass_def``, ``mass_name``,
  ``mass_unit``).

Halos lighter than ``M_min`` (by default 10 particles) are left unchanged. The profiles are convolved with the
HEALPix pixel, since the displacement is applied to the shell maps.

Matter power spectrum
---------------------

Predictions for baryonified products use the nonlinear matter power spectrum multiplied by the halo-model boost
``P_dmb / P_dmo`` of the same profiles, with the same mass cut
(:meth:`~pollux.baryons.BaryonifiedShellBuilder.matter_power_spectrum`). Every builder and catalog made from the
baryonified shells uses it.

Outputs
-------

The baryonified shells, and all products made from them, go to ``dmb/{tag}/`` next to the dark-matter-only
products in ``dmo/`` (see :doc:`outputs`), with the parameters of the profiles in their ``config.yaml``
(``profile_parameters``), so that shells of different models are not mixed.
