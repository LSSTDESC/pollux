Lensing
=======

Lensing builders (:mod:`pollux.lensing`) make convergence maps by summing the density contrast of the shells with
lensing weights, in the Born approximation (:class:`~pollux.lensing.BornLensingBuilder`), then shear maps from
them.

Convergence
-----------

``nzs`` sets the sources:

- a list of redshift distributions, one per tomographic bin: ``(z, n(z))`` tuples or functions of ``z`` (they need
  not be normalized). The maps of bin ``i`` are ``kappa_bin{i}``, and ``tag`` names the set of distributions
  (outputs in ``kappa/{tag}/``);
- ``'zsnapshots'``: one source plane at the far edge of each shell, as made by
  :meth:`~pollux.shells.ShellBuilder.snapshot_lensing_builder` (:class:`~pollux.lensing.SnapshotBornLensingBuilder`,
  outputs in ``kappa/snapshots/``). Catalogs interpolate these maps between the shells around each galaxy.

The weight of each shell is the integral of the lensing kernel over the shell (``multiple_planes=False``, default),
or that of a thin lens plane at the mean redshift of the shell (``multiple_planes=True``). The mean density of each
shell is that expected from the simulation, or measured on its map with ``nbar_from_maps=True``.

``construct_maps`` builds the convergence maps.

Shear
-----

:meth:`~pollux.lensing.LensingBuilder.compute_shear_maps` derives the shear ``gamma1``, ``gamma2`` from the
spherical harmonics of the convergence, with the spin-2 factors of the full sky. With ``smooth_Mpc``, both are
smoothed by a Gaussian of that comoving scale.

Noise
-----

``sigma_e`` and ``nbar`` (galaxies per arcmin²) set the shape noise of the predicted power spectra of the maps,
for error bars. They do not change the maps.
