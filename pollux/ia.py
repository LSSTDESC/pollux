"""Catalog augmentors: intrinsic alignments (:class:`IACalculator`), and reduced shear and observed shapes (:class:`CatalogAugmentorShear`)."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pyccl

from .catalog import CatalogAugmentor


class IACalculator(CatalogAugmentor):
    """Calculator for intrinsic alignment (IA) contributions to galaxy shapes.

    This class computes intrinsic alignment shear components from density and
    tidal fields using various IA models including TA (Tidal Alignment, also
    known as Nonlinear Linear Alignment or NLA), TT (Tidal Torque), or a
    combination of both. It augments galaxy catalogs by adding IA shear columns
    based on interpolated density and tidal fields at galaxy positions.

    Field names are resolved automatically: the catalog always stores
    interpolated values under base names (``t11``, ``t22``, ``t12``,
    ``density``), regardless of any Gaussian smoothing applied at map-loading
    time.

    Parameters
    ----------
    model : str, optional
        IA model to use. Supported values are 'TA', 'TT', or 'TATT'.
        Default is 'TA'.
    model_params : dict, optional
        Model parameters:

        - ``'A_TA'``: amplitude of the TA model (default 1.0);
        - ``'A_TT'``: amplitude of the TT model (default 1.0);
        - ``'delta_b_TA'``: bias of the density-dependent TA model (default 0.0);
        - ``'delta_b_TT'``: bias of the density-dependent TT model (default 0.0).

    Notes
    -----
    The TA (Tidal Alignment) model computes IA shear proportional to the
    tidal field, while the TT model uses quadratic tidal field terms. The
    normalization is CCL's (:meth:`coefficients`), with Cbar1 = 5e-14 h^-2 Msun^-1 Mpc^3
    (Brown et al. 2002), so that theory predictions of CCL use the same amplitudes.

    References
    ----------
    For the full model including density-dependent modulation, see
    https://arxiv.org/abs/2509.25166v1
    """

    def __init__(self, model: str = 'TA', model_params: dict | None = None) -> None:
        if model not in ['TA', 'TT', 'TATT']:
            raise ValueError(f"IA model '{model}' not recognized. Supported: 'TA', 'TT', 'TATT'.")

        self.model = model
        self.model_params = model_params or {}
        self.name = f"IA_{model}"

        required_fields = ['t11', 't22', 't12']
        if self.model_params.get('delta_b_TA', 0.0) != 0.0 or self.model_params.get('delta_b_TT', 0.0) != 0.0:
            required_fields.append('density')
        self.required_interp_fields = required_fields

        new_columns = []
        if 'TA' in model:
            new_columns += ['e1_ia_ta', 'e2_ia_ta']
        if 'TT' in model:
            new_columns += ['e1_ia_tt', 'e2_ia_tt']
        self.new_columns = new_columns

    def augment_catalog(self, cosmo: pyccl.Cosmology, catalog: dict[str, np.ndarray], **kwargs: Any) -> None:
        """Compute and add intrinsic alignment shear components to a catalog.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology.
        catalog : dict[str, np.ndarray]
            Catalog of one tomographic bin, updated in place. Must include ``'z'``
            (redshift), ``'t11'``, ``'t22'``, ``'t12'`` (tidal field), and
            ``'density'`` (density contrast) if a ``delta_b`` parameter is non-zero.
        **kwargs
            Unused (other augmentor options).

        Notes
        -----
        Adds ``'e1_ia_ta'``, ``'e2_ia_ta'`` (TA model) and/or ``'e1_ia_tt'``,
        ``'e2_ia_tt'`` (TT model), depending on the model.
        """
        z = catalog['z']
        t11 = catalog['t11']
        t22 = catalog['t22']
        t12 = catalog['t12']
        C1, C2 = self.coefficients(cosmo, z)

        if 'TA' in self.model:
            # TA (Tidal Alignment) model: e = C1 s, with s the traceless tidal field (e1 = s11 - s22, e2 = 2 s12 = t12)
            catalog['e1_ia_ta'] = C1 * (t11 - t22)
            catalog['e2_ia_ta'] = C1 * t12

        if 'TT' in self.model:
            # TT (tidal torque) model: e = C2 (s s), with (s s)_11 - (s s)_22 = s11^2 - s22^2 and 2 (s s)_12 = t12 (s11 + s22)
            catalog['e1_ia_tt'] = C2 * (t11**2 - t22**2)
            catalog['e2_ia_tt'] = C2 * t12 * (t11 + t22)

        # delta-TA/TT model
        delta_b_TA = self.model_params.get('delta_b_TA', 0.0)
        delta_b_TT = self.model_params.get('delta_b_TT', 0.0)

        if delta_b_TA != 0.0 or delta_b_TT != 0.0:
            delta = catalog['density']
            bias = kwargs.get('bias', 0.0)

            for model, delta_b in (('TA', delta_b_TA), ('TT', delta_b_TT)):
                if delta_b != 0.0 and model in self.model:
                    factor = _density_factor(delta, bias, delta_b)
                    catalog[f'e1_ia_{model.lower()}'] *= factor
                    catalog[f'e2_ia_{model.lower()}'] *= factor

    def config(self) -> dict:
        """Return the type and parameters of the augmentor, for the ``config.yaml`` of the catalogs."""
        return {'type': type(self).__name__, 'model': self.model, 'model_params': self.model_params}

    def coefficients(self, cosmo: pyccl.Cosmology, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the coefficients of the TA and TT terms, as CCL's :func:`pyccl.nl_pt.translate_IA_norm`.

        ``C1 = -A_TA Cbar1 rho_crit Omega_m / D(z)`` and ``C2 = 5 A_TT Cbar1 rho_crit Omega_m / D(z)^2``
        (DES convention), with ``Cbar1 = 5e-14 h^-2 Msun^-1 Mpc^3``, the critical density today in
        ``h^2 Msun Mpc^-3`` (so that ``Cbar1 rho_crit = 0.0139``), and the growth factor ``D``
        normalized to 1 today. The tidal field is that of the (nonlinear) density contrast at
        redshift ``z``, hence the growth factors: the NLA model with ``A_TA``.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology.
        z : array_like
            Redshifts.

        Returns
        -------
        C1, C2 : ndarray
            Coefficients at ``z`` (zero for a term not in the model).
        """
        z = np.asarray(z, dtype=float)
        a1 = self.model_params.get('A_TA', 1.0) if 'TA' in self.model else 0.
        a2 = self.model_params.get('A_TT', 1.0) if 'TT' in self.model else 0.
        C1, _, C2 = pyccl.nl_pt.translate_IA_norm(cosmo, z=z, a1=a1, a2=a2)
        return C1 * np.ones_like(z), C2 * np.ones_like(z)

    def theory_amplitudes(self, bias: float) -> dict[str, float]:
        """Return the amplitudes of the model for the predictions (``ia`` of :class:`pollux.cosmo.Tracer`).

        Galaxies are placed with ``1 + bias delta``, and with ``delta_b_TA`` the tidal alignment
        of each galaxy is multiplied by ``(1 + (bias + delta_b_TA) delta) / (1 + bias delta)``: the
        shapes summed over galaxies are those of ``C1 (1 + (bias + delta_b_TA) delta) s``, the
        density-weighted term of TATT with ``a1delta = (bias + delta_b_TA) a1``. The density
        weighting of the tidal torquing term (``bias`` and ``delta_b_TT``) is not modelled.

        Parameters
        ----------
        bias : float
            Linear bias of the galaxies (that placing them).

        Returns
        -------
        dict
            ``a1``, and ``a2`` and ``a1delta`` for the terms beyond NLA.
        """
        a1 = self.model_params.get('A_TA', 1.0) if 'TA' in self.model else 0.
        amplitudes = {'a1': a1, 'a1delta': (bias + self.model_params.get('delta_b_TA', 0.0)) * a1}
        if 'TT' in self.model:
            amplitudes['a2'] = self.model_params.get('A_TT', 1.0)
        return amplitudes


def _density_factor(delta: np.ndarray, bias: float, delta_b: float) -> np.ndarray:
    """Return the density modulation of the intrinsic alignments, ``max(0, 1 + (bias + delta_b) delta) / (1 + bias delta)``.

    Galaxies are placed with the density smoothed for their positions, and ``delta`` (the
    ``density`` field of the catalog) may have another smoothing: where ``1 + bias delta <= 0``,
    the factor is 1 (no modulation) rather than infinite or of the wrong sign.
    """
    weight = 1. + bias * delta
    factor = np.ones_like(delta, dtype=float)
    ok = weight > 0.
    factor[ok] = np.maximum(0., 1. + (bias + delta_b) * delta[ok]) / weight[ok]
    return factor


class CatalogAugmentorShear(CatalogAugmentor):
    """Augmentor that computes shear-derived quantities and shapes.

    Options include reduced shear, magnification, intrinsic shapes, and
    observed shapes. Intrinsic alignment (IA) shear contributions are included
    when available and requested.

    Parameters
    ----------
    compute_reduced_shear : bool, optional
        If True, compute reduced shear g1, g2. Default is True.
    compute_magnification : bool, optional
        If True, compute magnification mu. Default is False.
    generate_intrinsic : bool, optional
        If True, generate random intrinsic shapes e1_int, e2_int. Default is False.
    compute_observed : bool, optional
        If True, compute observed shapes by adding lensing and IA contributions
        to intrinsic shapes. Requires generate_intrinsic=True. Default is False.
    include_ia : bool, optional
        If True, include IA contributions in lensed shapes. Default is True.
    sigma_e : float, optional
        Standard deviation of the intrinsic ellipticity distribution. Default is 0.3.
    seed : int or None, optional
        Extra seed of the intrinsic shapes, if not None. The shapes of the galaxies of
        tomographic bin ``ikappa`` in shell ``ishell`` are drawn from a stream of their own,
        seeded by the seed of the realization (given by the catalog generator), ``ikappa`` and
        ``ishell``: they differ between realizations, and do not depend on how the catalog is
        split into shells. Default is None.
    """

    # stream of the random numbers of the intrinsic shapes, distinct from those of the catalog generators
    _SHAPE_STREAM = 10

    def __init__(
        self,
        compute_reduced_shear: bool = True,
        compute_magnification: bool = False,
        generate_intrinsic: bool = False,
        compute_observed: bool = False,
        include_ia: bool = True,
        sigma_e: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if compute_observed and not generate_intrinsic:
            raise ValueError("Intrinsic shapes must be generated to compute observed shapes.")

        self.compute_reduced_shear = compute_reduced_shear
        self.compute_magnification = compute_magnification
        self.generate_intrinsic = generate_intrinsic
        self.compute_observed = compute_observed
        self.include_ia = include_ia
        self.sigma_e = sigma_e
        self.seed = seed
        self.name = 'shear_derived'
        self.required_interp_fields = ['kappa', 'gamma1', 'gamma2']

        new_columns = []
        if compute_reduced_shear:
            new_columns += ['g1', 'g2']
        if compute_magnification:
            new_columns += ['mu']
        if generate_intrinsic:
            new_columns += ['e1_int', 'e2_int']
        if compute_observed:
            new_columns += ['e1_obs', 'e2_obs']
        self.new_columns = new_columns

    def config(self) -> dict:
        """Return the type and parameters of the augmentor, for the ``config.yaml`` of the catalogs.

        The parameters setting the values of the shapes (``sigma_e``, ``include_ia``, ``seed``) are
        recorded when they differ from their defaults, so that the configurations of catalogs made
        before they were recorded stay valid.
        """
        defaults = inspect.signature(type(self)).parameters
        return {'type': type(self).__name__, **{k: getattr(self, k) for k in ('sigma_e', 'include_ia', 'seed') if getattr(self, k) != defaults[k].default}}

    def augment_catalog(self, cosmo: pyccl.Cosmology, catalog: dict[str, np.ndarray], ikappa: int = 0, realization_seed: int = 0, **kwargs: Any) -> None:
        """Compute derived shear quantities and add them to the catalog.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology (unused).
        catalog : dict[str, np.ndarray]
            Catalog of one tomographic bin, with ``'kappa'``, ``'gamma1'``,
            ``'gamma2'`` and ``'idx_shell'``; updated in place.
        ikappa : int, optional
            Tomographic-bin index, used to seed the intrinsic shapes.
        realization_seed : int, optional
            Seed of the realization (that of the shell builder), used to seed the intrinsic shapes.
        **kwargs
            Unused (other augmentor options).
        """
        kappa = catalog['kappa']
        gamma1, gamma2 = catalog['gamma1'], catalog['gamma2']

        if self.compute_magnification:
            # magnification 1 / ((1 - kappa)^2 - |gamma|^2)
            catalog['mu'] = 1. / ((1. - kappa)**2 - gamma1**2 - gamma2**2)

        if self.compute_reduced_shear:
            # reduced shear g = gamma / (1 - kappa)
            denom = 1. - kappa
            g1 = gamma1 / denom
            g2 = gamma2 / denom
            catalog['g1'] = g1
            catalog['g2'] = g2
        else:
            g1 = gamma1
            g2 = gamma2

        if self.generate_intrinsic:
            # Gaussian intrinsic shapes, drawn per shell so that they do not depend on how the generation is split
            idx_shell = catalog['idx_shell']
            catalog['e1_int'] = np.empty(len(idx_shell))
            catalog['e2_int'] = np.empty(len(idx_shell))
            for ishell in np.unique(idx_shell):
                w = np.flatnonzero(idx_shell == ishell)
                rng = np.random.default_rng([realization_seed, self._SHAPE_STREAM, ikappa, ishell] + ([] if self.seed is None else [self.seed]))
                catalog['e1_int'][w] = rng.normal(0., self.sigma_e, size=len(w))
                catalog['e2_int'][w] = rng.normal(0., self.sigma_e, size=len(w))

        if self.compute_observed:
            # source ellipticity, plus the intrinsic alignments if computed
            e_int = catalog['e1_int'] + 1j * catalog['e2_int']
            if self.include_ia:
                for model in ('ta', 'tt'):
                    if f'e1_ia_{model}' in catalog:
                        e_int += catalog[f'e1_ia_{model}'] + 1j * catalog[f'e2_ia_{model}']
            # observed ellipticity e = (g + e_s) / (1 + g* e_s)
            g = g1 + 1j * g2
            e_obs = (g + e_int) / (1. + np.conj(g) * e_int)
            catalog['e1_obs'] = e_obs.real
            catalog['e2_obs'] = e_obs.imag
