"""Intrinsic alignments of the galaxies of catalogs (:class:`CatalogAugmentorIA`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyccl

from .catalog import CatalogAugmentor


class CatalogAugmentorIA(CatalogAugmentor):
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
        Model parameters, with the defaults of :attr:`PARAMETERS`:

        - ``'A_TA'``: amplitude of the TA model;
        - ``'A_TT'``: amplitude of the TT model;
        - ``'delta_b_TA'``: bias of the density-dependent TA model;
        - ``'delta_b_TT'``: bias of the density-dependent TT model.

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

    # model parameters and their defaults
    PARAMETERS = {'A_TA': 1.0, 'A_TT': 1.0, 'delta_b_TA': 0.0, 'delta_b_TT': 0.0}

    def __init__(self, model: str = 'TA', model_params: dict | None = None) -> None:
        if model not in ['TA', 'TT', 'TATT']:
            raise ValueError(f"IA model '{model}' not recognized. Supported: 'TA', 'TT', 'TATT'.")
        unknown = set(model_params or {}) - set(self.PARAMETERS)
        if unknown:
            raise ValueError(f"Unknown IA parameters {sorted(unknown)}, must be among {list(self.PARAMETERS)}")

        self.model = model
        self.model_params = model_params or {}  # as given, for the configuration of the catalogs
        self.parameters = {**self.PARAMETERS, **self.model_params}  # with the defaults
        self.name = f"IA_{model}"

        required_fields = ['t11', 't22', 't12']
        if self.parameters['delta_b_TA'] != 0.0 or self.parameters['delta_b_TT'] != 0.0:
            required_fields.append('density')
        self.required_interp_fields = required_fields

        new_columns = []
        if 'TA' in model:
            new_columns += ['e1_ia_ta', 'e2_ia_ta']
        if 'TT' in model:
            new_columns += ['e1_ia_tt', 'e2_ia_tt']
        self.new_columns = new_columns

    def augment_catalog(self, cosmo: pyccl.Cosmology, catalog: dict[str, np.ndarray], **kwargs: Any) -> None:
        r"""Compute and add intrinsic alignment shear components to a catalog.

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

        With :math:`C_1` and :math:`C_2` of :meth:`coefficients` and the tidal field :math:`t_{ab}` of the galaxy's shell
        (see :meth:`~pollux.shells.ShellBuilder.compute_tidal_tensor_maps`),

        .. math::

            e^{\rm TA} = C_1 \left(t_{11} - t_{22},\; t_{12}\right), \qquad
            e^{\rm TT} = C_2 \left(t_{11}^2 - t_{22}^2,\; t_{12} (t_{11} + t_{22})\right),

        each multiplied, with ``delta_b_TA`` (``delta_b_TT``), by the density weighting

        .. math::

            F = \frac{\max\left(0, 1 + (b + \Delta b)\, \delta\right)}{1 + b \delta},

        with :math:`b` the bias placing the galaxies (1 where :math:`1 + b \delta \le 0`): summed over galaxies, placed
        with :math:`1 + b \delta`, the alignments are weighted by :math:`1 + (b + \Delta b)\, \delta`.
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

        # density modulation (delta-TA/TT model)
        if 'density' in self.required_interp_fields:
            delta = catalog['density']
            bias = kwargs.get('bias', 0.0)
            for model in ('TA', 'TT'):
                delta_b = self.parameters[f'delta_b_{model}']
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
        a1 = self.parameters['A_TA'] if 'TA' in self.model else 0.
        a2 = self.parameters['A_TT'] if 'TT' in self.model else 0.
        C1, _, C2 = pyccl.nl_pt.translate_IA_norm(cosmo, z=z, a1=a1, a2=a2)
        return C1 * np.ones_like(z), C2 * np.ones_like(z)

    def tracer_amplitudes(self, bias: float) -> dict[str, float]:
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
        a1 = self.parameters['A_TA'] if 'TA' in self.model else 0.
        amplitudes = {'a1': a1, 'a1delta': (bias + self.parameters['delta_b_TA']) * a1}
        if 'TT' in self.model:
            amplitudes['a2'] = self.parameters['A_TT']
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
