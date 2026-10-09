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
        Dictionary of model parameters. Supported keys include:
        - 'A_TA' : float, amplitude for TA model (default 1.0)
        - 'A_TT' : float, amplitude for TT model (default 1.0)
        - 'delta_b_TA' : float, bias parameter for density-dependent TA model (default 0.0)
        - 'delta_b_TT' : float, bias parameter for density-dependent TT model (default 0.0)
    Notes
    -----
    The TA (Tidal Alignment) model computes IA shear proportional to the
    tidal field, while the TT model uses quadratic tidal field terms. The
    normalization follows Brown et al. 2014, with C1bar = 5e-14 h^2 Msun^-1 Mpc^3.

    References
    ----------
    For the full model including density-dependent modulation, see
    https://arxiv.org/abs/2509.25166v1
    """

    def __init__(self, model='TA', model_params=None):
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

    def augment_catalog(self, cosmo: 'pyccl.Cosmology', catalog: dict[str, np.ndarray], **kwargs) -> None:
        """Compute and add intrinsic alignment shear components to a catalog.

        Parameters
        ----------
        cosmo : ``pyccl.Cosmology`` object
            Cosmology object with methods ``rho_x(a, species='matter')`` and
            ``growth_factor(a)``, and attribute ``h`` (reduced Hubble constant).
        catalog : dict[str, np.ndarray]
            Dictionary containing galaxy catalog data. Must include:
            - 'z' : redshift array
            - 't11', 't22', 't12' : tidal field components
            - 'density' : density contrast (only if delta_b terms are non-zero)

        Notes
        -----
        This method modifies the input catalog in-place by adding new columns:
        - 'e1_ia_ta', 'e2_ia_ta' : IA ellipticity from TA model
        - 'e1_ia_tt', 'e2_ia_tt' : IA ellipticity from TT model

        The columns added depend on the selected model ('TA', 'TT', or 'TATT').
        """
        z = catalog['z']
        a = 1.0 / (1.0 + z)
        C1bar = 5e-14 * cosmo['h']**2  # normalization constant in (Msun^-1 Mpc^3) from value of 5e14 (Msun^-1 h^2 Mpc^3) found in Brown et al. 2014
        t11 = catalog['t11']
        t22 = catalog['t22']
        t12 = catalog['t12']

        if 'TA' in self.model:
            # TA (Tidal Alignment) model
            C1 = -(self.model_params.get('A_TA', 1.0) * C1bar) * (cosmo.rho_x(a, species='matter') / cosmo.growth_factor(a))
            catalog['e1_ia_ta'] = C1 * (t11 - t22)
            catalog['e2_ia_ta'] = C1 * t12

        if 'TT' in self.model:
            # TT (tidal torque) model
            C2 = 5 * self.model_params.get('A_TT', 1.0) * C1bar * (cosmo.rho_x(a, species='matter') / cosmo.growth_factor(a)**2)
            catalog['e1_ia_tt'] = C2 * (t11**2 - t22**2)
            catalog['e2_ia_tt'] = C2 * t12 * (t11 + t22)

        # delta-TA/TT model
        delta_b_TA = self.model_params.get('delta_b_TA', 0.0)
        delta_b_TT = self.model_params.get('delta_b_TT', 0.0)

        if delta_b_TA != 0.0 or delta_b_TT != 0.0:
            delta = catalog['density']
            bias = kwargs.get('bias', 0.0)

            if delta_b_TA != 0.0 and 'TA' in self.model:
                tmp_ta = (1. + (bias + delta_b_TA) * delta) / (1. + bias * delta)
                catalog['e1_ia_ta'] *= tmp_ta
                catalog['e2_ia_ta'] *= tmp_ta

            if delta_b_TT != 0.0 and 'TT' in self.model:
                tmp_tt = (1. + (bias + delta_b_TT) * delta) / (1. + bias * delta)
                catalog['e1_ia_tt'] *= tmp_tt
                catalog['e2_ia_tt'] *= tmp_tt


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
    rng_seed : int, optional
        Seed of the intrinsic shapes: the galaxies of tomographic bin ``ikappa``
        in shell ``ishell`` use ``np.random.default_rng([rng_seed, ikappa, ishell])``,
        so shapes do not depend on how the catalog is split into shells. Default is 42.
    """

    def __init__(
        self,
        compute_reduced_shear=True,
        compute_magnification=False,
        generate_intrinsic=False,
        compute_observed=False,
        include_ia=True,
        sigma_e=0.3,
        rng_seed=42,
    ):
        if compute_observed and not generate_intrinsic:
            raise ValueError("Intrinsic shapes must be generated to compute observed shapes.")

        self.compute_reduced_shear = compute_reduced_shear
        self.compute_magnification = compute_magnification
        self.generate_intrinsic = generate_intrinsic
        self.compute_observed = compute_observed
        self.include_ia = include_ia
        self.sigma_e = sigma_e
        self.rng_seed = rng_seed
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

    def augment_catalog(self, cosmo, catalog, ikappa=0, **kwargs):
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
        """
        kappa = catalog['kappa']
        gamma1, gamma2 = catalog['gamma1'], catalog['gamma2']

        if self.compute_magnification:
            catalog['mu'] = 1. / ((1. - kappa)**2 - gamma1**2 - gamma2**2)

        if self.compute_reduced_shear:
            denom = 1. - kappa
            g1 = gamma1 / denom
            g2 = gamma2 / denom
            catalog['g1'] = g1
            catalog['g2'] = g2
        else:
            g1 = gamma1
            g2 = gamma2

        if self.generate_intrinsic:
            idx_shell = catalog['idx_shell']
            catalog['e1_int'] = np.empty(len(idx_shell))
            catalog['e2_int'] = np.empty(len(idx_shell))
            for ishell in np.unique(idx_shell):
                w = np.flatnonzero(idx_shell == ishell)
                rng = np.random.default_rng([self.rng_seed, ikappa, ishell])
                catalog['e1_int'][w] = rng.normal(0., self.sigma_e, size=len(w))
                catalog['e2_int'][w] = rng.normal(0., self.sigma_e, size=len(w))

        if self.compute_observed:
            e_int = catalog['e1_int'] + 1j * catalog['e2_int']
            if 'e1_ia_ta' in catalog:
                e_int += catalog['e1_ia_ta'] + 1j * catalog['e2_ia_ta']
            if 'e1_ia_tt' in catalog:
                e_int += catalog['e1_ia_tt'] + 1j * catalog['e2_ia_tt']
            g = g1 + 1j * g2
            e_obs = (g + e_int) / (1. + np.conj(g) * e_int)
            catalog['e1_obs'] = e_obs.real
            catalog['e2_obs'] = e_obs.imag
