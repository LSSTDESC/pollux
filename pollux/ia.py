"""Intrinsic alignment utilities."""

import numpy as np
import healpy as hp
import healsparse
from astropy.io import fits

from .utils import radec2thetaphi
from .catalog import CatalogAugmentor

class IACalculator(CatalogAugmentor):
    """Calculator for intrinsic alignment (IA) contributions to galaxy shapes.

    This class computes intrinsic alignment shear components from density and
    tidal fields using various IA models including NLA (Nonlinear Linear
    Alignment), TT (Tidal Torque), or a combination of both. It augments
    galaxy catalogs by adding IA shear columns based on interpolated density
    and tidal fields at galaxy positions.

    Parameters
    ----------
    density_field : str, optional
        Name of the density contrast field in the catalog. Default is 'delta'.
    tidal_fields : tuple of str, optional
        Names of the three tidal field components in the catalog, expected in
        the order (t11, t22, t12). Default is ('t11', 't22', 't12').
    model : str, optional
        IA model to use. Supported values are 'NLA', 'TT', or 'NLA+TT'.
        Default is 'NLA'.
    model_params : dict, optional
        Dictionary of model parameters. Supported keys include:
        - 'A_IA' : float, amplitude for NLA model (default 1.0)
        - 'A_TT' : float, amplitude for TT model (default 1.0)
        - 'b_TA' : float, bias parameter for density-dependent IA (default 0.0)

    Attributes
    ----------
    density_field : str
        Name of the density contrast field.
    tidal_fields : tuple of str
        Names of the tidal field components.
    model : str
        IA model identifier.
    model_params : dict
        Model parameter values.

    Notes
    -----
    The NLA model computes IA shear proportional to the tidal field, while the
    TT model uses quadratic tidal field terms. The normalization follows
    Brown et al. 2014, with C1bar = 5e-14 h^2 Msun^-1 Mpc^3.
    """

    def __init__(self, density_field='delta', tidal_fields=('t11', 't22', 't12'), model='NLA', model_params={}):
        self.density_field = density_field
        self.tidal_fields = tuple(tidal_fields) if tidal_fields is not None else tuple()
        
        assert model in ['NLA', 'TT', 'NLA+TT'], f"IA model '{model}' not recognized."
        self.model = model
        self.model_params = model_params

        # required fields for interpolation
        required_fields = list(tidal_fields) if tidal_fields is not None else []
        if model_params.get('b_TA', 0.0) != 0.0:
            required_fields.append(density_field)

        # new columns to be added to the catalog
        new_columns = []
        if 'NLA' in model:
            new_columns += ['gamma1_ia_ta', 'gamma2_ia_ta']
        if 'TT' in model:
            new_columns += ['gamma1_ia_tt', 'gamma2_ia_tt']
        super().__init__(name=f"IA_{model}", required_interp_fields=required_fields, new_columns=new_columns)

    def augment_catalog(self, cosmo, catalog):
        """Compute and add intrinsic alignment shear components to a catalog.

        Parameters
        ----------
        cosmo : ``pyccl.Cosmology`` object
            Cosmology object with methods ``rho_x(a, species='matter')`` and
            ``growth_factor(a)``, and attribute ``h`` (reduced Hubble constant).
        catalog : dict
            Dictionary containing galaxy catalog data. Must include:
            - 'z' : redshift array
            - tidal field components specified by ``self.tidal_fields``
            - density field specified by ``self.density_field`` (if b_TA != 0)

        Notes
        -----
        This method modifies the input catalog in-place by adding new columns:
        - 'gamma1_ia_ta', 'gamma2_ia_ta' : IA shear from NLA model
        - 'gamma1_ia_tt', 'gamma2_ia_tt' : IA shear from TT model

        The columns added depend on the selected model ('NLA', 'TT', or 'NLA+TT').
        """        
        z = catalog['z']
        a = 1.0 / (1.0 + z)
        C1bar = 5e-14 * cosmo['h']**2  # normalization constant in (Msun^-1 Mpc^3) from value of 5e14 (Msun^-1 h^2 Mpc^3) found in Brown et al. 2014
        t11 = catalog[self.tidal_fields[0]]
        t22 = catalog[self.tidal_fields[1]]
        t12 = catalog[self.tidal_fields[2]]

        if 'NLA' in self.model:
            # NLA (nonlinear linear alignment) model
            C1 = -(self.model_params.get('A_IA', 1.0) * C1bar) * (cosmo.rho_x(a, species='matter') / cosmo.growth_factor(a))
            e1 = C1 * (t11 - t22)
            e2 = C1 * t12
            catalog['gamma1_ia_ta'] = e1
            catalog['gamma2_ia_ta'] = e2

        if 'TT' in self.model:
            # TT (tidal torque) model
            C2 = 5 * self.model_params.get('A_TT', 1.0) * C1bar * (cosmo.rho_x(a, species='matter') / cosmo.growth_factor(a)**2)
            e1 = C2 * (t11**2 - t22**2)
            e2 = C2 * t12 * (t11 + t22)
            catalog['gamma1_ia_tt'] = e1
            catalog['gamma2_ia_tt'] = e2

        # delta-NLA/TT model
        b_TA = self.model_params.get('b_TA', 0.0)
        if b_TA != 0.0:
            delta = catalog[self.density_field]
            tmp = 1. + b_TA * delta

            if 'NLA' in self.model:
                catalog['gamma1_ia_ta'] *= tmp
                catalog['gamma2_ia_ta'] *= tmp
            if 'TT' in self.model:
                catalog['gamma1_ia_tt'] *= tmp
                catalog['gamma2_ia_tt'] *= tmp
