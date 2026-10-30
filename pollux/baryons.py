from __future__ import annotations

import copy
import logging
import os
from typing import Any, Iterable

import numpy as np
import pyccl
import healpy as hp
import healsparse as hsp
import BaryonForge as bfg
from tqdm.auto import trange
from .shells import ShellBuilder, SnapshotShellBuilder

logger = logging.getLogger(__name__)


class BaryonifiedShellBuilder(ShellBuilder):
    """Apply a baryonification model to shell maps using halo catalogs.

    This builder reads per-shell halo catalogs and shell count maps from the
    provided builders, applies the BaryonForge baryonification model, and
    writes baryonified shell maps to disk. The matter power spectrum of the
    predictions (:meth:`matter_power_spectrum`) is corrected by the halo-model
    boost of the same profiles.

    Inherits all ShellBuilder methods and attributes from the input shell_builder.

    Parameters
    ----------
    shell_builder : ShellBuilder
        Builder providing shell count maps. All shell construction parameters
        and methods are inherited from this builder.
    halo_builder : SnapshotShellBuilder
        Builder providing halo catalogs per shell.
    dmo, dmb : BaryonForge profiles
        Dark-matter-only and baryonified profiles (e.g. :func:`s19_profiles`),
        used both to baryonify the shells and to correct the matter power
        spectrum of the predictions (:meth:`matter_power_spectrum`).
    mass_def : tuple
        Halo mass definition, passed to ``pyccl.halos.massdef.MassDef(*mass_def)``
        (e.g. ``('200', 'critical')``).
    mass_name : str
        Halo mass column of the catalogs (e.g. ``'sod_halo_mass'``).
    mass_unit : {'Msun', 'Msun_over_h'}
        Unit of the halo mass column.
    M_min : float or None, optional
        Minimum halo mass (Msun) of the baryonified halos, in the shells and in
        :meth:`matter_power_spectrum`. Default is ``10 * reader.mass_particles``.
    tag : str or None, optional
        Tag of the output subdirectory ``baryons_{tag}/`` (``baryons/`` if None).
    **interpolator_kwargs
        Forwarded to ``model.setup_interpolator`` for every shell. ``z_min``,
        ``z_max`` and ``N_samples_z`` are computed per shell (see
        :meth:`_setup_interpolator`) but can be overridden here for all shells.
    """

    def __init__(
        self,
        shell_builder: ShellBuilder,
        halo_builder: SnapshotShellBuilder,
        dmo: bfg.Profiles.Base.BaseBFGProfiles,
        dmb: bfg.Profiles.Base.BaseBFGProfiles,
        mass_def: tuple,
        mass_name: str,
        mass_unit: str,
        M_min: float | None = None,
        tag: str | None = None,
        **interpolator_kwargs: Any,
    ) -> None:
        if mass_unit not in ('Msun', 'Msun_over_h'):
            raise ValueError(f"mass_unit must be 'Msun' or 'Msun_over_h', got {mass_unit!r}")
        self.shell_builder_ref = shell_builder
        self.halo_builder = halo_builder
        self.dmo = dmo  # unconvolved, also used for the matter power spectrum prediction
        self.dmb = dmb
        self.mass_def = mass_def
        self.mass_name = mass_name
        self.mass_unit = mass_unit
        self.M_min = 10 * shell_builder.reader.mass_particles if M_min is None else M_min
        self.tag = tag
        self._interpolator_kwargs = {**interpolator_kwargs, 'M_min': self.M_min}

        baryon_dir = f'baryons_{tag}' if tag is not None else 'baryons'

        # Initialize parent ShellBuilder with the same configuration
        if self.shell_builder_ref.kind == 'snapshot':
            self.shell_edges_match = self.shell_builder_ref.shell_edges_match
        super().__init__(
            reader=shell_builder.reader,
            nshells=shell_builder.nshells,
            nside=shell_builder.nside,
            save_dir=os.path.join(shell_builder.save_dir, baryon_dir),
            seed=shell_builder.seed,
            dtype=shell_builder.dtype,
            apply_grid_isometry=(shell_builder.Abs if shell_builder.apply_grid_isometry else False),
            apply_replicate_rotflip=(shell_builder.replicate_rotflip if shell_builder.apply_replicate_rotflip else False),
            subsample_box=shell_builder.subsample_box,
            nside_c=shell_builder.nside_c,
            mask=shell_builder.footprint,
            save_format=shell_builder.save_format,
            wsp_dir=shell_builder.wsp_dir,
            b=shell_builder.b,
            nlb=shell_builder.nlb,
            remove_monopole=shell_builder.remove_monopole,
            padding_deg=shell_builder.padding_deg,
            apotype=shell_builder.apotype,
            padding_nside=shell_builder.padding_nside,
            replicate_octant=shell_builder.replicate_octant,
            nocompress=shell_builder.nocompress,
            f_ell=shell_builder.f_ell,
            base_fname='shell')

        # Set up baryonification model
        self.cosmo_dict = self.cosmo.to_dict()
        self.cosmo_dict.update({'matter_power_spectrum': 'linear'})  # Use linear power spectrum for baryonification
        self.cosmo_linear = pyccl.Cosmology(**self.cosmo_dict)
        self.cosmo_dict.update({'Omega_m': self.cosmo['Omega_m']})

        pix = bfg.HealPixel(self.nside)
        self.model = bfg.Profiles.Baryonification2D(
            bfg.ConvolvedProfile(dmo, pix), bfg.ConvolvedProfile(dmb, pix), self.cosmo_linear, mass_def=pyccl.halos.massdef.MassDef(*mass_def))

        self._interpolator_ishell = None  # shell index currently tabulated in self.model
        self._pk2d = None

    def construct_maps(
        self,
        ishells: int | Iterable[int] | None = None,
    ) -> hsp.HealSparseMap | dict[int, hsp.HealSparseMap]:
        """Baryonify shell count maps using the halo catalogs.

        Parameters
        ----------
        ishells : int or array_like of int or None, optional
            Shell indices to process. If None, processes all shells.

        Returns
        -------
        hsp.HealSparseMap or dict
            - If ``ishells`` is a single int: returns the baryonified shell map.
            - If ``ishells`` is ``None`` or an iterable: returns a dict keyed by
              shell index with baryonified shell maps.

        Notes
        -----
        Only halos heavier than ``M_min`` are baryonified. Shells already on disk
        are read instead of recomputed; the interpolator is rebuilt for the others
        (see :meth:`_setup_interpolator`).
        """
        shells = {}

        _, ipix = self.get_mask(nest=False, return_ipix=True, padded=True)
        for ishell in self.tqdm(ishells, desc="[construct_maps]"):
            fname = self.get_fname(ishell)
            shell = self.check_maps(read=True, fname=fname)
            if shell is not None:
                shells[ishell] = shell
                continue

            self._setup_interpolator(ishell)

            # Load the shell, turn into count map (in RING ordering), and set unseen/empty pixels to 0.
            count = self.shell_builder_ref.load_maps(ishell).generate_healpix_map(nest=False)
            count[count == hp.UNSEEN] = 0.
            lightcone_shell = bfg.utils.LightconeShell(map=count, cosmo=self.cosmo_dict)

            # Halo masses of the shell, in Msun
            halos = self.halo_builder.load_catalogs(ishell, variables=['ra', 'dec', 'z', self.mass_name])
            M = halos[self.mass_name] / self.cosmo['h'] if self.mass_unit == 'Msun_over_h' else halos[self.mass_name]

            keep = M > self.M_min
            if keep.any():
                logger.debug("Shell %d: using masses in range [%.3e, %.3e] Msun.", ishell, M[keep].min(), M[keep].max())
                lightcone_halos = bfg.utils.HaloLightConeCatalog(halos['ra'][keep], halos['dec'][keep], M[keep], halos['z'][keep], self.cosmo_dict)
                runner = bfg.Runners.BaryonifyShell(lightcone_halos, lightcone_shell, epsilon_max=20., model=self.model)
                baryonified_count = runner.process()
            else:
                # No halos above M_min in this shell
                baryonified_count = count

            # Save the baryonified map
            shell = self._make_hspmap(ipix, baryonified_count[ipix])
            self._save_map(shell, fname)

            shells[ishell] = shell

        return shells[ishells] if np.isscalar(ishells) else shells

    def _setup_interpolator(self, ishell: int) -> None:
        """(Re)build the interpolator of the baryonification model for one shell.

        Parameters
        ----------
        ishell : int
            Shell index.

        Notes
        -----
        Sets ``proj_cutoff`` to half the shell's comoving thickness (Anbajagane,
        Pandey & Chang 2024, Eq. 25), since the projection depth must match the
        shell's line-of-sight extent, so the interpolator is rebuilt per shell.
        """
        if self._interpolator_ishell == ishell:
            return

        z_edges = self.shell_edges_redshift
        chi_edges = self.shell_edges_comoving_distance
        proj_cutoff = (chi_edges[ishell + 1] - chi_edges[ishell]) / 2.
        self.model.DMO.set_parameter('proj_cutoff', proj_cutoff)
        self.model.DMB.set_parameter('proj_cutoff', proj_cutoff)

        interpolator_kwargs = dict(self._interpolator_kwargs)
        interpolator_kwargs.setdefault('N_samples_z', 2)
        interpolator_kwargs.setdefault('z_linear_sampling', True)
        interpolator_kwargs.setdefault('z_min', max(1e-3, z_edges[ishell]))  # Avoid z=0 because binning is log-spaced in z
        interpolator_kwargs.setdefault('z_max', max(z_edges[ishell + 1], z_edges[ishell] + 1e-3))
        self.model.setup_interpolator(**interpolator_kwargs)
        self._interpolator_ishell = ishell

    def matter_power_spectrum(self, nk: int = 128, na: int = 32) -> pyccl.Pk2D:
        """Return the baryon-corrected matter power spectrum consistent with the baryonified shells.

        CCL's nonlinear P(k) multiplied by the boost ``R(k, a) = P_DMB / P_DMO`` of the
        profiles used to baryonify the shells, in CCL's halo model (BaryonForge's
        ``FlexibleHMCalculator``, with the mass definition of the shells). As in the
        shells, halos lighter than ``M_min`` keep their dark-matter-only profile. The
        profiles' two-halo term is replaced by zero (on copies) since the halo model
        adds its own. Computed once, then cached.

        Parameters
        ----------
        nk, na : int, optional
            Number of wavenumbers in ``[1e-4, 20]`` 1/Mpc and of scale factors up to
            the shells' maximum redshift.

        Returns
        -------
        pyccl.Pk2D
        """
        # cached (it takes a few minutes)
        if self._pk2d is not None:
            return self._pk2d
        cosmo = self.cosmo
        rho_m = pyccl.rho_x(cosmo, 1.0, 'matter', is_comoving=True)

        # 1-halo terms only, with the baryonified profiles above M_min as in the shells
        dmo, dmb = copy.copy(self.dmo), copy.copy(self.dmb)
        for prof in (dmo, dmb):
            if hasattr(prof, 'TwoHalo'):
                prof.TwoHalo = bfg.Profiles.misc.Zeros()
        above = _MassThreshold(M_threshold=self.M_min)
        dmb = dmb * above + dmo * (1. - above)
        # profiles extend well beyond R200c: weight the 2-halo term with the total mass
        m_to_mtot = bfg.Profiles.misc.Mdelta_to_Mtot(dmo, r_min=1e-6, r_max=1e2, N_int=100)

        # overdensity profiles, truncated, with FFTLog precision high enough for extended profiles
        truncation = bfg.Profiles.misc.Truncation(epsilon_trunc=100)
        dmo = dmo / rho_m * truncation
        dmb = dmb / rho_m * truncation
        for prof in (dmo, dmb):
            prof.update_precision_fftlog(padding_lo_fftlog=1e-8, padding_hi_fftlog=1e8, n_per_decade=100)

        # halo model
        hmc = bfg.utils.FlexibleHMCalculator(
            mass_function='Tinker08',
            halo_bias='Tinker10',
            halo_m_to_mtot=m_to_mtot,
            mass_def=pyccl.halos.massdef.MassDef(*self.mass_def),
            log10M_min=9,
            log10M_max=16,
            nM=100)

        # boost of the halo-model power spectrum over the redshifts of the shells, applied to CCL's nonlinear one
        a_arr = np.linspace(1. / (1. + self.shell_edges_redshift.max()), 1., na)
        k_arr = np.geomspace(1e-4, 20, nk)
        cosmo.compute_nonlin_power()
        boost = np.empty((na, nk))
        for ia in trange(na, desc='[matter_power_spectrum]'):
            p_dmo = pyccl.halos.pk_2pt.halomod_power_spectrum(cosmo, hmc, k_arr, a_arr[ia], dmo, suppress_1h=lambda k: 1e-2)
            p_dmb = pyccl.halos.pk_2pt.halomod_power_spectrum(cosmo, hmc, k_arr, a_arr[ia], dmb, suppress_1h=lambda k: 1e-2)
            boost[ia] = p_dmb / p_dmo
        pk = np.array([cosmo.nonlin_matter_power(k_arr, a) for a in a_arr]) * boost
        self._pk2d = pyccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk), is_logp=True)
        return self._pk2d


def s19_profiles(
    bfg_params: dict,
    mass_def: tuple,
    **kwargs: Any,
) -> tuple[bfg.Profiles.Base.BaseBFGProfiles, bfg.Profiles.Base.BaseBFGProfiles]:
    """Return the S19 dark-matter-only and baryonified halo profiles.

    Both profiles share their components (dark matter, gas, stars, collisionless
    matter, two-halo term), with BaryonForge's default numerical settings unless
    overridden by ``kwargs``.

    Parameters
    ----------
    bfg_params : dict
        S19 model parameters (``baryon.bfg_params`` of the run configuration).
    mass_def : tuple
        Halo mass definition, passed to ``pyccl.halos.massdef.MassDef(*mass_def)``
        (e.g. ``('200', 'critical')``).
    **kwargs
        Other options of the BaryonForge S19 profiles, passed to every component
        (e.g. ``r_min_int``, ``r_max_int``, ``r_steps``, ``c_M_relation``).

    Returns
    -------
    dmo, dmb : BaryonForge profiles
        ``DarkMatterOnly`` and ``DarkMatterBaryon`` profiles.
    """
    S19 = bfg.Profiles.Schneider19
    kwargs = dict(bfg_params, mass_def=pyccl.halos.massdef.MassDef(*mass_def), **kwargs)
    dm = S19.DarkMatter(**kwargs)
    gas = S19.Gas(**kwargs)
    stars = S19.Stars(**kwargs)
    clm = S19.CollisionlessMatter(gas=gas, stars=stars, darkmatter=dm, **kwargs)
    two = S19.TwoHalo(**kwargs)
    dmo = S19.DarkMatterOnly(darkmatter=dm, twohalo=two, **kwargs)
    dmb = S19.DarkMatterBaryon(gas=gas, stars=stars, collisionlessmatter=clm, darkmatter=dm, twohalo=two, **kwargs)
    return dmo, dmb


class _MassThreshold(bfg.Profiles.Base.BaseBFGProfiles):
    """Profile equal to 1 for halo masses above ``M_threshold`` and 0 otherwise (to select halos in profile arithmetic)."""

    # a hyper parameter, so that profile arithmetic (which rebuilds profiles from their parameters) keeps it
    hyper_param_names = [*bfg.Profiles.Base.hyper_params, 'M_threshold']

    def __init__(self, M_threshold: float | None = None, **kwargs: Any) -> None:
        self.M_threshold = M_threshold
        super().__init__(**kwargs)

    def _real(self, cosmo: pyccl.Cosmology, r: float | np.ndarray, M: float | np.ndarray, a: float) -> np.ndarray:
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        prof = np.broadcast_to((M_use > self.M_threshold)[:, None], (len(M_use), len(r_use))).astype(float)
        # mirror the dimensions of the inputs
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
