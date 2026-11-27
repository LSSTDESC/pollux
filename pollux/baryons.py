"""Baryonification of the shells with BaryonForge (:class:`BaryonifiedShellBuilder`), with the matching correction of the matter power spectrum."""

from __future__ import annotations

import collections
import copy
import logging
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, Iterator

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
    shell_builder : SnapshotShellBuilder
        Builder providing shell count maps, of the same kind as ``halo_builder`` (the only one with halo
        catalogs), so that the halos and the particles of a shell match. All shell construction parameters
        and methods are inherited from this builder.
    halo_builder : SnapshotShellBuilder
        Builder providing halo catalogs per shell.
    dmo, dmb : BaryonForge profiles
        Dark-matter-only and baryonified profiles (e.g. :func:`s19_profiles` or :func:`mead20_profiles`),
        used both to baryonify the shells and to correct the matter power
        spectrum of the predictions (:meth:`matter_power_spectrum`).
    profile_parameters : dict or None, optional
        Parameters of the profiles (e.g. the ``bfg_params`` of :func:`s19_profiles`), recorded in
        the ``config.yaml`` of the shells, so that shells of other parameters are not mixed with them.
        Its ``epsilon_max`` (default 20) is the radius, in units of the halo radius, within which each halo
        displaces the shell's mass (``epsilon_max`` of BaryonForge's ``BaryonifyShell``).
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
    nworkers : int, optional
        Processes displacing the pixels around the halos of a shell, each for chunks of halos (see :meth:`_baryonify`).
        Not recorded in the ``config.yaml``. Default is 1 (no other process).
    tag : str or None, optional
        Tag of the baryon model: the shells go to ``{realization}/dmb/{tag}/shells`` (see
        :mod:`pollux.io.files`), ``dmb/default`` if None.
    **interpolator_kwargs
        Forwarded to ``model.setup_interpolator`` for every shell. ``z_min``,
        ``z_max`` and ``N_samples_z`` are computed per shell (see
        :meth:`_setup_interpolator`) but can be overridden here for all shells.
    """

    def __init__(
        self,
        shell_builder: SnapshotShellBuilder,
        halo_builder: SnapshotShellBuilder,
        dmo: bfg.Profiles.Base.BaseBFGProfiles,
        dmb: bfg.Profiles.Base.BaseBFGProfiles,
        profile_parameters: dict | None,
        mass_def: tuple,
        mass_name: str,
        mass_unit: str,
        M_min: float | None = None,
        nworkers: int = 1,
        tag: str | None = None,
        **interpolator_kwargs: Any,
    ) -> None:
        if not isinstance(shell_builder, SnapshotShellBuilder):
            raise TypeError(f"shell_builder must be a SnapshotShellBuilder, like the halo builder, not {type(shell_builder).__name__}.")
        if mass_unit not in ('Msun', 'Msun_over_h'):
            raise ValueError(f"mass_unit must be 'Msun' or 'Msun_over_h', got {mass_unit!r}")
        self.shell_builder_ref = shell_builder
        self.halo_builder = halo_builder
        self.dmo = dmo  # unconvolved, also used for the matter power spectrum prediction
        self.dmb = dmb
        self.profile_parameters = profile_parameters
        # radius of the baryonification around each halo, in units of R200c (the BaryonForge runner's epsilon_max)
        self.epsilon_max = float((profile_parameters or {}).get('epsilon_max', 20.))
        self.mass_def = mass_def
        self.mass_name = mass_name
        self.mass_unit = mass_unit
        self.M_min = 10 * shell_builder.reader.mass_particles if M_min is None else M_min
        self.nworkers = int(nworkers)
        self.tag = tag
        self._interpolator_kwargs = {**interpolator_kwargs, 'M_min': self.M_min}

        # Initialize parent ShellBuilder with the same configuration
        self.shell_edges_match = shell_builder.shell_edges_match
        super().__init__(
            reader=shell_builder.reader,
            nshells=shell_builder.nshells,
            footprint=shell_builder.footprint,
            # products of the baryon model, next to the dark-matter-only ones (dmb/{tag})
            io=shell_builder.io.with_options(baryon_tag=tag or 'default'),
            seed=shell_builder.seed,
            dtype=shell_builder.dtype,
            apply_grid_isometry=(shell_builder.Abs if shell_builder.apply_grid_isometry else False),
            apply_replicate_rotflip=(shell_builder.replicate_rotflip if shell_builder.apply_replicate_rotflip else False),
            subsample_box=shell_builder.subsample_box,
            nside_c=shell_builder.nside_c,
            map_format=shell_builder.map_format,
            sub_format=shell_builder.sub_format,
            b=shell_builder.b,
            remove_monopole=shell_builder.remove_monopole,
            nocompress=shell_builder.nocompress,
            pixel_window=shell_builder.pixel_window)

        # Set up baryonification model
        self.cosmo_linear = pyccl.Cosmology(**{**self.cosmo.to_dict(), 'matter_power_spectrum': 'linear'})  # of the model and the halos

        pix = bfg.HealPixel(self.nside)
        self.model = bfg.Profiles.Baryonification2D(
            bfg.ConvolvedProfile(dmo, pix), bfg.ConvolvedProfile(dmb, pix), self.cosmo_linear, mass_def=pyccl.halos.massdef.MassDef(*mass_def))

        self._interpolator_ishell = None  # shell index currently tabulated in self.model
        self._pk2d = None  # matter power spectrum with baryons, cached by matter_power_spectrum

    def config(self) -> dict:
        """Return the configuration of the baryonified shells, for their ``config.yaml``.

        Returns
        -------
        dict
            ``product`` (``'shells'``), ``parameters`` (those of the shells, :meth:`pollux.shells.ShellBuilder.config`,
            and of the baryon model) and ``inputs`` (the dark-matter-only shells and the halo catalogs).
        """
        config = super().config()
        parameters = config['parameters']
        parameters.update(
            baryon_tag=self.tag,
            dmo_profiles=_profile_name(self.dmo),
            dmb_profiles=_profile_name(self.dmb),
            mass_def=self.mass_def,
            mass_name=self.mass_name,
            mass_unit=self.mass_unit,
            M_min=self.M_min)
        if self.profile_parameters is not None:
            parameters['profile_parameters'] = self.profile_parameters
        return dict(config, product='shells', inputs=[self.shell_builder_ref.save_dir, self.halo_builder.save_dir])

    # ==================================================================================================================
    # region Maps

    def construct_maps(self, imaps: int | Iterable[int] | None = None) -> np.ndarray | hsp.HealSparseMap | dict[int, np.ndarray | hsp.HealSparseMap]:
        """Baryonify shell count maps using the halo catalogs.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Shell indices to process. If None, processes all shells.

        Returns
        -------
        ndarray or hsp.HealSparseMap, or dict
            The maps as written (:attr:`~pollux.map_builder.MapBuilder.map_format`): a single map if ``imaps`` is an
            int, else a dict keyed by map index.

        Notes
        -----
        Only halos heavier than ``M_min`` are baryonified (:meth:`_baryonify`). The interpolator is
        rebuilt for each shell (see :meth:`_setup_interpolator`).
        """
        shells = {}

        for ishell in self.tqdm(imaps, desc="[construct_maps]"):
            self._setup_interpolator(ishell)

            # Load the shell as a count map (in RING ordering), and set unseen/empty pixels to 0.
            count = self.shell_builder_ref.read_maps(ishell, dense=True)
            count[count == hp.UNSEEN] = 0.

            # Halo masses of the shell, in Msun
            halos = self.halo_builder.read_catalogs(ishell, variables=['ra', 'dec', 'z', self.mass_name])
            M = halos[self.mass_name] / self.cosmo['h'] if self.mass_unit == 'Msun_over_h' else halos[self.mass_name]

            keep = M > self.M_min
            if keep.any():
                logger.debug("Shell %d: using masses in range [%.3e, %.3e] Msun.", ishell, M[keep].min(), M[keep].max())
                baryonified_count = self._baryonify(count, halos['ra'][keep], halos['dec'][keep], M[keep], halos['z'][keep])
            else:
                # No halos above M_min in this shell
                baryonified_count = count

            # Save the baryonified map, on the padded footprint
            shells[ishell] = self._write_map(self.path(ishell), self._to_dense(baryonified_count))

        return shells[imaps] if np.isscalar(imaps) else shells

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

    def _baryonify(self, count: np.ndarray, ra: np.ndarray, dec: np.ndarray, M: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the baryonified map of a shell: BaryonForge's ``BaryonifyShell.process``, with the halos in parallel
        and only the displaced pixels regridded.

        Each halo displaces the pixels within ``epsilon_max`` times its radius (the displacement of the model, with the
        computations of BaryonForge's runner, in the cosmology of the builder). The displacements of the unit vectors of
        the pixels add up over the halos, so the halos are split in chunks of nearby halos, processed by
        :attr:`nworkers` processes (forked by a fork server, each given the model and the halos), each returning the sum
        of its displacements on the pixels it reaches; the chunks are then summed. The mass of each displaced pixel goes
        to the four pixels around its new position, with bilinear weights, as in BaryonForge (by blocks of pixels, also
        by the processes); that of the other pixels stays in place.

        Parameters
        ----------
        count : ndarray
            Dense RING count map of the shell (zero outside the footprint).
        ra, dec : ndarray
            Positions of the halos (degrees).
        M : ndarray
            Masses of the halos (Msun, :attr:`mass_def`).
        z : ndarray
            Redshifts of the halos.

        Returns
        -------
        ndarray
            Baryonified dense RING count map (float64), with the mass of ``count``.
        """
        if np.allclose(count, 0):
            return count
        if vars(self.model).get('p_keys'):
            raise NotImplementedError("Baryonification models with halo properties (p_keys) are not supported.")
        ra, dec, M, z = (np.asarray(x, dtype=np.float64) for x in (ra, dec, M, z))  # as BaryonForge's catalogs

        # compute the radii and distances of the halos (physical Mpc)
        a = 1 / (1 + z)
        mass_def = pyccl.halos.massdef.MassDef(*self.mass_def)  # as MassDef.get_radius, for all scale factors at once
        R = (M / (4. / 3. * np.pi * mass_def.get_Delta(self.cosmo_linear, a) * self.cosmo_linear.rho_x(a, mass_def.rho_type)))**(1. / 3.)
        D = pyccl.angular_diameter_distance(self.cosmo_linear, a)
        halos = dict(vec=np.atleast_2d(hp.ang2vec(ra, dec, lonlat=True)), M=M, a=a, R=R, D=D, ra=ra, dec=dec)
        state = dict(halos=halos, nside=self.nside, model=self.model, epsilon_max=self.epsilon_max)

        # start the processes, forked by a fork server that imports this module once (not from this process, whose
        # threads could hold locks), and give each the state once
        if self.nworkers > 1:
            context = multiprocessing.get_context('forkserver')
            context.set_forkserver_preload([__name__])
            executor = ProcessPoolExecutor(self.nworkers, mp_context=context, initializer=_set_state, initargs=(state,))
        else:
            executor = None
            _set_state(state)
        window = 2 * self.nworkers  # tasks sent ahead
        try:
            # displace the pixels around chunks of nearby halos (sorted by coarse pixel), and sum their displacements
            t0 = time.time()
            order = np.argsort(hp.ang2pix(16, ra, dec, lonlat=True, nest=True), kind='stable')
            chunks = np.array_split(order, min(M.size, 32 * self.nworkers))
            offsets = np.zeros((count.size, 3))
            for pix, offset in _imap(executor, _displace_pixels, chunks, window):
                offsets[pix] += offset

            # move the mass of the displaced pixels to the pixels around their new positions, by blocks of pixels
            t1 = time.time()
            moved = np.flatnonzero((count != 0) & np.any(offsets != 0, axis=1))
            new_map = count.astype(np.float64)
            new_map[moved] = 0.  # the other pixels keep their mass
            blocks = (moved[i:i + 2**22] for i in range(0, moved.size, 2**22))
            for pix, mass in _imap(executor, _regrid_pixels, ((block, offsets[block], count[block]) for block in blocks), window):
                new_map[pix] += mass
        finally:
            if executor is None:
                _set_state(None)
            else:
                executor.shutdown()

        # check that the mass is conserved
        if not np.isclose(np.sum(new_map), np.sum(count)):
            warnings.warn(f"Baryonification: the mass of the shell changed from {np.sum(count):.14e} to {np.sum(new_map):.14e}.")
        logger.debug(
            "Baryonification: %d halos displaced the pixels in %.1f s (%d processes), %d pixels with mass regridded in %.1f s.", M.size, t1 - t0, self.nworkers,
            moved.size,
            time.time() - t1)
        return new_map

    # endregion

    # ==================================================================================================================
    # region Predictions

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

    # endregion


# ======================================================================================================================
# region Profiles


def s19_profiles(
    bfg_params: dict,
    mass_def: tuple,
    **kwargs: Any,
) -> tuple[bfg.Profiles.Base.BaseBFGProfiles, bfg.Profiles.Base.BaseBFGProfiles]:
    """Return the S19 dark-matter-only and baryonified halo profiles.

    The profiles of Schneider et al. (2019), BaryonForge's default ones: dark matter and two-halo term
    without baryons, and gas, stars and collisionless matter with them, built by the profiles from
    the same parameters, with BaryonForge's default numerical settings unless overridden by ``kwargs``.

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
    return S19.DarkMatterOnly(**kwargs), S19.DarkMatterBaryon(**kwargs)


def mead20_profiles(
    bfg_params: dict,
    mass_def: tuple,
    **kwargs: Any,
) -> tuple[bfg.Profiles.Base.BaseBFGProfiles, bfg.Profiles.Base.BaseBFGProfiles]:
    """Return the Mead20 (HMcode-2020) dark-matter-only and baryonified halo profiles.

    The simplest pair of BaryonForge: an NFW profile without baryons, and bound gas,
    ejected gas, stars and collisionless matter with them, without two-halo term.
    The parameters may be set by a single AGN temperature, interpolated in the calibration of
    Mead et al. (2020, Table 2) by BaryonForge's ``Tagn2pars``.

    Parameters
    ----------
    bfg_params : dict
        Mead20 model parameters. ``log10_T_AGN`` (e.g. 7.8) sets them all from the calibration,
        with ``calibration`` (``'All'``, default, or ``'MatterPressure'``); other keys override
        them (e.g. ``cutoff`` and ``proj_cutoff``, in Mpc, or ``epsilon_max`` of
        :class:`BaryonifiedShellBuilder`).
    mass_def : tuple
        Halo mass definition, passed to ``pyccl.halos.massdef.MassDef(*mass_def)``
        (e.g. ``('200', 'critical')``).
    **kwargs
        Other options of the BaryonForge Mead20 profiles (e.g. ``r_min_int``, ``r_max_int``,
        ``r_steps``, ``c_M_relation``).

    Returns
    -------
    dmo, dmb : BaryonForge profiles
        ``DarkMatterOnly`` and ``DarkMatterBaryon`` profiles.

    Notes
    -----
    The masses of the calibration are converted from Msun/h to Msun by BaryonForge with h = 0.7.
    """
    from BaryonForge.Profiles.Mead20 import Tagn2pars
    M20 = bfg.Profiles.Mead20
    params = dict(bfg_params)
    if 'log10_T_AGN' in params:
        params = {**Tagn2pars(float(params.pop('log10_T_AGN')), mode=params.pop('calibration', 'All')), **params}
    kwargs = dict(params, mass_def=pyccl.halos.massdef.MassDef(*mass_def), **kwargs)
    return M20.DarkMatterOnly(**kwargs), M20.DarkMatterBaryon(**kwargs)


def _profile_name(profile: bfg.Profiles.Base.BaseBFGProfiles) -> str:
    """Return the model and class of a profile, for the ``config.yaml`` of the shells.

    Parameters
    ----------
    profile : BaryonForge profile

    Returns
    -------
    str
        E.g. ``'Schneider19.DarkMatterOnly'``.
    """
    return f'{type(profile).__module__.rsplit(".", 1)[-1]}.{type(profile).__name__}'


class _MassThreshold(bfg.Profiles.Base.BaseBFGProfiles):
    """Profile equal to 1 for halo masses above ``M_threshold`` and 0 otherwise (to select halos in profile arithmetic)."""

    # a hyper parameter, so that profile arithmetic (which rebuilds profiles from their parameters) keeps it
    hyper_param_names = [*bfg.Profiles.Base.hyper_params, 'M_threshold']

    def __init__(self, M_threshold: float | None = None, **kwargs: Any) -> None:
        self.M_threshold = M_threshold
        super().__init__(**kwargs)

    def _real(self, cosmo: pyccl.Cosmology, r: float | np.ndarray, M: float | np.ndarray, a: float) -> np.ndarray:
        """Return the profile in real space: 1 for the halos heavier than ``M_threshold``, 0 otherwise.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology (unused).
        r : float or ndarray
            Radii.
        M : float or ndarray
            Halo masses.
        a : float
            Scale factor (unused).

        Returns
        -------
        ndarray
            Shape ``(len(M), len(r))``, without the dimensions of scalar inputs.
        """
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        prof = np.broadcast_to((M_use > self.M_threshold)[:, None], (len(M_use), len(r_use))).astype(float)
        # mirror the dimensions of the inputs
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


# endregion

# ======================================================================================================================
# region Workers

# state of the processes of BaryonifiedShellBuilder._baryonify: the halos, the model, nside and epsilon_max
_BARYONIFY = None


def _set_state(state: dict | None) -> None:
    """Set the state of the processes of :meth:`BaryonifiedShellBuilder._baryonify`.

    The initializer of its workers, and called in its own process when it has no workers.

    Parameters
    ----------
    state : dict or None
        Halos (positions, masses, scale factors, radii, distances), model, ``nside`` and ``epsilon_max``; None to clear
        it.
    """
    global _BARYONIFY
    _BARYONIFY = state


def _displace_pixels(ihalos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the pixels displaced by some halos, and the sums of their displacements (of their unit vectors).

    The computations of BaryonForge's ``BaryonifyShell.process`` for each halo, on the state set by
    :meth:`BaryonifiedShellBuilder._baryonify`.

    Parameters
    ----------
    ihalos : ndarray
        Indices of the halos.

    Returns
    -------
    pix : ndarray
        RING pixels, sorted.
    offsets : ndarray
        Sum of the displacements of the unit vectors of the pixels, shape ``(len(pix), 3)``.
    """
    state = _BARYONIFY
    halos, nside, model = state['halos'], state['nside'], state['model']
    pixels, offsets = [], []
    for j in ihalos:
        vec_j, M_j, a_j, R_j, D_j = halos['vec'][j], halos['M'][j], halos['a'][j], halos['R'][j], halos['D'][j]
        # find the pixels within epsilon_max halo radii (at least the 4 nearest)
        pixind = hp.query_disc(nside, vec_j, R_j * state['epsilon_max'] / D_j, inclusive=False, nest=False)
        if pixind.size < 4:
            pixind = hp.get_interp_weights(nside, halos['ra'][j], halos['dec'][j], lonlat=True)[0]
        # compute their separations from the halo (physical distances, flat cosmology)
        vec = np.stack(hp.pix2vec(nside, pixind), axis=1)
        pos = vec * D_j
        diff = pos - vec_j * D_j
        r_sep = np.sqrt(np.sum(diff**2, axis=1))
        # displace them radially by the model (in comoving distances)
        offset = model.displacement(r_sep / a_j, M_j, a_j) * a_j
        offset = offset[:, None] * (diff / r_sep[:, None])
        offset = np.where(np.isfinite(offset), offset, 0)
        # keep the change of their unit vectors
        new_pos = pos + offset
        pixels.append(pixind)
        offsets.append(new_pos / np.sqrt(np.sum(new_pos**2, axis=1))[:, None] - vec)
    if not pixels:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    # sum the changes of each pixel
    upix, inverse = np.unique(np.concatenate(pixels), return_inverse=True)
    offsets = np.concatenate(offsets)
    return upix, np.stack([np.bincount(inverse, weights=offsets[:, k], minlength=upix.size) for k in range(3)], axis=1)


def _regrid_pixels(args: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return where the mass of displaced pixels goes: the pixels around their new positions, and the mass each gets.

    As BaryonForge's ``regrid_pixels_hpix``, for a block of pixels: the mass of each goes to the four pixels around its
    new position (unit vector plus displacement), with bilinear weights.

    Parameters
    ----------
    args : tuple
        RING pixels, their displacements (shape ``(n, 3)``) and their mass.

    Returns
    -------
    pix : ndarray
        RING pixels receiving mass, sorted.
    mass : ndarray
        Mass each receives.
    """
    pix, offsets, values = args
    nside = _BARYONIFY['nside']
    # find the 4 pixels around the new positions, with bilinear weights
    lon, lat = hp.vec2ang(np.stack(hp.pix2vec(nside, pix), axis=1) + offsets, lonlat=True)
    c_pix, c_weight = hp.get_interp_weights(nside, lon, lat, lonlat=True)
    # sum the mass each pixel receives
    upix, inverse = np.unique(c_pix.T.ravel(), return_inverse=True)
    return upix, np.bincount(inverse, weights=(c_weight * values).T.ravel(), minlength=upix.size)


def _imap(executor: ProcessPoolExecutor | None, func: Callable, args: Iterable, window: int) -> Iterator:
    """Yield ``func`` of each of ``args``, in order, computed by the processes of ``executor`` or in this one.

    Arguments are sent as results are taken, at most ``window`` ahead, so that they are made (e.g. sliced from large
    arrays) only as needed and results don't pile up.

    Parameters
    ----------
    executor : ProcessPoolExecutor or None
        Processes computing ``func``; None to compute it in this process.
    func : callable
        Function of one argument (pickled to the processes by reference).
    args : iterable
        Arguments, consumed lazily.
    window : int
        Arguments sent and not yet returned, at most.

    Yields
    ------
    object
        ``func(arg)`` for each of ``args``, in their order.
    """
    if executor is None:
        yield from map(func, args)
        return
    pending = collections.deque()
    for arg in args:
        pending.append(executor.submit(func, arg))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


# endregion
