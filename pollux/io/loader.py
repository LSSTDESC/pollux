"""Loader of the products of one realization, from their files only (:class:`PolluxLoader`)."""

from __future__ import annotations

import logging
import os

import healpy as hp
import healsparse as hsp
import numpy as np
import pyccl
import sacc
import yaml
from astropy.io import fits

from . import layout

logger = logging.getLogger(__name__)


class PolluxLoader:
    """Loader of the products of one realization, from their files only (see :mod:`pollux.io.layout`).

    No ``pollux`` builder is created, so the products can be read with the I/O dependencies only.

    Parameters
    ----------
    realization_dir : str
        Directory of the realization, ``{root_dir}/nside{nside}/seed{seed}[/pad{padding}]``.
    baryon_tag : str or None, optional
        Tag of the baryon model, for the products of ``dmb/{baryon_tag}``. Default is None, the
        dark-matter-only products (``dmo``).
    nz : str or None, optional
        Name of the n(z) of the convergence and shear maps and of the catalogs (``kappa/{nz}``,
        ``catalogs/{nz}``), ``'snapshots'`` for one source plane per shell. Default is ``'SRDv1_nz'``.
    catalog_tag : str or None, optional
        Tag of the catalogs (``catalogs/{nz}/{catalog_tag}``). Default is None (``default``).

    Attributes
    ----------
    simulation : dict
        Simulation of the shells (path, kind, box size, particles, snapshots), from their ``config.yaml``.
    cosmo : pyccl.Cosmology
        Cosmology of the simulation.
    nside, nshells : int
        Resolution and number of the shells.
    shell_info : FITS_rec
        Table of the shells (``shells.fits``, see :meth:`pollux.shells.ShellBuilder.info_table`).
    nzs : list of (ndarray, ndarray)
        Redshift distributions, unless ``nz`` is None or ``'snapshots'``.
    """

    def __init__(self, realization_dir: str, baryon_tag: str | None = None, nz: str | None = 'SRDv1_nz', catalog_tag: str | None = None) -> None:
        self.realization_dir = realization_dir
        self.baryon_tag = baryon_tag
        self.nz = nz
        self.catalog_tag = catalog_tag
        self.matter_dir = layout.matter_dir(realization_dir, baryon_tag)
        self.shells_dir = os.path.join(self.matter_dir, 'shells')

        logger.info("*** PolluxLoader ***")
        logger.info("  - Path = %s.", self.matter_dir)

        # simulation, cosmology and table of the shells (config.yaml and shells.fits of the shells)
        with open(os.path.join(self.shells_dir, 'config.yaml')) as f:
            parameters = yaml.safe_load(f)['parameters']
        self.simulation = parameters['simulation']
        self.nside = parameters['nside']
        self.nshells = parameters['nshells']
        self.cosmo = pyccl.Cosmology(**parameters['cosmology'])
        self.shell_info = fits.getdata(os.path.join(self.shells_dir, 'shells.fits'))
        logger.info("  - Simulation = %s, nside = %d, %d shells.", self.simulation['path'], self.nside, self.nshells)
        logger.info("  - Cosmology = %s.", self.cosmo)

        # redshift distributions: list of (z, n(z)) per bin
        if nz not in (None, 'snapshots'):
            fname = os.path.join(self._kappa_dir(), 'nzs.txt')
            if os.path.exists(fname):
                table = np.loadtxt(fname)
                self.nzs = [(table[:, 0], table[:, i]) for i in range(1, table.shape[1])]
            else:
                self.nzs = list(np.load(os.path.join(self._kappa_dir(), 'nzs.npy'), allow_pickle=True))

    def _kappa_dir(self) -> str:
        """Return the directory of the convergence and shear maps of the n(z)."""
        if self.nz is None:
            raise ValueError('nz must be set to reach the convergence, shear and catalog products.')
        return os.path.join(self.matter_dir, 'kappa', self.nz)

    def _kappa_fname(self, what: str, ikappa: int, smooth_Mpc: float | None = None) -> str:
        """Return the file of a convergence or shear map (``kappa_bin{b}``, or ``kappa_{iii}`` for snapshots)."""
        index = f'{ikappa:03d}' if self.nz == 'snapshots' else f'bin{ikappa}'
        folder = self._kappa_dir() if smooth_Mpc is None else os.path.join(self._kappa_dir(), 'smoothed', f'{smooth_Mpc:g}Mpc')
        return os.path.join(folder, f'{what}_{index}.fits')

    def _shell_fname(self, what: str, ishell: int, smooth_Mpc: float | None = None) -> str:
        """Return the file of a shell map, smoothed maps being by scale in ``smoothed/{s}Mpc``."""
        folder = self.shells_dir if smooth_Mpc is None else os.path.join(self.matter_dir, 'smoothed', f'{smooth_Mpc:g}Mpc')
        return os.path.join(folder, f'{what}_{ishell:03d}.fits')

    # ------------------------------------------------------------------
    # Shells
    # ------------------------------------------------------------------

    def load_shell(self, ishell: int, density: bool = False) -> hsp.HealSparseMap:
        """Load the particle counts of one shell.

        Parameters
        ----------
        ishell : int
            Shell index.
        density : bool, optional
            If True, return the density contrast, ``counts / pixel_density - 1``.

        Returns
        -------
        hsp.HealSparseMap
        """
        shell = hsp.HealSparseMap.read(self._shell_fname('shell', ishell))
        return shell / self.shell_info['pixel_density'][ishell] - 1. if density else shell

    def load_shell_alms(self, ishell: int, *, replicate_octant: bool = False) -> np.ndarray:
        """Load the a_lm of one shell (intermediate file).

        Parameters
        ----------
        ishell : int
            Shell index.
        replicate_octant : bool, optional
            If True, the a_lm of the octant replicated over the full sky (``alms_octrep``, see
            ``replicate_octant`` of :class:`~pollux.map_builder.MapBuilder`).

        Returns
        -------
        ndarray
            a_lm (healpy ordering).
        """
        folder = layout.intermediate_dir(self.shells_dir, self.realization_dir)
        return hp.read_alm(os.path.join(folder, f'{_alms_what(replicate_octant)}_{ishell:03d}.fits'))

    def load_shell_cls(self) -> sacc.Sacc:
        """Load the measured angular power spectra of the shells.

        Returns
        -------
        sacc.Sacc
        """
        return sacc.Sacc.load_fits(os.path.join(self.shells_dir, 'cls.fits'))

    def load_density(self, ishell: int, smooth_Mpc: float | None = None) -> hsp.HealSparseMap:
        """Load the density contrast of one shell.

        Parameters
        ----------
        ishell : int
            Shell index.
        smooth_Mpc : float or None, optional
            Gaussian smoothing scale in Mpc of the map. If None, the unsmoothed density
            contrast of the shell (:meth:`load_shell`).

        Returns
        -------
        hsp.HealSparseMap
        """
        if smooth_Mpc is None:
            return self.load_shell(ishell, density=True)
        return hsp.HealSparseMap.read(self._shell_fname('density', ishell, smooth_Mpc))

    def load_tidal(self, ishell: int, smooth_Mpc: float | None = None) -> tuple[hsp.HealSparseMap, hsp.HealSparseMap, hsp.HealSparseMap]:
        """Load the tidal-tensor components (t11, t22, t12) of one shell.

        Parameters
        ----------
        ishell : int
            Shell index.
        smooth_Mpc : float or None, optional
            Gaussian smoothing scale in Mpc of the maps.

        Returns
        -------
        tuple of hsp.HealSparseMap
            ``(t11, t22, t12)``.
        """
        return tuple(hsp.HealSparseMap.read(self._shell_fname(what, ishell, smooth_Mpc)) for what in ('t11', 't22', 't12'))

    # ------------------------------------------------------------------
    # Halo catalogs
    # ------------------------------------------------------------------

    def load_haloes(self, ishell: int) -> dict[str, np.ndarray]:
        """Load the halo catalog of one shell, shared by all baryon models.

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        dict
            Concatenated arrays by column (``'ihaloes'``, ``'ra'``, ``'dec'``, ``'chi'``, ``'z'``,
            and the extra columns saved at construction).
        """
        fname = os.path.join(self.realization_dir, 'halos', f'catalog_{ishell:03d}.fits')
        with fits.open(fname) as hdul:
            columns = hdul[1].columns.names
            return {k: np.concatenate([hdu.data[k] for hdu in hdul[1:]]) for k in columns}

    # ------------------------------------------------------------------
    # Convergence and shear
    # ------------------------------------------------------------------

    def load_kappa(self, ikappa: int, smooth_Mpc: float | None = None) -> hsp.HealSparseMap:
        """Load one convergence map.

        Parameters
        ----------
        ikappa : int
            Tomographic bin (or shell, for ``nz='snapshots'``).
        smooth_Mpc : float or None, optional
            Gaussian smoothing scale in Mpc of the map (snapshots only).

        Returns
        -------
        hsp.HealSparseMap
        """
        return hsp.HealSparseMap.read(self._kappa_fname('kappa', ikappa, smooth_Mpc))

    def load_kappa_alms(self, ikappa: int, *, replicate_octant: bool = False) -> np.ndarray:
        """Load the a_lm of one convergence map (intermediate file).

        Parameters
        ----------
        ikappa : int
            Tomographic bin (or shell, for ``nz='snapshots'``).
        replicate_octant : bool, optional
            If True, the a_lm of the octant replicated over the full sky (``alms_octrep``).

        Returns
        -------
        ndarray
            a_lm (healpy ordering).
        """
        fname = os.path.basename(self._kappa_fname(_alms_what(replicate_octant), ikappa))
        return hp.read_alm(os.path.join(layout.intermediate_dir(self._kappa_dir(), self.realization_dir), fname))

    def load_shear(self, ikappa: int, smooth_Mpc: float | None = None) -> tuple[hsp.HealSparseMap, hsp.HealSparseMap]:
        """Load the shear maps of one tomographic bin (or shell).

        Parameters
        ----------
        ikappa : int
            Tomographic bin (or shell, for ``nz='snapshots'``).
        smooth_Mpc : float or None, optional
            Gaussian smoothing scale in Mpc of the maps (snapshots only).

        Returns
        -------
        tuple of hsp.HealSparseMap
            ``(gamma1, gamma2)``.
        """
        return tuple(hsp.HealSparseMap.read(self._kappa_fname(what, ikappa, smooth_Mpc)) for what in ('gamma1', 'gamma2'))

    def load_kappa_cls(self) -> sacc.Sacc:
        """Load the measured angular power spectra of the convergence maps.

        Returns
        -------
        sacc.Sacc
        """
        return sacc.Sacc.load_fits(os.path.join(self._kappa_dir(), 'cls.fits'))

    # ------------------------------------------------------------------
    # Source catalogs
    # ------------------------------------------------------------------

    def catalog_dir(self) -> str:
        """Return the directory of the catalogs (``catalogs/{nz}/{catalog_tag}``)."""
        return os.path.join(self.matter_dir, 'catalogs', self.nz, self.catalog_tag or 'default')

    def load_source_catalog(self, ibin: int) -> fits.HDUList:
        """Open the source catalog of one tomographic bin.

        Parameters
        ----------
        ibin : int
            Tomographic bin.

        Returns
        -------
        astropy.io.fits.HDUList
        """
        return fits.open(os.path.join(self.catalog_dir(), f'cat_bin{ibin}.fits'))

    def load_catalog_2pt(self) -> sacc.Sacc:
        """Load the measured two-point functions of the catalogs.

        Returns
        -------
        sacc.Sacc
        """
        return sacc.Sacc.load_fits(os.path.join(self.catalog_dir(), 'cat_2pt.fits'))

    def load_catalog_cls(self) -> sacc.Sacc:
        """Load the measured angular power spectra of the catalogs.

        Returns
        -------
        sacc.Sacc
        """
        return sacc.Sacc.load_fits(os.path.join(self.catalog_dir(), 'cat_cls.fits'))


def _alms_what(replicate_octant: bool) -> str:
    """Return the name of the a_lm files, of the octant replicated over the full sky or not (as ``MapBuilder``)."""
    return 'alms_octrep' if replicate_octant else 'alms'
