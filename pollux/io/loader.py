"""Loader of the products of one realization, from their files only (:class:`PolluxLoader`)."""

from __future__ import annotations

import logging
import os

import healsparse as hsp
import numpy as np
import pyccl
import sacc
from astropy.io import fits

from .files import PolluxIO

logger = logging.getLogger(__name__)


class PolluxLoader(PolluxIO):
    """Loader of the products of one realization, from their files only: :class:`~pollux.io.files.PolluxIO` with named readers.

    No ``pollux`` builder is created, so the products can be read with the I/O dependencies only. Any file
    can also be read with the generic readers of :class:`~pollux.io.files.PolluxIO` (e.g.
    ``loader.read_map(loader.path('kappa', 'gamma1', 'bin3'))``).

    Parameters
    ----------
    root_dir : str
        Root directory of the products (see :mod:`pollux.io.files`).
    seed : int, optional
        Seed of the realization (``seed{seed}/``). Default is 0.
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
    footprint : dict
        Parameters of the footprint (``nside``, ``mask``, padding), from its ``config.yaml``.
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

    def __init__(self, root_dir: str, seed: int = 0, baryon_tag: str | None = None, nz: str | None = 'SRDv1_nz', catalog_tag: str | None = None) -> None:
        super().__init__(root_dir, seed=seed, baryon_tag=baryon_tag, nz=nz, tag=catalog_tag or 'default')
        logger.info("*** PolluxLoader ***")
        logger.info("  - Path = %s.", self.dir('realization'))

        # footprint (its config.yaml), then simulation, cosmology and table of the shells (config.yaml and shells.fits of the shells)
        self.footprint = self.read_config('footprint')['parameters']
        self.nside = self.footprint['nside']
        parameters = self.read_config('shells')['parameters']
        self.simulation = parameters['simulation']
        self.nshells = parameters['nshells']
        self.cosmo = pyccl.Cosmology(**parameters['cosmology'])
        self.shell_info = fits.getdata(self.path('shells', 'shells'))
        logger.info("  - Simulation = %s, nside = %d, %d shells.", self.simulation['path'], self.nside, self.nshells)
        logger.info("  - Cosmology = %s.", self.cosmo)

        # redshift distributions: list of (z, n(z)) per bin
        if nz not in (None, 'snapshots'):
            fname = self.path('kappa', 'nzs', ext='txt')
            if os.path.exists(fname):
                table = np.loadtxt(fname)
                self.nzs = [(table[:, 0], table[:, i]) for i in range(1, table.shape[1])]
            else:
                self.nzs = list(self.read_npy(self.path('kappa', 'nzs', ext='npy'), allow_pickle=True))

    # ==================================================================================================================
    # region Shells

    def load_shell(self, ishell: int, density: bool = False) -> hsp.HealSparseMap:
        """Load the particle counts of one shell, or its density contrast.

        Parameters
        ----------
        ishell : int
            Shell index.
        density : bool, optional
            Return the density contrast, ``counts / pixel_density - 1``. Default is False.

        Returns
        -------
        hsp.HealSparseMap
        """
        shell = self.read_map(self.path('shells', 'shell', self.SHELL_INDEX.format(ishell)))
        return shell / self.shell_info['pixel_density'][ishell] - 1. if density else shell

    def load_shell_alms(self, ishell: int) -> np.ndarray:
        """Load the a_lm of one shell (intermediate file).

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        ndarray
        """
        return self.read_alm(self.path('shells', 'alms', self.SHELL_INDEX.format(ishell), intermediate=True))

    def load_shell_cls(self) -> sacc.Sacc:
        """Load the measured angular power spectra of the shells.

        Returns
        -------
        sacc.Sacc
        """
        return self.read_sacc(self.path('shells', 'cls'))

    def load_density(self, ishell: int, smooth_Mpc: float | None = None) -> hsp.HealSparseMap:
        """Load the density contrast of one shell, smoothed or not.

        Parameters
        ----------
        ishell : int
            Shell index.
        smooth_Mpc : float or None, optional
            Smoothing scale (Mpc) of the map, None (default) for the unsmoothed map.

        Returns
        -------
        hsp.HealSparseMap
        """
        if smooth_Mpc is None:
            return self.load_shell(ishell, density=True)
        return self.read_map(self.path('shells', 'density', self.SHELL_INDEX.format(ishell), smooth_Mpc=smooth_Mpc))

    def load_tidal(self, ishell: int, smooth_Mpc: float | None = None) -> tuple[hsp.HealSparseMap, hsp.HealSparseMap, hsp.HealSparseMap]:
        """Load the components of the tidal field of one shell.

        Parameters
        ----------
        ishell : int
            Shell index.
        smooth_Mpc : float or None, optional
            Smoothing scale (Mpc) of the map, None (default) for the unsmoothed map.

        Returns
        -------
        tuple of hsp.HealSparseMap
            ``(t11, t22, t12)``.
        """
        return tuple(self.read_map(self.path('shells', what, self.SHELL_INDEX.format(ishell), smooth_Mpc=smooth_Mpc)) for what in ('t11', 't22', 't12'))

    def load_haloes(self, ishell: int) -> dict[str, np.ndarray]:
        """Load the halo catalog of one shell, shared by all baryon models.

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        dict
            Arrays by column, concatenated over the sub-volumes.
        """
        with self.read_table(self.path('halos', 'catalog', self.SHELL_INDEX.format(ishell))) as hdul:
            columns = hdul[1].columns.names
            return {k: np.concatenate([hdu.data[k] for hdu in hdul[1:]]) for k in columns}

    # endregion

    # ==================================================================================================================
    # region Convergence

    def load_kappa(self, ikappa: int, smooth_Mpc: float | None = None) -> hsp.HealSparseMap:
        """Load one convergence map.

        Parameters
        ----------
        ikappa : int
            Tomographic bin (or shell, for ``nz='snapshots'``).
        smooth_Mpc : float or None, optional
            Smoothing scale (Mpc) of the map, None (default) for the unsmoothed map.

        Returns
        -------
        hsp.HealSparseMap
        """
        return self.read_map(self.path('kappa', 'kappa', self._kappa_index(ikappa), smooth_Mpc=smooth_Mpc))

    def _kappa_index(self, ikappa: int) -> str:
        """Return the index of a convergence map in file names: ``bin{b}``, or ``{iii}`` for snapshots.

        Parameters
        ----------
        ikappa : int
            Tomographic bin (or shell, for ``nz='snapshots'``).

        Returns
        -------
        str
        """
        if self.options.get('nz') is None:
            raise ValueError('nz must be set to reach the convergence, shear and catalog products.')
        return (self.SHELL_INDEX if self.options['nz'] == 'snapshots' else self.BIN_INDEX).format(ikappa)

    def load_kappa_alms(self, ikappa: int) -> np.ndarray:
        """Load the a_lm of one convergence map (intermediate file).

        Parameters
        ----------
        ikappa : int
            Tomographic bin (or shell, for ``nz='snapshots'``).

        Returns
        -------
        ndarray
        """
        return self.read_alm(self.path('kappa', 'alms', self._kappa_index(ikappa), intermediate=True))

    def load_shear(self, ikappa: int, smooth_Mpc: float | None = None) -> tuple[hsp.HealSparseMap, hsp.HealSparseMap]:
        """Load the shear maps of one tomographic bin (or shell).

        Parameters
        ----------
        ikappa : int
            Tomographic bin (or shell, for ``nz='snapshots'``).
        smooth_Mpc : float or None, optional
            Smoothing scale (Mpc) of the map, None (default) for the unsmoothed map.

        Returns
        -------
        tuple of hsp.HealSparseMap
            ``(gamma1, gamma2)``.
        """
        return tuple(self.read_map(self.path('kappa', what, self._kappa_index(ikappa), smooth_Mpc=smooth_Mpc)) for what in ('gamma1', 'gamma2'))

    def load_kappa_cls(self) -> sacc.Sacc:
        """Load the measured angular power spectra of the convergence maps.

        Returns
        -------
        sacc.Sacc
        """
        return self.read_sacc(self.path('kappa', 'cls'))

    # endregion

    # ==================================================================================================================
    # region Catalogs

    def load_catalog(self, ibin: int) -> fits.HDUList:
        """Open the source catalog of one tomographic bin.

        Parameters
        ----------
        ibin : int
            Tomographic bin.

        Returns
        -------
        astropy.io.fits.HDUList
            To close, e.g. ``with loader.load_catalog(ibin) as hdul:``.
        """
        return self.read_table(self.path('catalogs', 'cat', self.BIN_INDEX.format(ibin)))

    def load_catalog_2pt(self, shapes: str = 'shear') -> sacc.Sacc:
        """Load the measured two-point functions of the catalogs (``cat_2pt[_{shapes}].fits``).

        Parameters
        ----------
        shapes : str, optional
            Shapes of the shear legs (``'shear'``, ``'ia'``, ``'shear_plus_ia'``, ``'observed'``). Default is ``'shear'``.

        Returns
        -------
        sacc.Sacc
        """
        return self.read_sacc(self.path('catalogs', *self.catalog_stat_name('2pt', shapes)))

    def load_catalog_cls(self, shapes: str = 'shear') -> sacc.Sacc:
        """Load the measured angular power spectra of the catalogs (``cat_cls[_{shapes}].fits``).

        Parameters
        ----------
        shapes : str, optional
            Shapes of the shear legs (``'shear'``, ``'ia'``, ``'shear_plus_ia'``, ``'observed'``). Default is ``'shear'``.

        Returns
        -------
        sacc.Sacc
        """
        return self.read_sacc(self.path('catalogs', *self.catalog_stat_name('cls', shapes)))

    # endregion
