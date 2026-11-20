"""Loader of the products of one realization, from their files only (:class:`PolluxLoader`)."""

from __future__ import annotations

import logging
import os

import healsparse as hsp
import numpy as np
import pyccl
import sacc
from astropy.io import fits

from .layout import PolluxIO

logger = logging.getLogger(__name__)


class PolluxLoader(PolluxIO):
    """Loader of the products of one realization, from their files only: :class:`~pollux.io.layout.PolluxIO` with named readers.

    No ``pollux`` builder is created, so the products can be read with the I/O dependencies only. Any file
    can also be read with the generic readers of :class:`~pollux.io.layout.PolluxIO` (e.g.
    ``loader.read_map(loader.path('kappa', 'gamma1', 'bin3'))``).

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
        super().__init__(realization_dir, baryon_tag=baryon_tag, nz=nz, tag=catalog_tag)
        logger.info("*** PolluxLoader ***")
        logger.info("  - Path = %s.", os.path.dirname(self.dir('shells')))

        # simulation, cosmology and table of the shells (config.yaml and shells.fits of the shells)
        parameters = self.read_config('shells')['parameters']
        self.simulation = parameters['simulation']
        self.nside = parameters['nside']
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

    def _kappa_index(self, ikappa: int) -> str:
        """Return the index of a convergence map in file names: ``bin{b}``, or ``{iii}`` for snapshots."""
        if self.nz is None:
            raise ValueError('nz must be set to reach the convergence, shear and catalog products.')
        return (self.SHELL_INDEX if self.nz == 'snapshots' else self.BIN_INDEX).format(ikappa)

    def load_shell(self, ishell: int, density: bool = False) -> hsp.HealSparseMap:
        """Load the particle counts of one shell, or with ``density`` its density contrast, ``counts / pixel_density - 1``."""
        shell = self.read_map(self.path('shells', 'shell', self.SHELL_INDEX.format(ishell)))
        return shell / self.shell_info['pixel_density'][ishell] - 1. if density else shell

    def load_shell_alms(self, ishell: int, *, replicate_octant: bool = False) -> np.ndarray:
        """Load the a_lm of one shell (intermediate file), of the octant replicated over the full sky with ``replicate_octant``."""
        return self.read_alm(self.path('shells', 'alms_octrep' if replicate_octant else 'alms', self.SHELL_INDEX.format(ishell), intermediate=True))

    def load_shell_cls(self) -> sacc.Sacc:
        """Load the measured angular power spectra of the shells."""
        return self.read_sacc(self.path('shells', 'cls'))

    def load_density(self, ishell: int, smooth_Mpc: float | None = None) -> hsp.HealSparseMap:
        """Load the density contrast of one shell, smoothed at ``smooth_Mpc`` (Mpc) if given."""
        if smooth_Mpc is None:
            return self.load_shell(ishell, density=True)
        return self.read_map(self.path('shells', 'density', self.SHELL_INDEX.format(ishell), smooth_Mpc=smooth_Mpc))

    def load_tidal(self, ishell: int, smooth_Mpc: float | None = None) -> tuple[hsp.HealSparseMap, hsp.HealSparseMap, hsp.HealSparseMap]:
        """Load the tidal-tensor components ``(t11, t22, t12)`` of one shell, smoothed at ``smooth_Mpc`` (Mpc)."""
        return tuple(self.read_map(self.path('shells', what, self.SHELL_INDEX.format(ishell), smooth_Mpc=smooth_Mpc)) for what in ('t11', 't22', 't12'))

    def load_haloes(self, ishell: int) -> dict[str, np.ndarray]:
        """Load the halo catalog of one shell, shared by all baryon models, as concatenated arrays by column."""
        with self.read_table(self.path('halos', 'catalog', self.SHELL_INDEX.format(ishell))) as hdul:
            columns = hdul[1].columns.names
            return {k: np.concatenate([hdu.data[k] for hdu in hdul[1:]]) for k in columns}

    def load_kappa(self, ikappa: int, smooth_Mpc: float | None = None) -> hsp.HealSparseMap:
        """Load one convergence map, of a tomographic bin (or shell, for ``nz='snapshots'``)."""
        return self.read_map(self.path('kappa', 'kappa', self._kappa_index(ikappa), smooth_Mpc=smooth_Mpc))

    def load_kappa_alms(self, ikappa: int, *, replicate_octant: bool = False) -> np.ndarray:
        """Load the a_lm of one convergence map (intermediate file)."""
        return self.read_alm(self.path('kappa', 'alms_octrep' if replicate_octant else 'alms', self._kappa_index(ikappa), intermediate=True))

    def load_shear(self, ikappa: int, smooth_Mpc: float | None = None) -> tuple[hsp.HealSparseMap, hsp.HealSparseMap]:
        """Load the shear maps ``(gamma1, gamma2)`` of one tomographic bin (or shell)."""
        return tuple(self.read_map(self.path('kappa', what, self._kappa_index(ikappa), smooth_Mpc=smooth_Mpc)) for what in ('gamma1', 'gamma2'))

    def load_kappa_cls(self) -> sacc.Sacc:
        """Load the measured angular power spectra of the convergence maps."""
        return self.read_sacc(self.path('kappa', 'cls'))

    def load_catalog(self, ibin: int) -> fits.HDUList:
        """Open the source catalog of one tomographic bin."""
        return self.read_table(self.path('catalogs', 'cat', f'bin{ibin}'))

    def load_catalog_2pt(self, shapes: str = 'shear') -> sacc.Sacc:
        """Load the measured two-point functions of the catalogs, with the shapes ``shapes`` of the shear legs (``cat_2pt[_{shapes}].fits``)."""
        return self.read_sacc(self.path('catalogs', 'cat_2pt', None if shapes == 'shear' else shapes))

    def load_catalog_cls(self, shapes: str = 'shear') -> sacc.Sacc:
        """Load the measured angular power spectra of the catalogs, with the shapes ``shapes`` (``cat_cls[_{shapes}].fits``)."""
        return self.read_sacc(self.path('catalogs', 'cat_cls', None if shapes == 'shear' else shapes))
