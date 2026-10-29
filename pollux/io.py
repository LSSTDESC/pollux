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

logger = logging.getLogger(__name__)


class PolluxIO:
    """Convenience loader for Pollux simulation data products.

    This class provides lightweight, file-only accessors (no ``pollux``
    builders are instantiated) to load shells, halo catalogs, lensing maps,
    intrinsic-alignment/tidal maps, density maps, angular power spectra,
    mock source catalogs, and baryonified variants of the above, from a
    Pollux output directory.

    Parameters
    ----------
    sim_path : str
        Path to the simulation shells directory containing ``sim_info.yaml``
        (typically ``{root_dir}/nside{nside}/seed{seed}/shells``).
    seed : int or None, optional
        Seed identifier, kept for reference only (already encoded in
        ``sim_path`` and not used to build filenames).
    tomo : str or None, optional
        Lensing tag / tomography directory name (e.g. ``'SRDv1_nz'``), i.e.
        the ``tag`` passed to the lensing builder. Products under
        ``{sim_path}/kappa_{tomo}/`` (kappa, shear, source catalogs, kappa
        cls) are only reachable when this is set.
    shell_kind : str, optional
        ``reader.kind`` used by the particle-shell reader (see
        :class:`pollux.reader.HACCReader`). Determines the ``shell_{kind}``
        filename prefix. Default is ``'hacc'``.
    halo_kind : str, optional
        ``reader.kind`` used by the halo reader (see
        :class:`pollux.reader.HACCHaloReader`). Default is ``'hacc_halo'``.
    catalog_tag : str or None, optional
        Tag passed to the source-catalog generator (``CatalogSourceGenerator``),
        determining the ``catalogs_{tag}`` subdirectory. If None, uses
        ``catalogs/`` (untagged).
    baryon_tag : str or None, optional
        Tag passed to :class:`~pollux.baryons.BaryonifiedShellBuilder`,
        determining the ``baryons_{tag}`` subdirectory. If None, uses
        ``baryons/`` (untagged).

    Notes
    -----
    Two independent reader "kinds" can share this same ``sim_path`` (the
    particle-shell reader and the halo reader), each with its own
    ``base_fname`` prefix (``shell_{shell_kind}`` / ``shell_{halo_kind}``).
    ``sim_info.yaml`` only records whichever builder wrote it first, so
    ``shell_kind``/``halo_kind`` are taken as explicit parameters here rather
    than inferred from that file.
    """

    def __init__(
        self,
        sim_path: str,
        seed: int | None = None,
        tomo: str | None = "SRDv1_nz",
        shell_kind: str = "hacc",
        halo_kind: str = "hacc_halo",
        catalog_tag: str | None = None,
        baryon_tag: str | None = None,
    ) -> None:
        self.sim_path = sim_path
        self.seed = seed
        self.tomo = tomo
        self.shell_kind = shell_kind
        self.halo_kind = halo_kind
        self.catalog_tag = catalog_tag
        self.baryon_tag = baryon_tag

        self.shell_prefix = f"shell_{shell_kind}"
        self.halo_prefix = f"shell_{halo_kind}"

        logger.info("*** PolluxIO ***")
        logger.info("  - Path = %s.", sim_path)

        # Basic simulation info
        with open(os.path.join(self.sim_path, "sim_info.yaml")) as f:
            info = yaml.unsafe_load(f)
        for k, v in info.items():
            setattr(self, k, v)
            logger.info("  - %s = %s.", k, v)

        # Shell info (per-shell pixel/sky density); each reader kind writes its
        # own file since it is named after base_fname (shell_{shell_kind}).
        self.shell_info = np.load(os.path.join(self.sim_path, f"{self.shell_prefix}_info.npy"))

        # Cosmology
        self.cosmo = pyccl.Cosmology.read_yaml(os.path.join(self.sim_path, "cosmo_ccl.yaml"))
        logger.info("  - Cosmology = %s.", self.cosmo)

        # Redshift distributions
        if tomo is not None:
            self.nzs = np.load(os.path.join(self.sim_path, f"kappa_{tomo}", "nzs.npy"))

    @staticmethod
    def _smooth_suffix(smooth_Mpc: float | None) -> str:
        """Return the filename suffix of a smoothing scale (empty without smoothing)."""
        return f"_{smooth_Mpc}Mpc" if smooth_Mpc is not None else ""

    def _baryon_dir(self) -> str:
        """Return the directory of the baryonified products."""
        sub = f"baryons_{self.baryon_tag}" if self.baryon_tag is not None else "baryons"
        return os.path.join(self.sim_path, sub)

    def _kappa_dir(self) -> str:
        """Return the directory of the lensing products of tomography ``tomo``."""
        if self.tomo is None:
            raise ValueError('tomo must be set to reach kappa/shear/source-catalog products.')
        return os.path.join(self.sim_path, f"kappa_{self.tomo}")

    # ------------------------------------------------------------------
    # Particle shells
    # ------------------------------------------------------------------

    def load_shell(self, ishell: int, density: bool = False) -> hsp.HealSparseMap:
        """Load a particle-shell count map.

        Parameters
        ----------
        ishell : int
            Shell index.
        density : bool, optional
            If True, return density contrast delta = shell/pixel_density - 1.

        Returns
        -------
        hsp.HealSparseMap
            Shell count map, or density contrast if ``density=True``.
        """
        fname = os.path.join(self.sim_path, f"{self.shell_prefix}_imap{ishell}.fits")
        shell = hsp.HealSparseMap.read(fname)

        if density:
            return shell / self.shell_info["pixel_density"][ishell] - 1.0
        return shell

    def load_shell_alms(self, ishell: int) -> np.ndarray:
        """Load the a_lm coefficients of one particle shell.

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        ndarray
            a_lm coefficients (healpy ordering).
        """
        fname = os.path.join(self.sim_path, f"{self.shell_prefix}_alms_imap{ishell}.fits")
        return hp.read_alm(fname)

    def load_shell_cls(self) -> sacc.Sacc:
        """Load the SACC file of measured shell (galaxy density) power spectra.

        Returns
        -------
        sacc.Sacc
        """
        fname = os.path.join(self.sim_path, f"{self.shell_prefix}_cls.fits")
        return sacc.Sacc.load_fits(fname)

    def load_density(self, ishell: int, smooth_Mpc: float | None = None) -> hsp.HealSparseMap:
        """Load a density-contrast map for one shell.

        Parameters
        ----------
        ishell : int
            Shell index.
        smooth_Mpc : float or None, optional
            Gaussian smoothing scale in Mpc of the map.

        Returns
        -------
        hsp.HealSparseMap
        """
        fname = os.path.join(self.sim_path, f"{self.shell_prefix}_density{self._smooth_suffix(smooth_Mpc)}_imap{ishell}.fits")
        return hsp.HealSparseMap.read(fname)

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
        suffix = self._smooth_suffix(smooth_Mpc)
        maps = [hsp.HealSparseMap.read(os.path.join(self.sim_path, f"{self.shell_prefix}_{what}{suffix}_imap{ishell}.fits")) for what in ("t11", "t22", "t12")]
        return tuple(maps)

    # ------------------------------------------------------------------
    # Halo catalogs
    # ------------------------------------------------------------------

    def load_haloes(self, ishell: int) -> dict[str, np.ndarray]:
        """Load the halo catalog for one shell as a dict of concatenated arrays.

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        dict
            Mapping from column name (``'ihaloes'``, ``'ra'``, ``'dec'``,
            ``'chi'``, ``'z'``, and any ``extra_columns`` saved at
            construction) to 1D concatenated numpy arrays.
        """
        fname = os.path.join(self.sim_path, f"{self.halo_prefix}_catalog_imap{ishell}.fits")
        with fits.open(fname) as hdul:
            columns = hdul[1].columns.names
            return {k: np.concatenate([hdu.data[k] for hdu in hdul[1:]]) for k in columns}

    # ------------------------------------------------------------------
    # Lensing (kappa/shear)
    # ------------------------------------------------------------------

    def load_kappa(self, ikappa: int, smooth_Mpc: float | None = None) -> hsp.HealSparseMap:
        """Load the convergence map of one tomographic/snapshot bin.

        Parameters
        ----------
        ikappa : int
            Bin index.
        smooth_Mpc : float or None, optional
            Gaussian smoothing scale in Mpc of the map.

        Returns
        -------
        hsp.HealSparseMap
        """
        fname = os.path.join(self._kappa_dir(), f"kappa{self._smooth_suffix(smooth_Mpc)}_imap{ikappa}.fits")
        return hsp.HealSparseMap.read(fname)

    def load_kappa_alms(self, ikappa: int) -> np.ndarray:
        """Load the a_lm coefficients of one convergence map.

        Parameters
        ----------
        ikappa : int
            Bin index.

        Returns
        -------
        ndarray
            a_lm coefficients (healpy ordering).
        """
        fname = os.path.join(self._kappa_dir(), f"kappa_alms_imap{ikappa}.fits")
        return hp.read_alm(fname)

    def load_shear(self, ikappa: int, smooth_Mpc: float | None = None) -> tuple[hsp.HealSparseMap, hsp.HealSparseMap]:
        """Load the shear maps of one tomographic/snapshot bin.

        Parameters
        ----------
        ikappa : int
            Bin index.
        smooth_Mpc : float or None, optional
            Gaussian smoothing scale in Mpc of the maps.

        Returns
        -------
        tuple of hsp.HealSparseMap
            ``(gamma1, gamma2)``.
        """
        suffix = self._smooth_suffix(smooth_Mpc)
        g1 = hsp.HealSparseMap.read(os.path.join(self._kappa_dir(), f"kappa_gamma1{suffix}_imap{ikappa}.fits"))
        g2 = hsp.HealSparseMap.read(os.path.join(self._kappa_dir(), f"kappa_gamma2{suffix}_imap{ikappa}.fits"))
        return g1, g2

    def load_kappa_cls(self) -> sacc.Sacc:
        """Load the SACC file of measured convergence power spectra.

        Returns
        -------
        sacc.Sacc
        """
        fname = os.path.join(self._kappa_dir(), "kappa_cls.fits")
        return sacc.Sacc.load_fits(fname)

    # ------------------------------------------------------------------
    # Mock source catalogs
    # ------------------------------------------------------------------

    def load_source_catalog(self, ikappa: int) -> fits.HDUList:
        """Open the mock source catalog FITS file of one tomographic bin.

        Parameters
        ----------
        ikappa : int
            Bin index.

        Returns
        -------
        astropy.io.fits.HDUList
        """
        cat_dir = f"catalogs_{self.catalog_tag}" if self.catalog_tag is not None else "catalogs"
        fname = os.path.join(self._kappa_dir(), cat_dir, f"cat_imap{ikappa}.fits")
        return fits.open(fname)

    # ------------------------------------------------------------------
    # Baryonified products (BaryonifiedShellBuilder; base_fname='shell')
    # ------------------------------------------------------------------

    def load_baryon_shell(self, ishell: int, density: bool = False) -> hsp.HealSparseMap:
        """Load a baryonified shell count map.

        Parameters
        ----------
        ishell : int
            Shell index.
        density : bool, optional
            If True, return density contrast delta = shell/pixel_density - 1.

        Returns
        -------
        hsp.HealSparseMap
            Baryonified count map, or density contrast if ``density=True``.
        """
        fname = os.path.join(self._baryon_dir(), f"shell_imap{ishell}.fits")
        shell = hsp.HealSparseMap.read(fname)

        if density:
            return shell / self.shell_info["pixel_density"][ishell] - 1.0
        return shell

    def load_baryon_density(self, ishell: int, smooth_Mpc: float | None = None) -> hsp.HealSparseMap:
        """Load a baryonified density-contrast map.

        Parameters
        ----------
        ishell : int
            Shell index.
        smooth_Mpc : float or None, optional
            Gaussian smoothing scale in Mpc of the map.

        Returns
        -------
        hsp.HealSparseMap
        """
        fname = os.path.join(self._baryon_dir(), f"shell_density{self._smooth_suffix(smooth_Mpc)}_imap{ishell}.fits")
        return hsp.HealSparseMap.read(fname)

    def load_baryon_tidal(self, ishell: int, smooth_Mpc: float | None = None) -> tuple[hsp.HealSparseMap, hsp.HealSparseMap, hsp.HealSparseMap]:
        """Load the baryonified tidal-tensor components (t11, t22, t12) of one shell.

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
        suffix = self._smooth_suffix(smooth_Mpc)
        maps = [hsp.HealSparseMap.read(os.path.join(self._baryon_dir(), f"shell_{what}{suffix}_imap{ishell}.fits")) for what in ("t11", "t22", "t12")]
        return tuple(maps)
