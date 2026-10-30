from __future__ import annotations

import logging
import os
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import healpy as hp
import pymaster as nmt
import sacc
import treecorr
import fitsio
from astropy.io import fits

from .cosmo import Tracer
from .utils import code_version, radec2thetaphi, quad, sample_pdf

if TYPE_CHECKING:
    import healsparse as hsp
    import pyccl
    from .lensing import LensingBuilder

logger = logging.getLogger(__name__)


class CatalogSourceGenerator(ABC):
    """Generic generator for mock source-galaxy catalogs for lensing analyses.

    This class generates positions and redshifts for galaxies in each tomographic
    bin and interpolates lensing quantities (e.g. kappa/gamma) from maps produced
    by a lensing builder.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder used to define tomographic bins and provide maps to
        interpolate.
    tag : str or None, optional
        Tag used to name the output directory ``catalogs_{tag}/``. Default is None.
    nbar_input : float, array-like, or None, optional
        Galaxy surface density in gal/arcmin^2. If a float, the same surface
        density is used for all tomographic bins; if an array-like, it must
        have length ``lensing.nmaps`` and provide a per-bin value. If ``None``,
        the values are taken from ``lensing.nbar``. Default is None.
    interp_fields : tuple of str, optional
        Fields interpolated at the galaxies, among ``FIELDS`` (``'kappa'``,
        ``'gamma1'``, ``'gamma2'``, ``'density'``, ``'t11'``, ``'t22'``, ``'t12'``);
        the fields required by the augmentors are added. Default is
        ``('kappa', 'gamma1', 'gamma2')``.
    interp_method : {'bilinear', 'nearest'}, optional
        Interpolation of the maps at the galaxy positions: bilinear across pixels
        (default), or the value of the nearest pixel.
    smooth_fields : tuple of str, optional
        Fields read from their maps smoothed by ``smooth_Mpc``, among ``FIELDS`` and
        ``'positions'`` (the density contrast used to place galaxies). Default is
        ``('positions', 'density', 't11', 't22', 't12')``. The smoothed maps are made
        by the builders (:meth:`~pollux.map_builder.MapBuilder.compute_maps_from_alms`,
        :meth:`~pollux.shells.ShellBuilder.compute_tidal_tensor_maps`,
        :meth:`~pollux.lensing.LensingBuilder.compute_shear_maps`).
    smooth_Mpc : float or dict or None, optional
        Comoving Gaussian smoothing scale in Mpc of ``smooth_fields``, or a dict of
        scales per field (then ``smooth_fields`` is ignored, and fields not in the dict
        are not smoothed). None (default) means no smoothing. Fields made together
        (``t11``, ``t22``, ``t12``; ``gamma1``, ``gamma2``) must share their scale.
    augmentors : list, optional
        List of catalog augmentors. Default is empty list.
    cls_b : pymaster.NmtBin or None, optional
        Binning of :meth:`compute_cls` and :meth:`predict_cls`, whose ``lmax`` is the
        maximum multipole of the catalog fields and workspaces. Default is the shell
        builder's binning (see :attr:`b`).
    cls_lmax_mask : int or None, optional
        Maximum multipole of the fields' masks (at least the ``lmax`` of :attr:`b`). Default is that ``lmax``.

    Attributes
    ----------
    Ngal : ndarray of int
        Number of galaxies drawn per tomographic bin.
    nbar_per_pix : ndarray
        Mean expected counts per pixel per tomographic bin.
    header_dict : dict
        FITS header keyword/value pairs stored in the output catalogs, including the
        code version: ``POLLUXV`` (git commit) and ``DIRTY`` (uncommitted changes).
    """

    # fields that can be interpolated at the galaxies
    FIELDS = ('kappa', 'gamma1', 'gamma2', 'density', 't11', 't22', 't12')
    # fields made together by the builders, which must share their smoothing
    _SMOOTHED_TOGETHER = (('t11', 't22', 't12'), ('gamma1', 'gamma2'))

    def __init__(
        self,
        lensing: LensingBuilder,
        tag: str | None = None,
        nbar_input: float | np.ndarray | None = None,
        interp_fields: tuple[str, ...] = ('kappa', 'gamma1', 'gamma2'),
        interp_method: str = 'bilinear',
        smooth_fields: tuple[str, ...] = ('positions', 'density', 't11', 't22', 't12'),
        smooth_Mpc: float | dict[str, float] | None = None,
        augmentors: list | None = None,
        cls_b: nmt.NmtBin | None = None,
        cls_lmax_mask: int | None = None,
    ) -> None:
        self.lensing = lensing
        self.tag = tag
        self.nbar_input = nbar_input
        if interp_method not in ('bilinear', 'nearest'):
            raise ValueError(f"interp_method must be 'bilinear' or 'nearest', got {interp_method!r}")
        if not set(interp_fields) <= set(self.FIELDS):
            raise ValueError(f"interp_fields must be among {self.FIELDS}, got {interp_fields!r}")
        self.interp_method = interp_method
        self.smooth_fields = tuple(smooth_fields)
        self.smooth_Mpc = smooth_Mpc
        self.smoothing = self.smoothing_scales(smooth_fields, smooth_Mpc)

        self.interp_fields = tuple(interp_fields)
        self.augmentors = augmentors if augmentors is not None else []

        # Catalogs live alongside the lensing maps; tag distinguishes parallel runs
        cat_dir = f'catalogs_{tag}' if tag is not None else 'catalogs'
        self._catalog_dir = os.path.join(lensing.save_dir, cat_dir)
        os.makedirs(self._catalog_dir, exist_ok=True)

        # Compute derived attributes
        self.cosmo = self.lensing.cosmo
        self.snap_lensing = self.lensing.shell_builder.snapshot_lensing_builder()
        self.nside = self.lensing.nside
        self.ncats = self.lensing.nmaps
        self.nshells = self.lensing.shell_builder.nshells

        # Handle augmentor fields
        for augmentor in self.augmentors:
            for field_name in augmentor.required_interp_fields:
                if field_name not in self.interp_fields:
                    warnings.warn(f"Adding field '{field_name}' to interp_fields for augmentor '{augmentor.name}'.")
                    self.interp_fields += (field_name,)
        self.augmentor_kwargs = {ikappa: {'ikappa': ikappa} for ikappa in self.lensing.iter()}
        self.cls_b = cls_b
        self.cls_lmax_mask = cls_lmax_mask
        self.predicted_cls = None
        self._wsp = {}

        # Handle nbar
        if self.nbar_input is None:
            nbar_arr = np.atleast_1d(self.lensing.nbar).astype(float)
        else:
            nbar_arr = np.atleast_1d(self.nbar_input).astype(float)
            if nbar_arr.size == 1:
                nbar_arr = np.full(self.lensing.nmaps, float(nbar_arr[0]))
            elif nbar_arr.size != self.lensing.nmaps:
                raise ValueError(f"nbar must be a scalar or array of length {self.lensing.nmaps}, got {nbar_arr.size}")
        self.nbar = nbar_arr

        # Compute galaxy counts and areas
        area_std = np.pi / 2. if self.lensing.octant_only else 4. * np.pi
        area_arcmin = area_std * (180. * 60 / np.pi)**2
        self.Ngal = (self.nbar * area_arcmin).astype(int)
        self.area_pix = hp.nside2pixarea(self.nside, degrees=False) * (180. * 60. / np.pi)**2
        self.nbar_per_pix = self.nbar * self.area_pix
        self.header_dict = {f'NBAR_{i}': n for i, n in enumerate(self.nbar)}
        version = code_version()  # code that produced the catalogs
        self.header_dict.update(POLLUXV=(version['commit'] or 'unknown')[:12], DIRTY=version['dirty'])

        logger.info("*** Catalog (%s) ***", self.tag if self.tag is not None else 'default')
        logger.info("  - Catalog class = %s.", self.__class__.__name__)
        logger.info("  - Lensing class = %s.", self.lensing.__class__.__name__)
        logger.info("  - Shell builder class = %s.", self.lensing.shell_builder.__class__.__name__)
        logger.info("  - Number of catalogs = %d.", self.ncats)
        logger.info("  - Catalogs saved to %s.", self._catalog_dir)

    @classmethod
    def smoothing_scales(cls, smooth_fields: tuple[str, ...], smooth_Mpc: float | dict[str, float] | None) -> dict[str, float | None]:
        """Return the smoothing scale of every field (see ``smooth_fields`` and ``smooth_Mpc``).

        Parameters
        ----------
        smooth_fields : tuple of str
            Fields smoothed by a single scale ``smooth_Mpc``.
        smooth_Mpc : float or dict or None
            Smoothing scale, or scales per field.

        Returns
        -------
        dict
            Scale (Mpc) or None of each of ``FIELDS`` and ``'positions'``, e.g. to
            schedule the smoothed maps needed by a catalog.
        """
        names = cls.FIELDS + ('positions',)
        if isinstance(smooth_Mpc, dict):
            scales = dict(smooth_Mpc)
        else:
            scales = {k: smooth_Mpc for k in smooth_fields} if smooth_Mpc else {}
        unknown = set(scales) - set(names)
        if unknown:
            raise ValueError(f"Unknown fields {sorted(unknown)} for smoothing, must be among {names}")
        scales = {k: scales.get(k) or None for k in names}
        for group in cls._SMOOTHED_TOGETHER:
            if len({scales[k] for k in group}) > 1:
                raise ValueError(f"Fields {group} are made together and must share their smoothing scale")
        return scales

    def generate(self, ikappas: int | Iterable[int] | None = None, ishell: int | None = None) -> None:
        """Generate mock catalogs and interpolate requested map quantities.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate. If None, generate for all bins.
        ishell : int or None, optional
            If given, only generate the galaxies of shell ``ishell`` and save them
            as partial catalogs (one per bin), to be combined with
            :meth:`combine_shells`. Only supported by subclasses drawing galaxies
            shell by shell (:class:`CatalogSourceClusteredPoisson`).

        Notes
        -----
        The generation proceeds in four steps:
        1) draw galaxies (positions/redshifts and shell membership),
        2) interpolate requested fields from maps,
        3) apply the augmentors,
        4) save one FITS file per tomographic bin.
        """
        catalog = {ikappa: {} for ikappa in self.lensing.iter(ikappas)}
        ishells = None if ishell is None else [ishell]

        # Generate ra, dec, z, etc.
        self._generate_galaxies(catalog, ikappas=ikappas, ishells=ishells)

        # Interpolate kappa/shear values
        self._interpolate_maps(catalog, ikappas=ikappas, ishells=ishells)

        # Augment catalogs if requested
        for augmentor in self.augmentors:
            for ikappa in self.lensing.tqdm(ikappas, desc=f"[generate] augmenting catalogs with {augmentor.name}", leave=False):
                augmentor.augment_catalog(self.cosmo, catalog[ikappa], **self.augmentor_kwargs[ikappa])

        # Save it!
        self._save_catalogs(catalog, ikappas=ikappas, ishell=ishell)

    def combine_shells(self, ikappas: int | Iterable[int] | None = None) -> None:
        """Concatenate the per-shell partial catalogs into one catalog per bin.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to merge. If None, merge all bins.

        Notes
        -----
        Partial catalogs are produced by :meth:`generate` with ``ishell`` and are
        concatenated in shell order, which gives the same catalog as generating
        all shells at once.
        """
        for ikappa in self.lensing.tqdm(ikappas, desc="[combine_shells]", leave=False):
            parts = [fits.getdata(self.get_fname(ikappa, ishell=ishell), 1) for ishell in range(self.nshells)]
            header = fits.Header()
            for k, v in self.header_dict.items():
                header[k] = v
            hdul = [fits.PrimaryHDU(), fits.BinTableHDU(data=np.concatenate(parts), header=header)]
            fname = self.get_fname(ikappa)
            fits.HDUList(hdul).writeto(fname, overwrite=True)
            logger.debug("Saved to file %s.", fname)

    @abstractmethod
    def _generate_galaxies(
            self, catalog: dict[int, dict[str, np.ndarray]], ikappas: int | Iterable[int] | None = None, ishells: list[int] | None = None) -> None:
        """Populate a catalog container with galaxy properties (redshifts and positions).

        Parameters
        ----------
        catalog : dict
            Dictionary containing per-bin catalog data that will be filled in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        ishells : list of int or None, optional
            Shells whose galaxies are generated. If None, all shells.
        """

    def _generate_z(self, catalog: dict[int, dict[str, np.ndarray]], ikappa: int) -> None:
        """Draw redshifts from the tracer n(z) via inverse-CDF sampling.

        Parameters
        ----------
        catalog : dict
            Per-bin output container.
        ikappa : int
            Tomographic-bin index.

        Notes
        -----
        Shell membership is set by digitizing redshifts onto the shell edges
        returned by ``shell_builder.shell_edges_redshift``.
        """
        builder = self.lensing.shell_builder
        z_edges = builder.shell_edges_redshift

        rng = np.random.default_rng(self.lensing.shell_builder.seed + ikappa * self.nshells)

        # Generate redshifts according to lensing object redshift distributions (using cdf sampling)
        t = self.lensing.tracers_ccl[ikappa]
        zz = np.linspace(0, z_edges[-1], 1000)
        nz = t.get_dndz(zz)

        z = sample_pdf(zz, nz, self.Ngal[ikappa], rng)
        chi = builder.cosmo.comoving_radial_distance(1. / (1. + z))

        catalog[ikappa]['z'] = z
        catalog[ikappa]['chi'] = chi
        catalog[ikappa]['idx_shell'] = np.digitize(z, z_edges) - 1

    def _interpolate_maps(
            self, catalog: dict[int, dict[str, np.ndarray]], ikappas: int | Iterable[int] | None = None, ishells: list[int] | None = None) -> None:
        """Interpolate map quantities at galaxy positions.

        Parameters
        ----------
        catalog : dict
            Per-bin catalog dictionaries containing at least ``'ra'``, ``'dec'``,
            ``'chi'`` and ``'idx_shell'``.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to process.
        ishells : list of int or None, optional
            Shells whose galaxies are interpolated. If None, all shells.

        Notes
        -----
        This performs linear interpolation between consecutive shells in comoving
        distance and optionally interpolates across pixels (via HealSparse
        ``interpolate_pos``). For ``k='delta'``, a density contrast map is built
        from snapshot-lensing density shells.
        """
        for k in self.interp_fields:
            if ishells is None:
                # all shells: each map is read once and reused as the lower map of the next shell
                map_high = None
                for ishell in self.lensing.shell_builder.tqdm(None, desc=f"[generate] interpolating maps {k}"):
                    map_low = map_high
                    map_high = self._load_interp_map(k, ishell)
                    self._interpolate_shell(catalog, k, ishell, map_low, map_high, ikappas=ikappas)
            else:
                for ishell in ishells:
                    map_low = self._load_interp_map(k, ishell - 1) if ishell > 0 and 'delta' not in k else None
                    map_high = self._load_interp_map(k, ishell)
                    self._interpolate_shell(catalog, k, ishell, map_low, map_high, ikappas=ikappas)

    def _density_map(self, ishell: int, smooth_Mpc: float | None) -> hsp.HealSparseMap:
        """Return the density contrast of one shell, smoothed by ``smooth_Mpc`` (Mpc) if not None.

        Parameters
        ----------
        ishell : int
            Shell index.
        smooth_Mpc : float or None
            Smoothing scale of the map (saved by the shell builder's
            :meth:`~pollux.shells.ShellBuilder.compute_tidal_tensor_maps`).

        Returns
        -------
        hsp.HealSparseMap
        """
        builder = self.lensing.shell_builder
        if smooth_Mpc is None:
            return builder.field_hspmap(ishell)
        return builder.load_maps(ishell, what='density', smooth_Mpc=smooth_Mpc)

    def _load_interp_map(self, k: str, ishell: int) -> hsp.HealSparseMap:
        """Load the map of field ``k`` for one shell.

        Parameters
        ----------
        k : str
            Field name (e.g. ``'kappa'``, ``'gamma1'``, ``'density'``, ``'t11'``).
        ishell : int
            Shell index.

        Returns
        -------
        hsp.HealSparseMap
            Map of field ``k``, smoothed as given by ``smooth_fields`` and ``smooth_Mpc``.
        """
        smooth_Mpc = self.smoothing[k]
        # Fields that live in the shell builder (density/tidal), vs the snapshot lensing builder
        if k == 'density':
            hspmap = self._density_map(ishell, smooth_Mpc)
        elif k in ('t11', 't22', 't12'):
            hspmap = self.lensing.shell_builder.load_maps(ishell, what=k, smooth_Mpc=smooth_Mpc)
        else:
            # kappa, gamma1, gamma2: live in the snapshot lensing builder.
            # kappa is the default product (what=None); gamma* use what=k.
            what = None if k == 'kappa' else k
            hspmap = self.snap_lensing.load_maps(ishell, what=what, smooth_Mpc=smooth_Mpc)

        return hspmap

    def _interpolate_shell(
            self,
            catalog: dict[int, dict[str, np.ndarray]],
            k: str,
            ishell: int,
            map_low: hsp.HealSparseMap | None,
            map_high: hsp.HealSparseMap,
            ikappas: int | Iterable[int] | None = None) -> None:
        """Interpolate field ``k`` at the positions of the galaxies of one shell.

        Parameters
        ----------
        catalog : dict
            Per-bin catalog dictionaries, updated in place.
        k : str
            Field name.
        ishell : int
            Shell index.
        map_low, map_high : hsp.HealSparseMap or None
            Maps of field ``k`` for shells ``ishell - 1`` and ``ishell``; ``map_low``
            is unused for ``ishell == 0`` and for ``'delta'`` fields.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to process.
        """
        chi_low, chi_high = self.lensing.shell_builder.shell_edges_comoving_distance[[ishell, ishell + 1]]

        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, disable=True):
            # select galaxies in current shell
            w = (catalog[ikappa]['idx_shell'] == ishell)

            if np.any(w):
                ra = catalog[ikappa]['ra'][w]
                dec = catalog[ikappa]['dec'][w]
                if 'delta' in k:
                    # no interpolation between shells for delta (integrated quantity)
                    if self.interp_method == 'bilinear':
                        catalog[ikappa][k][w] = map_high.interpolate_pos(
                            ra,
                            dec,
                            allow_partial=True,
                        )
                    else:
                        idx = hp.ang2pix(
                            self.nside,
                            *radec2thetaphi(ra, dec),
                            nest=True,
                        )
                        catalog[ikappa][k][w] = map_high[idx]
                else:
                    # prepare for interpolation
                    dchi = (catalog[ikappa]['chi'][w] - chi_low)  # distance

                    # interpolation between shells and (optionnally) accross pixels
                    if self.interp_method == 'bilinear':
                        if ishell == 0:
                            catalog[ikappa][k][w] = dchi * map_high.interpolate_pos(ra, dec, allow_partial=True) / (chi_high - chi_low)
                        else:
                            map_low_interp = map_low.interpolate_pos(ra, dec, allow_partial=True)
                            map_high_interp = map_high.interpolate_pos(ra, dec, allow_partial=True)
                            catalog[ikappa][k][w] = map_low_interp + dchi * (map_high_interp - map_low_interp) / (chi_high - chi_low)
                    else:
                        idx = hp.ang2pix(self.nside, *radec2thetaphi(ra, dec), nest=True)
                        if ishell == 0:
                            catalog[ikappa][k][w] = dchi * map_high[idx] / (chi_high - chi_low)
                        else:
                            catalog[ikappa][k][w] = map_low[idx] + dchi * (map_high[idx] - map_low[idx]) / (chi_high - chi_low)

    def get_fname(self, ikappa: int, format: str = 'fits', ishell: int | None = None) -> str:
        """Return the path for the saved catalog of one tomographic bin.

        Parameters
        ----------
        ikappa : int
            Tomographic-bin index.
        format : str, optional
            File extension without the leading dot. Default is ``'fits'``.
        ishell : int or None, optional
            If given, path of the partial catalog of shell ``ishell``.

        Returns
        -------
        str
            ``{lensing.save_dir}/catalogs_{tag}/cat_imap{ikappa}.{format}``
            (or ``catalogs/`` when tag is None), or
            ``.../shells/cat_imap{ikappa}_shell{ishell}.{format}`` for a partial catalog.
        """
        if ishell is None:
            return os.path.join(self._catalog_dir, f'cat_imap{ikappa}.{format}')
        return os.path.join(self._catalog_dir, 'shells', f'cat_imap{ikappa}_shell{ishell}.{format}')

    def _save_catalogs(self, catalog: dict[int, dict[str, np.ndarray]], ikappas: int | Iterable[int] | None = None, ishell: int | None = None) -> None:
        """Write per-bin catalogs to FITS files.

        Parameters
        ----------
        catalog : dict
            Per-bin outputs containing ``'ra'``, ``'dec'``, ``'z'``, ``'chi'`` and
            any fields listed in ``self.interp_fields``.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to save.
        ishell : int or None, optional
            If given, save as the partial catalogs of shell ``ishell``.
        """
        columns = list(self.interp_fields)
        for augmentor in self.augmentors:
            columns += augmentor.new_columns
        if ishell is not None:
            os.makedirs(os.path.dirname(self.get_fname(0, ishell=ishell)), exist_ok=True)

        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] saving", leave=False):

            ra = fits.Column(name='ra', format='E', array=catalog[ikappa]['ra'])
            dec = fits.Column(name='dec', format='E', array=catalog[ikappa]['dec'])
            z = fits.Column(name='z', format='E', array=catalog[ikappa]['z'])
            chi = fits.Column(name='chi', format='E', array=catalog[ikappa]['chi'])

            interp_colums = [fits.Column(name=k, format='E', array=catalog[ikappa][k]) for k in columns]

            header = fits.Header()
            for k, v in self.header_dict.items():
                header[k] = v

            hdul = [fits.PrimaryHDU(), fits.BinTableHDU.from_columns([ra, dec, z, chi] + interp_colums, header)]

            fname = self.get_fname(ikappa, format='fits', ishell=ishell)
            fits.HDUList(hdul).writeto(fname, overwrite=True)
            logger.debug("Saved to file %s.", fname)

    def load_catalogs(
            self,
            ikappas: int | Iterable[int] | None = None,
            return_header: bool = False,
            subsample: float | None = None,
            seed: int | None = None,
            columns: list[str] | None = None,
            remove_mean: list[str] | None = None) -> np.ndarray | dict[int, np.ndarray] | tuple:
        """Load saved FITS catalog(s) for one or more tomographic bins.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to load. If None, loads all bins.
        return_header : bool, optional
            If True, also return the FITS header(s).
        subsample : float or None, optional
            If not None, randomly draw (without replacement, seeded by
            ``seed + ikappa``) this fraction of galaxies from each loaded
            catalog, with ``0 < subsample < 1``. A random draw is used rather
            than a fixed stride, since galaxies are stored in a fixed
            (non-random) order and a stride would sample a specific,
            non-representative pattern. Default is None (no subsampling).
        seed : int or None, optional
            Seed for the subsampling RNG. Only used when ``subsample`` is not
            None. Default is the shell builder's seed.
        columns : list of str or None, optional
            Columns to read. If None (default), all columns.
        remove_mean : list of str or None, optional
            Columns whose mean over each loaded catalog is subtracted.

        Returns
        -------
        header : astropy.io.fits.Header or dict, optional
            Only if ``return_header=True``. FITS header from the binary table
            HDU if ``ikappas`` is a single int, otherwise a dict keyed by bin index.
        table : numpy.ndarray or dict
            Structured array (native byte order) if ``ikappas`` is a single int,
            otherwise a dict keyed by bin index.

        Notes
        -----
        Only the requested rows and columns are read (with ``fitsio``).
        """
        seed = self.lensing.shell_builder.seed if seed is None else seed

        # read the requested columns of each bin, optionally a random subsample of its rows
        headers = {}
        tables = {}
        for ikappa in self.lensing.tqdm(ikappas, desc="[load_catalogs]"):
            fname = self.get_fname(ikappa, format='fits')
            headers[ikappa] = fits.getheader(fname, ext=1)
            rows = None
            if subsample is not None:
                # random rows (galaxies are stored in shell order, so a stride would not be a fair sample)
                if not 0.0 < subsample < 1.0:
                    raise ValueError('subsample must satisfy 0 < subsample < 1')
                nrows = headers[ikappa]['NAXIS2']
                rows = np.sort(np.random.default_rng(seed + ikappa).choice(nrows, size=int(subsample * nrows), replace=False))
            table = fitsio.read(fname, ext=1, columns=columns, rows=rows)
            # native byte order (FITS is big-endian), for numpy operations
            table = table.astype(table.dtype.newbyteorder('='))
            # e.g. remove the mean shear of the bin
            for col in remove_mean or []:
                table[col] -= table[col].mean()
            tables[ikappa] = table
            logger.debug("Found and loaded catalog at %s.", fname)

        # a single table for a single bin, else dicts keyed by bin
        if np.isscalar(ikappas):
            header, table = headers[ikappas], tables[ikappas]
        else:
            header, table = headers, tables

        if return_header:
            return header, table
        else:
            return table

    @property
    def lensing_tracers(self) -> list[Tracer]:
        """Lensing tracers of the catalog, those of the lensing builder (:attr:`MapBuilder.tracers`).

        Returns
        -------
        list of pollux.cosmo.Tracer
            Shear tracers (one per bin) with the n(z) of the catalog's galaxies.
        """
        return self.lensing.tracers

    @property
    def density_tracers(self) -> list[Tracer]:
        """Density tracers of the catalog: each bin's n(z) and linear bias.

        Returns
        -------
        list of pollux.cosmo.Tracer
            ``'density'`` tracers named ``density_{i}``, with the n(z) of the lensing
            tracers (the galaxies are both lenses and sources), and the smoothing of
            the density used to place them (``'positions'`` smoothing).
        """
        bias = getattr(self, 'bias', np.ones(self.ncats))
        smooth_Mpc = self.smoothing['positions']
        return [Tracer(f"density_{i}", 'density', z=t.z, nz=t.nz, bias=bias[i], smooth_Mpc=smooth_Mpc) for i, t in enumerate(self.lensing_tracers)]

    def _noise_cls(self, tracer: Tracer, shear_cols: tuple[str, str] = ('gamma1', 'gamma2'), subsample: float | None = None) -> float:
        """Return the white noise level of one catalog tracer.

        Parameters
        ----------
        tracer : pollux.cosmo.Tracer
            Density (``density_{i}``) or shear (``kappa_{i}``) tracer of bin ``i``.
        shear_cols : tuple of str, optional
            Shear columns used for the measurement (see :meth:`compute_cls`): shape
            noise only for ellipticities (``e1_*``), none for the lensing shear.
        subsample : float or None, optional
            Fraction of galaxies used for the measurement.

        Returns
        -------
        float
            Shot noise ``1 / n`` (density) or shape noise ``sigma_e**2 / n`` (shear),
            with ``n`` the bin's galaxy density per steradian.
        """
        i = int(tracer.name.rsplit('_', 1)[-1])
        n_sr = self.nbar[i] * (180. * 60. / np.pi)**2 * (subsample or 1.)
        if tracer.kind == 'density':
            return 1. / n_sr
        if not shear_cols[0].startswith('e'):
            return 0.
        sigma_e = next((a.sigma_e for a in self.augmentors if hasattr(a, 'sigma_e')), 0.)
        return sigma_e**2 / n_sr

    @cached_property
    def b(self) -> nmt.NmtBin:
        """Return (and lazily create) the NaMaster binning of :meth:`compute_cls` and :meth:`predict_cls`.

        Returns
        -------
        pymaster.NmtBin
            ``cls_b``, or else the shell builder's binning (:attr:`MapBuilder.b`)
            without its pixel-window deconvolution, since catalog fields are sampled
            at exact positions.
        """
        if self.cls_b is not None:
            return self.cls_b
        b_ref = self.lensing.shell_builder.b
        n_bands = b_ref.get_n_bands()
        ells = np.concatenate([b_ref.get_ell_list(i) for i in range(n_bands)])
        weights = np.concatenate([b_ref.get_weight_list(i) for i in range(n_bands)])
        bpws = np.concatenate([np.full(len(b_ref.get_ell_list(i)), i) for i in range(n_bands)])
        return nmt.NmtBin(ells=ells, bpws=bpws, weights=weights, f_ell=None, lmax=b_ref.lmax)

    @property
    def lmax_mask(self) -> int:
        """Maximum multipole of the masks of the catalog fields and of the footprint (``cls_lmax_mask``)."""
        lmax_mask = self.cls_lmax_mask if self.cls_lmax_mask is not None else self.b.lmax
        if lmax_mask < self.b.lmax:
            raise ValueError(f"cls_lmax_mask={lmax_mask} is below the fields' lmax={self.b.lmax}")
        return lmax_mask

    @cached_property
    def footprint(self) -> np.ndarray:
        """Footprint mask (RING, float, at the shell builder's ``nside``), the mask of map-based fields."""
        return self.lensing.shell_builder.get_mask(nest=False).astype(np.float64)

    def _cls_fields(
            self,
            ikappas: list[int],
            what: list[str],
            shear_cols: tuple[str, str] = ('gamma1', 'gamma2'),
            subsample: float | None = None,
            remove_mean: bool = True,
            randoms: bool = False,
            nran_factor: int = 5,
            shared_randoms: bool = True,
            seed: int | None = None) -> tuple[dict[int, nmt.NmtFieldCatalogClustering], dict[int, nmt.NmtFieldCatalog]]:
        """Build the NaMaster catalog fields used by :meth:`compute_cls`.

        Parameters
        ----------
        ikappas : list of int
            Tomographic bins.
        what : list of str
            Statistics to compute (see :meth:`compute_cls`), which set the fields needed.
        shear_cols, subsample, remove_mean, randoms, nran_factor, shared_randoms, seed : optional
            See :meth:`compute_cls`.

        Returns
        -------
        f_dens, f_shear : dict
            Density (``NmtFieldCatalogClustering``) and spin-2 shear
            (``NmtFieldCatalog``) fields keyed by bin, if needed by ``what``.
            Each field has a ``mask_amplitude`` attribute, the ratio of its mask
            to :attr:`footprint` (its weighted number of galaxies or randoms per
            steradian of footprint, or 1 for density fields with the footprint mask).
        """
        builder = self.lensing.shell_builder
        lmax, lmax_mask = self.b.lmax, self.lmax_mask
        seed = builder.seed if seed is None else seed
        # footprint pixels (to draw randoms) and area (to normalize the mask amplitudes)
        _, ipix_mask = builder.get_mask(nest=True, return_ipix=True)
        area = ipix_mask.size * hp.nside2pixarea(builder.nside)

        # fields needed by the statistics
        need_density = 'clustering' in what or 'shear_clustering' in what
        need_shear = 'shear' in what or 'shear_clustering' in what
        col1, col2 = shear_cols

        # Build each bin's catalog field(s) once, reused across all pairs
        f_dens, f_shear = {}, {}
        shared_radec = None
        for ikappa in self.lensing.tqdm(ikappas, desc="[compute_cls] building fields"):
            cat = self.load_catalogs(
                ikappa, subsample=subsample, seed=seed, columns=['ra', 'dec', *shear_cols], remove_mean=list(shear_cols) if remove_mean else None)
            positions = [cat['ra'], cat['dec']]
            weights_gal = np.ones(cat['ra'].size)

            if need_density:
                # density: footprint mask, or randoms drawn over the footprint
                if not randoms:
                    f = nmt.NmtFieldCatalogClustering(positions, weights_gal, None, None, lmax=lmax, lmax_mask=lmax_mask, lonlat=True, mask=self.footprint)
                    f.mask_amplitude = 1.
                else:
                    # one set of randoms for all bins, or one per bin
                    if shared_randoms:
                        if shared_radec is None:
                            rng = np.random.default_rng(seed)
                            nran = int(nran_factor * cat['ra'].size)
                            shared_radec = generate_radec_in_pix(self.nside, rng.choice(ipix_mask, size=nran), rng)
                        ra_rand, dec_rand = shared_radec
                    else:
                        rng = np.random.default_rng(seed + ikappa)
                        nran = int(nran_factor * cat['ra'].size)
                        ra_rand, dec_rand = generate_radec_in_pix(self.nside, rng.choice(ipix_mask, size=nran), rng)
                    f = nmt.NmtFieldCatalogClustering(
                        positions, weights_gal, [ra_rand, dec_rand], np.ones(len(ra_rand)), lmax=lmax, lmax_mask=lmax_mask, lonlat=True)
                    # randoms per steradian of footprint
                    f.mask_amplitude = len(ra_rand) / area
                f_dens[ikappa] = f

            if need_shear:
                # shear: spin-2 field of the galaxies, whose mask amplitude is their number per steradian
                f = nmt.NmtFieldCatalog(positions, weights_gal, [cat[col1], cat[col2]], spin=2, lmax=lmax, lmax_mask=lmax_mask, lonlat=True)
                f.mask_amplitude = weights_gal.sum() / area
                f_shear[ikappa] = f

        return f_dens, f_shear

    def _wsp_fname(self, stat: str, pair: tuple[int, int] | None = None, randoms: bool = False) -> str:
        """Return the path of a workspace (see :meth:`get_wsp`).

        Parameters
        ----------
        stat : {'clustering', 'shear', 'shear_clustering'}
            Statistic.
        pair : tuple of int or None, optional
            Bin pair, or None for the workspace shared by all pairs.
        randoms : bool, optional
            For a bin pair, whether density fields are masked by randoms (see :meth:`compute_cls`).

        Returns
        -------
        str
            ``{catalog_dir}/wsp_{stat}_lmax{lmax}.wsp``, or
            ``{catalog_dir}/wsp_{stat}_{i}_{j}[_randoms]_lmax{lmax}.wsp`` (with ``i >= j`` for
            the symmetric clustering and shear statistics). Bandpower windows are saved
            next to it, with suffix ``_bpws.npy`` (see :meth:`get_bpws`).
        """
        if pair is None:
            name = stat
        else:
            i, j = pair
            if stat != 'shear_clustering':
                i, j = max(i, j), min(i, j)
            name = f'{stat}_{i}_{j}' + ('_randoms' if randoms and stat != 'shear' else '')
        return os.path.join(self._catalog_dir, f'wsp_{name}_lmax{self.b.lmax}.wsp')

    def get_wsp(
            self,
            stat: str,
            pair: tuple[int, int] | None = None,
            fields: tuple | None = None,
            read_unbinned_MCM: bool = True,
            **field_kwargs: Any) -> nmt.NmtWorkspace:
        """Return (and lazily create) the workspace of one statistic and bin pair.

        The workspace stores the mode-coupling matrix and bandpower windows. It is
        either specific to a bin pair, computed from the masks of the pair's catalog
        fields (the weighted galaxy positions, and the footprint or randoms for
        density), or shared by all bin pairs (``pair=None``), computed from
        :attr:`footprint`. The shared workspace is exact in expectation for galaxies
        uniformly distributed over the footprint once pseudo-spectra are divided by
        the fields' ``mask_amplitude`` (see :meth:`compute_cls`); with clustered
        sources, it misses the coupling from the clustering of galaxy positions.

        Workspaces are cached on disk (:meth:`_wsp_fname`), with their bandpower
        windows when computed; shared workspaces are also cached in memory.

        Parameters
        ----------
        stat : {'clustering', 'shear', 'shear_clustering'}
            Statistic.
        pair : tuple of int or None, optional
            Bin pair, or None (default) for the shared workspace.
        fields : tuple or None, optional
            For a bin pair, the two fields to compute the workspace from, if not
            on disk. By default, the fields of the pair.
        read_unbinned_MCM : bool, optional
            When reading from disk, also read the unbinned mode-coupling matrix,
            needed for bandpower windows but not to decouple spectra, and large
            (~77 GB for spin-2 fields at lmax 24575). Default is True.
        **field_kwargs
            Field options for a bin pair (``shear_cols``, ``randoms``, ``nran_factor``,
            ``seed``, ``subsample``, ``shared_randoms``, ``remove_mean``), see :meth:`compute_cls`.

        Returns
        -------
        pymaster.NmtWorkspace
        """
        # shared workspace already in memory (with the unbinned coupling matrix, if needed)
        if pair is None and stat in self._wsp and (self._wsp[stat][1] or not read_unbinned_MCM):
            return self._wsp[stat][0]

        # otherwise read it from disk, or compute and save it (with its bandpower windows)
        fname = self._wsp_fname(stat, pair, field_kwargs.get('randoms', False))
        wsp = nmt.NmtWorkspace()
        if os.path.exists(fname):
            wsp.read_from(fname, read_unbinned_MCM=read_unbinned_MCM)
            logger.info("Found and loaded workspace at %s.", fname)
        else:
            if pair is None:
                # shared: fields of the footprint mask, with the spins of the statistic
                spins = {'clustering': (0, 0), 'shear': (2, 2), 'shear_clustering': (0, 2)}[stat]
                f_spin = {spin: nmt.NmtField(self.footprint, None, spin=spin, lmax=self.b.lmax, lmax_mask=self.lmax_mask) for spin in set(spins)}
                fields = (f_spin[spins[0]], f_spin[spins[1]])
            elif fields is None:
                # per pair: the catalog fields of the two bins
                i, j = pair
                f_dens, f_shear = self._cls_fields(sorted({i, j}), [stat], **field_kwargs)
                f1, f2 = {'clustering': (f_dens, f_dens), 'shear': (f_shear, f_shear), 'shear_clustering': (f_dens, f_shear)}[stat]
                fields = (f1[i], f2[j])
            wsp.compute_coupling_matrix(*fields, self.b)
            wsp.write_to(fname)
            np.save(fname.replace('.wsp', '_bpws.npy'), wsp.get_bandpower_windows())
            logger.info("Computed and saved workspace to file %s.", fname)
            read_unbinned_MCM = True

        # keep the shared workspace in memory
        if pair is None:
            self._wsp[stat] = (wsp, read_unbinned_MCM)
        return wsp

    def get_bpws(self, stat: str, pair: tuple[int, int] | None = None, randoms: bool = False) -> np.ndarray:
        """Return the bandpower windows of a workspace (see :meth:`get_wsp`).

        Read from the file saved with the workspace, else from the workspace
        (and saved). The shared workspace is computed if missing.

        Parameters
        ----------
        stat : {'clustering', 'shear', 'shear_clustering'}
            Statistic.
        pair : tuple of int or None, optional
            Bin pair, or None (default) for the shared workspace.
        randoms : bool, optional
            For a bin pair, whether density fields are masked by randoms (see :meth:`compute_cls`).

        Returns
        -------
        ndarray
            Shape ``(n_cl, n_bands, n_cl, lmax + 1)``.

        Raises
        ------
        FileNotFoundError
            If the workspace of a bin pair has not been computed (see :meth:`compute_cls`).
        """
        fname = self._wsp_fname(stat, pair, randoms)
        fname_bpws = fname.replace('.wsp', '_bpws.npy')
        if os.path.exists(fname_bpws):
            return np.load(fname_bpws)
        if pair is not None and not os.path.exists(fname):
            raise FileNotFoundError(f"No workspace at {fname}, compute it with compute_cls or get_wsp.")
        bpws = self.get_wsp(stat, pair, randoms=randoms).get_bandpower_windows()
        np.save(fname_bpws, bpws)
        return bpws

    # tracer roles (density or shear) of the (i, j) legs of each harmonic-space SACC data type
    _SACC_TYPE_LEGS_CLS = {
        sacc.standard_types.galaxy_density_cl: ('density', 'density'),
        sacc.standard_types.galaxy_shear_cl_ee: ('shear', 'shear'),
        sacc.standard_types.galaxy_shear_cl_bb: ('shear', 'shear'),
        sacc.standard_types.galaxy_shearDensity_cl_e: ('density', 'shear'),
        sacc.standard_types.galaxy_shearDensity_cl_b: ('density', 'shear'),
    }

    # index of each harmonic-space SACC data type among the coupled spectra of its statistic
    _SACC_TYPE_COMPONENT_CLS = {
        sacc.standard_types.galaxy_density_cl: 0,
        sacc.standard_types.galaxy_shear_cl_ee: 0,
        sacc.standard_types.galaxy_shear_cl_bb: 3,
        sacc.standard_types.galaxy_shearDensity_cl_e: 0,
        sacc.standard_types.galaxy_shearDensity_cl_b: 1,
    }

    # harmonic-space SACC data types of each statistic
    _STAT_TYPES_CLS = {
        'clustering': (sacc.standard_types.galaxy_density_cl,),
        'shear': (sacc.standard_types.galaxy_shear_cl_ee, sacc.standard_types.galaxy_shear_cl_bb),
        'shear_clustering': (sacc.standard_types.galaxy_shearDensity_cl_e, sacc.standard_types.galaxy_shearDensity_cl_b),
    }

    def compute_cls(
            self,
            ikappas: int | Iterable[int] | None = None,
            what: str | list[str] = 'all',
            *,
            auto_only: bool = False,
            pairs: list[tuple[int, int]] | None = None,
            shared_wsp: bool = False,
            randoms: bool = False,
            shear_cols: tuple[str, str] = ('gamma1', 'gamma2'),
            subsample: float | None = None,
            remove_mean: bool = True,
            nran_factor: int = 5,
            shared_randoms: bool = True,
            seed: int | None = None,
            save: bool = True) -> tuple[np.ndarray, dict]:
        """Measure angular power spectra directly from the generated catalog(s),
        including all cross-tomographic-bin combinations.

        Uses NaMaster's catalog-based pseudo-C_ell estimators (``NmtFieldCatalog``
        for shear, ``NmtFieldCatalogClustering`` for clustering), binned with the
        shell builder's own NmtBin (:attr:`ShellBuilder.b`) so that the
        resulting ``ell`` matches other C_ell products in the codebase for direct
        comparability. The pixel-window deconvolution folded into the shell
        builder's binning (appropriate for map-based estimators) is stripped out,
        since catalog fields are sampled at exact positions and have no pixel
        window to correct for.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic bin indices to process. If None, uses all bins.
        what : {'shear', 'clustering', 'shear_clustering', 'all'} or list of str, optional
            Which spectra to compute:
            - 'clustering': galaxy density auto/cross-spectra (dd), symmetric i<=j pairs.
            - 'shear': shear E/B-mode auto/cross-spectra (ee, bb), symmetric i<=j pairs.
            - 'shear_clustering': density-shear cross-spectrum (de, db), all (i, j)
              ordered pairs since lens-bin i / source-bin j is not symmetric.
            - 'all': compute all three. Default.
        auto_only : bool, optional
            If True, restrict to same-bin pairs (i, i) only, skipping cross-bin
            spectra. Default is False (compute all cross-bin combinations).
        pairs : list of tuple of int or None, optional
            If given, only compute these ``(i, j)`` bin pairs, for every
            statistic in ``what`` (overrides ``ikappas`` and ``auto_only``).
            Only the catalogs of the bins involved are loaded.
        shared_wsp : bool, optional
            If True, use one mode-coupling matrix per statistic (dd, shear-shear,
            density-shear) for all bin pairs, computed from the footprint, with
            pseudo-spectra divided by the fields' mask amplitudes (see :meth:`get_wsp`).
            This is exact in expectation for galaxies uniformly distributed over the
            footprint, but misses the coupling from the clustering of galaxy
            positions. If False (default), use one workspace per pair, computed from
            the masks of its fields (exact, slower). Workspaces are cached on disk
            and reused by later calls and by :meth:`predict_cls`; delete them if the
            catalogs or settings change.
        randoms : bool, optional
            If False (default), the mask of density fields is the :attr:`footprint`.
            If True, it is a random catalog (see ``nran_factor`` and ``shared_randoms``).
        shear_cols : tuple of str, optional
            Names of the two catalog columns to use as the spin-2 shear field,
            e.g. ``('gamma1', 'gamma2')`` (lensing shear, default), ``('g1', 'g2')``
            (reduced shear), ``('e1_obs', 'e2_obs')`` (observed ellipticity), or
            any IA-related pair such as ``('e1_ia_ta', 'e2_ia_ta')``. Must be
            present in the loaded catalog (see the augmentors used at generation
            time, e.g. :class:`~pollux.ia.CatalogAugmentorShear`, :class:`~pollux.ia.IACalculator`).
        subsample : float or None, optional
            If not None, randomly draw (without replacement, seeded by
            ``seed + ikappa``) this fraction of galaxies from each loaded
            catalog before building fields, with ``0 < subsample < 1``. A
            random draw is used rather than a fixed stride, since galaxies
            are stored in a fixed (non-random) order and a stride would
            sample a specific, non-representative pattern. Default is None
            (no subsampling).
        remove_mean : bool, optional
            If True (default), subtract the mean of the shear columns over each bin
            before the measurement.
        nran_factor : int, optional
            With ``randoms``, ratio of random points to galaxies for each bin's
            clustering random catalog. Default is 5.
        shared_randoms : bool, optional
            With ``randoms``, if True (default), draw a single random-point realization (sized
            from the first processed bin) and reuse it for every bin's
            density field, rather than drawing independent randoms per bin.
            This matches the fact that all bins share the same survey
            footprint, and is cheaper since randoms are only drawn once. If
            False, each bin gets its own randoms, seeded by ``seed + ikappa``.
        seed : int or None, optional
            Seed for the random catalogs' RNG. Default is the shell builder's seed.
        save : bool, optional
            If True (default), save the spectra to a SACC FITS file via
            :meth:`_save_cls`, at ``{self._catalog_dir}/cls.fits``.

        Returns
        -------
        ell : ndarray
            Effective multipoles of the bandpowers (shared with other C_ell
            products via the same underlying binning, minus pixel-window
            deconvolution).
        cl : dict
            Keyed by SACC data type (e.g. ``'galaxy_density_cl'``,
            ``'galaxy_shear_cl_ee'``, ``'galaxy_shear_cl_bb'``,
            ``'galaxy_shearDensity_cl_e'``, ``'galaxy_shearDensity_cl_b'``,
            depending on ``what``), each mapping to a dict keyed by ``(i, j)``.
        """
        if isinstance(what, str):
            what = list(self._STAT_TYPES_CLS) if what == 'all' else [what]

        # bins, and symmetric pairs (i >= j) for same-type correlations
        ikappas, sym_pairs = self.lensing._pairs(ikappas, auto_only, pairs)
        # bandpowers, and the catalog fields of each bin (built once, reused by all pairs)
        b = self.b
        ell = b.get_effective_ells()
        f_dens, f_shear = self._cls_fields(
            ikappas,
            what,
            shear_cols=shear_cols,
            subsample=subsample,
            remove_mean=remove_mean,
            randoms=randoms,
            nran_factor=nran_factor,
            shared_randoms=shared_randoms,
            seed=seed)

        # decoupled spectra, keyed by SACC data type then bin pair
        cl = {}

        def decoupled(f1: nmt.NmtField, f2: nmt.NmtField, name: str, pair: tuple[int, int]) -> np.ndarray:
            # the shared workspace is that of the footprint, to which pseudo-spectra are rescaled (see get_wsp)
            pcl = nmt.compute_coupled_cell(f1, f2)
            if shared_wsp:
                return self.get_wsp(name, read_unbinned_MCM=False).decouple_cell(pcl / (f1.mask_amplitude * f2.mask_amplitude))
            return self.get_wsp(name, pair=pair, fields=(f1, f2), read_unbinned_MCM=False, randoms=randoms).decouple_cell(pcl)

        if 'clustering' in what:
            for i, j in self.lensing.tqdm(sym_pairs, desc="[compute_cls] clustering"):
                c = decoupled(f_dens[i], f_dens[j], 'clustering', (i, j))
                for data_type in self._STAT_TYPES_CLS['clustering']:
                    cl.setdefault(data_type, {})[i, j] = c[self._SACC_TYPE_COMPONENT_CLS[data_type]]

        if 'shear' in what:
            for i, j in self.lensing.tqdm(sym_pairs, desc="[compute_cls] shear"):
                c = decoupled(f_shear[i], f_shear[j], 'shear', (i, j))
                for data_type in self._STAT_TYPES_CLS['shear']:
                    cl.setdefault(data_type, {})[i, j] = c[self._SACC_TYPE_COMPONENT_CLS[data_type]]

        if 'shear_clustering' in what:
            # not symmetric under bin exchange (lens bin i, source bin j) -> all ordered pairs
            de_pairs = sym_pairs if (pairs is not None or auto_only) else [(i, j) for i in ikappas for j in ikappas]
            for i, j in self.lensing.tqdm(de_pairs, desc="[compute_cls] shear x clustering"):
                c = decoupled(f_dens[i], f_shear[j], 'shear_clustering', (i, j))
                for data_type in self._STAT_TYPES_CLS['shear_clustering']:
                    cl.setdefault(data_type, {})[i, j] = c[self._SACC_TYPE_COMPONENT_CLS[data_type]]

        if save:
            self._save_cls(ell, cl)

        return ell, cl

    def _save_cls(self, ell: np.ndarray, cl: dict) -> str:
        """Save catalog-based angular power spectra to a SACC FITS file.

        The catalog's density and shear fields share the same underlying n(z)
        per bin (galaxy redshifts are drawn from the lensing tracers' own
        dndz, see :meth:`_generate_z`), but are registered under distinct
        tracer names (``density_{i}`` vs. the lensing builder's ``kappa_{i}``)
        so that clustering data points don't reference a lensing-named tracer.

        Parameters
        ----------
        ell : ndarray
            Effective multipoles (as returned by :meth:`compute_cls`).
        cl : dict
            Dictionary keyed by SACC data type, each mapping to a dict keyed
            by ``(i, j)`` (as returned by :meth:`compute_cls`).

        Returns
        -------
        str
            Path to the saved SACC FITS file.
        """
        s = sacc.Sacc()
        shear_tracers = self.lensing.tracers_sacc
        density_tracers = [t.sacc() for t in self.density_tracers]
        for t in shear_tracers + density_tracers:
            s.add_tracer_object(t)

        tracers = {'density': density_tracers, 'shear': shear_tracers}

        for data_type, pairs in cl.items():
            role_i, role_j = self._SACC_TYPE_LEGS_CLS[data_type]
            tracers_i, tracers_j = tracers[role_i], tracers[role_j]
            for (i, j), _cl in pairs.items():
                for _ell, _cl_ell in zip(ell, _cl, strict=True):
                    s.add_data_point(data_type, [tracers_i[i].name, tracers_j[j].name], _cl_ell, ell=_ell)
        fname = os.path.join(self._catalog_dir, 'cat_cls.fits')
        s.save_fits(fname, overwrite=True)
        logger.debug("Saved to file %s.", fname)
        return fname

    def load_cls(self, return_dict: bool = False) -> sacc.Sacc | tuple[np.ndarray, dict[str, dict[tuple[int, int], np.ndarray]]]:
        """Load catalog-based angular power spectra previously saved by :meth:`_save_cls`.

        Parameters
        ----------
        return_dict : bool, optional
            If True, return ``(ell, cl)``. Otherwise, return the SACC object.

        Returns
        -------
        sacc.Sacc or (ell, dict)
            If ``return_dict=False``, returns the loaded SACC object.
            If ``return_dict=True``, returns ``(ell, cl)`` where ``ell`` is the
            multipole array and ``cl`` is a dict keyed by SACC data type, each
            mapping to a dict keyed by ``(i, j)`` (same structure as returned
            by :meth:`compute_cls`). Reads only the data types and tracer
            pairs actually present in the file (:meth:`compute_cls` may not
            have computed all of them), with no dependency on ``self.lensing``
            beyond the file path.
        """
        fname = os.path.join(self._catalog_dir, 'cat_cls.fits')
        s = sacc.Sacc.load_fits(fname)
        logger.debug("Found and loaded file at %s.", fname)
        if not return_dict:
            return s

        def tracer_index(name: str) -> int:
            # bin index of a tracer name (e.g. 'kappa_3')
            return int(name.rsplit('_', 1)[-1])

        ell = None
        cl = {}
        for data_type in s.get_data_types():
            pairs = {}
            for name_i, name_j in s.get_tracer_combinations(data_type):
                ell, _cl = s.get_ell_cl(data_type, name_i, name_j)
                pairs[tracer_index(name_i), tracer_index(name_j)] = _cl
            cl[data_type] = pairs

        return ell, cl

    def predict_cls(
            self,
            ikappas: int | Iterable[int] | None = None,
            what: str | list[str] = 'all',
            *,
            auto_only: bool = False,
            pairs: list[tuple[int, int]] | None = None,
            binning: str = 'wsp',
            shared_wsp: bool = False,
            randoms: bool = False,
            shear_cols: tuple[str, str] = ('gamma1', 'gamma2'),
            subsample: float | None = None,
            error_bars: bool = False,
            keep_cls: bool = False,
            stacked: bool = False,
            ell: np.ndarray | None = None) -> tuple:
        """Predict the catalog angular power spectra.

        Predictions use the lensing builder's :attr:`~pollux.map_builder.MapBuilder.theory`,
        i.e. the matter power spectrum of the catalog's shells (CCL's nonlinear P(k)
        for dark-matter-only shells, S19-corrected for baryonified shells). Predicted
        spectra are noise-free, like the measured ones (NaMaster subtracts the noise
        bias of catalog auto-spectra).

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic bins. If None, all bins.
        what : {'shear', 'clustering', 'shear_clustering', 'all'} or list of str, optional
            Statistics to predict (see :meth:`compute_cls`).
        auto_only : bool, optional
            If True, only same-bin pairs ``(i, i)``.
        pairs : list of tuple of int or None, optional
            If given, only these ``(i, j)`` bin pairs, for every statistic in ``what``
            (overrides ``ikappas`` and ``auto_only``).
        binning : {'wsp', 'average', 'unbinned'}, optional
            ``'wsp'`` (default): bandpowers binned with the bandpower windows of the
            workspaces of :meth:`compute_cls` (see ``shared_wsp`` and :meth:`get_bpws`). ``'average'``: bandpowers averaged over each bandpower.
            ``'unbinned'``: spectra at ``ell``.
        shared_wsp : bool, optional
            With ``binning='wsp'``, as in :meth:`compute_cls`: use the workspace shared
            by all pairs of each statistic if True, else that of each pair (default).
        randoms : bool, optional
            With per-pair workspaces, as in :meth:`compute_cls`.
        shear_cols, subsample : optional
            As in :meth:`compute_cls`, setting the noise (see :meth:`_noise_cls`).
        error_bars : bool, optional
            If True, also return Gaussian error bars (E-mode and density spectra),
            with ``C_ell + N_ell`` for auto-spectra (:meth:`_noise_cls`) and the
            lensing builder's sky fraction.
        keep_cls : bool, optional
            If True, cache the results on ``self.predicted_cls``.
        stacked : bool, optional
            If True, also return the E-mode and density spectra of all statistics
            stacked into one vector (in the order of ``what``, then of the pairs), and
            with ``error_bars`` their joint covariance.
        ell : ndarray or None, optional
            Multipoles for ``binning='unbinned'``. Default is the effective bandpower multipoles.

        Returns
        -------
        ell : ndarray
            Multipoles.
        cl : dict
            Keyed by SACC data type, then by ``(i, j)`` (same structure as :meth:`compute_cls`).
        dcl : dict
            Only with ``error_bars``, keyed like ``cl`` (E-mode and density types).
        cl_stacked, cov : ndarray
            Only with ``stacked`` (``cov`` only with ``error_bars`` too).

        Notes
        -----
        Wraps :meth:`pollux.cosmo.Theory.predict_cls` (one call for all statistics,
        so that the covariance includes cross-statistic blocks).
        """
        # cached prediction (see keep_cls)
        if self.predicted_cls is not None:
            return self.predicted_cls
        if isinstance(what, str):
            what = list(self._STAT_TYPES_CLS) if what == 'all' else [what]
        # tracers of the legs of each statistic
        tracers = {'density': self.density_tracers, 'shear': self.lensing_tracers}
        # bins, and symmetric pairs (i >= j) for same-type correlations
        ikappas, sym_pairs = self.lensing._pairs(ikappas, auto_only, pairs)

        # tracer pairs of all statistics, with the bandpower windows of their workspaces
        tracer_pairs, bpws = {}, {}
        for stat in what:
            role_i, role_j = self._SACC_TYPE_LEGS_CLS[self._STAT_TYPES_CLS[stat][0]]
            # shear x clustering is not symmetric under bin exchange (lens bin i, source bin j) -> all ordered pairs
            stat_pairs = sym_pairs
            if stat == 'shear_clustering' and pairs is None and not auto_only:
                stat_pairs = [(i, j) for i in ikappas for j in ikappas]
            windows = {}  # bandpower windows by workspace (bin pair, or None if shared)
            for i, j in stat_pairs:
                tracer_pairs[stat, i, j] = (tracers[role_i][i], tracers[role_j][j])
                if binning == 'wsp':
                    pair = None if shared_wsp else (i, j)
                    if pair not in windows:
                        windows[pair] = self.get_bpws(stat, pair, randoms)
                    bpws[stat, i, j] = windows[pair]

        # noise of each tracer, for the error bars
        noise = None
        if error_bars:
            noise = {t.name: self._noise_cls(t, shear_cols, subsample) for t1, t2 in tracer_pairs.values() for t in (t1, t2)}

        # predict all statistics at once (for their joint covariance)
        res = self.lensing.theory.predict_cls(
            tracer_pairs,
            b=self.b,
            ell=ell,
            binning=binning,
            bpws=bpws,
            noise=noise,
            error_bars=error_bars,
            stacked=stacked,
            fsky=self.lensing.fsky,
            all_components=True)

        # split the coupled components into SACC data types, as in compute_cls
        cl = {}
        for (stat, i, j), c in res[1].items():
            c = np.atleast_2d(c)  # components (a single one without bandpower windows)
            for data_type in self._STAT_TYPES_CLS[stat]:
                k = self._SACC_TYPE_COMPONENT_CLS[data_type]
                cl.setdefault(data_type, {})[i, j] = c[k] if k < len(c) else np.zeros_like(c[0])

        # outputs: ell, cl, error bars of the first data type of each statistic, then stacked spectra and covariance
        out = [res[0], cl]
        if error_bars:
            dcl = {}
            for (stat, i, j), d in res[2].items():
                dcl.setdefault(self._STAT_TYPES_CLS[stat][0], {})[i, j] = d
            out.append(dcl)
        out += list(res[2 + error_bars:])
        out = tuple(out)
        if keep_cls:
            self.predicted_cls = out
        return out

    def plot_cls(
            self,
            plot_dir: str | None = None,
            *,
            modes: tuple[str, ...] = ('values', 'ratio'),
            ell_factor: int = 1,
            logy: bool = False,
            error_bars: bool = False,
            **predict_kwargs: Any) -> None:
        """Plot the saved angular power spectra against their prediction (:meth:`predict_cls`), one panel per bin pair.

        Parameters
        ----------
        plot_dir : str or None, optional
            Output directory. Default is ``{catalog_dir}/plots``.
        modes : tuple of str, optional
            Plots to make, see :func:`pollux.plotting.plot_pairs` (``'values'``, ``'ratio'``,
            ``'diff_over_error'``). Default is ``('values', 'ratio')``.
        ell_factor : {0, 1, 2}, optional
            In ``'values'`` mode, plot ``C_ell`` (0), ``ell C_ell`` (1, default) or
            ``ell (ell + 1) C_ell / 2 pi`` (2).
        logy : bool, optional
            If True, use a log y-scale in ``'values'`` mode.
        error_bars : bool, optional
            If True, draw Gaussian errors of the prediction as bands (always used by
            ``'diff_over_error'``).
        **predict_kwargs
            Options of :meth:`predict_cls`, e.g. ``auto_only``, ``pairs``, or ``shared_wsp`` and
            ``randoms`` as used by :meth:`compute_cls`: the predicted pairs are plotted.

        Notes
        -----
        Saves ``cl_shear_ee``, ``cl_shear_density`` and ``cl_density``, with suffix
        ``_{mode}`` for modes other than ``'values'``.
        """
        from .plotting import plot_pairs
        plot_dir = plot_dir or os.path.join(self._catalog_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # measured spectra, and their prediction for the statistics found
        ell, cl = self.load_cls(return_dict=True)
        what = [stat for stat, types in self._STAT_TYPES_CLS.items() if types[0] in cl]
        need_errors = error_bars or 'diff_over_error' in modes
        _, cl_th, *dcl = self.predict_cls(what=what, error_bars=need_errors, **predict_kwargs)
        dcl = dcl[0] if need_errors else {}

        # one figure per E-mode/density data type and mode
        scales = self.lensing.reference_scales()
        st = sacc.standard_types
        for data_type, name, color in ((st.galaxy_shear_cl_ee, 'cl_shear_ee', 'orange'), (st.galaxy_shearDensity_cl_e, 'cl_shear_density', 'red'),
                                       (st.galaxy_density_cl, 'cl_density', 'green')):
            if data_type not in cl:
                continue
            for mode in modes:
                plot_pairs(
                    ell,
                    data={ij: cl[data_type][ij] for ij in cl_th[data_type] if ij in cl[data_type]},
                    theory=cl_th[data_type],
                    theory_errors=dcl.get(data_type) if (error_bars or mode == 'diff_over_error') else None,
                    mode=mode,
                    auto_only=all(i == j for i, j in cl_th[data_type]),
                    ell_factor=ell_factor,
                    scales=scales,
                    color=color,
                    logy=logy,
                    sharey=False if mode == 'values' else 'all',
                    fname=os.path.join(plot_dir, name + ('' if mode == 'values' else f'_{mode}') + '.png'))

    # tracer roles (density or shear) of the (i, j) legs of each real-space SACC data type
    _SACC_TYPE_LEGS_2PT = {
        sacc.standard_types.galaxy_density_xi: ('density', 'density'),
        sacc.standard_types.galaxy_shear_xi_plus: ('shear', 'shear'),
        sacc.standard_types.galaxy_shear_xi_minus: ('shear', 'shear'),
        sacc.standard_types.galaxy_shearDensity_xi_t: ('density', 'shear'),
        sacc.standard_types.galaxy_shearDensity_xi_x: ('density', 'shear'),
    }

    # component of the predicted correlation (xi+ or xi-) of each real-space SACC data type (None if a single one; zero if absent)
    _SACC_TYPE_COMPONENT_2PT = {
        sacc.standard_types.galaxy_density_xi: None,
        sacc.standard_types.galaxy_shear_xi_plus: 0,
        sacc.standard_types.galaxy_shear_xi_minus: 1,
        sacc.standard_types.galaxy_shearDensity_xi_t: None,
    }

    # real-space SACC data types of each statistic, in the order of the correlations of TreeCorr
    _STAT_TYPES_2PT = {
        'clustering': (sacc.standard_types.galaxy_density_xi,),
        'shear': (sacc.standard_types.galaxy_shear_xi_plus, sacc.standard_types.galaxy_shear_xi_minus),
        'shear_clustering': (sacc.standard_types.galaxy_shearDensity_xi_t, sacc.standard_types.galaxy_shearDensity_xi_x),
    }

    def compute_2pt(
            self,
            ikappas: int | Iterable[int] | None = None,
            what: str | list[str] = 'all',
            *,
            auto_only: bool = False,
            pairs: list[tuple[int, int]] | None = None,
            theta_min: float = 1.,
            theta_max: float = 100.,
            ntheta: int = 20,
            sep_units: str = 'arcmin',
            bin_slop: float | None = None,
            shear_cols: tuple[str, str] = ('gamma1', 'gamma2'),
            subsample: float | None = None,
            remove_mean: bool = True,
            nran_factor: int = 5,
            shared_randoms: bool = True,
            seed: int | None = None,
            error_bars: bool = False,
            theta_per_pair: bool = False,
            save: bool = True) -> tuple:
        """Measure real-space two-point correlation functions directly from the
        generated catalog(s), including all cross-tomographic-bin combinations.

        Uses TreeCorr's angular correlation estimators (``NNCorrelation`` for
        clustering, ``GGCorrelation`` for shear, ``NGCorrelation`` for the
        shear-density cross-correlation), binned in angular separation between
        ``theta_min`` and ``theta_max``. This is the real-space analog of
        :meth:`compute_cls`; ``ikappas``, ``what``, ``auto_only``,
        ``shear_cols``, ``nran_factor``, ``seed`` and ``subsample`` have the
        same meaning there.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic bin indices to process. If None, uses all bins.
        what : {'shear', 'clustering', 'shear_clustering', 'all'} or list of str, optional
            Which correlations to compute:
            - 'clustering': galaxy density autocorrelation ``w(theta)``, symmetric i<=j pairs.
            - 'shear': shear autocorrelation ``xi_+/-(theta)``, symmetric i<=j pairs.
            - 'shear_clustering': density-shear cross-correlation ``gamma_t/x(theta)``,
              all (i, j) ordered pairs since lens-bin i / source-bin j is not symmetric.
            - 'all': compute all three. Default.
        auto_only : bool, optional
            If True, restrict to same-bin pairs (i, i) only, skipping cross-bin
            correlations. Default is False.
        pairs : list of tuple of int or None, optional
            If given, only compute these ``(i, j)`` bin pairs, for every
            statistic in ``what`` (overrides ``ikappas`` and ``auto_only``).
            Only the catalogs of the bins involved are loaded.
        theta_min, theta_max : float, optional
            Angular separation range, in ``sep_units``. Defaults are 1 and 100 arcmin.
        ntheta : int, optional
            Number of log-spaced angular bins. Default is 20.
        sep_units : str, optional
            Units of ``theta_min``/``theta_max`` and of the returned ``theta``.
            Default is 'arcmin'.
        bin_slop : float or None, optional
            TreeCorr ``bin_slop`` (binning approximation tolerance). If None,
            uses TreeCorr's own default.
        shear_cols : tuple of str, optional
            Names of the two catalog columns to use as the spin-2 shear field.
            See :meth:`compute_cls`.
        subsample : float or None, optional
            If not None, randomly draw (without replacement, seeded by
            ``seed + ikappa``) this fraction of galaxies from each loaded
            catalog before building fields, with ``0 < subsample < 1``. A
            random draw is used rather than a fixed stride, since galaxies
            are stored in a fixed (non-random) order and a stride would
            sample a specific, non-representative pattern. Default is None
            (no subsampling).
        remove_mean : bool, optional
            If True (default), subtract the mean of the shear columns over each bin
            before the measurement.
        nran_factor : int, optional
            Ratio of random points to galaxies for each bin's random catalog,
            used for the Landy-Szalay ``w(theta)`` estimator and for
            subtracting additive bias in ``gamma_t/x(theta)``. Default is 5.
        shared_randoms : bool, optional
            If True (default), draw a single random-point realization (sized
            from the first processed bin) and reuse it for every bin's
            random catalog (used both for the ``w(theta)`` estimator and for
            the ``gamma_t/x(theta)`` random-lens subtraction), rather than
            drawing independent randoms per bin. This matches the fact that
            all bins share the same survey footprint, and is cheaper since
            randoms (and their TreeCorr catalog/tree) are only built once.
            If False, each bin gets its own randoms, seeded by ``seed + ikappa``.
        seed : int or None, optional
            Seed for the random catalogs' RNG. Default is the shell builder's seed.
        error_bars : bool, optional
            If True, also return ``dxi``: standard deviations (shot-noise
            error bars from TreeCorr), with the same structure as ``xi``.
            Default is False.
        theta_per_pair : bool, optional
            If True, return ``theta`` as a dict with the same structure as
            ``xi`` (each pair's own mean separation from its correlation
            object's ``meanlogr``) instead of a single shared array. The
            exact mean separation is catalog-dependent and can differ
            slightly from one pair to another; this is always what gets
            written to the SACC file regardless of this flag. Default is
            False (return a single array, taken from the first pair computed).
        save : bool, optional
            If True (default), save the correlation functions to a SACC FITS
            file via :meth:`_save_2pt`, at ``{self._catalog_dir}/cat_2pt.fits``.

        Returns
        -------
        theta : ndarray or dict
            Mean angular separation, in ``sep_units``. A single shared array,
            or a dict keyed like ``xi`` if ``theta_per_pair=True``.
        xi : dict
            Keyed by SACC data type (e.g. ``'galaxy_density_xi'``,
            ``'galaxy_shear_xi_plus'``, ``'galaxy_shear_xi_minus'``,
            ``'galaxy_shearDensity_xi_t'``, ``'galaxy_shearDensity_xi_x'``,
            depending on ``what``), each mapping to a dict keyed by ``(i, j)``.
        dxi : dict
            Only with ``error_bars``: shot-noise error bars, keyed like ``xi``.
        """
        if isinstance(what, str):
            what = list(self._STAT_TYPES_2PT) if what == 'all' else [what]

        builder = self.lensing.shell_builder
        seed = builder.seed if seed is None else seed

        # footprint pixels, to draw randoms
        _, ipix_mask = builder.get_mask(nest=True, return_ipix=True)

        # bins, and symmetric pairs (i >= j) for same-type correlations
        ikappas, sym_pairs = self.lensing._pairs(ikappas, auto_only, pairs)

        # catalogs needed by the statistics, and the angular binning
        need_density = 'clustering' in what or 'shear_clustering' in what
        need_shear = 'shear' in what or 'shear_clustering' in what
        col1, col2 = shear_cols

        corr_kwargs = dict(min_sep=theta_min, max_sep=theta_max, nbins=ntheta, sep_units=sep_units, bin_slop=bin_slop)

        # Build each bin's catalog(s) once, reused across all pairs below.
        # A single galaxy catalog serves as both the density tracer (for NN)
        # and the shear tracer (for GG/NG): NNCorrelation only looks at
        # positions and simply ignores the g1/g2 fields when present.
        cat_gal, cat_rand = {}, {}
        shared_cat_rand = None
        for ikappa in self.lensing.tqdm(ikappas, desc="[compute_2pt] building catalogs"):
            cat = self.load_catalogs(
                ikappa, subsample=subsample, seed=seed, columns=['ra', 'dec', *shear_cols], remove_mean=list(shear_cols) if remove_mean else None)

            # galaxies, with their shear if needed (TreeCorr's g2 convention is flipped)
            gal_kwargs = dict(ra=cat['ra'], dec=cat['dec'], ra_units='deg', dec_units='deg')
            if need_shear:
                gal_kwargs.update(g1=cat[col1], g2=cat[col2], flip_g2=True)
            cat_gal[ikappa] = treecorr.Catalog(**gal_kwargs)

            if need_density:
                # randoms over the footprint (for w(theta) and the random-lens subtraction of gamma_t): one set for all bins, or one per bin
                if shared_randoms:
                    if shared_cat_rand is None:
                        rng = np.random.default_rng(seed)
                        nran = int(nran_factor * cat['ra'].size)
                        ra_rand, dec_rand = generate_radec_in_pix(self.nside, rng.choice(ipix_mask, size=nran), rng)
                        shared_cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units='deg', dec_units='deg')
                    cat_rand[ikappa] = shared_cat_rand
                else:
                    rng = np.random.default_rng(seed + ikappa)
                    nran = int(nran_factor * cat['ra'].size)
                    ra_rand, dec_rand = generate_radec_in_pix(self.nside, rng.choice(ipix_mask, size=nran), rng)
                    cat_rand[ikappa] = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units='deg', dec_units='deg')

        # correlations, errors and mean separations, keyed by SACC data type then bin pair
        xi = {}
        dxi = {}
        theta_pairs = {}
        theta = None

        if 'clustering' in what:
            (xi_nn, dxi_nn, theta_nn), = ([d.setdefault(t, {}) for d in (xi, dxi, theta_pairs)] for t in self._STAT_TYPES_2PT['clustering'])
            for i, j in self.lensing.tqdm(sym_pairs, desc="[compute_2pt] clustering"):
                dd = treecorr.NNCorrelation(**corr_kwargs)
                rr = treecorr.NNCorrelation(**corr_kwargs)
                dr = treecorr.NNCorrelation(**corr_kwargs)
                # Landy-Szalay estimator
                if i == j:
                    dd.process(cat_gal[i])
                    rr.process(cat_rand[i])
                    dr.process(cat_gal[i], cat_rand[i])
                    xi_nn[i, j], var_nn = dd.calculateXi(rr=rr, dr=dr)
                else:
                    rd = treecorr.NNCorrelation(**corr_kwargs)
                    dd.process(cat_gal[i], cat_gal[j])
                    rr.process(cat_rand[i], cat_rand[j])
                    dr.process(cat_gal[i], cat_rand[j])
                    rd.process(cat_rand[i], cat_gal[j])
                    xi_nn[i, j], var_nn = dd.calculateXi(rr=rr, dr=dr, rd=rd)
                dxi_nn[i, j] = np.sqrt(var_nn)
                theta_nn[i, j] = np.exp(dd.meanlogr)
                if theta is None:
                    theta = theta_nn[i, j]

        if 'shear' in what:
            (xi_p, dxi_p, theta_p), (xi_m, dxi_m, theta_m) = ([d.setdefault(t, {}) for d in (xi, dxi, theta_pairs)] for t in self._STAT_TYPES_2PT['shear'])
            for i, j in self.lensing.tqdm(sym_pairs, desc="[compute_2pt] shear"):
                gg = treecorr.GGCorrelation(**corr_kwargs)
                if i == j:
                    gg.process(cat_gal[i])
                else:
                    gg.process(cat_gal[i], cat_gal[j])
                xi_p[i, j] = gg.xip
                xi_m[i, j] = gg.xim
                dxi_p[i, j] = np.sqrt(gg.varxip)
                dxi_m[i, j] = np.sqrt(gg.varxim)
                theta_p[i, j] = theta_m[i, j] = np.exp(gg.meanlogr)
                if theta is None:
                    theta = theta_p[i, j]

        if 'shear_clustering' in what:
            # not symmetric under bin exchange (lens bin i, source bin j) -> all ordered pairs
            de_pairs = sym_pairs if (pairs is not None or auto_only) else [(i, j) for i in ikappas for j in ikappas]
            (xi_t, dxi_t, theta_t), (xi_x, dxi_x,
                                     theta_x) = ([d.setdefault(t, {}) for d in (xi, dxi, theta_pairs)] for t in self._STAT_TYPES_2PT['shear_clustering'])
            for i, j in self.lensing.tqdm(de_pairs, desc="[compute_2pt] shear x clustering"):
                ng = treecorr.NGCorrelation(**corr_kwargs)
                rg = treecorr.NGCorrelation(**corr_kwargs)
                # lens bin i, source bin j, with the signal around randoms subtracted
                ng.process(cat_gal[i], cat_gal[j])
                rg.process(cat_rand[i], cat_gal[j])
                xi_t[i, j], xi_x[i, j], var_ng = ng.calculateXi(rg=rg)
                dxi_t[i, j] = dxi_x[i, j] = np.sqrt(var_ng)
                theta_t[i, j] = theta_x[i, j] = np.exp(ng.meanlogr)
                if theta is None:
                    theta = theta_t[i, j]

        if save:
            self._save_2pt(theta_pairs, xi, dxi=dxi)

        theta_out = theta_pairs if theta_per_pair else theta

        if error_bars:
            return theta_out, xi, dxi
        else:
            return theta_out, xi

    def _save_2pt(self, theta: dict, xi: dict, dxi: dict | None = None) -> str:
        """Save catalog-based real-space two-point functions to a SACC FITS file.

        Mirrors :meth:`_save_cls`: the catalog's density and shear fields
        share the same underlying n(z) per bin but are registered under
        distinct tracer names (``density_{i}`` vs. the lensing builder's
        ``kappa_{i}``).

        Parameters
        ----------
        theta : dict
            Angular separations, keyed by SACC data type then by ``(i, j)``
            (the ``theta_pairs`` structure computed internally by
            :meth:`compute_2pt`, i.e. each pair's own mean separation, since
            it is catalog-dependent and can differ slightly between pairs).
        xi : dict
            Dictionary keyed by SACC data type, each mapping to a dict keyed
            by ``(i, j)`` (as returned by :meth:`compute_2pt`).
        dxi : dict or None, optional
            Error bars with the same structure as ``xi``; if given, saved as the
            diagonal of the SACC covariance.

        Returns
        -------
        str
            Path to the saved SACC FITS file.
        """
        s = sacc.Sacc()
        shear_tracers = self.lensing.tracers_sacc
        density_tracers = [t.sacc() for t in self.density_tracers]
        for t in shear_tracers + density_tracers:
            s.add_tracer_object(t)

        tracers = {'density': density_tracers, 'shear': shear_tracers}

        for data_type, pairs in xi.items():
            role_i, role_j = self._SACC_TYPE_LEGS_2PT[data_type]
            tracers_i, tracers_j = tracers[role_i], tracers[role_j]
            for (i, j), _xi in pairs.items():
                _theta = theta[data_type][i, j]
                for _theta_k, _xi_k in zip(_theta, _xi, strict=True):
                    s.add_data_point(data_type, [tracers_i[i].name, tracers_j[j].name], _xi_k, theta=_theta_k)
        if dxi is not None:
            # same order as the data points added above
            var = np.concatenate([dxi[data_type][ij]**2 for data_type, pairs in xi.items() for ij in pairs])
            s.add_covariance(np.diag(var))
        fname = os.path.join(self._catalog_dir, 'cat_2pt.fits')
        s.save_fits(fname, overwrite=True)
        logger.debug("Saved to file %s.", fname)
        return fname

    def load_2pt(self, return_dict: bool = False, return_errors: bool = False) -> sacc.Sacc | tuple:
        """Load catalog-based two-point functions previously saved by :meth:`_save_2pt`.

        Parameters
        ----------
        return_dict : bool, optional
            If True, return ``(theta, xi)``. Otherwise, return the SACC object.
        return_errors : bool, optional
            With ``return_dict``, also return the error bars (square root of the
            diagonal of the saved covariance), with the same structure as ``xi``.

        Returns
        -------
        sacc.Sacc or (dict, dict)
            If ``return_dict=False``, returns the loaded SACC object.
            If ``return_dict=True``, returns ``(theta, xi)``, both dicts keyed
            by SACC data type then by ``(i, j)`` (same structure as
            ``theta_pairs``/``xi`` in :meth:`compute_2pt`; each pair's own
            mean separation is preserved, since it is catalog-dependent).
            Reads only the data types and tracer pairs actually present in
            the file (:meth:`compute_2pt` may not have computed all of them),
            with no dependency on ``self.lensing`` beyond the file path.
        """
        fname = os.path.join(self._catalog_dir, 'cat_2pt.fits')
        s = sacc.Sacc.load_fits(fname)
        logger.debug("Found and loaded file at %s.", fname)
        if not return_dict:
            return s

        def tracer_index(name: str) -> int:
            # bin index of a tracer name (e.g. 'kappa_3')
            return int(name.rsplit('_', 1)[-1])

        theta = {}
        xi = {}
        dxi = {}
        sigma = np.sqrt(np.diag(s.covariance.covmat)) if return_errors and s.covariance is not None else None
        for data_type in s.get_data_types():
            theta_pairs, xi_pairs, dxi_pairs = {}, {}, {}
            for name_i, name_j in s.get_tracer_combinations(data_type):
                key = tracer_index(name_i), tracer_index(name_j)
                theta_pairs[key], xi_pairs[key] = s.get_theta_xi(data_type, name_i, name_j)
                if sigma is not None:
                    dxi_pairs[key] = sigma[s.indices(data_type, (name_i, name_j))]
            theta[data_type] = theta_pairs
            xi[data_type] = xi_pairs
            dxi[data_type] = dxi_pairs

        if return_errors:
            return theta, xi, dxi
        return theta, xi

    def predict_2pt(self, theta: dict, nell: int = 500) -> dict:
        """Predict the catalog two-point correlation functions.

        Predictions use the matter power spectrum of the catalog's shells (see :meth:`predict_cls`).

        Parameters
        ----------
        theta : dict
            Angular separations in arcmin keyed by SACC data type, then by ``(i, j)``
            (as returned by :meth:`load_2pt` or :meth:`compute_2pt` with ``theta_per_pair=True``).
        nell : int, optional
            Number of log-spaced multipoles up to the bandpower ``lmax`` used for the Hankel transforms.

        Returns
        -------
        dict
            Same structure as ``theta``; ``galaxy_shearDensity_xi_x`` is zero.
        """
        tracers = {'density': self.density_tracers, 'shear': self.lensing_tracers}
        xi = {}
        for data_type, type_pairs in theta.items():
            if data_type not in self._SACC_TYPE_COMPONENT_2PT:
                xi[data_type] = {ij: np.zeros(len(tt)) for ij, tt in type_pairs.items()}
                continue
            role_i, role_j = self._SACC_TYPE_LEGS_2PT[data_type]
            component = self._SACC_TYPE_COMPONENT_2PT[data_type]
            res = self.lensing.theory.predict_2pt({(i, j): (tracers[role_i][i], tracers[role_j][j]) for i, j in type_pairs}, type_pairs, self.b.lmax, nell=nell)
            xi[data_type] = {ij: r if component is None else r[component] for ij, r in res.items()}
        return xi

    def plot_2pt(self, plot_dir: str | None = None, *, modes: tuple[str, ...] = ('values', 'ratio'), reference: CatalogSourceGenerator | None = None) -> None:
        """Plot the saved two-point functions against their prediction (:meth:`predict_2pt`), one panel per bin pair.

        Parameters
        ----------
        plot_dir : str or None, optional
            Output directory. Default is ``{catalog_dir}/plots``.
        modes : tuple of str, optional
            Plots to make, see :func:`pollux.plotting.plot_pairs` (``'values'``, ``'ratio'``,
            ``'diff_over_error'``, with the measurement errors). Default is ``('values', 'ratio')``.
            Values are multiplied by ``theta``.
        reference : CatalogSourceGenerator or None, optional
            Catalog with saved two-point functions (e.g. dark-matter-only, for a
            baryonified catalog): adds the ``'ratio_to_reference'`` mode, the ratio of the
            two catalogs' functions against the ratio of their predictions.

        Notes
        -----
        Saves ``xi_shear``, ``xi_shear_density`` and ``xi_density``, with suffix ``_{mode}``
        for modes other than ``'values'``.
        """
        from .plotting import plot_pairs
        plot_dir = plot_dir or os.path.join(self._catalog_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # measurements and predictions (and those of the reference catalog)
        theta, xi, dxi = self.load_2pt(return_dict=True, return_errors=True)
        xi_th = self.predict_2pt(theta)
        xi_ref, xi_th_ref = {}, {}
        if reference is not None:
            _, xi_ref = reference.load_2pt(return_dict=True)
            xi_th_ref = reference.predict_2pt(theta)
            modes = (*modes, 'ratio_to_reference')

        # one figure per group of data types and mode, the data types of a group overplotted
        scales = self.lensing.reference_scales()
        st = sacc.standard_types
        for data_types, name, ylabel in (({st.galaxy_shear_xi_plus: 'dodgerblue', st.galaxy_shear_xi_minus: 'orange'}, 'xi_shear', r'$\theta \xi_\pm(\theta)$'),
                                         ({st.galaxy_shearDensity_xi_t: 'red'}, 'xi_shear_density',
                                          r'$\theta \gamma_{\rm t}(\theta)$'), ({st.galaxy_density_xi: 'green'}, 'xi_density', r'$\theta w(\theta)$')):
            data_types = {t: c for t, c in data_types.items() if t in xi}
            for mode in modes:
                # the last data type of the group saves the figure
                figaxes = None
                for k, (data_type, color) in enumerate(data_types.items()):
                    last = k == len(data_types) - 1
                    figaxes = plot_pairs(
                        theta[data_type],
                        data=xi[data_type],
                        theory=xi_th[data_type],
                        data_errors=dxi.get(data_type) if dxi else None,
                        data_ref=xi_ref.get(data_type),
                        theory_ref=xi_th_ref.get(data_type),
                        mode=mode,
                        auto_only=all(i == j for i, j in xi[data_type]),
                        ell_factor=1,
                        scales=scales,
                        color=color,
                        xlabel=r'$\theta$ (arcmin)',
                        ylabel=ylabel if mode == 'values' else None,
                        sharey=False if mode == 'values' else 'all',
                        figaxes=figaxes,
                        fname=os.path.join(plot_dir, name + ('' if mode == 'values' else f'_{mode}') + '.png') if last else None)


class CatalogSourceUniform(CatalogSourceGenerator):
    """Generate source galaxies uniformly over the footprint.

    Redshifts are drawn from the tomographic tracer distributions. Angular
    positions are drawn uniformly over the observed sky region (octant or full
    sky depending on ``lensing.octant_only``).

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag used to name the output directory ``catalogs_{tag}/``. Default is None.
    nbar_input : float, array-like, or None, optional
        Galaxy surface density. Default is None.
    interp_fields, interp_method, smooth_fields, smooth_Mpc
        Interpolation and smoothing of the fields, see :class:`CatalogSourceGenerator`.
    augmentors : list, optional
        List of augmentors. Default is empty list.
    """

    def _generate_galaxies(
            self, catalog: dict[int, dict[str, np.ndarray]], ikappas: int | Iterable[int] | None = None, ishells: list[int] | None = None) -> None:
        """Generate redshifts and positions for all selected tomographic bins.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        ishells : None, optional
            Per-shell generation is not supported.
        """
        if ishells is not None:
            raise NotImplementedError(f"Per-shell generation is not supported by {self.__class__.__name__}.")

        # Generate ra, dec and redshift
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] generating ra, dec, z"):
            self._generate_z(catalog, ikappa)
            self._generate_pos(catalog, ikappa)

            for k in self.interp_fields:
                catalog[ikappa][k] = np.zeros_like(catalog[ikappa]['z'])

    def _generate_pos(self, catalog: dict[int, dict[str, np.ndarray]], ikappa: int) -> None:
        """Draw uniform (RA, DEC) positions over the footprint.

        Parameters
        ----------
        catalog : dict
            Per-bin output container.
        ikappa : int
            Tomographic-bin index.
        """
        rng = np.random.default_rng(self.lensing.shell_builder.seed + ikappa * self.nshells)

        # Generate random positions with a uniform distribution over the footprint
        if self.lensing.octant_only:
            ra = rng.uniform(0., 90., self.Ngal[ikappa])
            dec = np.rad2deg(np.arcsin(rng.uniform(0., 1., self.Ngal[ikappa])))
        else:
            ra = rng.uniform(0., 360., self.Ngal[ikappa])
            dec = np.rad2deg(np.arcsin(rng.uniform(-1., 1., self.Ngal[ikappa])))

        catalog[ikappa]['ra'] = ra
        catalog[ikappa]['dec'] = dec


class CatalogSourceClustered(CatalogSourceGenerator):
    """Base class for clustered source galaxy catalogs.

    This class provides common functionality for generating clustered catalogs
    where galaxy positions are biased by a density field.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag used to name the output directory ``catalogs_{tag}/``. Default is None.
    nbar_input : float, array-like, or None, optional
        Galaxy surface density in gal/arcmin^2. Default is None.
    bias : float or array-like, optional
        Linear galaxy bias per tomographic bin. If a float, the same bias is used
        for all tomographic bins; if an array-like, it must have length
        ``lensing.nmaps`` and provide a per-bin value.
    interp_fields, interp_method, smooth_fields, smooth_Mpc : optional
        Interpolation and smoothing of the fields (including the density used to
        place galaxies, ``'positions'``), see :class:`CatalogSourceGenerator`.
    augmentors : list, optional
        List of augmentors applied after map interpolation. Default is empty list.
    cls_b, cls_lmax_mask : optional
        Binning of the catalog angular power spectra, see :class:`CatalogSourceGenerator`.
    """

    def __init__(
        self,
        lensing: LensingBuilder,
        tag: str | None = None,
        nbar_input: float | np.ndarray | None = None,
        bias: float | np.ndarray | None = None,
        interp_fields: tuple[str, ...] = ('kappa', 'gamma1', 'gamma2'),
        interp_method: str = 'bilinear',
        smooth_fields: tuple[str, ...] = ('positions', 'density', 't11', 't22', 't12'),
        smooth_Mpc: float | dict[str, float] | None = None,
        augmentors: list | None = None,
        cls_b: nmt.NmtBin | None = None,
        cls_lmax_mask: int | None = None,
    ) -> None:
        super().__init__(
            lensing=lensing,
            tag=tag,
            nbar_input=nbar_input,
            interp_fields=interp_fields,
            interp_method=interp_method,
            smooth_fields=smooth_fields,
            smooth_Mpc=smooth_Mpc,
            augmentors=augmentors,
            cls_b=cls_b,
            cls_lmax_mask=cls_lmax_mask,
        )

        if bias is None:
            raise ValueError(f'bias must be provided for {self.__class__.__name__}')

        # Normalize bias: accept scalar or per-bin array
        bias_arr = np.atleast_1d(bias).astype(float)
        if bias_arr.size == 1:
            bias_arr = np.full(self.lensing.nmaps, float(bias_arr[0]))
        elif bias_arr.size != self.lensing.nmaps:
            raise ValueError(f"bias must be a scalar or array of length {self.lensing.nmaps}, got {bias_arr.size}")
        self.bias = bias_arr

        # Store bias values in augmentor kwargs
        for ikappa in self.lensing.iter():
            self.augmentor_kwargs[ikappa]['bias'] = self.bias[ikappa]

        # Store bias values in header dict
        for i, b in enumerate(self.bias):
            self.header_dict[f'BIAS_{i}'] = b

    def _density_weight(self, density: np.ndarray, ikappa: int, ishell: int) -> np.ndarray:
        """Return the relative galaxy density ``1 + b * delta`` per pixel, clipped at zero.

        With ``b > 1`` the linear bias model is negative in underdense pixels, where no
        galaxies can be placed; a warning gives the fraction of such pixels.

        Parameters
        ----------
        density : ndarray
            Density contrast per pixel.
        ikappa : int
            Tomographic-bin index, setting the bias.
        ishell : int
            Shell index, for the warning.

        Returns
        -------
        ndarray
            Non-negative weights, same shape as ``density``.
        """
        weight = 1. + self.bias[ikappa] * density
        fraction = np.mean(weight < 0.)
        if fraction > 0.:
            warnings.warn(f"1 + b * delta < 0 in {100. * fraction:.3g}% of the pixels (bin {ikappa}, shell {ishell}, b = {self.bias[ikappa]:g}): clipped to 0.")
        return np.clip(weight, 0., None)


class CatalogSourceClusteredMultinomial(CatalogSourceClustered):
    """Generate clustered source galaxies using multinomial sampling.

    The total number of galaxies per tomographic bin is fixed, and positions are
    drawn per shell from a multinomial distribution over pixels, proportional to
    ``1 + b * delta`` where ``delta`` is the density contrast field.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag used to name the output directory ``catalogs_{tag}/``. Default is None.
    nbar_input : float, array-like, or None, optional
        Galaxy surface density in gal/arcmin^2. Default is None.
    bias : float or array-like, optional
        Linear galaxy bias per tomographic bin.
    interp_fields, interp_method, smooth_fields, smooth_Mpc
        Interpolation and smoothing of the fields, see :class:`CatalogSourceGenerator`.
    augmentors : list, optional
        List of augmentors. Default is empty list.
    """

    def _generate_galaxies(
            self, catalog: dict[int, dict[str, np.ndarray]], ikappas: int | Iterable[int] | None = None, ishells: list[int] | None = None) -> None:
        """Generate redshifts first, then positions conditioned on density maps.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        ishells : None, optional
            Per-shell generation is not supported (the number of galaxies per bin
            is fixed over all shells).
        """
        if ishells is not None:
            raise NotImplementedError(f"Per-shell generation is not supported by {self.__class__.__name__}.")

        # Generate redshifts first, needed to know what shells galaxies are in
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] generating z"):
            self._generate_z(catalog, ikappa)
            for k in ['ra', 'dec', *self.interp_fields]:
                catalog[ikappa][k] = np.zeros_like(catalog[ikappa]['z'])

        # Then generate positions, looping over shells for speed
        for ishell in self.lensing.shell_builder.tqdm(None, desc="[generate] generating ra, dec from density"):
            self._generate_pos(catalog, ishell, ikappas=ikappas)

    def _generate_pos(self, catalog: dict[int, dict[str, np.ndarray]], ishell: int, ikappas: int | Iterable[int] | None = None) -> None:
        """Draw positions in a given shell using multinomial pixel sampling.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place.
        ishell : int
            Shell index.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to process.
        """
        # density contrast used to place galaxies ('positions' smoothing)
        _, ipix_mask = self.snap_lensing.shell_builder.get_mask(return_ipix=True)
        density_masked = self._density_map(ishell, self.smoothing['positions']).get_values_pix(ipix_mask)

        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, leave=False):
            rng = np.random.default_rng(self.lensing.shell_builder.seed + ishell + ikappa * self.nshells)

            # select galaxies in current shell
            w = (catalog[ikappa]['idx_shell'] == ishell)

            if np.any(w):
                # Multinomial sampling of number of galaxies per pixels
                n = np.sum(w)
                p = self._density_weight(density_masked, ikappa, ishell)
                p /= np.sum(p)

                ipix = rng.choice(ipix_mask, size=n, replace=True, p=p)

                # Sampling ra, dec within pixel
                ra, dec = generate_radec_in_pix(self.nside, ipix, rng)
                catalog[ikappa]['ra'][w] = ra
                catalog[ikappa]['dec'][w] = dec


class CatalogSourceClusteredPoisson(CatalogSourceClustered):
    """Generate clustered source galaxies using Poisson sampling.

    In each shell and tomographic bin, the number of galaxies per pixel is drawn
    as a Poisson random variable with mean proportional to
    ``(1 + b * delta) * <nbar>``. The total number of galaxies therefore
    fluctuates.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag used to name the output directory ``catalogs_{tag}/``. Default is None.
    nbar_input : float, array-like, or None, optional
        Galaxy surface density in gal/arcmin^2. Default is None.
    bias : float or array-like, optional
        Linear galaxy bias per tomographic bin.
    interp_fields, interp_method, smooth_fields, smooth_Mpc
        Interpolation and smoothing of the fields, see :class:`CatalogSourceGenerator`.
    augmentors : list, optional
        List of augmentors. Default is empty list.
    """

    def _generate_galaxies(
            self, catalog: dict[int, dict[str, np.ndarray]], ikappas: int | Iterable[int] | None = None, ishells: list[int] | None = None) -> None:
        """Generate a clustered catalog by shell-wise Poisson sampling.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        ishells : list of int or None, optional
            Shells whose galaxies are generated. If None, all shells.

        Notes
        -----
        This method:
        1) computes the expected mean counts per pixel per shell from each
           tracer n(z),
        2) samples positions conditioned on density maps,
        3) samples redshifts within each shell for each bin.
        """
        builder = self.lensing.shell_builder
        z_edges = builder.shell_edges_redshift

        # First compute mean density per shell for each tracer's n(z) (all bins needed for nbar)
        nbar_per_pix_per_bin = []
        for ikappa, t in enumerate(self.lensing.tracers_ccl):
            nz_int = np.array([quad(lambda z: t.get_dndz(z), z_edges[ishell], z_edges[ishell + 1]) for ishell in range(self.snap_lensing.nmaps)])
            nbar_per_pix_per_bin.append(nz_int / np.sum(nz_int) * self.nbar_per_pix[ikappa])
        self.nbar_per_pix_per_bin = np.array(nbar_per_pix_per_bin)

        # Initialize catalog entries only for the requested bins
        for ikappa in self.lensing.iter(ikappas):
            catalog[ikappa]['ra'] = []
            catalog[ikappa]['dec'] = []
            catalog[ikappa]['z'] = []
            catalog[ikappa]['idx_shell'] = []

        # Second generate positions, looping over shells
        ishells = list(range(self.nshells)) if ishells is None else ishells
        for ishell in self.lensing.shell_builder.tqdm(ishells, desc="[generate] generating ra, dec from density"):
            self._generate_pos(catalog, ishell, ikappas=ikappas)

        # Third sample redshifts within shells
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] generating z"):
            self._generate_z(catalog, ikappa, ishells)

        for ikappa in self.lensing.tqdm(ikappas, disable=True):
            catalog[ikappa]['ra'] = np.concatenate(catalog[ikappa]['ra'])
            catalog[ikappa]['dec'] = np.concatenate(catalog[ikappa]['dec'])
            catalog[ikappa]['z'] = np.concatenate(catalog[ikappa]['z'])
            catalog[ikappa]['idx_shell'] = np.concatenate(catalog[ikappa]['idx_shell'])
            catalog[ikappa]['chi'] = builder.cosmo.comoving_radial_distance(1. / (1. + catalog[ikappa]['z']))
            for k in self.interp_fields:
                catalog[ikappa][k] = np.zeros_like(catalog[ikappa]['z'])

    def _generate_pos(self, catalog: dict[int, dict[str, np.ndarray]], ishell: int, ikappas: int | Iterable[int] | None = None) -> None:
        """Sample galaxy positions in one shell using pixel-wise Poisson draws.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place.
        ishell : int
            Shell index.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to process.
        """
        # density contrast used to place galaxies ('positions' smoothing)
        _, ipix_mask = self.snap_lensing.shell_builder.get_mask(return_ipix=True)
        density_masked = self._density_map(ishell, self.smoothing['positions']).get_values_pix(ipix_mask)

        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, leave=False):
            rng = np.random.default_rng(self.lensing.shell_builder.seed + ishell + ikappa * self.nshells)

            # no galaxy of this bin in this shell
            if self.nbar_per_pix_per_bin[ikappa][ishell] == 0.:
                catalog[ikappa]['ra'].append(np.zeros(0))
                catalog[ikappa]['dec'].append(np.zeros(0))
                continue

            # Poisson sampling
            p = self._density_weight(density_masked, ikappa, ishell)
            lam = p * self.nbar_per_pix_per_bin[ikappa][ishell]
            n_per_pix = rng.poisson(lam=lam)
            ipix = np.repeat(ipix_mask, n_per_pix)

            # Sampling ra, dec within pixel
            ra, dec = generate_radec_in_pix(self.nside, ipix, rng)

            catalog[ikappa]['ra'].append(ra)
            catalog[ikappa]['dec'].append(dec)

    def _generate_z(self, catalog: dict[int, dict[str, np.ndarray]], ikappa: int, ishells: list[int] | None) -> None:
        """Sample galaxy redshifts within each shell for one tomographic bin.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place, with one entry of
            ``catalog[ikappa]['ra']`` per shell of ``ishells``.
        ikappa : int
            Tomographic-bin index.
        ishells : list of int
            Shells whose galaxies were generated.

        Notes
        -----
        For each shell, redshifts are sampled from the tracer n(z) restricted to
        that shell's redshift interval.
        """
        builder = self.lensing.shell_builder
        z_edges = builder.shell_edges_redshift

        t = self.lensing.tracers_ccl[ikappa]
        for i, ishell in enumerate(ishells):
            rng = np.random.default_rng(self.lensing.shell_builder.seed + ishell + ikappa * self.nshells)

            zz = np.linspace(z_edges[ishell], z_edges[ishell + 1], 100)
            nz = t.get_dndz(zz)
            ngal = len(catalog[ikappa]['ra'][i])

            catalog[ikappa]['z'].append(sample_pdf(zz, nz, ngal, rng))
            catalog[ikappa]['idx_shell'].append(np.full(ngal, ishell))


def generate_radec_in_pix(nside: int, ipix: np.ndarray, rng: np.random.Generator, log2nside_high: int = 29) -> tuple[np.ndarray, np.ndarray]:
    """Sample random (RA, DEC) coordinates uniformly within Healpix pixels.

    This uses an approximation: pixels are upsampled to a very high nside
    (``2**log2nside_high``) and one sub-pixel is selected uniformly at random.
    See https://astro.pages.rwth-aachen.de/astrotools/_modules/healpytools.html#rand_vec_from_map.

    Parameters
    ----------
    nside : int
        Target Healpix nside of the input pixel indices.
    ipix : array_like of int
        Pixel indices at resolution ``nside`` (assumed NESTED).
    rng : numpy.random.Generator
        Random number generator.
    log2nside_high : int, optional
        Log2 of the high resolution nside used internally. Default is 29.

    Returns
    -------
    ra, dec : ndarray
        Longitudes and latitudes in degrees (Healpy ``lonlat=True`` convention).
    """
    order = hp.nside2order(nside)
    nside_high = 2**log2nside_high
    n_up = log2nside_high - order

    ipix_high = ipix * (4**n_up)
    ipix_high += rng.integers(0, 4**n_up, size=np.size(ipix))

    return hp.pix2ang(nside_high, ipix_high, nest=True, lonlat=True)  # ra, dec


class CatalogAugmentor(ABC):
    """Base class for catalog augmentors that add fields to existing catalogs.

    Subclasses must implement the ``augment_catalog`` method and set
    ``self.name``, ``self.required_interp_fields``, and ``self.new_columns``
    in their ``__init__``.
    """

    @abstractmethod
    def augment_catalog(self, cosmo: pyccl.Cosmology, catalog: dict[str, np.ndarray], **kwargs: Any) -> None:
        """Add the augmentor's columns to the catalog of one tomographic bin, in place.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology.
        catalog : dict[str, np.ndarray]
            Catalog of one tomographic bin.
        **kwargs
            Per-bin options (``augmentor_kwargs`` of the catalog generator, e.g. ``ikappa``).
        """
