"""Tomographic source catalogs from the shells and lensing maps, by uniform or clustered sampling of the galaxies.

Also measures their two-point statistics (Cls with NaMaster, two-point functions with TreeCorr) and predicts them.
"""

from __future__ import annotations

import logging
import os
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import healpy as hp
import pyccl
import pymaster as nmt
import sacc
import treecorr
import fitsio
from astropy.io import fits

from .io import layout
from .io.code_version import code_version
from .cosmo import Tracer
from .utils.binning import binning_id
from .utils.numerics import quad, sample_pdf
from .utils.sky import radec2thetaphi

if TYPE_CHECKING:
    import healsparse as hsp
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
        Tag of the catalogs: they go to ``catalogs/{nz}/{tag}/`` next to the maps (see :mod:`pollux.io.layout`), ``default`` if None.
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
    smooth_Mpc : float, array-like, dict or None, optional
        Comoving Gaussian smoothing scale in Mpc of ``smooth_fields``, or one scale per
        shell (e.g. :meth:`~pollux.cosmo.Theory.lognormal_smoothing`), or a dict of
        either per field (then ``smooth_fields`` is ignored, and fields not in the dict
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
        Number of galaxies drawn per tomographic bin over the footprint (``nbar`` times its area).
    nbar_per_pix : ndarray
        Mean expected counts per pixel per tomographic bin.
    header_dict : dict
        FITS header keyword/value pairs stored in the output catalogs, including the
        code version: ``POLLUXV`` (git commit) and ``DIRTY`` (uncommitted changes).
    """

    # fields that can be interpolated at the galaxies
    FIELDS = ('kappa', 'gamma1', 'gamma2', 'density', 't11', 't22', 't12')
    # fields averaged over each shell, not interpolated between shells (the others are at the shells' far edges)
    SHELL_FIELDS = ('density', 't11', 't22', 't12')
    # streams of random numbers, per (seed of the realization, stream, bin, shell)
    _STREAM_Z, _STREAM_POS, _STREAM_COUNTS = 0, 1, 2
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
        self.smoothing = self.smoothing_scales(smooth_fields, smooth_Mpc, self.lensing.shell_builder.nshells)

        self.interp_fields = tuple(interp_fields)
        self.augmentors = augmentors if augmentors is not None else []

        # catalogs by n(z) and tag, next to the maps they come from (see pollux.io.layout)
        shell_builder = self.lensing.shell_builder
        self._catalog_dir = os.path.join(shell_builder.matter_dir, 'catalogs', os.path.basename(lensing.save_dir), tag or 'default')
        self._intermediate_dir = layout.intermediate_dir(self._catalog_dir, shell_builder.realization_dir)
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
        self.augmentor_kwargs = {ikappa: {'ikappa': ikappa, 'realization_seed': self.lensing.shell_builder.seed} for ikappa in self.lensing.iter()}
        self.cls_b = cls_b
        self.cls_lmax_mask = cls_lmax_mask
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

        # galaxy counts over the footprint, and per pixel
        area_arcmin = 4. * np.pi * self.lensing.fsky * (180. * 60 / np.pi)**2
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
    def smoothing_scales(cls, smooth_fields: tuple[str, ...], smooth_Mpc: float | np.ndarray | dict | None, nshells: int) -> dict[str, tuple]:
        """Return the smoothing scale of every field and shell (see ``smooth_fields`` and ``smooth_Mpc``).

        Parameters
        ----------
        smooth_fields : tuple of str
            Fields smoothed by ``smooth_Mpc``.
        smooth_Mpc : float, array-like, dict or None
            Smoothing scale, one scale per shell, or either per field.
        nshells : int
            Number of shells.

        Returns
        -------
        dict
            Tuple of the scales (Mpc, or None if not smoothed) of the shells, for each of
            ``FIELDS`` and ``'positions'``, e.g. to schedule the smoothed maps needed by a catalog.
        """

        def per_shell(scale):
            # tuple of nshells floats, or of None
            if scale is None or np.isscalar(scale):
                return (float(scale) if scale else None,) * nshells
            if len(scale) != nshells:
                raise ValueError(f"{len(scale)} smoothing scales for {nshells} shells")
            return tuple(float(s) for s in scale)

        names = cls.FIELDS + ('positions',)
        if isinstance(smooth_Mpc, dict):
            scales = dict(smooth_Mpc)
        else:
            scales = {k: smooth_Mpc for k in smooth_fields} if smooth_Mpc is not None else {}
        unknown = set(scales) - set(names)
        if unknown:
            raise ValueError(f"Unknown fields {sorted(unknown)} for smoothing, must be among {names}")
        scales = {k: per_shell(scales.get(k)) for k in names}
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
        self._write_config()
        for ikappa in self.lensing.tqdm(ikappas, desc="[combine_shells]", leave=False):
            parts = [fits.getdata(self.get_fname(ikappa, ishell=ishell), 1) for ishell in range(self.nshells)]
            header = fits.Header()
            for k, v in self.header_dict.items():
                header[k] = v
            hdul = [fits.PrimaryHDU(), fits.BinTableHDU(data=np.concatenate(parts), header=header)]
            fname = self.get_fname(ikappa)
            with layout.atomic_write(fname) as tmp:
                fits.HDUList(hdul).writeto(tmp, overwrite=True)
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

    def _rng(self, stream: int, ikappa: int, ishell: int = -1) -> np.random.Generator:
        """Return the random generator of a stream (redshifts, positions, counts) of one bin and shell.

        Seeded by ``[seed of the realization, stream, ikappa, ishell + 1]``: independent between
        realizations, streams, bins and shells (``ishell=-1`` for draws over all shells).
        """
        return np.random.default_rng([self.lensing.shell_builder.seed, stream, ikappa, ishell + 1])

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

        rng = self._rng(self._STREAM_Z, ikappa)

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
        Fields at the far edges of the shells (convergence, shear) are interpolated
        linearly between consecutive shells in comoving distance; fields averaged over
        a shell (:attr:`SHELL_FIELDS`: density, tidal field) take the value of the
        galaxy's shell. Values are optionally interpolated across pixels (HealSparse
        ``interpolate_pos``).
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
                    map_low = self._load_interp_map(k, ishell - 1) if ishell > 0 and k not in self.SHELL_FIELDS else None
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
        smooth_Mpc = self.smoothing[k][ishell]
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
            is unused for ``ishell == 0`` and for :attr:`SHELL_FIELDS`.
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
                if k in self.SHELL_FIELDS:
                    # no interpolation between shells for fields averaged over the shell (density, tidal field)
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
            ``{catalog_dir}/cat_bin{ikappa}.{format}``, with ``catalog_dir`` the
            ``catalogs/{nz}/{tag}`` directory of the maps' products (see :mod:`pollux.io.layout`), or
            ``{intermediate}/parts/cat_bin{ikappa}_{ishell:03d}.{format}`` for a partial catalog.
        """
        if ishell is None:
            return os.path.join(self._catalog_dir, f'cat_bin{ikappa}.{format}')
        return os.path.join(self._intermediate_dir, 'parts', f'cat_bin{ikappa}_{ishell:03d}.{format}')

    def _write_config(self) -> None:
        """Write the ``config.yaml`` of the catalogs, or check it against the existing one (:func:`pollux.io.layout.write_config`)."""
        shell_builder = self.lensing.shell_builder
        augmentors = [a.config() for a in self.augmentors]
        parameters = dict(
            generator=type(self).__name__,
            nbar_input=self.nbar_input,
            bias=getattr(self, 'bias', None),
            bias_model=getattr(self, 'bias_model', None),
            interp_fields=self.interp_fields,
            interp_method=self.interp_method,
            smoothing=self.smoothing,
            augmentors=augmentors,
        )
        inputs = [self.lensing.save_dir, self.snap_lensing.save_dir, shell_builder.save_dir, shell_builder.smoothed_dir]
        layout.write_config(self._catalog_dir, 'catalogs', parameters, inputs=inputs, realization=shell_builder.realization_dir)

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
        self._write_config()
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
            with layout.atomic_write(fname) as tmp:
                fits.HDUList(hdul).writeto(tmp, overwrite=True)
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
        Only the requested columns are read (with ``fitsio``), and with ``subsample`` only the
        selected rows are kept, read in large contiguous blocks (see :func:`_read_rows`).
        """
        seed = self.lensing.shell_builder.seed if seed is None else seed

        # read the requested columns of each bin, optionally a random subsample of its rows
        headers = {}
        tables = {}
        for ikappa in self.lensing.tqdm(ikappas, desc="[load_catalogs]"):
            fname = self.get_fname(ikappa, format='fits')
            headers[ikappa] = fits.getheader(fname, ext=1)
            if subsample is None:
                table = fitsio.read(fname, ext=1, columns=columns)
            else:
                # random rows (galaxies are stored in shell order, so a stride would not be a fair sample)
                if not 0.0 < subsample < 1.0:
                    raise ValueError('subsample must satisfy 0 < subsample < 1')
                nrows = headers[ikappa]['NAXIS2']
                rows = np.sort(np.random.default_rng(seed + ikappa).choice(nrows, size=int(subsample * nrows), replace=False))
                table = _read_rows(fname, rows, columns)
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
            the density used to place them (``'positions'`` smoothing, at least the pixel size).
        """
        bias = getattr(self, 'bias', np.ones(self.ncats))
        # one scale per shell, between the shells' comoving distances
        scales = self.smoothing['positions']
        smooth_Mpc = None if scales[0] is None else (tuple(self.lensing.shell_builder.shell_edges_comoving_distance), scales)
        return [
            Tracer(f"density_{i}", 'density', z=t.z, nz=t.nz, bias=bias[i], smooth_Mpc=smooth_Mpc, smooth_nside=self.lensing.nside)
            for i, t in enumerate(self.lensing_tracers)
        ]

    @property
    def ia_tracers(self) -> list[Tracer] | None:
        """Intrinsic-alignment tracers of the catalog, from its :class:`~pollux.ia.IACalculator` augmentor.

        Returns
        -------
        list of pollux.cosmo.Tracer or None
            ``'ia'`` tracers named ``ia_{i}``, with the n(z) of the lensing tracers, the amplitudes
            of the augmentor (:meth:`~pollux.ia.IACalculator.tracer_amplitudes`, with the bias of
            each bin) and the smoothing of the tidal field; None without intrinsic alignments.
        """
        calculator = self._ia_calculator
        if calculator is None:
            return None
        bias = getattr(self, 'bias', np.ones(self.ncats))
        # smoothing of the tidal field, one scale per shell
        scales = self.smoothing['t11']
        smooth_Mpc = None if scales[0] is None else (tuple(self.lensing.shell_builder.shell_edges_comoving_distance), scales)
        tracers = []
        for i, t in enumerate(self.lensing_tracers):
            amplitudes = calculator.tracer_amplitudes(float(bias[i]))
            tracers.append(Tracer(f"ia_{i}", 'ia', z=t.z, nz=t.nz, smooth_Mpc=smooth_Mpc, smooth_nside=self.lensing.nside, ia=amplitudes))
        return tracers

    @property
    def _ia_calculator(self) -> Any:
        """The :class:`~pollux.ia.IACalculator` augmentor of the catalog, or None without intrinsic alignments."""
        from .ia import IACalculator  # here: pollux.ia imports this module
        return next((a for a in self.augmentors if isinstance(a, IACalculator)), None)

    def _shape_columns(self, shapes: str, shear_cols: tuple[str, str]) -> tuple[list[str], list[str]]:
        """Return the catalog columns summed into the two components of the shapes of the shear legs.

        Parameters
        ----------
        shapes : {'shear', 'ia', 'observed'}
            Shapes, as in :meth:`predict_2pt`: the lensing shear (``shear_cols``), the intrinsic
            alignments (the columns of the IA augmentor, e.g. ``e1_ia_ta``), or their sum (without
            shape noise, unlike the ``e1_obs`` columns).
        shear_cols : tuple of str
            Columns of the lensing shear.

        Returns
        -------
        list of str, list of str
            Columns of the first and of the second component.
        """
        if shapes not in self._SHAPES:
            raise ValueError(f"shapes must be one of {sorted(self._SHAPES)}, got {shapes!r}")
        cols1, cols2 = [], []
        if 'shear' in self._SHAPES[shapes]:
            cols1.append(shear_cols[0])
            cols2.append(shear_cols[1])
        if 'ia' in self._SHAPES[shapes]:
            if self._ia_calculator is None:
                raise ValueError(f"shapes={shapes!r} needs an intrinsic-alignment augmentor")
            # e1_ia_*, e2_ia_* of each IA model (TA, TT)
            ia_cols = self._ia_calculator.new_columns
            cols1 += ia_cols[0::2]
            cols2 += ia_cols[1::2]
        return cols1, cols2

    def _2pt_fname(self, shapes: str = 'shear') -> str:
        """Return the file of the two-point functions measured with the shapes ``shapes`` (see :meth:`compute_2pt`)."""
        return os.path.join(self._catalog_dir, 'cat_2pt.fits' if shapes == 'shear' else f'cat_2pt_{shapes}.fits')

    def _reference_scales(self, roles: tuple[str, str], shapes: str = 'shear') -> dict:
        """Return the reference scales of a statistic: of the lensing maps, and the smoothing of the fields of its legs.

        Parameters
        ----------
        roles : tuple of str
            Roles of the two legs (``'density'`` or ``'shear'``), as in ``_SACC_TYPE_LEGS_2PT``.
        shapes : str, optional
            Shapes of the shear legs (see :meth:`compute_2pt`): for ``'ia'``, their smoothing is
            that of the tidal field (the lensing shear is not smoothed).

        Returns
        -------
        dict
            :meth:`~pollux.map_builder.MapBuilder.reference_scales` of the lensing builder, with,
            if a leg is smoothed, ``theta_smooth`` (arcmin) and ``ell_smooth`` per bin pair: the
            smoothing of the density placing the galaxies of a density leg, or of the tidal field
            of an IA leg (``shapes='ia'``) (:meth:`~pollux.cosmo.Tracer.effective_smooth_Mpc`, at the mean redshift
            of the bin's n(z)), the geometric mean of the two legs if both are smoothed.
        """
        scales = self.lensing.reference_scales()

        # tracers carrying the smoothing of each leg (none for the lensing shear)
        leg_tracers = {'density': self.density_tracers, 'shear': self.ia_tracers if shapes == 'ia' else None}
        smoothed = [role for role in dict.fromkeys(roles) if leg_tracers[role] is not None and leg_tracers[role][0].smooth_Mpc is not None]
        if not smoothed:
            return scales

        # angle of the smoothing of each bin and smoothed leg, at the mean redshift of its galaxies
        theta = {}
        for role in smoothed:
            angles = []
            for t in leg_tracers[role]:
                chi = pyccl.comoving_radial_distance(self.lensing.cosmo, 1. / (1. + t.mean_z_nz()))
                angles.append(float(Tracer.effective_smooth_Mpc(t.smooth_Mpc, chi, t.smooth_nside)) / chi)
            theta[role] = np.array(angles)

        # per pair: geometric mean over the smoothed legs
        nbins = len(self.density_tracers)
        theta_pair = {}
        for i in range(nbins):
            for j in range(nbins):
                legs = [theta[role][k] for role, k in zip(roles, (i, j), strict=True) if role in theta]
                theta_pair[i, j] = float(np.exp(np.mean(np.log(legs))))
        rad2arcmin = 180. * 60. / np.pi
        scales['theta_smooth'] = {ij: rad2arcmin * t for ij, t in theta_pair.items()}
        scales['ell_smooth'] = {ij: np.pi / t for ij, t in theta_pair.items()}
        return scales

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
            without its pixel-window deconvolution: the pixel window is in the predictions
            instead (see :meth:`predict_cls`).
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
            ``{catalog_dir}/wsp_{stat}_lmax{lmax}_b{id}.wsp``, or
            ``{catalog_dir}/wsp_{stat}_{i}_{j}[_randoms]_lmax{lmax}_b{id}.wsp`` (with ``i >= j`` for
            the symmetric clustering and shear statistics), with ``id`` the identifier of the
            binning and of ``cls_lmax_mask`` (:func:`pollux.utils.binning_id`). Bandpower windows are saved
            next to it, with suffix ``_bpws.npy`` (see :meth:`get_bpws`).
        """
        if pair is None:
            name = stat
        else:
            i, j = pair
            if stat != 'shear_clustering':
                i, j = max(i, j), min(i, j)
            name = f'{stat}_{i}_{j}' + ('_randoms' if randoms and stat != 'shear' else '')
        # binning of the bandpowers and of the masks
        return os.path.join(self._intermediate_dir, f'wsp_{name}_lmax{self.b.lmax}_b{binning_id(self.b, extra=(self.lmax_mask,))}.wsp')

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
            with layout.atomic_write(fname) as tmp:
                wsp.write_to(tmp)
            with layout.atomic_write(fname.replace('.wsp', '_bpws.npy')) as tmp:
                np.save(tmp, wsp.get_bandpower_windows())
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
        with layout.atomic_write(fname_bpws) as tmp:
            np.save(tmp, bpws)
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

            - 'clustering': galaxy density auto/cross-spectra (dd), symmetric i>=j pairs.
            - 'shear': shear E/B-mode auto/cross-spectra (ee, bb), symmetric i>=j pairs.
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
        self._write_config()
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
        with layout.atomic_write(fname) as tmp:
            s.save_fits(tmp, overwrite=True)
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
            stacked: bool = False,
            ell: np.ndarray | None = None,
            pixel_window: bool = True) -> tuple:
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
        stacked : bool, optional
            If True, also return the E-mode and density spectra of all statistics
            stacked into one vector (in the order of ``what``, then of the pairs), and
            with ``error_bars`` their joint covariance.
        ell : ndarray or None, optional
            Multipoles for ``binning='unbinned'``. Default is the effective bandpower multipoles.
        pixel_window : bool, optional
            If True (default), multiply the spectra by the pixel window of the maps (one per
            field): galaxies are placed in the pixels of the density maps, and the fields are
            interpolated from pixelized maps.

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
            f_ell=hp.pixwin(self.lensing.nside, lmax=self.b.lmax) if pixel_window else None,
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
        return tuple(out)

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
                    scales=self._reference_scales(self._SACC_TYPE_LEGS_CLS[data_type]),
                    color=color,
                    logy=logy,
                    xlim=(self.b.get_ell_min(0), self.b.get_ell_max(self.b.get_n_bands() - 1)),
                    sharey=False,
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
            shapes: str = 'shear',
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

            - 'clustering': galaxy density autocorrelation ``w(theta)``, symmetric i>=j pairs.
            - 'shear': shear autocorrelation ``xi_+/-(theta)``, symmetric i>=j pairs.
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
        shapes : {'shear', 'ia', 'observed'}, optional
            Shapes of the shear legs, as in :meth:`predict_2pt`: the lensing shear
            (``shear_cols``, default), the intrinsic alignments (II and gI), or their
            sum without shape noise (GG + GI + IG + II and gG + gI). The same galaxies
            are used for all (same ``subsample`` and ``seed``), so that GI is exactly the
            difference of the three.
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
            file via :meth:`_save_2pt`, at ``{self._catalog_dir}/cat_2pt.fits``
            (``cat_2pt_{shapes}.fits`` for shapes other than ``'shear'``).

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
        cols1, cols2 = self._shape_columns(shapes, shear_cols)

        corr_kwargs = dict(min_sep=theta_min, max_sep=theta_max, nbins=ntheta, sep_units=sep_units, bin_slop=bin_slop)

        # Build each bin's catalog(s) once, reused across all pairs below.
        # A single galaxy catalog serves as both the density tracer (for NN)
        # and the shear tracer (for GG/NG): NNCorrelation only looks at
        # positions and simply ignores the g1/g2 fields when present.
        cat_gal, cat_rand = {}, {}
        shared_cat_rand = None
        for ikappa in self.lensing.tqdm(ikappas, desc="[compute_2pt] building catalogs"):
            columns = cols1 + cols2
            cat = self.load_catalogs(ikappa, subsample=subsample, seed=seed, columns=['ra', 'dec', *columns], remove_mean=columns if remove_mean else None)

            # galaxies, with their shapes if needed (sum of the columns; TreeCorr's g2 convention is flipped)
            gal_kwargs = dict(ra=cat['ra'], dec=cat['dec'], ra_units='deg', dec_units='deg')
            if need_shear:
                gal_kwargs.update(g1=sum(cat[c] for c in cols1), g2=sum(cat[c] for c in cols2), flip_g2=True)
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
            self._save_2pt(theta_pairs, xi, dxi=dxi, shapes=shapes)

        theta_out = theta_pairs if theta_per_pair else theta

        if error_bars:
            return theta_out, xi, dxi
        else:
            return theta_out, xi

    def _save_2pt(self, theta: dict, xi: dict, dxi: dict | None = None, shapes: str = 'shear') -> str:
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
        shapes : str, optional
            Shapes of the shear legs (see :meth:`compute_2pt`), which set the file name.

        Returns
        -------
        str
            Path to the saved SACC FITS file.
        """
        self._write_config()
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
        fname = self._2pt_fname(shapes)
        with layout.atomic_write(fname) as tmp:
            s.save_fits(tmp, overwrite=True)
        logger.debug("Saved to file %s.", fname)
        return fname

    def load_2pt(self, return_dict: bool = False, return_errors: bool = False, shapes: str = 'shear') -> sacc.Sacc | tuple:
        """Load catalog-based two-point functions previously saved by :meth:`_save_2pt`.

        Parameters
        ----------
        return_dict : bool, optional
            If True, return ``(theta, xi)``. Otherwise, return the SACC object.
        return_errors : bool, optional
            With ``return_dict``, also return the error bars (square root of the
            diagonal of the saved covariance), with the same structure as ``xi``.
        shapes : str, optional
            Shapes of the shear legs of the measurement (see :meth:`compute_2pt`).

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
        fname = self._2pt_fname(shapes)
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

    # shape fields of the shear legs of the predictions: lensing shear (G) and intrinsic alignments (I)
    _SHAPES = {'shear': ('shear',), 'observed': ('shear', 'ia'), 'ia': ('ia',)}

    def predict_2pt(self, theta: dict, nell: int = 500, *, shapes: str = 'shear') -> dict:
        """Predict the catalog two-point correlation functions.

        Predictions use the matter power spectrum of the catalog's shells (see :meth:`predict_cls`).

        Parameters
        ----------
        theta : dict
            Angular separations in arcmin keyed by SACC data type, then by ``(i, j)``
            (as returned by :meth:`load_2pt` or :meth:`compute_2pt` with ``theta_per_pair=True``).
        nell : int, optional
            Number of log-spaced multipoles up to the bandpower ``lmax`` used for the Hankel transforms.
        shapes : {'shear', 'observed', 'ia'}, optional
            Shapes of the shear legs: ``'shear'`` (default), the lensing shear (columns ``gamma1``,
            ``gamma2``); ``'observed'``, lensing shear plus intrinsic alignments (GG + GI + IG + II
            for cosmic shear, gG + gI for galaxy-galaxy lensing; also columns ``e1_obs``, ``e2_obs``, with shape noise);
            ``'ia'``, the intrinsic alignments alone (II, gI; columns ``e1_ia_*``, ``e2_ia_*``).
            See :attr:`ia_tracers`.

        Returns
        -------
        dict
            Same structure as ``theta``; ``galaxy_shearDensity_xi_x`` is zero.
        """
        if shapes not in self._SHAPES:
            raise ValueError(f"shapes must be one of {sorted(self._SHAPES)}, got {shapes!r}")
        tracers = {'density': self.density_tracers, 'shear': self.lensing_tracers}
        if shapes != 'shear':
            tracers['ia'] = self.ia_tracers
            if tracers['ia'] is None:
                raise ValueError(f"shapes={shapes!r} needs an intrinsic-alignment augmentor")
        xi = {}
        for data_type, type_pairs in theta.items():
            if data_type not in self._SACC_TYPE_COMPONENT_2PT:
                xi[data_type] = {ij: np.zeros(len(tt)) for ij, tt in type_pairs.items()}
                continue
            legs = [self._SHAPES[shapes] if role == 'shear' else (role,) for role in self._SACC_TYPE_LEGS_2PT[data_type]]
            component = self._SACC_TYPE_COMPONENT_2PT[data_type]
            # sum over the fields of the two legs (e.g. GG + GI + IG + II)
            xi[data_type] = {ij: 0. for ij in type_pairs}
            for kind_i in legs[0]:
                for kind_j in legs[1]:
                    pairs = {(i, j): (tracers[kind_i][i], tracers[kind_j][j]) for i, j in type_pairs}
                    res = self.lensing.theory.predict_2pt(pairs, type_pairs, self.b.lmax, nell=nell)
                    for ij, r in res.items():
                        xi[data_type][ij] = xi[data_type][ij] + (r if component is None else r[component])
        return xi

    def plot_2pt(
            self,
            plot_dir: str | None = None,
            *,
            modes: tuple[str, ...] = ('values', 'ratio'),
            reference: CatalogSourceGenerator | None = None,
            xlim: tuple[float, float] | None = None,
            shapes: str = 'shear') -> None:
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
        xlim : tuple of float or None, optional
            Limits of the x axis, e.g. the edges of the binning, ``(theta_min, theta_max)`` of
            :meth:`compute_2pt`. Default is the range of the measured ``theta``.
        shapes : {'shear', 'ia', 'observed'}, optional
            Shapes of the shear legs of the measurement (see :meth:`compute_2pt`).

        Notes
        -----
        Saves, with suffix ``_{mode}`` for modes other than ``'values'``: ``xi_shear``,
        ``xi_shear_density`` and ``xi_density`` for the lensing shear; ``xi_ia`` (II) and
        ``xi_ia_density`` (gI) for the intrinsic alignments, and ``xi_gi`` (GI: observed minus
        shear minus IA, when the three are measured); ``xi_observed`` and
        ``xi_observed_density`` for their sum.
        """
        from .plotting import plot_pairs
        plot_dir = plot_dir or os.path.join(self._catalog_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        st = sacc.standard_types
        shear_types = (st.galaxy_shear_xi_plus, st.galaxy_shear_xi_minus)
        colors = {st.galaxy_shear_xi_plus: 'dodgerblue', st.galaxy_shear_xi_minus: 'orange', st.galaxy_shearDensity_xi_t: 'red', st.galaxy_density_xi: 'green'}

        # measurements and predictions (and those of the reference catalog)
        theta, xi, dxi = self.load_2pt(return_dict=True, return_errors=True, shapes=shapes)
        xi_th = self.predict_2pt(theta, shapes=shapes)
        xi_ref, xi_th_ref = {}, {}
        if reference is not None:
            _, xi_ref = reference.load_2pt(return_dict=True, shapes=shapes)
            xi_th_ref = reference.predict_2pt(theta, shapes=shapes)
            modes = (*modes, 'ratio_to_reference')

        # figures: name, y label, and the values of the data types overplotted (theta, data, theory,
        # data errors, reference data and theory)
        label, label_t = {'shear': ('', ''), 'ia': (r'^{\rm II}', r'^{\rm gI}'), 'observed': (r'^{\rm obs}', r'^{\rm obs}')}[shapes]
        groups = [(f'xi_{shapes}', rf'$\theta \xi{label}_\pm(\theta)$', shear_types),
                  (f'xi_{shapes}_density', rf'$\theta \gamma{label_t}_{{\rm t}}(\theta)$', (st.galaxy_shearDensity_xi_t,))]
        if shapes == 'shear':
            groups.append(('xi_density', r'$\theta w(\theta)$', (st.galaxy_density_xi,)))
        figures = []
        for fig_name, ylabel, data_types in groups:
            figures.append((fig_name, ylabel, {t: (theta[t], xi[t], xi_th[t], dxi.get(t), xi_ref.get(t), xi_th_ref.get(t)) for t in data_types if t in xi}))

        # GI = observed - shear - IA, from the same galaxies (data) and from the predictions at the same theta
        if shapes == 'ia' and all(os.path.exists(self._2pt_fname(s)) for s in ('shear', 'observed')):
            _, xi_obs = self.load_2pt(return_dict=True, shapes='observed')
            _, xi_g = self.load_2pt(return_dict=True, shapes='shear')
            th_obs, th_g = self.predict_2pt(theta, shapes='observed'), self.predict_2pt(theta, shapes='shear')
            gi = {}
            for t in shear_types:
                if t in xi and t in xi_obs and t in xi_g:
                    pairs = [ij for ij in xi[t] if ij in xi_obs[t] and ij in xi_g[t]]
                    data = {ij: xi_obs[t][ij] - xi_g[t][ij] - xi[t][ij] for ij in pairs}
                    pred = {ij: th_obs[t][ij] - th_g[t][ij] - xi_th[t][ij] for ij in pairs}
                    gi[t] = ({ij: theta[t][ij] for ij in pairs}, data, pred, None, None, None)
            figures.append(('xi_gi', r'$\theta \xi^{\rm GI}_\pm(\theta)$', gi))

        # one figure per group of data types and mode, the data types of a group overplotted
        for fig_name, ylabel, data_types in figures:
            for mode in modes:
                if mode == 'ratio_to_reference' and fig_name == 'xi_gi':
                    continue
                # the last data type of the group saves the figure
                figaxes = None
                for k, (data_type, (th, data, theory, errors, data_ref, theory_ref)) in enumerate(data_types.items()):
                    last = k == len(data_types) - 1
                    figaxes = plot_pairs(
                        th,
                        data=data,
                        theory=theory,
                        data_errors=errors,
                        data_ref=data_ref,
                        theory_ref=theory_ref,
                        mode=mode,
                        auto_only=all(i == j for i, j in data),
                        ell_factor=1,
                        scales=self._reference_scales(self._SACC_TYPE_LEGS_2PT[data_type], shapes='ia' if fig_name == 'xi_gi' else shapes),
                        color=colors[data_type],
                        xlabel=r'$\theta$ (arcmin)',
                        ylabel=ylabel if mode == 'values' else None,
                        xlim=xlim,
                        sharey=False,
                        figaxes=figaxes,
                        fname=os.path.join(plot_dir, fig_name + ('' if mode == 'values' else f'_{mode}') + '.png') if last else None)


class CatalogSourceUniform(CatalogSourceGenerator):
    """Generate source galaxies uniformly over the footprint.

    Redshifts are drawn from the tomographic tracer distributions. Angular
    positions are drawn uniformly over the footprint of the maps.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag of the catalogs: they go to ``catalogs/{nz}/{tag}/`` next to the maps (see :mod:`pollux.io.layout`), ``default`` if None.
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
        rng = self._rng(self._STREAM_POS, ikappa)

        # uniform over the footprint: equal-area pixels drawn at random, then a position within each
        _, ipix = self.lensing.shell_builder.get_mask(return_ipix=True)
        ra, dec = generate_radec_in_pix(self.nside, rng.choice(ipix, size=self.Ngal[ikappa]), rng)
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
        Tag of the catalogs: they go to ``catalogs/{nz}/{tag}/`` next to the maps (see :mod:`pollux.io.layout`), ``default`` if None.
    nbar_input : float, array-like, or None, optional
        Galaxy surface density in gal/arcmin^2. Default is None.
    bias : float or array-like, optional
        Galaxy bias per tomographic bin. If a float, the same bias is used
        for all tomographic bins; if an array-like, it must have length
        ``lensing.nmaps`` and provide a per-bin value.
    bias_model : {'linear', 'power_law'}, optional
        Relative galaxy density per pixel, given the density contrast ``delta``
        (see :meth:`_density_weight`):

        - ``'linear'`` (default): ``1 + b * delta``, clipped at zero where negative
          (underdense pixels with ``b > 1``);
        - ``'power_law'``: ``(1 + delta)**b / <(1 + delta)**b>``, the mean over the
          footprint pixels of the shell, positive everywhere (no clipping), and with
          linear bias ``b`` on large scales.
    interp_fields, interp_method, smooth_fields, smooth_Mpc : optional
        Interpolation and smoothing of the fields (including the density used to
        place galaxies, ``'positions'``), see :class:`CatalogSourceGenerator`.
    augmentors : list, optional
        List of augmentors applied after map interpolation. Default is empty list.
    cls_b, cls_lmax_mask : optional
        Binning of the catalog angular power spectra, see :class:`CatalogSourceGenerator`.
    """

    BIAS_MODELS = ('linear', 'power_law')

    def __init__(
        self,
        lensing: LensingBuilder,
        tag: str | None = None,
        nbar_input: float | np.ndarray | None = None,
        bias: float | np.ndarray | None = None,
        bias_model: str = 'linear',
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
        if bias_model not in self.BIAS_MODELS:
            raise ValueError(f"bias_model must be one of {self.BIAS_MODELS}, not {bias_model!r}")
        self.bias_model = bias_model
        self.header_dict['BIASMOD'] = bias_model

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
        """Return the relative galaxy density per pixel of the bias model (``bias_model``).

        - ``'linear'``: ``1 + b * delta``. With ``b > 1`` it is negative in underdense
          pixels, where no galaxies can be placed: clipped to zero, with a warning
          giving the fraction of such pixels.
        - ``'power_law'``: ``(1 + delta)**b`` divided by its mean over the pixels, so
          that the mean density is unchanged. It is positive wherever ``1 + delta`` is;
          pixels with ``1 + delta < 0`` (ringing of the smoothed map) are set to zero,
          with a warning.

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
        b = self.bias[ikappa]
        if self.bias_model == 'power_law':
            weight = 1. + density.astype(float)
            fraction = np.mean(weight < 0.)
            if fraction > 0.:
                warnings.warn(f"1 + delta < 0 in {100. * fraction:.3g}% of the pixels (bin {ikappa}, shell {ishell}): set to 0.")
            weight = np.clip(weight, 0., None)**b
            return weight / np.mean(weight)

        weight = 1. + b * density
        fraction = np.mean(weight < 0.)
        if fraction > 0.:
            warnings.warn(f"1 + b * delta < 0 in {100. * fraction:.3g}% of the pixels (bin {ikappa}, shell {ishell}, b = {b:g}): clipped to 0.")
        return np.clip(weight, 0., None)


class CatalogSourceClusteredMultinomial(CatalogSourceClustered):
    """Generate clustered source galaxies using multinomial sampling.

    The total number of galaxies per tomographic bin is fixed, and positions are
    drawn per shell from a multinomial distribution over pixels, proportional to
    the relative density of the bias model (``1 + b * delta`` by default, where ``delta``
    is the density contrast field, see :class:`CatalogSourceClustered`).

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag of the catalogs: they go to ``catalogs/{nz}/{tag}/`` next to the maps (see :mod:`pollux.io.layout`), ``default`` if None.
    nbar_input : float, array-like, or None, optional
        Galaxy surface density in gal/arcmin^2. Default is None.
    bias, bias_model : optional
        Galaxy bias per tomographic bin, and bias model, see :class:`CatalogSourceClustered`.
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
        density_masked = self._density_map(ishell, self.smoothing['positions'][ishell]).get_values_pix(ipix_mask)

        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, leave=False):
            rng = self._rng(self._STREAM_COUNTS, ikappa, ishell)

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
    as a Poisson random variable with mean ``<nbar>`` times the relative density of
    the bias model (``1 + b * delta`` by default, see :class:`CatalogSourceClustered`).
    The total number of galaxies therefore fluctuates.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag of the catalogs: they go to ``catalogs/{nz}/{tag}/`` next to the maps (see :mod:`pollux.io.layout`), ``default`` if None.
    nbar_input : float, array-like, or None, optional
        Galaxy surface density in gal/arcmin^2. Default is None.
    bias, bias_model : optional
        Galaxy bias per tomographic bin, and bias model, see :class:`CatalogSourceClustered`.
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
        density_masked = self._density_map(ishell, self.smoothing['positions'][ishell]).get_values_pix(ipix_mask)

        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, leave=False):
            rng = self._rng(self._STREAM_COUNTS, ikappa, ishell)

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
            rng = self._rng(self._STREAM_Z, ikappa, ishell)

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


def _read_rows(fname: str, rows: np.ndarray, columns: list[str] | None = None, block: int = 5_000_000) -> np.ndarray:
    """Read given rows of the table of a FITS file, by contiguous blocks of rows.

    Reading scattered rows directly makes one small read per row; reading blocks and keeping
    the rows of each is several times faster for subsamples of large catalogs (e.g. 1 min
    instead of 5 for 0.5% of 2e8 rows on Lustre), and gives the same rows.

    Parameters
    ----------
    fname : str
        FITS file, with the table in HDU 1.
    rows : ndarray of int
        Sorted row indices.
    columns : list of str or None, optional
        Columns to read. If None, all columns.
    block : int, optional
        Number of rows per block.

    Returns
    -------
    numpy.ndarray
        Structured array of the rows.
    """
    with fitsio.FITS(fname) as f:
        hdu = f[1] if columns is None else f[1][columns]
        parts = []
        for start in range(0, int(rows[-1]) + 1 if len(rows) else 0, block):
            i0, i1 = np.searchsorted(rows, [start, start + block])
            if i1 > i0:
                parts.append(hdu[start:min(start + block, int(rows[-1]) + 1)][rows[i0:i1] - start])
        return np.concatenate(parts) if parts else f[1].read(columns=columns, rows=rows)


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

    def config(self) -> dict:
        """Return the type and parameters of the augmentor, for the ``config.yaml`` of the catalogs."""
        return {'type': type(self).__name__}
