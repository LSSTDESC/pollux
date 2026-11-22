"""Tomographic source catalogs from the shells and lensing maps, by uniform or clustered sampling of the galaxies.

Their two-point statistics (Cls with NaMaster, two-point functions with TreeCorr) and predictions are methods inherited from
:class:`~pollux.statistics.catalogs.CatalogStatistics`.
"""

from __future__ import annotations

import inspect
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import healpy as hp
import pyccl
import pymaster as nmt
import fitsio
from astropy.io import fits

from .io.code_version import code_version
from .utils.numerics import quad, sample_pdf
from .utils.fits import read_fits_rows
from .utils.healpix import interpolate_pos, values_pos
from .utils.sky import generate_radec_in_pix
from .statistics.catalogs import CatalogStatistics

if TYPE_CHECKING:
    import healsparse as hsp
    from .lensing import LensingBuilder

logger = logging.getLogger(__name__)


class CatalogGenerator(CatalogStatistics, ABC):
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
        Tag of the catalogs: they go to ``catalogs/{nz}/{tag}/`` next to the maps (see :class:`pollux.io.files.PolluxIO`), ``default`` if None.
    nbar_input : float or array-like
        Galaxy surface density in gal/arcmin^2 (required, keyword only). If a float, the same surface
        density is used for all tomographic bins; if an array-like, it must
        have length ``lensing.nmaps`` and provide a per-bin value.
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
        *,
        nbar_input: float | np.ndarray,
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
        # augmentors of the shapes and intrinsic alignments, which the statistics model
        by_kind = {a.kind: a for a in self.augmentors if a.kind is not None}
        self.shear_augmentor = by_kind.get('shear')
        self.ia_augmentor = by_kind.get('ia')

        # catalogs by n(z) and tag, next to the maps they come from
        self.io = lensing.io.with_options(tag=tag or 'default')

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
        self.augmentor_kwargs = {ibin: {'ibin': ibin, 'realization_seed': self.lensing.shell_builder.seed} for ibin in self.lensing.iter()}
        self.cls_b = cls_b
        self.cls_lmax_mask = cls_lmax_mask
        self._wsp = {}

        # Handle nbar
        nbar_arr = np.atleast_1d(self.nbar_input).astype(float)
        if nbar_arr.size == 1:
            nbar_arr = np.full(self.lensing.nmaps, float(nbar_arr[0]))
        elif nbar_arr.size != self.lensing.nmaps:
            raise ValueError(f"nbar must be a scalar or array of length {self.lensing.nmaps}, got {nbar_arr.size}")
        self.nbar = nbar_arr

        # galaxy counts over the footprint, and per pixel
        area_arcmin = 4. * np.pi * self.lensing.footprint.fsky * (180. * 60 / np.pi)**2
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
        logger.info("  - Catalogs saved to %s.", self.io.dir('catalogs'))

        # configuration of the catalogs (checked against an existing one), before any product is read or written
        self.io.write_config('catalogs', self.config())

    def config(self) -> dict:
        """Return the configuration of the catalogs, for their ``config.yaml`` (:func:`pollux.io.config.write_config`).

        Returns
        -------
        dict
            ``product`` (``'catalogs'``), ``parameters`` (generator, galaxy densities and bias, interpolated fields
            and their smoothing, augmentors: their :meth:`CatalogAugmentor.config`) and ``inputs`` (the
            convergence maps of the n(z) and per shell, shells and smoothed maps).
        """
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
        inputs = [self.lensing.save_dir, self.snap_lensing.save_dir, shell_builder.save_dir]
        return dict(product='catalogs', parameters=parameters, inputs=inputs)

    def path(self, ibin: int, ext: str = 'fits', ishell: int | None = None) -> str:
        """Return the path for the saved catalog of one tomographic bin.

        Parameters
        ----------
        ibin : int
            Tomographic-bin index.
        ext : str, optional
            File extension without the leading dot. Default is ``'fits'``.
        ishell : int or None, optional
            If given, path of the catalog of shell ``ishell`` (:meth:`generate` with ``ishell``).

        Returns
        -------
        str
            ``{catalog_dir}/cat_bin{ibin}.{ext}``, with ``catalog_dir`` the
            ``catalogs/{nz}/{tag}`` directory of the maps' products (see :mod:`pollux.io.files`), or
            ``{intermediate}/shells/cat_bin{ibin}_{ishell:03d}.{ext}`` for the catalog of one shell.
        """
        if ishell is None:
            return self.io.path('catalogs', 'cat', self.io.BIN_INDEX.format(ibin), ext=ext)
        return self.io.path('catalogs', 'cat', self.io.BIN_INDEX.format(ibin), self.io.SHELL_INDEX.format(ishell), ext=ext, intermediate=True, folder='shells')

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

    # ==================================================================================================================
    # region Galaxies

    def generate(self, ibins: int | Iterable[int] | None = None, ishell: int | None = None) -> None:
        """Generate mock catalogs and interpolate requested map quantities.

        Parameters
        ----------
        ibins : int or array_like of int or None, optional
            Tomographic-bin indices to generate. If None, generate for all bins.
        ishell : int or None, optional
            If given, only generate the galaxies of shell ``ishell`` and save them
            as the catalogs of the shell (one per bin), to be merged with
            :meth:`merge_shell_catalogs`. Only supported by subclasses drawing galaxies
            shell by shell (:class:`CatalogClusteredPoisson`).

        Notes
        -----
        The generation proceeds in four steps:
        1) draw galaxies (positions/redshifts and shell membership),
        2) interpolate requested fields from maps,
        3) apply the augmentors,
        4) save one FITS file per tomographic bin.
        """
        catalog = {ibin: {} for ibin in self.lensing.iter(ibins)}
        ishells = None if ishell is None else [ishell]

        # Generate ra, dec, z, etc.
        self._generate_galaxies(catalog, ibins=ibins, ishells=ishells)

        # Interpolate kappa/shear values
        self._interpolate_maps(catalog, ibins=ibins, ishells=ishells)

        # Augment catalogs if requested
        for augmentor in self.augmentors:
            for ibin in self.lensing.tqdm(ibins, desc=f"[generate] augmenting catalogs with {augmentor.name}", leave=False):
                augmentor.augment_catalog(self.cosmo, catalog[ibin], **self.augmentor_kwargs[ibin])

        # Save it!
        self._write_catalogs(catalog, ibins=ibins, ishell=ishell)

    @abstractmethod
    def _generate_galaxies(self, catalog: dict[int, dict[str, np.ndarray]], ibins: int | Iterable[int] | None = None, ishells: list[int] | None = None) -> None:
        """Populate a catalog container with galaxy properties (redshifts and positions).

        Parameters
        ----------
        catalog : dict
            Dictionary containing per-bin catalog data that will be filled in-place.
        ibins : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        ishells : list of int or None, optional
            Shells whose galaxies are generated. If None, all shells.
        """

    def _generate_z(self, catalog: dict[int, dict[str, np.ndarray]], ibin: int) -> None:
        """Draw redshifts from the tracer n(z) via inverse-CDF sampling.

        Parameters
        ----------
        catalog : dict
            Per-bin output container.
        ibin : int
            Tomographic-bin index.

        Notes
        -----
        Shell membership is set by digitizing redshifts onto the shell edges
        returned by ``shell_builder.shell_edges_redshift``.
        """
        builder = self.lensing.shell_builder
        z_edges = builder.shell_edges_redshift

        rng = self._rng(self._STREAM_Z, ibin)

        # Generate redshifts according to lensing object redshift distributions (using cdf sampling)
        t = self.lensing.tracers[ibin].ccl(self.lensing.cosmo)
        zz = np.linspace(0, z_edges[-1], 1000)
        nz = t.get_dndz(zz)

        z = sample_pdf(zz, nz, self.Ngal[ibin], rng)
        chi = builder.cosmo.comoving_radial_distance(1. / (1. + z))

        catalog[ibin]['z'] = z
        catalog[ibin]['chi'] = chi
        catalog[ibin]['idx_shell'] = np.digitize(z, z_edges) - 1

    def _rng(self, stream: int, ibin: int, ishell: int = -1) -> np.random.Generator:
        """Return the random generator of a stream (redshifts, positions, counts) of one bin and shell.

        Seeded by ``[seed of the realization, stream, ibin, ishell + 1]``: independent between
        realizations, streams, bins and shells (``ishell=-1`` for draws over all shells).

        Parameters
        ----------
        stream : int
            Stream (``_STREAM_Z``, ``_STREAM_POS``, ``_STREAM_COUNTS``).
        ibin : int
            Tomographic bin.
        ishell : int, optional
            Shell, -1 (default) for draws over all shells.

        Returns
        -------
        numpy.random.Generator
        """
        return np.random.default_rng([self.lensing.shell_builder.seed, stream, ibin, ishell + 1])

    # endregion

    # ==================================================================================================================
    # region Interpolation

    def _interpolate_maps(self, catalog: dict[int, dict[str, np.ndarray]], ibins: int | Iterable[int] | None = None, ishells: list[int] | None = None) -> None:
        r"""Interpolate map quantities at galaxy positions.

        Parameters
        ----------
        catalog : dict
            Per-bin catalog dictionaries containing at least ``'ra'``, ``'dec'``,
            ``'chi'`` and ``'idx_shell'``.
        ibins : int or array_like of int or None, optional
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

        For a galaxy of shell :math:`i` at comoving distance :math:`\chi`, a field known at the far edges of the shells
        (:math:`f_i` at :math:`\chi_{i+1}`, and :math:`f_{-1} = 0` at the observer) is

        .. math::

            f(\chi) = f_{i-1} + \frac{\chi - \chi_i}{\chi_{i+1} - \chi_i} \left(f_i - f_{i-1}\right),

        and a field averaged over the shells is :math:`f(\chi) = f_i`.
        """
        for k in self.interp_fields:
            if ishells is None:
                # all shells: each map is read once and reused as the lower map of the next shell
                map_high = None
                for ishell in self.lensing.shell_builder.tqdm(None, desc=f"[generate] interpolating maps {k}"):
                    map_low = map_high
                    map_high = self._read_interp_map(k, ishell)
                    self._interpolate_shell(catalog, k, ishell, map_low, map_high, ibins=ibins)
            else:
                for ishell in ishells:
                    map_low = self._read_interp_map(k, ishell - 1) if ishell > 0 and k not in self.SHELL_FIELDS else None
                    map_high = self._read_interp_map(k, ishell)
                    self._interpolate_shell(catalog, k, ishell, map_low, map_high, ibins=ibins)

    def _read_interp_map(self, k: str, ishell: int) -> np.ndarray | hsp.HealSparseMap:
        """Read the map of field ``k`` for one shell, in the format of its file (dense RING or HealSparse).

        Parameters
        ----------
        k : str
            Field name (e.g. ``'kappa'``, ``'gamma1'``, ``'density'``, ``'t11'``).
        ishell : int
            Shell index.

        Returns
        -------
        ndarray or hsp.HealSparseMap
            Map of field ``k``, smoothed as given by ``smooth_fields`` and ``smooth_Mpc``; the unsmoothed density
            contrast is computed from the counts of the shell (dense RING map).
        """
        smooth_Mpc = self.smoothing[k][ishell]
        # density and tidal field from the shells, convergence and shear from the lensing builder of each shell
        builder = self.lensing.shell_builder if k in self.SHELL_FIELDS else self.snap_lensing
        if k == 'density' and smooth_Mpc is None:
            return builder.field_map(ishell)
        return builder.read_maps(ishell, what=None if k == 'kappa' else k, smooth_Mpc=smooth_Mpc, dense=None)

    def _interpolate_shell(
            self,
            catalog: dict[int, dict[str, np.ndarray]],
            k: str,
            ishell: int,
            map_low: np.ndarray | hsp.HealSparseMap | None,
            map_high: np.ndarray | hsp.HealSparseMap,
            ibins: int | Iterable[int] | None = None) -> None:
        """Interpolate field ``k`` at the positions of the galaxies of one shell.

        Parameters
        ----------
        catalog : dict
            Per-bin catalog dictionaries, updated in place.
        k : str
            Field name.
        ishell : int
            Shell index.
        map_low, map_high : ndarray or hsp.HealSparseMap or None
            Maps of field ``k`` (dense RING or HealSparse) for shells ``ishell - 1`` and ``ishell``; ``map_low``
            is unused for ``ishell == 0`` and for :attr:`SHELL_FIELDS`.
        ibins : int or array_like of int or None, optional
            Tomographic-bin indices to process.
        """
        chi_low, chi_high = self.lensing.shell_builder.shell_edges_comoving_distance[[ishell, ishell + 1]]

        # Loop over tomographic bins
        for ibin in self.lensing.tqdm(ibins, disable=True):
            # select galaxies in current shell
            w = (catalog[ibin]['idx_shell'] == ishell)

            if np.any(w):
                ra = catalog[ibin]['ra'][w]
                dec = catalog[ibin]['dec'][w]
                if k in self.SHELL_FIELDS:
                    # no interpolation between shells for fields averaged over the shell (density, tidal field)
                    if self.interp_method == 'bilinear':
                        catalog[ibin][k][w] = interpolate_pos(map_high, ra, dec, allow_partial=True)
                    else:
                        catalog[ibin][k][w] = values_pos(map_high, ra, dec)
                else:
                    # prepare for interpolation
                    dchi = (catalog[ibin]['chi'][w] - chi_low)  # distance

                    # interpolation between shells and (optionnally) accross pixels
                    if self.interp_method == 'bilinear':
                        if ishell == 0:
                            catalog[ibin][k][w] = dchi * interpolate_pos(map_high, ra, dec, allow_partial=True) / (chi_high - chi_low)
                        else:
                            map_low_interp = interpolate_pos(map_low, ra, dec, allow_partial=True)
                            map_high_interp = interpolate_pos(map_high, ra, dec, allow_partial=True)
                            catalog[ibin][k][w] = map_low_interp + dchi * (map_high_interp - map_low_interp) / (chi_high - chi_low)
                    else:
                        high = values_pos(map_high, ra, dec)
                        if ishell == 0:
                            catalog[ibin][k][w] = dchi * high / (chi_high - chi_low)
                        else:
                            low = values_pos(map_low, ra, dec)
                            catalog[ibin][k][w] = low + dchi * (high - low) / (chi_high - chi_low)

    # endregion

    # ==================================================================================================================
    # region Catalogs

    def _write_catalogs(self, catalog: dict[int, dict[str, np.ndarray]], ibins: int | Iterable[int] | None = None, ishell: int | None = None) -> None:
        """Write per-bin catalogs to FITS files.

        Parameters
        ----------
        catalog : dict
            Per-bin outputs containing ``'ra'``, ``'dec'``, ``'z'``, ``'chi'`` and
            any fields listed in ``self.interp_fields``.
        ibins : int or array_like of int or None, optional
            Tomographic-bin indices to save.
        ishell : int or None, optional
            If given, save as the catalogs of shell ``ishell`` (one per bin).
        """
        columns = list(self.interp_fields)
        for augmentor in self.augmentors:
            columns += augmentor.new_columns
        if ishell is not None:
            os.makedirs(os.path.dirname(self.path(0, ishell=ishell)), exist_ok=True)

        for ibin in self.lensing.tqdm(ibins, desc="[generate] saving", leave=False):

            ra = fits.Column(name='ra', format='E', array=catalog[ibin]['ra'])
            dec = fits.Column(name='dec', format='E', array=catalog[ibin]['dec'])
            z = fits.Column(name='z', format='E', array=catalog[ibin]['z'])
            chi = fits.Column(name='chi', format='E', array=catalog[ibin]['chi'])

            interp_colums = [fits.Column(name=k, format='E', array=catalog[ibin][k]) for k in columns]

            header = fits.Header()
            for k, v in self.header_dict.items():
                header[k] = v

            hdul = [fits.PrimaryHDU(), fits.BinTableHDU.from_columns([ra, dec, z, chi] + interp_colums, header)]

            self.io.write_table(self.path(ibin, ext='fits', ishell=ishell), hdul)

    def merge_shell_catalogs(self, ibins: int | Iterable[int] | None = None) -> None:
        """Merge the catalogs of the shells (:meth:`generate` with ``ishell``, saved in ``shells/``) into one catalog per bin.

        Parameters
        ----------
        ibins : int or array_like of int or None, optional
            Tomographic-bin indices to merge. If None, merge all bins.

        Notes
        -----
        Partial catalogs are produced by :meth:`generate` with ``ishell`` and are
        concatenated in shell order, which gives the same catalog as generating
        all shells at once.
        """
        for ibin in self.lensing.tqdm(ibins, desc="[merge_shell_catalogs]", leave=False):
            shells = [fits.getdata(self.path(ibin, ishell=ishell), 1) for ishell in range(self.nshells)]
            header = fits.Header()
            for k, v in self.header_dict.items():
                header[k] = v
            hdul = [fits.PrimaryHDU(), fits.BinTableHDU(data=np.concatenate(shells), header=header)]
            self.io.write_table(self.path(ibin), hdul)

    def read_catalogs(
            self,
            ibins: int | Iterable[int] | None = None,
            return_header: bool = False,
            subsample: float | None = None,
            seed: int | None = None,
            columns: list[str] | None = None,
            remove_mean: list[str] | None = None) -> np.ndarray | dict[int, np.ndarray] | tuple:
        """Read saved FITS catalog(s) for one or more tomographic bins.

        Parameters
        ----------
        ibins : int or array_like of int or None, optional
            Tomographic-bin indices to load. If None, loads all bins.
        return_header : bool, optional
            If True, also return the FITS header(s).
        subsample : float or None, optional
            If not None, randomly draw (without replacement, seeded by
            ``seed + ibin``) this fraction of galaxies from each loaded
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
            HDU if ``ibins`` is a single int, otherwise a dict keyed by bin index.
        table : numpy.ndarray or dict
            Structured array (native byte order) if ``ibins`` is a single int,
            otherwise a dict keyed by bin index.

        Notes
        -----
        Only the requested columns are read (with ``fitsio``), and with ``subsample`` only the
        selected rows are kept, read in large contiguous blocks (see :func:`~pollux.utils.fits.read_fits_rows`).
        """
        seed = self.lensing.shell_builder.seed if seed is None else seed

        # read the requested columns of each bin, optionally a random subsample of its rows
        headers = {}
        tables = {}
        for ibin in self.lensing.tqdm(ibins, desc="[read_catalogs]"):
            fname = self.path(ibin, ext='fits')
            headers[ibin] = fits.getheader(fname, ext=1)
            if subsample is None:
                table = fitsio.read(fname, ext=1, columns=columns)
            else:
                # random rows (galaxies are stored in shell order, so a stride would not be a fair sample)
                if not 0.0 < subsample < 1.0:
                    raise ValueError('subsample must satisfy 0 < subsample < 1')
                nrows = headers[ibin]['NAXIS2']
                rows = np.sort(np.random.default_rng(seed + ibin).choice(nrows, size=int(subsample * nrows), replace=False))
                table = read_fits_rows(fname, rows, columns)
            # native byte order (FITS is big-endian), for numpy operations
            table = table.astype(table.dtype.newbyteorder('='))
            # e.g. remove the mean shear of the bin
            for col in remove_mean or []:
                table[col] -= table[col].mean()
            tables[ibin] = table
            logger.debug("Found and loaded catalog at %s.", fname)

        # a single table for a single bin, else dicts keyed by bin
        if np.isscalar(ibins):
            header, table = headers[ibins], tables[ibins]
        else:
            header, table = headers, tables

        if return_header:
            return header, table
        else:
            return table

    # endregion


# ======================================================================================================================
# region Generators


class CatalogUniform(CatalogGenerator):
    """Generate source galaxies uniformly over the footprint.

    Redshifts are drawn from the tomographic tracer distributions. Angular
    positions are drawn uniformly over the footprint of the maps.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag of the catalogs: they go to ``catalogs/{nz}/{tag}/`` next to the maps (see :class:`pollux.io.files.PolluxIO`), ``default`` if None.
    nbar_input : float or array-like
        Galaxy surface density in gal/arcmin^2, per bin or for all (required, keyword only).
    interp_fields, interp_method, smooth_fields, smooth_Mpc
        Interpolation and smoothing of the fields, see :class:`CatalogGenerator`.
    augmentors : list, optional
        List of augmentors. Default is empty list.
    """

    def _generate_galaxies(self, catalog: dict[int, dict[str, np.ndarray]], ibins: int | Iterable[int] | None = None, ishells: list[int] | None = None) -> None:
        """Generate redshifts and positions for all selected tomographic bins.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place.
        ibins : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        ishells : None, optional
            Per-shell generation is not supported.
        """
        if ishells is not None:
            raise NotImplementedError(f"Per-shell generation is not supported by {self.__class__.__name__}.")

        # Generate ra, dec and redshift
        for ibin in self.lensing.tqdm(ibins, desc="[generate] generating ra, dec, z"):
            self._generate_z(catalog, ibin)
            self._generate_pos(catalog, ibin)

            for k in self.interp_fields:
                catalog[ibin][k] = np.zeros_like(catalog[ibin]['z'])

    def _generate_pos(self, catalog: dict[int, dict[str, np.ndarray]], ibin: int) -> None:
        """Draw uniform (RA, DEC) positions over the footprint.

        Parameters
        ----------
        catalog : dict
            Per-bin output container.
        ibin : int
            Tomographic-bin index.
        """
        rng = self._rng(self._STREAM_POS, ibin)

        # uniform over the footprint: equal-area pixels drawn at random, then a position within each
        ipix = self.lensing.shell_builder.footprint.ipix()
        ra, dec = generate_radec_in_pix(self.nside, rng.choice(ipix, size=self.Ngal[ibin]), rng)
        catalog[ibin]['ra'] = ra
        catalog[ibin]['dec'] = dec


class CatalogClustered(CatalogGenerator):
    """Base class for clustered source galaxy catalogs.

    This class provides common functionality for generating clustered catalogs
    where galaxy positions are biased by a density field.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag of the catalogs: they go to ``catalogs/{nz}/{tag}/`` next to the maps (see :class:`pollux.io.files.PolluxIO`), ``default`` if None.
    nbar_input : float or array-like
        Galaxy surface density in gal/arcmin^2, per bin or for all (required, keyword only).
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
        place galaxies, ``'positions'``), see :class:`CatalogGenerator`.
    augmentors : list, optional
        List of augmentors applied after map interpolation. Default is empty list.
    cls_b, cls_lmax_mask : optional
        Binning of the catalog angular power spectra, see :class:`CatalogGenerator`.
    """

    BIAS_MODELS = ('linear', 'power_law')

    def __init__(
        self,
        lensing: LensingBuilder,
        tag: str | None = None,
        *,
        nbar_input: float | np.ndarray,
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
        # bias of the galaxies, a scalar or one per bin, set before the generator's configuration is written
        if bias is None:
            raise ValueError(f'bias must be provided for {self.__class__.__name__}')
        if bias_model not in self.BIAS_MODELS:
            raise ValueError(f"bias_model must be one of {self.BIAS_MODELS}, not {bias_model!r}")
        self.bias_model = bias_model
        bias_arr = np.atleast_1d(bias).astype(float)
        if bias_arr.size == 1:
            bias_arr = np.full(lensing.nmaps, float(bias_arr[0]))
        elif bias_arr.size != lensing.nmaps:
            raise ValueError(f"bias must be a scalar or array of length {lensing.nmaps}, got {bias_arr.size}")
        self.bias = bias_arr

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
        self.header_dict['BIASMOD'] = bias_model

        # Store bias values in augmentor kwargs
        for ibin in self.lensing.iter():
            self.augmentor_kwargs[ibin]['bias'] = self.bias[ibin]

        # Store bias values in header dict
        for i, b in enumerate(self.bias):
            self.header_dict[f'BIAS_{i}'] = b

    def _density_weight(self, density: np.ndarray, ibin: int, ishell: int) -> np.ndarray:
        r"""Return the relative galaxy density per pixel of the bias model (``bias_model``).

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
        ibin : int
            Tomographic-bin index, setting the bias.
        ishell : int
            Shell index, for the warning.

        Returns
        -------
        ndarray
            Non-negative weights, same shape as ``density``.

        Notes
        -----
        The relative density of galaxies in a pixel of density contrast :math:`\delta` is

        .. math::

            w = \max(0, 1 + b \delta) \quad (\text{linear}), \qquad
            w = \frac{\max(0, 1 + \delta)^b}{\langle \max(0, 1 + \delta)^b \rangle} \quad (\text{power law}).
        """
        b = self.bias[ibin]
        if self.bias_model == 'power_law':
            weight = 1. + density.astype(float)
            fraction = np.mean(weight < 0.)
            if fraction > 0.:
                warnings.warn(f"1 + delta < 0 in {100. * fraction:.3g}% of the pixels (bin {ibin}, shell {ishell}): set to 0.")
            weight = np.clip(weight, 0., None)**b
            return weight / np.mean(weight)

        weight = 1. + b * density
        fraction = np.mean(weight < 0.)
        if fraction > 0.:
            warnings.warn(f"1 + b * delta < 0 in {100. * fraction:.3g}% of the pixels (bin {ibin}, shell {ishell}, b = {b:g}): clipped to 0.")
        return np.clip(weight, 0., None)


class CatalogClusteredMultinomial(CatalogClustered):
    """Generate clustered source galaxies using multinomial sampling.

    The total number of galaxies per tomographic bin is fixed, and positions are
    drawn per shell from a multinomial distribution over pixels, proportional to
    the relative density of the bias model (``1 + b * delta`` by default, where ``delta``
    is the density contrast field, see :class:`CatalogClustered`).

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag of the catalogs: they go to ``catalogs/{nz}/{tag}/`` next to the maps (see :class:`pollux.io.files.PolluxIO`), ``default`` if None.
    nbar_input : float or array-like
        Galaxy surface density in gal/arcmin^2, per bin or for all (required, keyword only).
    bias, bias_model : optional
        Galaxy bias per tomographic bin, and bias model, see :class:`CatalogClustered`.
    interp_fields, interp_method, smooth_fields, smooth_Mpc
        Interpolation and smoothing of the fields, see :class:`CatalogGenerator`.
    augmentors : list, optional
        List of augmentors. Default is empty list.
    """

    def _generate_galaxies(self, catalog: dict[int, dict[str, np.ndarray]], ibins: int | Iterable[int] | None = None, ishells: list[int] | None = None) -> None:
        """Generate redshifts first, then positions conditioned on density maps.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place.
        ibins : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        ishells : None, optional
            Per-shell generation is not supported (the number of galaxies per bin
            is fixed over all shells).
        """
        if ishells is not None:
            raise NotImplementedError(f"Per-shell generation is not supported by {self.__class__.__name__}.")

        # Generate redshifts first, needed to know what shells galaxies are in
        for ibin in self.lensing.tqdm(ibins, desc="[generate] generating z"):
            self._generate_z(catalog, ibin)
            for k in ['ra', 'dec', *self.interp_fields]:
                catalog[ibin][k] = np.zeros_like(catalog[ibin]['z'])

        # Then generate positions, looping over shells for speed
        for ishell in self.lensing.shell_builder.tqdm(None, desc="[generate] generating ra, dec from density"):
            self._generate_pos(catalog, ishell, ibins=ibins)

    def _generate_pos(self, catalog: dict[int, dict[str, np.ndarray]], ishell: int, ibins: int | Iterable[int] | None = None) -> None:
        """Draw positions in a given shell using multinomial pixel sampling.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place.
        ishell : int
            Shell index.
        ibins : int or array_like of int or None, optional
            Tomographic-bin indices to process.
        """
        # density contrast used to place galaxies ('positions' smoothing)
        footprint = self.lensing.shell_builder.footprint
        ipix_mask = footprint.ipix()
        density = self.lensing.shell_builder.field_map(ishell, self.smoothing['positions'][ishell])
        density_masked = density[footprint.ipix(nest=False)]  # in the order of ipix_mask: RING index of each NESTED pixel

        # Loop over tomographic bins
        for ibin in self.lensing.tqdm(ibins, leave=False):
            rng = self._rng(self._STREAM_COUNTS, ibin, ishell)

            # select galaxies in current shell
            w = (catalog[ibin]['idx_shell'] == ishell)

            if np.any(w):
                # Multinomial sampling of number of galaxies per pixels
                n = np.sum(w)
                p = self._density_weight(density_masked, ibin, ishell)
                p /= np.sum(p)

                ipix = rng.choice(ipix_mask, size=n, replace=True, p=p)

                # Sampling ra, dec within pixel
                ra, dec = generate_radec_in_pix(self.nside, ipix, rng)
                catalog[ibin]['ra'][w] = ra
                catalog[ibin]['dec'][w] = dec


class CatalogClusteredPoisson(CatalogClustered):
    r"""Generate clustered source galaxies using Poisson sampling.

    In each shell and tomographic bin, the number of galaxies per pixel is drawn
    as a Poisson random variable with mean ``<nbar>`` times the relative density of
    the bias model (``1 + b * delta`` by default, see :class:`CatalogClustered`).
    The total number of galaxies therefore fluctuates.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag of the catalogs: they go to ``catalogs/{nz}/{tag}/`` next to the maps (see :class:`pollux.io.files.PolluxIO`), ``default`` if None.
    nbar_input : float or array-like
        Galaxy surface density in gal/arcmin^2, per bin or for all (required, keyword only).
    bias, bias_model : optional
        Galaxy bias per tomographic bin, and bias model, see :class:`CatalogClustered`.
    interp_fields, interp_method, smooth_fields, smooth_Mpc
        Interpolation and smoothing of the fields, see :class:`CatalogGenerator`.
    augmentors : list, optional
        List of augmentors. Default is empty list.

    Notes
    -----
    In shell :math:`i` and bin :math:`j`, pixel :math:`p` gets

    .. math::

        N_p \sim \mathrm{Poisson}\left(\bar N_{ij}\, w(\delta_p)\right), \qquad
        \bar N_{ij} = \bar n_j\, \Omega_{\rm pix} \frac{\int_{z_i}^{z_{i+1}} n_j(z)\, dz}{\sum_k \int_{z_k}^{z_{k+1}} n_j(z)\, dz},

    with :math:`\bar n_j` the density of the bin (``nbar_input``), :math:`w` the relative density of
    :meth:`_density_weight`, and :math:`\delta_p` the density contrast smoothed for the positions. Galaxies are placed
    uniformly within their pixel, at redshifts drawn from :math:`n_j(z)` within the shell.
    """

    def _generate_galaxies(self, catalog: dict[int, dict[str, np.ndarray]], ibins: int | Iterable[int] | None = None, ishells: list[int] | None = None) -> None:
        """Generate a clustered catalog by shell-wise Poisson sampling.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place.
        ibins : int or array_like of int or None, optional
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
        for ibin, t in enumerate(tr.ccl(self.lensing.cosmo) for tr in self.lensing.tracers):
            nz_int = np.array([quad(lambda z: t.get_dndz(z), z_edges[ishell], z_edges[ishell + 1]) for ishell in range(self.snap_lensing.nmaps)])
            nbar_per_pix_per_bin.append(nz_int / np.sum(nz_int) * self.nbar_per_pix[ibin])
        self.nbar_per_pix_per_bin = np.array(nbar_per_pix_per_bin)

        # Initialize catalog entries only for the requested bins
        for ibin in self.lensing.iter(ibins):
            catalog[ibin]['ra'] = []
            catalog[ibin]['dec'] = []
            catalog[ibin]['z'] = []
            catalog[ibin]['idx_shell'] = []

        # Second generate positions, looping over shells
        ishells = list(range(self.nshells)) if ishells is None else ishells
        for ishell in self.lensing.shell_builder.tqdm(ishells, desc="[generate] generating ra, dec from density"):
            self._generate_pos(catalog, ishell, ibins=ibins)

        # Third sample redshifts within shells
        for ibin in self.lensing.tqdm(ibins, desc="[generate] generating z"):
            self._generate_z(catalog, ibin, ishells)

        for ibin in self.lensing.tqdm(ibins, disable=True):
            catalog[ibin]['ra'] = np.concatenate(catalog[ibin]['ra'])
            catalog[ibin]['dec'] = np.concatenate(catalog[ibin]['dec'])
            catalog[ibin]['z'] = np.concatenate(catalog[ibin]['z'])
            catalog[ibin]['idx_shell'] = np.concatenate(catalog[ibin]['idx_shell'])
            catalog[ibin]['chi'] = builder.cosmo.comoving_radial_distance(1. / (1. + catalog[ibin]['z']))
            for k in self.interp_fields:
                catalog[ibin][k] = np.zeros_like(catalog[ibin]['z'])

    def _generate_pos(self, catalog: dict[int, dict[str, np.ndarray]], ishell: int, ibins: int | Iterable[int] | None = None) -> None:
        """Sample galaxy positions in one shell using pixel-wise Poisson draws.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place.
        ishell : int
            Shell index.
        ibins : int or array_like of int or None, optional
            Tomographic-bin indices to process.
        """
        # density contrast used to place galaxies ('positions' smoothing)
        footprint = self.lensing.shell_builder.footprint
        ipix_mask = footprint.ipix()
        density = self.lensing.shell_builder.field_map(ishell, self.smoothing['positions'][ishell])
        density_masked = density[footprint.ipix(nest=False)]  # in the order of ipix_mask: RING index of each NESTED pixel

        # Loop over tomographic bins
        for ibin in self.lensing.tqdm(ibins, leave=False):
            rng = self._rng(self._STREAM_COUNTS, ibin, ishell)

            # no galaxy of this bin in this shell
            if self.nbar_per_pix_per_bin[ibin][ishell] == 0.:
                catalog[ibin]['ra'].append(np.zeros(0))
                catalog[ibin]['dec'].append(np.zeros(0))
                continue

            # Poisson sampling
            p = self._density_weight(density_masked, ibin, ishell)
            lam = p * self.nbar_per_pix_per_bin[ibin][ishell]
            n_per_pix = rng.poisson(lam=lam)
            ipix = np.repeat(ipix_mask, n_per_pix)

            # Sampling ra, dec within pixel
            ra, dec = generate_radec_in_pix(self.nside, ipix, rng)

            catalog[ibin]['ra'].append(ra)
            catalog[ibin]['dec'].append(dec)

    def _generate_z(self, catalog: dict[int, dict[str, np.ndarray]], ibin: int, ishells: list[int] | None) -> None:
        """Sample galaxy redshifts within each shell for one tomographic bin.

        Parameters
        ----------
        catalog : dict
            Per-bin dictionaries updated in-place, with one entry of
            ``catalog[ibin]['ra']`` per shell of ``ishells``.
        ibin : int
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

        t = self.lensing.tracers[ibin].ccl(self.lensing.cosmo)
        for i, ishell in enumerate(ishells):
            rng = self._rng(self._STREAM_Z, ibin, ishell)

            zz = np.linspace(z_edges[ishell], z_edges[ishell + 1], 100)
            nz = t.get_dndz(zz)
            ngal = len(catalog[ibin]['ra'][i])

            catalog[ibin]['z'].append(sample_pdf(zz, nz, ngal, rng))
            catalog[ibin]['idx_shell'].append(np.full(ngal, ishell))


# endregion

# ======================================================================================================================
# region Augmentors


class CatalogAugmentor(ABC):
    """Base class for catalog augmentors that add fields to existing catalogs.

    Subclasses must implement the ``augment_catalog`` method and set
    ``self.name``, ``self.required_interp_fields``, and ``self.new_columns``
    in their ``__init__``.
    """

    # what the augmentor models, for the statistics of the catalogs: 'shear' (shapes), 'ia', or None
    kind: str | None = None

    def config(self) -> dict:
        """Return the type and parameters of the augmentor, for the ``config.yaml`` of the catalogs.

        Returns
        -------
        dict
            ``type`` (the class of the augmentor), extended by subclasses with their parameters.
        """
        return {'type': type(self).__name__}

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
            Per-bin options (``augmentor_kwargs`` of the catalog generator, e.g. ``ibin``).
        """


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
    seed : int or None, optional
        Extra seed of the intrinsic shapes, if not None. The shapes of the galaxies of
        tomographic bin ``ibin`` in shell ``ishell`` are drawn from a stream of their own,
        seeded by the seed of the realization (given by the catalog generator), ``ibin`` and
        ``ishell``: they differ between realizations, and do not depend on how the catalog is
        split into shells. Default is None.
    """

    kind = 'shear'

    # stream of the random numbers of the intrinsic shapes, distinct from those of the catalog generators
    _SHAPE_STREAM = 10

    def __init__(
        self,
        compute_reduced_shear: bool = True,
        compute_magnification: bool = False,
        generate_intrinsic: bool = False,
        compute_observed: bool = False,
        include_ia: bool = True,
        sigma_e: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if compute_observed and not generate_intrinsic:
            raise ValueError("Intrinsic shapes must be generated to compute observed shapes.")

        self.compute_reduced_shear = compute_reduced_shear
        self.compute_magnification = compute_magnification
        self.generate_intrinsic = generate_intrinsic
        self.compute_observed = compute_observed
        self.include_ia = include_ia
        self.sigma_e = sigma_e
        self.seed = seed
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

    def config(self) -> dict:
        """Return the type and parameters of the augmentor, for the ``config.yaml`` of the catalogs.

        The parameters setting the values of the shapes (``sigma_e``, ``include_ia``, ``seed``) are
        recorded when they differ from their defaults, so that the configurations of catalogs made
        before they were recorded stay valid.

        Returns
        -------
        dict
            ``type`` and the parameters that differ from their defaults.
        """
        defaults = inspect.signature(type(self)).parameters
        return {'type': type(self).__name__, **{k: getattr(self, k) for k in ('sigma_e', 'include_ia', 'seed') if getattr(self, k) != defaults[k].default}}

    def augment_catalog(self, cosmo: pyccl.Cosmology, catalog: dict[str, np.ndarray], ibin: int = 0, realization_seed: int = 0, **kwargs: Any) -> None:
        """Compute derived shear quantities and add them to the catalog.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology (unused).
        catalog : dict[str, np.ndarray]
            Catalog of one tomographic bin, with ``'kappa'``, ``'gamma1'``,
            ``'gamma2'`` and ``'idx_shell'``; updated in place.
        ibin : int, optional
            Tomographic-bin index, used to seed the intrinsic shapes.
        realization_seed : int, optional
            Seed of the realization (that of the shell builder), used to seed the intrinsic shapes.
        **kwargs
            Unused (other augmentor options).
        """
        kappa = catalog['kappa']
        gamma1, gamma2 = catalog['gamma1'], catalog['gamma2']

        if self.compute_magnification:
            # magnification 1 / ((1 - kappa)^2 - |gamma|^2)
            catalog['mu'] = 1. / ((1. - kappa)**2 - gamma1**2 - gamma2**2)

        if self.compute_reduced_shear:
            # reduced shear g = gamma / (1 - kappa)
            denom = 1. - kappa
            g1 = gamma1 / denom
            g2 = gamma2 / denom
            catalog['g1'] = g1
            catalog['g2'] = g2
        else:
            g1 = gamma1
            g2 = gamma2

        if self.generate_intrinsic:
            # Gaussian intrinsic shapes, drawn per shell so that they do not depend on how the generation is split
            idx_shell = catalog['idx_shell']
            catalog['e1_int'] = np.empty(len(idx_shell))
            catalog['e2_int'] = np.empty(len(idx_shell))
            for ishell in np.unique(idx_shell):
                w = np.flatnonzero(idx_shell == ishell)
                rng = np.random.default_rng([realization_seed, self._SHAPE_STREAM, ibin, ishell] + ([] if self.seed is None else [self.seed]))
                catalog['e1_int'][w] = rng.normal(0., self.sigma_e, size=len(w))
                catalog['e2_int'][w] = rng.normal(0., self.sigma_e, size=len(w))

        if self.compute_observed:
            # source ellipticity, plus the intrinsic alignments if computed
            e_int = catalog['e1_int'] + 1j * catalog['e2_int']
            if self.include_ia:
                for model in ('ta', 'tt'):
                    if f'e1_ia_{model}' in catalog:
                        e_int += catalog[f'e1_ia_{model}'] + 1j * catalog[f'e2_ia_{model}']
            # observed ellipticity e = (g + e_s) / (1 + g* e_s)
            g = g1 + 1j * g2
            e_obs = (g + e_int) / (1. + np.conj(g) * e_int)
            catalog['e1_obs'] = e_obs.real
            catalog['e2_obs'] = e_obs.imag


# endregion
