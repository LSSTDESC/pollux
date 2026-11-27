"""Base class of the map builders (:class:`MapBuilder`).

Handles file names and I/O of the maps (HealSparse or dense HEALPix files, :attr:`MapBuilder.map_format`), spherical
harmonic transforms and smoothing on their footprint (:class:`~pollux.footprint.Footprint`), and plots of the maps.
Their statistics (tracers and predictions, power spectra, moments) are methods inherited from
:class:`~pollux.statistics.maps.MapStatistics`.
"""

from __future__ import annotations

import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator

import healpy as hp
import healsparse as hsp
import skyproj
import matplotlib.pyplot as plt
import numba
import numpy as np
import pymaster as nmt
from tqdm.auto import tqdm, trange

from .footprint import Footprint
from .io import PolluxIO
from .cosmo import Tracer
from .statistics.maps import MapStatistics
from .utils.healpix import dense_to_hsp, hsp_to_dense, is_dense

logger = logging.getLogger(__name__)


class MapBuilder(MapStatistics, ABC):
    """Base class to build, store, and analyze Healpix/HealSparse maps.

    This class centralizes:
    - file naming and I/O for HealSparse maps,
    - alm computation (Healpy), on its footprint (:class:`~pollux.footprint.Footprint`),
    - plots of the maps,

    and inherits their statistics from :class:`~pollux.statistics.maps.MapStatistics` (pseudo-C_ell and
    decoupled bandpowers with NaMaster, SACC export, predictions and plots).

    Subclasses must implement :meth:`_create_tracers` and
    :meth:`construct_maps` (abstract).
    :meth:`_noise_cls` is optional, needed only for noise-aware predictions.

    Parameters
    ----------
    nmaps : int
        Number of maps handled/produced by this builder.
    dtype : data-type
        Data type used to store map values (dtype for arrays written to
        HealSparse/Healpix maps and intermediate map buffers).
    footprint : pollux.footprint.Footprint
        Footprint of the maps (resolution, mask, padding, weights of the harmonic transforms), shared with
        the builders made from this one.
    io : pollux.io.files.PolluxIO
        Files of the products, with the options of the builder (e.g. its seed, matter model or n(z)).
    product : str
        Product of the builder (e.g. ``'shells'``, ``'kappa'``): its files are ``io.path(product, ...)``.
    nside_c : int, optional
        HealSparse coverage nside. Default is 32.
    map_format : {'healsparse', 'healpix'}, optional
        Format of the map files: HealSparse maps (NESTED, ``.fits``, the default), or dense full-sky HEALPix
        maps in RING ordering (healpy's, ``.npy``), with ``hp.UNSEEN`` outside the padded footprint. The maps are
        computed as dense RING maps (as the harmonic transforms need them) and converted only when written or read in
        the other format (:mod:`pollux.utils.healpix`). Files of either format are read (:meth:`iter_maps`).
    nocompress : bool, optional
        If True, disable HealSparse compression. Default is True.
    sacc_type : str or None, optional
        SACC data type name used when saving spectra.
    remove_monopole : bool, optional
        If True (default), subtract the (weighted) mean of maps over the alm
        weights before harmonic transforms (:meth:`_alm_field`). Other uses of
        maps (:meth:`field_map`) keep their mean.
    b : pymaster.NmtBin or None, optional
        NaMaster binning of the power spectra. Default is None: 20 logarithmic bands from ell = 8 to
        ``2 * nside``, dividing out the pixel window with ``pixel_window`` (:attr:`b`). The names of the
        workspaces identify a binning by its bands and ``pixel_window``, not by a window of its own (not
        readable from a ``NmtBin``): give distinct binnings distinct bands.
    pixel_window : bool, optional
        Whether the maps carry the Healpix pixel window (True by default: maps of pixel averages). The
        default binning divides it out of the bandpowers, and the predictions apply it to the theory before
        binning, so that they match the measured bandpowers with any binning.

    Notes
    -----
    Healpix ordering: HealSparse maps and the footprint's pixels are NESTED, dense maps (and the harmonic transforms)
    RING.
    """

    # ``what`` of the smoothed field maps (e.g. 'density' for shells; None for convergence maps)
    smoothed_field_what: str | None = None

    # map indices in file names, and the files needed only to make the products (see :mod:`pollux.io.files`)
    _index_format = PolluxIO.SHELL_INDEX
    _INTERMEDIATE = ('alms', 'catalog_props')

    # formats of the map files, and the ``what`` of the map files (the other files are FITS files)
    MAP_FORMATS = ('healsparse', 'healpix')
    _MAP_WHATS = (None, 'density', 'kappa', 't11', 't22', 't12', 'gamma1', 'gamma2')

    def __init__(
        self,
        nmaps: int,
        dtype: np.dtype,
        footprint: Footprint,
        io: PolluxIO,
        product: str,
        nside_c: int = 32,
        map_format: str = 'healsparse',
        nocompress: bool = True,
        sacc_type: str | None = None,
        remove_monopole: bool = True,
        b: 'nmt.NmtBin' | None = None,
        pixel_window: bool = True,
    ) -> None:
        self.nmaps = nmaps
        self.dtype = dtype
        self.footprint = footprint
        self.nside = footprint.nside
        self.npix = footprint.npix
        self.io = io
        self.product = product
        self.nside_c = nside_c
        if map_format not in self.MAP_FORMATS:
            raise ValueError(f"map_format must be one of {self.MAP_FORMATS}, not {map_format!r}")
        self.map_format = map_format
        self.nocompress = nocompress
        self.sacc_type = sacc_type
        self.remove_monopole = remove_monopole
        if b is not None:
            self.b = b  # otherwise built lazily
        self.pixel_window = pixel_window

        # Make sure folder exists
        os.makedirs(self.save_dir, exist_ok=True)

        logger.info("*** Map builder (%s) ***", self.map_name)
        logger.info("  - Builder class = %s.", self.__class__.__name__)
        logger.info("  - Number of maps = %d.", self.nmaps)
        logger.info("  - Maps saved to %s.", self.save_dir)

    @property
    def save_dir(self) -> str:
        """Directory of the products of the builder (``io.dir(product)``)."""
        return self.io.dir(self.product)

    @property
    def map_name(self) -> str:
        """Name of the maps in their files (e.g. ``'shell'``, ``'kappa'``, :attr:`pollux.io.files.PolluxIO.MAP_NAMES`)."""
        return self.io.MAP_NAMES[self.product]

    @property
    def map_dtype(self) -> np.dtype:
        """dtype of the maps of fields: the builder's ``dtype``, or float32 for an integer one (particle counts)."""
        return np.dtype(np.float32) if np.issubdtype(self.dtype, np.integer) else np.dtype(self.dtype)

    @property
    def _map_ext(self) -> str:
        """Extension of the map files: ``'fits'`` for HealSparse maps, ``'npy'`` for dense maps."""
        return 'fits' if self.map_format == 'healsparse' else 'npy'

    @property
    def _sub_ext(self) -> str:
        """Extension of the files of the sub-maps (e.g. sub-volumes of shells), that of the maps by default."""
        return self._map_ext

    def path(
        self,
        imap: int | None = None,
        sub: int | None = None,
        what: str | None = None,
        smooth_Mpc: float | None = None,
        extra: str | None = None,
        ext: str | None = None,
    ) -> str:
        """Return the path of a file of the products (or of their intermediate files).

        Parameters
        ----------
        imap : int or None, optional
            Map index (e.g. shell, or tomographic bin).
        sub : int or str or None, optional
            Sub-volume of the map (intermediate file).
        what : str or None, optional
            Field or product (e.g. ``'gamma1'``, ``'density'``, ``'cls'``, ``'alms'``), replacing
            ``map_name`` in the name. The smoothing scale goes in ``smooth_Mpc``.
        smooth_Mpc : float or None, optional
            Comoving Gaussian smoothing scale in Mpc of the map.
        extra : str or None, optional
            Additional string at the end of the name.
        ext : str or None, optional
            File extension (without leading dot). Default is that of the format of the maps (:attr:`map_format`,
            or that of the sub-volume maps, :attr:`_sub_ext`) for map files, ``'fits'`` for the others.

        Returns
        -------
        str
            ``{what or map_name}[_{index}][_sub{sub}][_{extra}].{ext}`` (:meth:`pollux.io.files.PolluxIO.path`),
            with the index formatted as ``_index_format`` (e.g. ``012`` or ``bin3``): in the directory of the
            maps smoothed at ``smooth_Mpc``, in the intermediate files (``subs/`` for sub-volumes, and a_lm),
            or in the product's directory.
        """
        index = None if imap is None else self._index_format.format(imap)
        name = (what or self.map_name, index, None if sub is None else f'sub{sub}', extra)
        if ext is None:
            if what not in self._MAP_WHATS:
                ext = 'fits'
            else:
                ext = self._map_ext if sub is None else self._sub_ext
        if smooth_Mpc is not None:
            return self.io.path(self.product, *name, ext=ext, smooth_Mpc=smooth_Mpc)
        intermediate = sub is not None or what in self._INTERMEDIATE
        return self.io.path(self.product, *name, ext=ext, intermediate=intermediate, folder='subs' if sub is not None else None)

    def iter(self, it: Iterable[int] | int | None = None) -> range | Iterable[int] | list[int]:
        """Return an iterator over the requested indices.

        Parameters
        ----------
        it : Iterable[int] or int or None, optional
            - If None: iterate over ``range(self.nmaps)``.
            - If Iterable[int]: return the iterable as is.
            - If int: return a singleton list ``[it]``.

        Returns
        -------
        range or Iterable[int] or list
            Iterable over the requested map indices.

        Raises
        ------
        ValueError
            If ``it`` is neither None, an int, nor an iterable of ints, or if
            indices are not in ``[0, self.nmaps)``.
        """
        if it is None:
            return range(self.nmaps)
        return [it] if np.isscalar(it) else it

    def tqdm(self, it: Iterable[int] | int | None = None, **tqdm_kwargs: Any) -> trange | tqdm | list[int]:
        """Wrap an iterable of ints (or default range) with tqdm progress bars.

        Parameters
        ----------
        it : Iterable[int] or int or None, optional
            - If None: iterate over ``range(self.nmaps)`` using :func:`trange`.
            - If Iterable[int]: wrap with :func:`tqdm`.
            - If int: return a singleton list ``[it]``.
        **tqdm_kwargs
            Options of the progress bar (e.g. ``desc``).

        Returns
        -------
        trange or tqdm or list
            Iterable with tqdm progress bar behavior.

        Raises
        ------
        ValueError
            If ``it`` is neither None, an int, nor an iterable of ints, or if
            indices are not in ``[0, self.nmaps)``.
        """
        if it is None:
            return trange(self.nmaps, **tqdm_kwargs)
        return [it] if np.isscalar(it) else tqdm(it, **tqdm_kwargs)

    # ==================================================================================================================
    # region Maps

    @abstractmethod
    def construct_maps(self, imaps: int | Iterable[int] | None = None) -> np.ndarray | hsp.HealSparseMap | dict[int, np.ndarray | hsp.HealSparseMap]:
        """Construct maps, write them and return them.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to construct. If None, constructs all maps.

        Returns
        -------
        ndarray or hsp.HealSparseMap, or dict
            The maps as written (:attr:`~pollux.map_builder.MapBuilder.map_format`): a single map if ``imaps`` is an
            int, else a dict keyed by map index.
        """

    def _to_dense(self, m: np.ndarray | hsp.HealSparseMap | tuple[np.ndarray, np.ndarray], dtype: Any = None) -> np.ndarray:
        """Return a dense RING map of the builder, from a dense RING map, a HealSparse map or a RING pixel-value pair.

        Parameters
        ----------
        m : ndarray or hsp.HealSparseMap or tuple
            Full-sky dense RING map (e.g. from a harmonic transform), cast to ``dtype`` (in place if already of
            ``dtype``) with no value outside the padded footprint; a HealSparse map, converted as it is
            (:func:`pollux.utils.healpix.hsp_to_dense`); or a RING pixel-value pair ``(ipix, values)``: values on
            RING pixels (e.g. the non-empty pixels of a sub-volume map), zero on the other pixels of the padded
            footprint.
        dtype : data-type or None, optional
            dtype of a map from a dense map. Default is :attr:`map_dtype` (``dtype`` for a pixel-value pair).

        Returns
        -------
        ndarray
            The map, with the sentinel of its dtype (``hp.UNSEEN`` for floats) where it has no value.
        """
        if isinstance(m, tuple):
            dtype = dtype or self.dtype
            out = self._to_dense(np.zeros(self.npix, dtype=dtype), dtype=dtype)
            out[m[0]] = m[1]
            return out
        if not is_dense(m):
            return hsp_to_dense(m)
        dtype = np.dtype(dtype or self.map_dtype)
        out = np.asarray(m, dtype=dtype)
        if not self.footprint.fullsky:
            out[~self.footprint.mask(nest=False, padded=True)] = hsp.utils.check_sentinel(dtype.type, None)
        return out

    def _to_hsp(self, m: np.ndarray | hsp.HealSparseMap | tuple[np.ndarray, np.ndarray]) -> hsp.HealSparseMap:
        """Return the HealSparse map of a dense RING map or a RING pixel-value pair (see :meth:`_to_dense`).

        Parameters
        ----------
        m : ndarray or hsp.HealSparseMap or tuple
            Full-sky dense RING map (valid where it has a value), RING pixel-value pair ``(ipix, values)`` (valid on
            ``ipix``), or HealSparse map (returned as it is).

        Returns
        -------
        hsp.HealSparseMap
        """
        if isinstance(m, tuple):
            valid = np.zeros(self.npix, dtype=bool)
            valid[m[0]] = True
            return dense_to_hsp(self._to_dense(m), self.nside_c, valid=valid)
        return dense_to_hsp(m, self.nside_c) if is_dense(m) else m

    def _write_map(self, fname: str, m: np.ndarray | hsp.HealSparseMap) -> np.ndarray | hsp.HealSparseMap:
        """Write a map in the format of the builder (:attr:`map_format`), converting it if needed.

        Parameters
        ----------
        fname : str
            Output path (typically from :meth:`path`).
        m : ndarray or hsp.HealSparseMap
            Dense RING map (as :meth:`_to_dense`) or HealSparse map.

        Returns
        -------
        ndarray or hsp.HealSparseMap
            The map as written: what the methods writing maps return, in the format of the builder.
        """
        if self.map_format == 'healpix':
            m = m if is_dense(m) else self._to_dense(m)
            self.io.write_npy(fname, m)
        else:
            m = self._to_hsp(m)
            self.io.write_map(fname, m, nocompress=self.nocompress)
        return m

    def iter_maps(
        self,
        imaps: int | Iterable[int] | None = None,
        isubs: Iterable[int | str] | None = None,
        allow_missing: bool = False,
        desc: str = "[iter_maps]",
        dense: bool | None = False,
        **path_kwargs: Any,
    ) -> Iterator[tuple[int | str, np.ndarray | hsp.HealSparseMap]]:
        """Read maps from disk one at a time.

        Unlike :meth:`read_maps`, only one map is held in memory at a time, which
        allows looping over many large maps (e.g. to sum them).

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to read. If None, reads all maps. Must be a single int
            if ``isubs`` is given.
        isubs : iterable of int or str, or None, optional
            If provided, read the sub-maps ``isubs`` of map ``imaps`` (e.g.
            sub-volume indices, or ``'orphan'``) instead of full maps.
        allow_missing : bool, optional
            If True, skip missing files (logged) instead of raising
            ``FileNotFoundError``.
        desc : str, optional
            Progress bar description.
        dense : bool or None, optional
            As dense RING maps (True), HealSparse maps (False, the default), or as stored (None): files of any
            format are read, by their extension: ``.npy`` for dense RING maps, ``.npz`` for RING pixel-value pairs
            ``(ipix, values)`` of sub-volumes (see :meth:`_to_dense`), HealSparse otherwise.
        **path_kwargs
            Extra keywords forwarded to :meth:`path` (e.g. ``what='kappa'``).

        Yields
        ------
        index : int or str
            Map index, or sub index if ``isubs`` is given.
        map : ndarray or hsp.HealSparseMap or tuple
            The map read from disk.
        """
        if isubs is None:
            items = [(imap, self.path(imap, **path_kwargs)) for imap in self.iter(imaps)]
        else:
            if not np.isscalar(imaps):
                raise ValueError('imaps must be a single map index when isubs is given.')
            items = [(isub, self.path(imaps, sub=isub, **path_kwargs)) for isub in isubs]

        for index, fname in tqdm(items, desc=desc, disable=len(items) == 1):
            if not os.path.exists(fname):
                if not allow_missing:
                    raise FileNotFoundError(f"Map not found at {fname}.")
                logger.info("Skipping missing map at %s.", fname)
                continue
            if fname.endswith('.npz'):
                with np.load(fname) as f:
                    m = (f['ipix'], f['counts'])
            elif fname.endswith('.npy'):
                m = self.io.read_npy(fname)
            else:
                m = self.io.read_map(fname, nside_coverage=self.nside_c)
            logger.debug("Found and loaded map at %s.", fname)
            if dense is not None:
                m = self._to_dense(m) if dense else self._to_hsp(m)
            yield index, m

    def read_maps(self,
                  imaps: int | Iterable[int] | None = None,
                  dense: bool | None = False,
                  **path_kwargs: Any) -> np.ndarray | hsp.HealSparseMap | dict[int, np.ndarray | hsp.HealSparseMap]:
        """Read one or multiple maps from disk.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to load. If None, loads all maps.
        dense : bool or None, optional
            As dense RING maps (True), HealSparse maps (False, the default), or as stored (None), see :meth:`iter_maps`.
        **path_kwargs
            Extra keywords forwarded to :meth:`path` (e.g. ``what='kappa'``).

        Returns
        -------
        ndarray or hsp.HealSparseMap, or dict
            - If ``imaps`` is a single int: returns a single map.
            - If ``imaps`` is ``None`` or an iterable: returns a dict keyed by
              map index with maps as values.
        """
        maps = dict(self.iter_maps(imaps, desc="[read_maps]", dense=dense, **path_kwargs))
        return maps[imaps] if np.isscalar(imaps) else maps

    def field_map(self, imap: int, smooth_Mpc: float | None = None) -> np.ndarray:
        """Return the field of one map as a dense RING map.

        Parameters
        ----------
        imap : int
            Map index.
        smooth_Mpc : float or None, optional
            Smoothing scale in Mpc: the smoothed field saved by :meth:`compute_maps_from_alms`
            (``what=smoothed_field_what``). Default is the unsmoothed field.

        Returns
        -------
        ndarray
            The map on disk (subclasses may convert it, e.g. counts to density contrast), with the sentinel of its
            dtype (``hp.UNSEEN``) outside the padded footprint.
        """
        if smooth_Mpc is not None:
            return self.read_maps(imap, what=self.smoothed_field_what, smooth_Mpc=smooth_Mpc, dense=True)
        return self.read_maps(imap, dense=True)

    # endregion

    # ==================================================================================================================
    # region Transforms

    def compute_alms(self, imaps: int | Iterable[int] | None = None, **map2alm_kwargs: Any) -> np.ndarray | dict[int, np.ndarray]:
        """Compute the spherical harmonic coefficients a_lm of maps, and save them (intermediate files).

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, uses all maps.
        **map2alm_kwargs
            Forwarded to :func:`healpy.map2alm`.

        Returns
        -------
        ndarray or dict
            The a_lm of ``imaps`` if an int, else ``{imap: alm}``.

        Notes
        -----
        The a_lm are those of :meth:`_alm_field` times the alm weights. They are read back with
        :meth:`read_alms`.
        """
        alms = {}
        for imap in self.tqdm(imaps, desc="[compute_alms]"):
            weights, _map, mean = self._alm_field(imap, return_mean=True)
            alms[imap] = hp.map2alm(_map * weights, **map2alm_kwargs)
            self.io.write_alm(self._path_alms(imap, legacy=False), alms[imap], mean=mean)
        return alms[imaps] if np.isscalar(imaps) else alms

    def _alm_field(self, imap: int, return_mean: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, float]:
        """Return the alm weights and the field prepared for its a_lm (RING ordering): ``map2alm(field * weights)``.

        Parameters
        ----------
        imap : int
            Map index.
        return_mean : bool, optional
            Also return the mean removed from the field.

        Returns
        -------
        weights : ndarray
            Alm weights (:attr:`pollux.footprint.Footprint.alm_weights`).
        _map : ndarray
            Field map, with its weighted mean over the weights removed if ``self.remove_monopole``, and
            zero where the weights vanish (not multiplied by the weights).
        mean : float
            With ``return_mean``: the mean removed, 0 without ``remove_monopole``.
        """
        weights = self.footprint.alm_weights
        good = weights > 0.
        _map = np.where(good, self.field_map(imap), 0.)
        mean = 0.
        if self.remove_monopole:
            mean = float(np.average(_map[good], weights=weights[good]))
            _map[good] -= mean
        return (weights, _map, mean) if return_mean else (weights, _map)

    def _path_alms(self, imap: int, legacy: bool = True) -> str:
        """Return the file of the a_lm of one map (intermediate, see :meth:`compute_alms`).

        Parameters
        ----------
        imap : int
            Map index.
        legacy : bool, optional
            Return the FITS file of the a_lm of earlier runs (without the mean) if only that one exists (default).

        Returns
        -------
        str
            ``.npz`` file of the a_lm and the mean removed from the field (:meth:`pollux.io.files.PolluxIO.write_alm`),
            or the legacy file.
        """
        fname = self.path(imap, what='alms', ext='npz')
        if legacy and not os.path.exists(fname):
            legacy_fname = self.path(imap, what='alms', ext='fits')
            if os.path.exists(legacy_fname):
                return legacy_fname
        return fname

    def read_alms(self, imaps: int | Iterable[int] | None = None) -> np.ndarray | dict[int, np.ndarray]:
        """Read the a_lm saved by :meth:`compute_alms`.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, uses all maps.

        Returns
        -------
        ndarray or dict
            The a_lm of ``imaps`` if an int, else ``{imap: alm}``.

        Raises
        ------
        FileNotFoundError
            If the a_lm of a map are not saved.
        """
        alms = {}
        for imap in self.iter(imaps):
            fname = self._path_alms(imap)
            if not os.path.exists(fname):
                raise FileNotFoundError(f"No a_lm of map {imap} at {fname}: compute them first (compute_alms).")
            alms[imap] = self.io.read_alm(fname)
            logger.debug("Found and loaded alms at %s.", fname)
        return alms[imaps] if np.isscalar(imaps) else alms

    # endregion

    # ==================================================================================================================
    # region Smoothed maps

    def compute_maps_from_alms(self,
                               imaps: int | Iterable[int] | None = None,
                               smooth_Mpc: float | None = None) -> np.ndarray | hsp.HealSparseMap | dict[int, np.ndarray | hsp.HealSparseMap]:
        """Compute field maps from their a_lm, optionally smoothed.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, uses all maps.
        smooth_Mpc : float or None, optional
            Comoving Gaussian smoothing scale in Mpc (converted to an angle per map,
            see :meth:`_smooth_sigma`). If given, the smoothed maps (pixel window
            deconvolved) are saved (``what=smoothed_field_what``); if None, the maps
            are only reconstructed in memory.

        Returns
        -------
        ndarray or hsp.HealSparseMap, or dict
            Map(s) on the padded footprint: as written with ``smooth_Mpc`` (:attr:`map_format`), dense RING maps
            otherwise (:meth:`_to_dense`).

        Notes
        -----
        The a_lm are those saved by :meth:`compute_alms` (of the weighted map, up to their
        ``lmax``), so the maps are the field (with its mean added back, see
        :meth:`_alm_field_mean`) on the footprint, but tapered in the padding.
        """
        maps = {}
        for imap in self.tqdm(imaps, desc="[compute_maps_from_alms]"):
            alm = self.read_alms(imap)
            if smooth_Mpc is not None:
                self._smooth_alms(imap, alm, smooth_Mpc, hp.Alm.getlmax(len(alm)))
            maps[imap] = self._field_from_alms(imap, alm)
            if smooth_Mpc is not None:
                maps[imap] = self._write_map(self.path(imap, what=self.smoothed_field_what, smooth_Mpc=smooth_Mpc), maps[imap])

        return maps[imaps] if np.isscalar(imaps) else maps

    def _smooth_alms(self, imap: int, alm: np.ndarray, smooth_Mpc: float, lmax: int) -> None:
        """Smooth the a_lm of one map in place, with a Gaussian of comoving scale ``smooth_Mpc``.

        The pixel window function is deconvolved, so that the Gaussian is the only
        smoothing of maps made from the smoothed a_lm (:meth:`_field_from_alms`). The width
        of the Gaussian is at least the pixel size (:meth:`_smooth_sigma`): below it, the
        deconvolution amplifies multipoles up to ``lmax`` and the maps ring (e.g. density
        contrasts below -1).

        Parameters
        ----------
        imap : int
            Map index, setting the angular scale of the smoothing (:meth:`_smooth_sigma`).
        alm : ndarray
            Spherical harmonic coefficients, modified in place.
        smooth_Mpc : float
            Comoving Gaussian smoothing scale in Mpc.
        lmax : int
            Maximum multipole of ``alm``.
        """
        ell = np.arange(lmax + 1)
        sigma = self._smooth_sigma(imap, smooth_Mpc)
        fl = np.exp(-0.5 * (ell * (ell + 1)) * sigma**2) / hp.pixwin(self.nside, pol=False, lmax=lmax)
        hp.almxfl(alm, fl, inplace=True)

    def _smooth_sigma(self, imap: int, smooth_Mpc: float) -> float:
        """Return the angular width (radians) of a comoving smoothing scale for one map, at least the pixel size.

        The comoving scale is :meth:`pollux.cosmo.Tracer.effective_smooth_Mpc` at the
        map's distance (:meth:`_smooth_distance`), with a warning when it is the pixel
        size rather than ``smooth_Mpc``.

        Parameters
        ----------
        imap : int
            Map index.
        smooth_Mpc : float
            Requested comoving Gaussian smoothing scale in Mpc.

        Returns
        -------
        float
            Angular width of the Gaussian in radians.
        """
        chi = self._smooth_distance(imap)
        smooth_eff = float(Tracer.effective_smooth_Mpc(smooth_Mpc, chi, self.nside))
        if smooth_eff > smooth_Mpc:
            warnings.warn(
                f"Smoothing of {smooth_Mpc} Mpc is below the pixel size for map {imap} at nside {self.nside}: "
                f"smoothing by the pixel size, {smooth_eff:.3g} Mpc, instead.")
        return smooth_eff / chi

    def _smooth_distance(self, imap: int) -> float:
        """Return the comoving distance (Mpc) converting a comoving smoothing scale to an angle for one map.

        Parameters
        ----------
        imap : int
            Map index.

        Returns
        -------
        float
            Implemented by builders whose maps have a distance (shells, per-shell convergence).
        """
        raise NotImplementedError(f"{type(self).__name__} maps have no distance to convert a smoothing scale in Mpc to an angle.")

    def _field_from_alms(self, imap: int, alm: np.ndarray) -> np.ndarray:
        """Return the field of one map from its a_lm (e.g. smoothed), on the padded footprint.

        Parameters
        ----------
        imap : int
            Map index.
        alm : ndarray
            Spherical harmonic coefficients of the map (:meth:`read_alms`), possibly smoothed (:meth:`_smooth_alms`).

        Returns
        -------
        ndarray
            Dense RING map of the field (:meth:`_to_dense`), with the mean removed before its transform added back
            (:meth:`_alm_field_mean`).
        """
        values = hp.alm2map(alm, self.nside, lmax=hp.Alm.getlmax(len(alm)), pol=False)
        values += self._alm_field_mean(imap)
        return self._to_dense(values)

    def _alm_field_mean(self, imap: int) -> float:
        """Return the weighted mean removed from the field by :meth:`_alm_field` (to add back after smoothing).

        Parameters
        ----------
        imap : int
            Map index.

        Returns
        -------
        float
            Mean of the field map weighted by the alm weights, or 0 without
            ``remove_monopole``. Adding it back to a smoothed map (a Gaussian
            smoothing keeps constants) restores the original mean. Saved with the a_lm (:meth:`compute_alms`), or
            computed from the field for the a_lm of earlier runs.
        """
        if not self.remove_monopole:
            return 0.
        fname = self._path_alms(imap)
        if fname.endswith('.npz'):
            return self.io.read_alm_mean(fname)
        weights = self.footprint.alm_weights
        good = weights > 0.
        return float(np.average(self.field_map(imap)[good], weights=weights[good]))

    # endregion

    # ==================================================================================================================
    # region Plots

    def plot_maps(
            self,
            maps_or_imaps: int | Iterable[int] | Iterable | None = None,
            plot_dir: str | None = None,
            nx: int | None = None,
            log: bool = False,
            title_prefix: str | None = None,
            cbar: bool = True,
            save: bool = True,
            backend: str = 'skyproj',
            nside_plot: int | None = 1024,
            **kwargs: Any) -> None:
        """Plot one or multiple maps.

        Parameters
        ----------
        maps_or_imaps : int or iterable or None, optional
            - If None: loads and plots all maps.
            - If int: loads and plots that map index.
            - If iterable of int: loads and plots those indices.
            - If iterable of maps: plots provided maps directly (HealSparse maps, dense RING maps or RING pixel-value
              pairs, see :meth:`_to_dense`).
        plot_dir : str or None, optional
            Output directory, with ``save``. Default is ``{save_dir}/plots``.
        nx : int or None, optional
            Number of columns in the subplot grid. Default is a near-square grid.
        log : bool, optional
            If True, plot the map on a logarithmic color scale. With
            ``backend='healpy'``, plots ``log10(map)`` when the map has
            positive values. With ``backend='skyproj'``, uses ``norm='log'``
            in :meth:`skyproj.Skyproj.draw_hspmap` if ``log(values)`` are all
            finite over the valid pixels, otherwise falls back to
            ``norm='symlog'``.
        title_prefix : str or None, optional
            Prefix used for subplot titles.
        cbar : bool, optional
            If True, display a colorbar.
        save : bool, optional
            If True, save the figure to ``{plot_dir}/{map_name}_maps.png``.
        backend : {'skyproj', 'healpy'}, optional
            Plotting backend. With ``'skyproj'`` (default), uses
            :class:`skyproj.MollweideSkyproj` for full sky and octant
            footprints, and :class:`skyproj.McBrydeSkyproj` otherwise. With
            ``'healpy'``, uses :func:`healpy.orthview` (octant footprint) or
            :func:`healpy.mollview` (full sky or custom mask).
        nside_plot : int or None, optional
            Maps of higher nside are degraded to it before drawing (default 1024, finer
            than the panels' resolution): the figure then only holds these copies, not
            the full-resolution maps. None draws the
            maps as they are.
        **kwargs
            Forwarded to :func:`healpy.orthview`/:func:`healpy.mollview` (for
            ``backend='healpy'``) or to
            :meth:`skyproj.Skyproj.draw_hspmap` (for ``backend='skyproj'``).
        """
        if backend not in ('healpy', 'skyproj'):
            raise ValueError(f"backend must be 'healpy' or 'skyproj', not {backend!r}")

        # map indices (as in iter), or maps given directly
        maps = None if maps_or_imaps is None or np.isscalar(maps_or_imaps) else list(maps_or_imaps)
        if maps is not None and np.isscalar(maps[0]):
            maps_or_imaps, maps = maps, None
        imaps = list(self.iter(maps_or_imaps)) if maps is None else range(len(maps))
        nmaps = len(imaps)

        # near-square grid of panels
        if nx is None:
            nx = int(np.ceil(np.sqrt(nmaps)))
        ny = int(np.ceil(nmaps / nx))

        if backend == 'skyproj':
            fig, axes = plt.subplots(ny, nx, figsize=(8 * nx, 5 * ny), squeeze=False)
            for ax in axes.flat[nmaps:]:
                ax.axis('off')
            axes = axes.flat
        else:
            fig = plt.figure(figsize=(4 * nx, 4 * ny))

        # draw each map in its panel
        sub = 1
        items = self.iter_maps(imaps, desc="[plot_maps]") if maps is None else ((imap, maps[imap]) for imap in self.tqdm(imaps))
        for imap, hp_map in items:
            title = title_prefix + f" {imap}" if title_prefix is not None else None

            # HealSparse map (maps given as dense RING maps or RING pixel-value pairs converted), degraded for
            # drawing, so that the full-resolution map is released after this panel
            hp_map = self._to_hsp(hp_map)
            if nside_plot is not None and hp_map.nside_sparse > nside_plot:
                hp_map = hp_map.degrade(nside_plot)

            if backend == 'skyproj':
                ax = next(axes)
                # Mollweide for the octant or the full sky, McBryde (zoomed on the footprint) otherwise
                if self.footprint.octant or self.footprint.fullsky:
                    sp = skyproj.MollweideSkyproj(ax=ax)
                else:
                    sp = skyproj.McBrydeSkyproj(ax=ax)

                # log scale, or symmetric log if some values are not positive
                draw_kwargs = dict(kwargs)
                if log:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        log_finite = np.all(np.isfinite(np.log(hp_map[hp_map.valid_pixels])))
                    draw_kwargs['norm'] = 'log' if log_finite else 'symlog'
                sp.draw_hspmap(hp_map, **draw_kwargs)

                if title is not None:
                    ax.set_title(title)
                if cbar:
                    sp.draw_colorbar(pad=0.05)
            else:
                # healpy backend: dense RING maps, orthographic view of the octant or Mollweide
                hp_map = hsp_to_dense(hp_map)
                if log and np.any(hp_map > 0.):
                    hp_map = np.log10(hp_map)
                if self.footprint.octant:
                    hp.orthview(
                        hp_map,
                        rot=(35, 20),
                        half_sky=True,
                        sub=(ny, nx, sub),
                        title=title,
                        cbar=cbar,
                        min=0. if log else None,
                        margins=(0., 0., 0., 0.),
                        nest=False,
                        **kwargs)
                else:
                    hp.mollview(hp_map, sub=(ny, nx, sub), title=title, cbar=cbar, min=0. if log else None, margins=(0., 0., 0., 0.), nest=False, **kwargs)
            sub += 1

        if backend == 'healpy':
            hp.graticule()

        if save:
            plot_dir = plot_dir or os.path.join(self.save_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            fname = os.path.join(plot_dir, f'{self.map_name}_maps.png')
            # matplotlib cannot render images beyond 2**16 pixels per side: lower the resolution of large grids
            dpi = min(150., 2**15 / max(fig.get_size_inches()))
            plt.savefig(fname, bbox_inches='tight', dpi=dpi, transparent=False)

    # endregion


class MapSum:
    """Weighted sum of maps over the padded footprint, accumulated in place, from maps of any kind.

    Maps are added one at a time, so only one input map needs to be in memory at a time
    (``hsp.operations.sum_union`` needs all maps at once, or rebuilds the output for every pairwise sum). The sum
    is in the representation of the first map added: HealSparse maps are summed in the output's own sparse array,
    by NESTED blocks, and dense RING maps and RING pixel-value pairs ``(ipix, values)`` (see
    :meth:`MapBuilder._to_dense`) in a dense RING map, without conversion.

    Parameters
    ----------
    footprint : pollux.footprint.Footprint
        Footprint of the maps: every pixel of its padded footprint has a value in the sum, starting at zero.
    nside_coverage : int
        Coverage nside of a HealSparse sum.
    dtype : data-type
        dtype of the sum.

    Notes
    -----
    In a HealSparse sum, built from sorted coverage pixels, block ``k`` of the sparse array (after the leading
    sentinel block) holds the NESTED pixels of coverage pixel ``cov_pix[k]``.
    """

    def __init__(self, footprint: Footprint, nside_coverage: int, dtype: np.dtype) -> None:
        self.footprint = footprint
        self.nside_coverage = nside_coverage
        self.dtype = np.dtype(dtype)
        self.sentinel = hsp.utils.check_sentinel(self.dtype.type, None)
        self.sparse = None  # HealSparse sum
        self.dense = None  # dense RING sum

    def _start_hsp(self) -> None:
        """Start a HealSparse sum: zero on the padded footprint, by NESTED blocks."""
        mask = self.footprint.mask(nest=True, padded=True)
        self.nfine = (self.footprint.nside // self.nside_coverage)**2
        mask_blocks = mask.reshape(-1, self.nfine)
        cov_pix = np.flatnonzero(mask_blocks.any(axis=1))
        self.cov = hsp.HealSparseCoverage.make_from_pixels(self.nside_coverage, self.footprint.nside, cov_pix)
        self.cov_index = self.cov[:]
        self.sparse = np.full((len(cov_pix) + 1) * self.nfine, self.sentinel, dtype=self.dtype)
        self.sparse[self.nfine:].reshape(-1, self.nfine)[mask_blocks[cov_pix]] = 0

    def _start_dense(self) -> None:
        """Start a dense RING sum: zero on the padded footprint, the sentinel outside."""
        self.dense = np.zeros(self.footprint.npix, dtype=self.dtype)
        if not self.footprint.fullsky:
            self.dense[~self.footprint.mask(nest=False, padded=True)] = self.sentinel

    def add(self, m: np.ndarray | hsp.HealSparseMap | tuple[np.ndarray, np.ndarray], scale: float = 1., offset: float = 0.) -> None:
        """Add ``scale * m + offset`` on the pixels of ``m`` with a value.

        Parameters
        ----------
        m : ndarray or hsp.HealSparseMap or tuple
            Map to add, of the nside of the footprint, with values inside the padded footprint: dense RING map
            (values where not its sentinel), HealSparse map (on its valid pixels, of coverage nside
            ``nside_coverage``), or RING pixel-value pair ``(ipix, values)`` (values on distinct RING pixels).
        scale : float, optional
            Factor applied to the map values.
        offset : float, optional
            Constant added on each pixel of ``m`` with a value.
        """
        if self.sparse is None and self.dense is None:
            self._start_hsp() if isinstance(m, hsp.HealSparseMap) else self._start_dense()
        if isinstance(m, hsp.HealSparseMap):
            if self.sparse is None:
                raise ValueError("A HealSparse map cannot be added to a sum of dense maps.")
            src_cov = np.flatnonzero(m.coverage_mask)
            src_cov = src_cov[self.cov.coverage_mask[src_cov]]  # inputs lie in the footprint, so this drops nothing
            _add_blocks(self.sparse, self.cov_index, m._sparse_map, m._cov_map[:], src_cov, self.nfine, m._sentinel, scale, offset)
        elif self.dense is None:
            raise ValueError("A dense map cannot be added to a sum of HealSparse maps.")
        elif isinstance(m, tuple):
            _add_pixels(self.dense, *m, scale, offset)
        else:
            _add_dense(self.dense, m, hsp.utils.check_sentinel(m.dtype.type, None), scale, offset)

    def to_map(self) -> np.ndarray | hsp.HealSparseMap:
        """Return the sum (sharing memory with this accumulator).

        Returns
        -------
        ndarray or hsp.HealSparseMap
            Sum of the added maps, with a value on every pixel of the padded footprint: a HealSparse map for
            HealSparse maps, a dense RING map otherwise (also without any map added).
        """
        if self.sparse is not None:
            return hsp.HealSparseMap(cov_map=self.cov, sparse_map=self.sparse, nside_sparse=self.footprint.nside, sentinel=self.sentinel)
        if self.dense is None:
            self._start_dense()
        return self.dense


@numba.njit(parallel=True)
def _add_dense(dst: np.ndarray, src: np.ndarray, src_sentinel: float, scale: float, offset: float) -> None:
    """Add ``scale * value + offset`` for the values of a dense map that are not its sentinel into another, in place.

    The same arithmetic as :func:`_add_blocks`.

    Parameters
    ----------
    dst : ndarray
        Destination dense map, updated in place.
    src : ndarray
        Source dense map, of the same length.
    src_sentinel : scalar
        Sentinel value of the source; such pixels are skipped.
    scale, offset : float
        Applied to each source value before adding.
    """
    for i in numba.prange(len(src)):
        v = src[i]
        if v != src_sentinel:
            dst[i] += scale * v + offset


@numba.njit(parallel=True)
def _add_pixels(dst: np.ndarray, ipix: np.ndarray, values: np.ndarray, scale: float, offset: float) -> None:
    """Add ``scale * value + offset`` for values on distinct pixels into a dense map, in place.

    The same arithmetic as :func:`_add_dense`.

    Parameters
    ----------
    dst : ndarray
        Destination dense map, updated in place.
    ipix : ndarray of int
        Distinct pixels of the values (e.g. the non-empty pixels of a sub-volume map).
    values : ndarray
        Values on ``ipix``.
    scale, offset : float
        Applied to each value before adding.
    """
    for i in numba.prange(len(ipix)):
        dst[ipix[i]] += scale * values[i] + offset


@numba.njit
def _add_blocks(
        dst: np.ndarray, dst_cov_index: np.ndarray, src: np.ndarray, src_cov_index: np.ndarray, src_cov_pix: np.ndarray, nfine: int, src_sentinel: float,
        scale: float, offset: float) -> None:
    """Add ``scale * value + offset`` for the valid values of a HealSparse sparse array into another, in place.

    Both arrays use the HealSparse layout, where the NESTED pixel ``p`` of
    coverage pixel ``c`` is stored at ``cov_index[c] + p``, i.e. at
    ``cov_index[c] + c * nfine + j`` for its ``j``-th fine pixel.

    Parameters
    ----------
    dst : ndarray
        Destination sparse array, updated in place.
    dst_cov_index : ndarray of int
        Coverage index map of the destination (covering ``src_cov_pix``).
    src : ndarray
        Source sparse array.
    src_cov_index : ndarray of int
        Coverage index map of the source.
    src_cov_pix : ndarray of int
        Coverage pixels of the source to add.
    nfine : int
        Number of fine pixels per coverage pixel.
    src_sentinel : scalar
        Sentinel value of the source; such pixels are skipped.
    scale, offset : float
        Applied to each valid source value before adding.
    """
    for c in src_cov_pix:
        s0 = src_cov_index[c] + c * nfine
        d0 = dst_cov_index[c] + c * nfine
        for j in range(nfine):
            v = src[s0 + j]
            if v != src_sentinel:
                dst[d0 + j] += scale * v + offset
