from __future__ import annotations

import logging
import os
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Iterable, Iterator

import healpy as hp
import healsparse as hsp
import skyproj
import matplotlib.pyplot as plt
import numba
import numpy as np
import pyccl
import pymaster as nmt
import sacc
from itertools import product
from tqdm.auto import tqdm, trange

from .cosmo import Theory
from .utils import make_mask_radec_cuts, make_nmtbin_logspaced

logger = logging.getLogger(__name__)


class MapBuilder(ABC):
    """Base class to build, store, and analyze Healpix/HealSparse maps.

    This class centralizes:
    - file naming and I/O for HealSparse maps,
    - mask/footprint creation,
    - alm computation (Healpy),
    - pseudo-C_ell and decoupled bandpowers (NaMaster),
    - SACC export and plotting.

    Subclasses must implement :meth:`_create_tracers` and
    :meth:`construct_maps` (abstract).
    :meth:`_noise_cls` is optional, needed only for noise-aware predictions.

    Parameters
    ----------
    nside : int
        Healpix resolution parameter of the maps.
    dtype : data-type
        Data type used to store map values (dtype for arrays written to
        HealSparse/Healpix maps and intermediate map buffers).
    save_dir : str
        Directory where outputs are written.
    base_fname : str
        Base filename prefix used by :meth:`get_fname`.
    nmaps : int
        Number of maps handled/produced by this builder.
    nside_c : int, optional
        HealSparse coverage nside. Default is 32.
    mask : ndarray or str or None, optional
        Footprint specification:

        - ``None``: full sky (no mask applied).
        - ``'octant'``: restrict to the positive octant (x>0, y>0, z>0).
        - ndarray: binary HEALPix mask of length ``hp.nside2npix(nside)``; pixels
          where the mask is non-zero are kept. Must be in NESTED ordering.
        - str (other than ``'octant'``): path to a FITS file containing a
          binary HEALPix mask in NESTED ordering, read lazily inside
          :meth:`_create_mask`.

        Default is ``None``.
    save_format : str, optional
        HealSparse file format (e.g. 'fits'). Default is 'fits'.
    wsp_dir : str or None, optional
        Directory used to cache NaMaster workspaces. Default is ``save_dir``.
    sacc_type : str or None, optional
        SACC data type name used when saving spectra.
    b : pymaster.NmtBin, optional
        NaMaster binning scheme. If None, it is constructed lazily.
    nlb : int, optional
        Number of bandpowers when constructing bins lazily. Default is 20.
    remove_monopole : bool, optional
        If True (default), subtract the (weighted) mean of maps over the alm
        weights before harmonic transforms (:meth:`_alm_field`). Other uses of
        maps (:meth:`field_map`) keep their mean.
    nocompress : bool, optional
        If True, disable HealSparse compression. Default is True.
    f_ell : {'pixwin'} or ndarray or None, optional
        Multiplicative factor applied to theory spectra before binning (e.g.
        pixel window). If 'pixwin', uses the Healpix pixel window. Default is
        'pixwin'.
    padding_deg : float or None, optional
        If given (and the footprint is not the full sky), maps are built on the
        footprint dilated by this angle (the padded footprint, see :meth:`get_mask`),
        and harmonic transforms use the padded footprint apodized over the padding
        (:attr:`_alm_weights`), equal to 1 on the footprint. Analyses
        (catalogs, sky fraction) use the footprint itself. Default is None (no padding).
    apotype : {'C1', 'C2', 'Smooth'} or None, optional
        Apodization of the padded footprint (see :func:`pymaster.mask_apodization`),
        over the padding width. If None, the padded footprint is used unapodized.
        Default is 'C1'.
    padding_nside : int, optional
        Resolution at which the padded footprint and its apodization are computed
        (then upgraded to ``nside``). Default is 512.
    replicate_octant : bool, optional
        If True, harmonic transforms use the octant footprint replicated to the
        full sky (by reflections, see :meth:`_octant_to_fullsky_ipix`) instead of
        the masked octant. Requires the octant footprint without padding. Default is False.

    Notes
    -----
    Healpix ordering: this codebase uses NESTED pixel ordering by default for
    internal Healpix/HealSparse operations and IO unless otherwise stated.
    """

    # ``what`` of the smoothed field maps (e.g. 'density' for shells; None for convergence maps)
    smoothed_field_what: str | None = None

    def __init__(
        self,
        nside: int,
        dtype: np.dtype,
        save_dir: str,
        base_fname: str,
        nmaps: int,
        nside_c: int = 32,
        mask: np.ndarray | str | None = None,
        save_format: str = 'fits',
        wsp_dir: str | None = None,
        sacc_type: str | None = None,
        b: 'nmt.NmtBin' | None = None,
        nlb: int = 20,
        remove_monopole: bool = True,
        nocompress: bool = True,
        f_ell: str | np.ndarray | None = 'pixwin',
        padding_deg: float | None = None,
        apotype: str | None = 'C1',
        padding_nside: int = 512,
        replicate_octant: bool = False,
    ) -> None:
        if isinstance(mask, np.ndarray) and len(mask) != hp.nside2npix(nside):
            raise ValueError(f"mask has {len(mask)} pixels but nside={nside} requires {hp.nside2npix(nside)}.")
        if isinstance(mask, str) and mask != 'octant' and not os.path.exists(mask):
            raise ValueError(f"mask path {mask} does not exist.")
        self.nside = nside
        self.dtype = dtype
        self.save_dir = save_dir
        self.base_fname = base_fname
        self.nmaps = nmaps
        self.nside_c = nside_c
        self.footprint = mask
        self.octant_only = isinstance(mask, str) and mask == 'octant'
        self.save_format = save_format
        self.wsp_dir = wsp_dir if wsp_dir is not None else save_dir
        self.sacc_type = sacc_type
        if b is not None:
            self.b = b  # otherwise built lazily
        self.nlb = nlb
        self.remove_monopole = remove_monopole
        self.nocompress = nocompress
        self.f_ell = f_ell
        self.padding_deg = padding_deg if (padding_deg and mask is not None) else None
        self.apotype = apotype
        self.padding_nside = min(padding_nside, nside)
        if replicate_octant and (not self.octant_only or self.padding_deg is not None):
            raise ValueError("replicate_octant requires the octant footprint without padding.")
        self.replicate_octant = replicate_octant

        # Derived attributes
        self.npix = hp.nside2npix(self.nside)

        # Cached attributes (lazy-initialized)
        self._masks = {}  # (padded, nest) -> [mask, ipix]
        self._ipix_oct_fullsky = None
        self.predicted_cls = None

        # Make sure folder exists
        os.makedirs(self.save_dir, exist_ok=True)

        logger.info("*** Map builder (%s) ***", self.base_fname)
        logger.info("  - Builder class = %s.", self.__class__.__name__)
        logger.info("  - Number of maps = %d.", self.nmaps)
        logger.info("  - Maps saved to %s.", self.save_dir)

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

    def get_fname(
        self,
        imap: int | None = None,
        sub: int | None = None,
        what: str | None = None,
        smooth_Mpc: float | None = None,
        extra: str | None = None,
        format: str | None = None,
    ) -> str:
        """Build an output filename for a given data product.

        Parameters
        ----------
        imap : int or None, optional
            Map index to include in the filename.
        sub : int or None, optional
            Sub-volume identifier. If provided, the file is placed under
            ``{save_dir}/subs/``.
        what : str or None, optional
            Field/product tag (e.g. ``'alms'``, ``'cls'``, ``'gamma1'``,
            ``'density'``).  Do *not* embed the smoothing scale here — pass it
            via ``smooth_Mpc`` instead so the convention is consistent.
        smooth_Mpc : float or None, optional
            Physical Gaussian smoothing scale in Mpc.  When provided,
            ``_{smooth_Mpc}Mpc`` is appended after ``what`` in the filename.
        extra : str or None, optional
            Additional string appended at the end of the stem.
        format : str or None, optional
            File extension (without leading dot). Defaults to
            ``self.save_format``.

        Returns
        -------
        str
            Full path to the requested product file.

        Notes
        -----
        The filename stem is built as::

            {base_fname}[_{what}][_{smooth_Mpc}Mpc][_imap{imap}][_sub{sub}][_{extra}]

        The folder encodes all context that varies (nside, seed, builder type,
        smoothing hierarchy); filenames are kept as short as possible.
        """
        parts = [self.base_fname]
        if what is not None:
            parts.append(what)
        if smooth_Mpc is not None:
            parts.append(f'{smooth_Mpc}Mpc')
        if imap is not None:
            parts.append(f'imap{imap}')
        if sub is not None:
            parts.append(f'sub{sub}')
        if extra is not None:
            parts.append(extra)
        fname = '_'.join(parts) + f'.{format or self.save_format}'

        if sub is None:
            return os.path.join(self.save_dir, fname)
        else:
            subs_dir = os.path.join(self.save_dir, 'subs')
            os.makedirs(subs_dir, exist_ok=True)
            return os.path.join(subs_dir, fname)

    @abstractmethod
    def construct_maps(self, imaps: int | Iterable[int] | None = None, **kwargs: Any) -> hsp.HealSparseMap | dict[int, hsp.HealSparseMap]:
        """Construct maps and return them as HealSparse objects.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to construct. If None, constructs all maps.
        **kwargs
            Forwarded to subclass implementation.

        Returns
        -------
        hsp.HealSparseMap or dict
            - If ``imaps`` is a single int: returns a single ``HealSparseMap``.
            - If ``imaps`` is ``None`` or an iterable: returns a dict keyed by
              map index with ``HealSparseMap`` values.
        """

    def check_maps(self,
                   imaps: int | Iterable[int] | None = None,
                   read: bool = False,
                   fname: str | None = None,
                   check_finite: bool = False) -> dict[int, bool | hsp.HealSparseMap | None] | bool | hsp.HealSparseMap | None:
        """Check existence/readability of map files.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to check (ignored if ``fname`` is provided).
        read : bool, optional
            If True, attempt to read the file(s) with HealSparse. When
            ``read=True`` the function returns loaded maps (or ``None`` for
            missing/unreadable files) rather than boolean existence flags.
        fname : str or None, optional
            If provided, check this single filename instead of using indices.
        check_finite : bool, optional
            If True, validates that values on valid pixels are finite and not
            identically zero (only used when ``read=True``).

        Returns
        -------
        dict or bool or hsp.HealSparseMap or None
            - If ``fname`` is provided: returns a single value:
                - when ``read=False``: ``True`` if file exists else ``False``;
                - when ``read=True``: loaded ``HealSparseMap`` on success or
                  ``None`` on failure.
            - If ``imaps`` is a single integer: same single-value behavior as
              above (based on ``read``).
            - If ``imaps`` is ``None`` or an iterable: returns a **dict**
              keyed by map index. Values are either booleans (when
              ``read=False``) or loaded ``HealSparseMap`` objects / ``None``
              (when ``read=True``).

        Notes
        -----
        Use the dict-returning form to obtain loaded maps for further
        processing; this makes it straightforward to build a unified mapping
        of loaded/computed maps keyed by index.
        """
        if not (imaps is None or fname is None):
            raise ValueError("Give imaps or fname, not both")

        def check_one(fname: str) -> bool | hsp.HealSparseMap | None:
            # existence (or map, with read) of one file
            if os.path.exists(fname):
                if read:
                    try:
                        m = hsp.HealSparseMap.read(fname, nside_coverage=self.nside_c)
                        if check_finite:
                            a = m[m.valid_pixels]
                            if not (np.all(np.isfinite(a)) and np.any(a)):
                                logger.debug("Map at %s failed the finite check, ignoring cached file.", fname)
                                return None
                        logger.debug("Found and loaded map at %s.", fname)
                        return m
                    except Exception as e:
                        warnings.warn(f"Cannot read map at {fname}: {e}.")
                        return None
                else:
                    logger.debug("Found map at %s.", fname)
                    return True
            else:
                logger.debug("Cannot find map at %s.", fname)
                return None if read else False

        # a single file, or the maps of the given indices
        if fname is not None:
            return check_one(fname)

        out = {}
        for imap in self.tqdm(imaps, desc="[check_maps]"):
            fname = self.get_fname(imap)
            out[imap] = check_one(fname)

        return out[imaps] if np.isscalar(imaps) else out

    def load_maps(self, imaps: int | Iterable[int] | None = None, **fname_kwargs: Any) -> hsp.HealSparseMap | dict[int, hsp.HealSparseMap]:
        """Load one or multiple HealSparse maps from disk.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to load. If None, loads all maps.
        **fname_kwargs
            Extra keywords forwarded to :meth:`get_fname` (e.g. ``what='kappa'``).

        Returns
        -------
        hsp.HealSparseMap or dict
            - If ``imaps`` is a single int: returns a single ``HealSparseMap``.
            - If ``imaps`` is ``None`` or an iterable: returns a dict keyed by
              map index with ``HealSparseMap`` values.
        """
        maps = dict(self.iter_maps(imaps, desc="[load_maps]", **fname_kwargs))
        return maps[imaps] if np.isscalar(imaps) else maps

    def iter_maps(
        self,
        imaps: int | Iterable[int] | None = None,
        isubs: Iterable[int | str] | None = None,
        allow_missing: bool = False,
        desc: str = "[iter_maps]",
        **fname_kwargs: Any,
    ) -> Iterator[tuple[int | str, hsp.HealSparseMap]]:
        """Read HealSparse maps from disk one at a time.

        Unlike :meth:`load_maps`, only one map is held in memory at a time, which
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
        **fname_kwargs
            Extra keywords forwarded to :meth:`get_fname` (e.g. ``what='kappa'``).

        Yields
        ------
        index : int or str
            Map index, or sub index if ``isubs`` is given.
        map : hsp.HealSparseMap
            The map read from disk.
        """
        if isubs is None:
            items = [(imap, self.get_fname(imap, **fname_kwargs)) for imap in self.iter(imaps)]
        else:
            if not np.isscalar(imaps):
                raise ValueError('imaps must be a single map index when isubs is given.')
            items = [(isub, self.get_fname(imaps, sub=isub, **fname_kwargs)) for isub in isubs]

        for index, fname in tqdm(items, desc=desc, disable=len(items) == 1):
            if not os.path.exists(fname):
                if not allow_missing:
                    raise FileNotFoundError(f"Map not found at {fname}.")
                logger.info("Skipping missing map at %s.", fname)
                continue
            hspmap = hsp.HealSparseMap.read(fname, nside_coverage=self.nside_c)
            logger.debug("Found and loaded map at %s.", fname)
            yield index, hspmap

    def _make_hspmap(self, ipix: np.ndarray | None = None, values: np.ndarray | None = None, nest: bool = False, dtype: Any = None) -> hsp.HealSparseMap:
        """Return a HealSparse map at the builder's resolution, empty or with values on given pixels.

        Parameters
        ----------
        ipix : ndarray of int or None, optional
            Pixels of the values (e.g. the padded footprint). If None, the map is empty.
        values : ndarray or None, optional
            Values on ``ipix`` (not a full-sky map), cast to ``dtype``.
        nest : bool, optional
            Ordering of ``ipix``. Default is False (RING).
        dtype : data-type or None, optional
            Data type of the map. Default is the builder's ``dtype``.

        Returns
        -------
        hsp.HealSparseMap
        """
        dtype = self.dtype if dtype is None else dtype
        hspmap = hsp.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=dtype)
        if ipix is not None:
            hspmap.update_values_pix(ipix, np.asarray(values).astype(dtype), nest=nest)
        return hspmap

    def _save_map(self, hspmap: hsp.HealSparseMap, fname: str) -> None:
        """Save a HealSparse map with the builder's format and compression.

        Parameters
        ----------
        hspmap : hsp.HealSparseMap
            Map to save.
        fname : str
            Output path (typically from :meth:`get_fname`).
        """
        hspmap.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
        logger.debug("Saved to file %s.", fname)

    def plot_maps(
            self,
            maps_or_imaps: int | Iterable[int] | Iterable | None = None,
            nx: int | None = None,
            log: bool = False,
            title_prefix: str | None = None,
            cbar: bool = True,
            save: bool = True,
            backend: str = 'skyproj',
            **kwargs: Any) -> None:
        """Plot one or multiple maps.

        Parameters
        ----------
        maps_or_imaps : int or iterable or None, optional
            - If None: loads and plots all maps.
            - If int: loads and plots that map index.
            - If iterable of int: loads and plots those indices.
            - If iterable of maps: plots provided maps directly.
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
            If True, save the figure to ``get_fname(format='png')``.
        backend : {'skyproj', 'healpy'}, optional
            Plotting backend. With ``'skyproj'`` (default), uses
            :class:`skyproj.MollweideSkyproj` for full sky and octant
            footprints, and :class:`skyproj.McBrydeSkyproj` otherwise. With
            ``'healpy'``, uses :func:`healpy.orthview` (octant footprint) or
            :func:`healpy.mollview` (full sky or custom mask).
        **kwargs
            Forwarded to :func:`healpy.orthview`/:func:`healpy.mollview` (for
            ``backend='healpy'``) or to
            :meth:`skyproj.Skyproj.draw_hspmap`/``draw_hpxmap`` (for
            ``backend='skyproj'``).
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

            if backend == 'skyproj':
                ax = next(axes)
                # Mollweide for the octant or the full sky, McBryde (zoomed on the footprint) otherwise
                if self.octant_only or self.footprint is None:
                    sp = skyproj.MollweideSkyproj(ax=ax)
                else:
                    sp = skyproj.McBrydeSkyproj(ax=ax)

                # log scale, or symmetric log if some values are not positive
                draw_kwargs = dict(kwargs)
                if log:
                    if isinstance(hp_map, hsp.HealSparseMap):
                        values = hp_map[hp_map.valid_pixels]
                    else:
                        values = hp_map[hp_map != hp.UNSEEN]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        log_finite = np.all(np.isfinite(np.log(values)))
                    draw_kwargs['norm'] = 'log' if log_finite else 'symlog'

                # healpy backend: dense maps, orthographic view of the octant or Mollweide
                if isinstance(hp_map, hsp.HealSparseMap):
                    sp.draw_hspmap(hp_map, **draw_kwargs)
                else:
                    sp.draw_hpxmap(hp_map, nest=True, **draw_kwargs)

                if title is not None:
                    ax.set_title(title)
                if cbar:
                    sp.draw_colorbar(pad=0.05)
            else:
                if isinstance(hp_map, hsp.HealSparseMap):
                    hp_map = hp_map.generate_healpix_map()
                if log and np.any(hp_map > 0.):
                    hp_map = np.log10(hp_map)
                if self.octant_only:
                    hp.orthview(
                        hp_map,
                        rot=(35, 20),
                        half_sky=True,
                        sub=(ny, nx, sub),
                        title=title,
                        cbar=cbar,
                        min=0. if log else None,
                        margins=(0., 0., 0., 0.),
                        nest=True,
                        **kwargs)
                else:
                    hp.mollview(hp_map, sub=(ny, nx, sub), title=title, cbar=cbar, min=0. if log else None, margins=(0., 0., 0., 0.), nest=True, **kwargs)
            sub += 1

        if backend == 'healpy':
            hp.graticule()

        if save:
            fname = self.get_fname(format='png')
            # matplotlib cannot render images beyond 2**16 pixels per side: lower the resolution of large grids
            dpi = min(150., 2**15 / max(fig.get_size_inches()))
            plt.savefig(fname, bbox_inches='tight', dpi=dpi, transparent=False)

    def get_mask(self, nest: bool = True, return_ipix: bool = False, padded: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return the footprint (or padded footprint) mask, and optionally its pixel indices.

        Parameters
        ----------
        nest : bool, optional
            If True, use NESTED ordering; otherwise RING. Default is True.
        return_ipix : bool, optional
            If True, also return the valid pixel indices. Default is False.
        padded : bool, optional
            If True, return the padded footprint (see ``padding_deg``), on which maps
            are built; it is the footprint without padding. Default is False.

        Returns
        -------
        mask : ndarray of bool
            Dense mask at ``self.nside`` in the requested ordering.
        ipix : ndarray of int, optional
            Valid pixel indices in the requested ordering (only if
            ``return_ipix=True``).

        Notes
        -----
        Results are cached. In RING ordering, ``ipix[k]`` is the RING index of
        the k-th NESTED valid pixel (so it is not sorted).
        """
        padded = padded and self.padding_deg is not None
        key = (padded, nest)
        if key not in self._masks:
            if nest:
                self._masks[key] = [self._create_padded_mask() if padded else self._create_mask(), None]
            else:
                self._masks[key] = [hp.reorder(self.get_mask(nest=True, padded=padded), n2r=True), None]
        cache = self._masks[key]
        if not return_ipix:
            return cache[0]
        if cache[1] is None:
            ipix_nest = np.flatnonzero(self.get_mask(nest=True, padded=padded))
            cache[1] = ipix_nest if nest else hp.nest2ring(self.nside, ipix_nest)
        return cache[0], cache[1]

    @property
    def fsky(self) -> float:
        """Sky fraction covered by the footprint mask.

        Computed from the actual number of kept pixels and their areas, so it
        accounts for edge effects (e.g. HEALPix pixels only partially inside
        the octant boundary that are dropped by :meth:`_create_mask`).

        Returns
        -------
        float
            Fraction of the full sky inside the footprint (between 0 and 1).
        """
        _, ipix = self.get_mask(return_ipix=True)
        return len(ipix) / self.npix

    def _create_mask(self) -> np.ndarray:
        """Create a footprint mask at ``self.nside``.

        Returns
        -------
        mask : ndarray of bool
            Dense mask array in NESTED ordering, length ``self.npix``.
        """
        if isinstance(self.footprint, str) and self.footprint == 'octant':
            mask = make_mask_radec_cuts(self.nside, 0, 90, 0, 90, eps=1e-10, nest=True, dtype=bool)
        elif isinstance(self.footprint, str):
            # path to a NESTED-ordering FITS mask
            mask, fits_header = hp.read_map(self.footprint, nest=True, h=True)
            ordering = dict(fits_header)['ORDERING']
            if ordering.upper() != 'NESTED':
                raise ValueError(f"mask at {self.footprint} has ORDERING={ordering!r} in its FITS header, expected 'NESTED'.")
            mask = mask.astype(bool)
            if len(mask) != self.npix:
                raise ValueError(f"mask at {self.footprint} has {len(mask)} pixels but nside={self.nside} requires {self.npix}.")
        elif isinstance(self.footprint, np.ndarray):
            mask = self.footprint.astype(bool)
        else:
            mask = np.ones(self.npix, dtype=bool)

        return mask

    def _create_padded_mask(self) -> np.ndarray:
        """Create the padded footprint: the footprint dilated by ``padding_deg``.

        Returns
        -------
        mask : ndarray of bool
            Dense mask array in NESTED ordering, length ``self.npix``.

        Notes
        -----
        The dilation is computed at ``padding_nside`` (pixels within ``padding_deg``
        of the footprint, where the C1 apodization of its complement is below 1,
        whatever ``apotype``) and upgraded to ``nside``, so the padding is accurate
        to about a ``padding_nside`` pixel. It is read from disk if saved by
        :meth:`compute_padding`.
        """
        mask = self.get_mask(nest=True)
        if self.padding_deg is None:
            return mask
        fname = self._padding_fname('padded_mask')
        if os.path.exists(fname):
            padded = np.zeros(self.npix, dtype=bool)
            padded[hsp.HealSparseMap.read(fname).valid_pixels] = True
            logger.debug("Found and loaded padded footprint at %s.", fname)
            return padded
        outside_lowres = (hp.ud_grade((~mask).astype(np.float64), self.padding_nside, order_in='NESTED', order_out='RING') > 0.).astype(np.float64)
        apo_lowres = nmt.mask_apodization(outside_lowres, self.padding_deg, apotype='C1')
        padded_lowres = hp.reorder((apo_lowres < 1.).astype(np.float64), r2n=True) > 0.
        return mask | (hp.ud_grade(padded_lowres.astype(np.float64), self.nside, order_in='NESTED', order_out='NESTED') > 0.)

    def _padding_fname(self, what: str) -> str:
        """Return the path of the padded footprint or alm weights saved by :meth:`compute_padding`.

        Parameters
        ----------
        what : {'padded_mask', 'alm_weights'}
            Product.

        Returns
        -------
        str
            ``{wsp_dir}/{what}_nside{nside}_pad{padding_deg}[{apotype}]_res{padding_nside}.fits``.
        """
        apotype = (self.apotype or '') if what == 'alm_weights' else ''
        return os.path.join(self.wsp_dir, f'{what}_nside{self.nside}_pad{self.padding_deg:g}{apotype}_res{self.padding_nside}.fits')

    def compute_padding(self) -> None:
        """Compute and save the padded footprint and the alm weights, if not on disk.

        Notes
        -----
        Does nothing without ``padding_deg``. The products (:meth:`get_mask` with
        ``padded=True``, :attr:`_alm_weights`) are then read from disk
        (:meth:`_padding_fname`) instead of recomputed by every builder.
        """
        if self.padding_deg is None:
            return
        _, ipix = self.get_mask(nest=True, return_ipix=True, padded=True)
        for what in ('padded_mask', 'alm_weights'):
            fname = self._padding_fname(what)
            if os.path.exists(fname) or (what == 'alm_weights' and self.apotype is None):
                continue
            values = np.ones(len(ipix), dtype=bool) if what == 'padded_mask' else self._alm_weights[hp.nest2ring(self.nside, ipix)]
            hspmap = self._make_hspmap(ipix, values, nest=True, dtype=values.dtype)
            # via a temporary file, so readers never see a partial file
            tmp = fname[:-len('.fits')] + '.tmp.fits'
            hspmap.write(tmp, clobber=True, nocompress=self.nocompress)
            os.replace(tmp, fname)
            logger.info("Saved to file %s.", fname)

    def _octant_to_fullsky_ipix(self, nest: bool = True) -> np.ndarray:
        """Return a mapping from octant pixels to their full-sky replicas.

        Parameters
        ----------
        nest : bool, optional
            Pixel ordering used for the mapping.

        Returns
        -------
        ndarray of int
            Array of length ``hp.nside2npix(self.nside)`` giving, for each pixel
            index on the full sky, the corresponding pixel index in the octant
            (by reflecting theta and wrapping phi).

        Raises
        ------
        ValueError
            If ``self.octant_only`` is False.
        """
        if not self.octant_only:
            raise ValueError("Only defined for octant builders (octant_only=True)")

        if self._ipix_oct_fullsky is None:
            # geometric mapping: reflect southern hemisphere and reflect phi into first quadrant
            # (triangle wave, continuous across quadrant boundaries)
            theta, phi = hp.pix2ang(self.nside, np.arange(self.npix), nest=nest)
            ipix_mapped = hp.ang2pix(
                self.nside, np.where(theta <= np.pi / 2., theta, np.pi - theta), np.pi / 2. - np.abs((phi % np.pi) - np.pi / 2.), nest=nest)

            # Pixels on the equator and the phi=0/pi/2 meridians are absent from
            # the octant mask. Replace each with a neighboring pixel in the mask.
            octant_mask = self.get_mask(nest=nest)
            bad = np.where(~octant_mask[ipix_mapped].astype(bool))[0]
            if len(bad) > 0:
                neighbors = hp.get_all_neighbours(self.nside, ipix_mapped[bad], nest=nest)  # (8, n_bad)
                for i, nbrs in enumerate(neighbors.T):
                    valid = nbrs[nbrs >= 0]
                    in_mask = octant_mask[valid].astype(bool)
                    if np.any(in_mask):
                        ipix_mapped[bad[i]] = valid[in_mask][0]

            self._ipix_oct_fullsky = ipix_mapped

        return self._ipix_oct_fullsky

    def field_hspmap(self, imap: int) -> hsp.HealSparseMap:
        """Return the field of one map as a HealSparse map.

        Parameters
        ----------
        imap : int
            Map index.

        Returns
        -------
        hsp.HealSparseMap
            The map on disk (subclasses may convert it, e.g. counts to density contrast).
        """
        return self.load_maps(imap)

    def field_map(self, imap: int, nest: bool = True) -> np.ndarray:
        """Return the field of one map (:meth:`field_hspmap`) as a dense Healpix array.

        Parameters
        ----------
        imap : int
            Map index.
        nest : bool, optional
            Ordering of the returned map.

        Returns
        -------
        ndarray
            Dense Healpix map of length ``hp.nside2npix(self.nside)``, UNSEEN outside the
            map's valid pixels, with its mean (see :meth:`_alm_field` for the field
            prepared for its a_lm).
        """
        return self.field_hspmap(imap).generate_healpix_map(nest=nest)

    @cached_property
    def _alm_weights(self) -> np.ndarray:
        """Weights applied to maps before computing their a_lm.

        Returns
        -------
        ndarray
            Dense weight map in RING ordering, length ``self.npix``:

            - without padding, the footprint mask (the full sky with ``replicate_octant``);
            - with ``padding_deg``, the padded footprint apodized over the padding
              (``apotype``, unapodized if None), equal to 1 on the footprint.

        Notes
        -----
        Padding keeps transforms (e.g. shear from convergence, smoothing) from being
        truncated at the footprint edges. The apodization is computed at ``padding_nside``
        and interpolated bilinearly at ``nside`` (avoiding steps at the ``padding_nside``
        pixel scale). The result is cached, and read from disk if saved by
        :meth:`compute_padding`.
        """
        fname = self._padding_fname('alm_weights') if self.padding_deg is not None else None
        if self.replicate_octant:
            weights = np.ones(self.npix)
        elif self.padding_deg is None or self.apotype is None:
            weights = self.get_mask(nest=False, padded=True).astype(np.float64)
        elif os.path.exists(fname):
            hspmap = hsp.HealSparseMap.read(fname)
            ipix = hspmap.valid_pixels
            weights = np.zeros(self.npix)
            weights[hp.nest2ring(self.nside, ipix)] = hspmap[ipix]
            logger.debug("Found and loaded alm weights at %s.", fname)
        else:
            padded_lowres = (
                hp.ud_grade(self.get_mask(nest=True, padded=True).astype(np.float64), self.padding_nside, order_in='NESTED', order_out='RING')
                > 0.).astype(np.float64)
            apo_lowres = nmt.mask_apodization(padded_lowres, self.padding_deg, apotype=self.apotype)
            # 1 on the footprint, and the apodization interpolated (bilinearly) in the padding
            footprint = self.get_mask(nest=False)
            weights = footprint.astype(np.float64)
            ipix_padding = np.flatnonzero(self.get_mask(nest=False, padded=True) & ~footprint)
            for ipix in np.array_split(ipix_padding, max(1, len(ipix_padding) // 10_000_000)):
                theta, phi = hp.pix2ang(self.nside, ipix)
                weights[ipix] = np.clip(hp.get_interp_val(apo_lowres, theta, phi), 0., 1.)
        return weights

    def _alm_field(self, imap: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the alm weights and the field prepared for its a_lm (RING ordering): ``map2alm(field * weights)``.

        Parameters
        ----------
        imap : int
            Map index.

        Returns
        -------
        weights : ndarray
            Alm weights (:attr:`_alm_weights`).
        _map : ndarray
            Field map (replicated to the full sky with ``replicate_octant``), with its
            weighted mean over the weights removed if ``self.remove_monopole``, and
            zero where the weights vanish (not multiplied by the weights).
        """
        weights = self._alm_weights
        _map = self.field_map(imap, nest=False)
        if self.replicate_octant:
            _map = _map[self._octant_to_fullsky_ipix(nest=False)]
        good = weights > 0.
        _map = np.where(good, _map, 0.)
        if self.remove_monopole:
            _map[good] -= np.average(_map[good], weights=weights[good])
        return weights, _map

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
            smoothing keeps constants) restores the original mean.
        """
        if not self.remove_monopole:
            return 0.
        weights = self._alm_weights
        if self.replicate_octant:
            values = self.field_map(imap, nest=False)[self._octant_to_fullsky_ipix(nest=False)]
            return float(np.average(values, weights=weights))
        ipix = np.flatnonzero(weights > 0.)
        values = self.field_hspmap(imap).get_values_pix(hp.ring2nest(self.nside, ipix), nest=True)
        return float(np.average(values, weights=weights[ipix]))

    def _alm_fname(self, imap: int) -> str:
        """Return the path of the cached a_lm of one map.

        Parameters
        ----------
        imap : int
            Map index.

        Returns
        -------
        str
            File name with ``what='alms'``, or ``'alms_octrep'`` with ``replicate_octant``.
        """
        return self.get_fname(imap, what='alms_octrep' if self.replicate_octant else 'alms', format='fits')

    def compute_alms(self, imaps: int | Iterable[int] | None = None, **map2alm_kwargs: Any) -> np.ndarray | dict[int, np.ndarray]:
        """Compute spherical harmonic coefficients a_lm from maps.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, uses all maps.
        **map2alm_kwargs
            Forwarded to :func:`healpy.map2alm`.

        Returns
        -------
        dict
            Mapping ``imap -> alm`` (complex ndarray).

        Notes
        -----
        The a_lm are those of :meth:`_alm_field` times the alm weights.
        Results are cached (:meth:`_alm_fname`): a pre-existing file is loaded
        instead of recomputing, and a freshly computed alm is saved for later calls.
        """
        alms = {}
        for imap in self.tqdm(imaps, desc="[compute_alms]"):
            fname = self._alm_fname(imap)
            if os.path.exists(fname):
                try:
                    alms[imap] = hp.read_alm(fname)
                    logger.debug("Found and loaded alms at %s.", fname)
                    continue
                except Exception as e:
                    warnings.warn(f"Cannot read alms at {fname}: {e}.")

            weights, _map = self._alm_field(imap)
            alms[imap] = hp.map2alm(_map * weights, **map2alm_kwargs)

            hp.write_alm(fname, alms[imap], overwrite=True)
            logger.debug("Saved alms to file %s.", fname)

        return alms[imaps] if np.isscalar(imaps) else alms

    def _smooth_sigma(self, imap: int, smooth_Mpc: float) -> float:
        """Return the angular width (radians) of a comoving smoothing scale for one map.

        Parameters
        ----------
        imap : int
            Map index.
        smooth_Mpc : float
            Comoving Gaussian smoothing scale in Mpc.

        Returns
        -------
        float
            Implemented by builders whose maps have a distance (shells, per-shell convergence).
        """
        raise NotImplementedError(f"{type(self).__name__} maps have no distance to convert a smoothing scale in Mpc to an angle.")

    def _smooth_alms(self, imap: int, alm: np.ndarray, smooth_Mpc: float, lmax: int) -> None:
        """Smooth the a_lm of one map in place, with a Gaussian of comoving scale ``smooth_Mpc``.

        The pixel window function is deconvolved, so that the Gaussian is the only
        smoothing of maps made from the smoothed a_lm (:meth:`_alm2hspmap`).

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

    def _alm2hspmap(self, alm: np.ndarray, ipix: np.ndarray, lmax: int, mean: float = 0.) -> hsp.HealSparseMap:
        """Return the map of a_lm as a HealSparse map on the given pixels.

        Parameters
        ----------
        alm : ndarray
            Spherical harmonic coefficients.
        ipix : ndarray of int
            RING pixels of the output map (e.g. the padded footprint).
        lmax : int
            Maximum multipole of ``alm``.
        mean : float, optional
            Constant added to the map, e.g. the mean removed before the transform
            (:meth:`_alm_field_mean`). Default is 0.

        Returns
        -------
        hsp.HealSparseMap
        """
        values = hp.alm2map(alm, self.nside, lmax=lmax, pol=False)[ipix] + mean
        hspmap = self._make_hspmap(ipix, values)
        return hspmap

    def compute_maps_from_alms(self,
                               imaps: int | Iterable[int] | None = None,
                               smooth_Mpc: float | None = None,
                               lmax: int | None = None) -> hsp.HealSparseMap | dict[int, hsp.HealSparseMap]:
        """Compute field maps from their a_lm, optionally smoothed.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, uses all maps.
        smooth_Mpc : float or None, optional
            Comoving Gaussian smoothing scale in Mpc (converted to an angle per map,
            see :meth:`_smooth_sigma`). If given, the smoothed maps (pixel window
            deconvolved) are saved (``what=smoothed_field_what``), or read if
            already on disk; if None, the maps are only reconstructed in memory.
        lmax : int or None, optional
            Maximum multipole of the transforms. Default is ``3*nside - 1``.

        Returns
        -------
        hsp.HealSparseMap or dict
            Map(s) on the padded footprint.

        Notes
        -----
        The a_lm are those of the weighted map (:meth:`compute_alms`), so the maps
        are the field (with its mean added back, see :meth:`_alm_field_mean`) on
        the footprint, but tapered in the padding.
        """
        if lmax is None:
            lmax = 3 * self.nside - 1
        _, ipix = self.get_mask(nest=False, return_ipix=True, padded=True)

        maps = {}
        for imap in self.tqdm(imaps, desc="[compute_maps_from_alms]"):
            # smoothed maps are saved, and reused if on disk
            fname = self.get_fname(imap, what=self.smoothed_field_what, smooth_Mpc=smooth_Mpc) if smooth_Mpc is not None else None
            if fname is not None and (existing := self.check_maps(read=True, fname=fname)) is not None:
                maps[imap] = existing
                continue

            alm = self.compute_alms(imap, pol=False, lmax=lmax)
            if smooth_Mpc is not None:
                self._smooth_alms(imap, alm, smooth_Mpc, lmax)
            maps[imap] = self._alm2hspmap(alm, ipix, lmax, mean=self._alm_field_mean(imap))
            if fname is not None:
                self._save_map(maps[imap], fname)

        return maps[imaps] if np.isscalar(imaps) else maps

    @abstractmethod
    def _create_tracers(self) -> list:
        """Create the tracers describing the maps.

        Returns
        -------
        list of pollux.cosmo.Tracer
            One tracer per map.
        """

    @cached_property
    def tracers(self) -> list:
        """Return (and lazily create) the tracers describing the maps.

        Returns
        -------
        list of pollux.cosmo.Tracer
            One tracer per map, from which CCL (:attr:`tracers_ccl`) and SACC
            (:attr:`tracers_sacc`) tracers are derived.
        """
        return self._create_tracers()

    @property
    def tracers_ccl(self) -> list:
        """Return the CCL tracers of the maps.

        Returns
        -------
        list of pyccl.Tracer
        """
        return [t.ccl(self.cosmo) for t in self.tracers]

    @cached_property
    def tracers_sacc(self) -> list:
        """Return the SACC tracers used to store measured spectra.

        Returns
        -------
        list of sacc.tracers.BaseTracer
        """
        return [t.sacc() for t in self.tracers]

    def matter_power_spectrum(self) -> pyccl.Pk2D | None:
        """Return the matter power spectrum consistent with the maps, for predictions.

        Returns
        -------
        pyccl.Pk2D or None
            None (CCL's nonlinear matter power spectrum) by default; overridden
            by builders whose maps follow another one (e.g.
            :class:`~pollux.baryons.BaryonifiedShellBuilder`, or lensing builders
            using their shell builder's).
        """
        return None

    @cached_property
    def theory(self) -> Theory:
        """Return the theory predictor of the maps.

        Returns
        -------
        pollux.cosmo.Theory
            With the builder's cosmology and :meth:`matter_power_spectrum`.
        """
        return Theory(self.cosmo, p_of_k_a=self.matter_power_spectrum())

    @cached_property
    def b(self) -> nmt.NmtBin:
        """Return (and lazily create) the NaMaster binning scheme.

        Returns
        -------
        pymaster.NmtBin
            Bandpower binning used for :meth:`compute_cls` and :meth:`predict_cls`.

        Notes
        -----
        Lazy, so that the builder stays light to pickle: the ``NmtBin`` is not
        constructed until first accessed (unless given at construction).
        """
        return make_nmtbin_logspaced(self.nside, 8, 2 * self.nside, self.nlb, f_ell=self.f_ell)

    @cached_property
    def wsp(self) -> nmt.NmtWorkspace:
        """Return (and lazily create) the NaMaster workspace.

        The workspace stores the mode-coupling matrix and bandpower windows for
        the current mask and binning, and is cached on disk under ``self.wsp_dir``.

        Returns
        -------
        pymaster.NmtWorkspace
            Workspace containing the coupling matrix and bandpower windows.

        Notes
        -----
        Lazy, so that the builder stays light to pickle: the ``NmtWorkspace`` is
        not constructed until first accessed.
        """
        padding = f'_pad{self.padding_deg:g}{self.apotype or ""}' if self.padding_deg is not None else ''
        if self.replicate_octant:
            padding = '_octrep'
        fname = os.path.join(self.wsp_dir, f'wsp_nside{self.nside}{padding}.wsp')
        wsp = nmt.NmtWorkspace()
        if os.path.exists(fname):
            wsp.read_from(fname)
            logger.info("Found and loaded workspace at %s.", fname)
        else:
            mask = self._alm_weights
            b = self.b
            dummy_f = nmt.NmtField(mask, [np.zeros_like(mask)], lite=True)
            wsp.compute_coupling_matrix(dummy_f, dummy_f, b)
            wsp.write_to(fname)
            logger.info("Computed and saved workspace to file %s.", fname)
        return wsp

    def _pairs(self, imaps: int | Iterable[int] | None = None, auto_only: bool = False, pairs: list | None = None) -> tuple[list[int], list[tuple[int, int]]]:
        """Return the map indices and ``(i, j)`` pairs selected by ``imaps``, ``auto_only`` or ``pairs``.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, all maps.
        auto_only : bool, optional
            If True, only auto-spectra.
        pairs : list of tuple of int or None, optional
            Explicit pairs, overriding ``imaps`` and ``auto_only``.

        Returns
        -------
        imaps : list of int
            Maps involved.
        pairs : list of tuple of int
            ``(i, j)`` with ``i >= j`` (unless given explicitly).
        """
        if pairs is not None:
            pairs = [tuple(pair) for pair in pairs]
            return sorted({k for pair in pairs for k in pair}), pairs
        imaps = list(self.iter(imaps))
        if auto_only:
            return imaps, [(i, i) for i in imaps]
        return imaps, [(i1, i2) for i, i2 in enumerate(imaps) for i1 in imaps[i:]]  # with this order, i2 is preserved

    def compute_cls(
            self,
            imaps: int | Iterable[int] | None = None,
            *,
            auto_only: bool = False,
            pairs: list | None = None,
            save: bool = True,
            use_precomputed_alms: bool = True) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
        """Compute binned pseudo-C_ell bandpowers for requested map pairs.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, uses all maps.
        auto_only : bool, optional
            If True, compute only auto-spectra.
        pairs : list of tuple of int or None, optional
            If given, only these ``(i, j)`` map pairs (overrides ``imaps`` and ``auto_only``).
        save : bool, optional
            If True, save spectra to SACC FITS (``what='cls'``).
        use_precomputed_alms : bool, optional
            If True, reads a_lm from disk and uses Healpy :func:`healpy.alm2cl`
            followed by NaMaster decoupling.
            If False, computes NaMaster fields from masked maps and runs full
            MASTER with :func:`pymaster.compute_full_master`.

        Returns
        -------
        ell : ndarray
            Effective multipoles of the bandpowers.
        cl : dict
            Dictionary keyed by (i, j) containing decoupled bandpowers.
        """
        imaps, pairs = self._pairs(imaps, auto_only, pairs)

        cl = {}
        b = self.b
        wsp = self.wsp

        if use_precomputed_alms:
            # pseudo-C_ell from the saved alms (of the weighted maps), decoupled with the shared workspace;
            # pairs come with j fixed and i increasing, so each alm is read at most once per pair change
            previous_pair = (-1, -1)
            for i, j in self.tqdm(pairs, desc="[compute_cls]"):
                if not i == previous_pair[0]:
                    almi_fname = self._alm_fname(i)
                    almi = hp.read_alm(almi_fname)
                    logger.debug("Found and loaded alms at %s.", almi_fname)
                if i == j:
                    almj = almi
                else:
                    if not j == previous_pair[1]:
                        almj_fname = self._alm_fname(j)
                        almj = hp.read_alm(almj_fname)
                        logger.debug("Found and loaded alms at %s.", almj_fname)
                pcl = hp.alm2cl(almi, almj)
                cl[i, j] = wsp.decouple_cell([pcl])[0]
                previous_pair = (i, j)
        else:
            # NaMaster fields of the weighted maps, built when first needed
            fields = {i: None for i in imaps}
            for i, j in self.tqdm(pairs, desc="[compute_cls]"):
                for k in (i, j):
                    if fields[k] is None:
                        weights, _map = self._alm_field(k)
                        fields[k] = nmt.NmtField(weights, [_map])
                cl[i, j] = nmt.compute_full_master(fields[i], fields[j], b, workspace=wsp)[0]

        ell = b.get_effective_ells()

        if save:
            self._save_cls(ell, cl)

        return ell, cl

    def _save_cls(self, ell: np.ndarray, cl: dict[tuple[int, int], np.ndarray]) -> str:
        """Save angular power spectra to a SACC FITS file.

        Parameters
        ----------
        ell : ndarray
            Effective multipoles (typically bandpower effective ells).
        cl : dict
            Dictionary keyed by (i, j) containing spectra arrays evaluated at
            ``ell``.

        Returns
        -------
        str
            Path to the saved SACC FITS file.
        """
        s = sacc.Sacc()
        tracers = self.tracers_sacc
        for t in tracers:
            s.add_tracer_object(t)
        for i, j in cl.keys():
            for _ell, _cl in zip(ell, cl[i, j], strict=True):
                s.add_data_point(self.sacc_type, [tracers[i].name, tracers[j].name], _cl, ell=_ell)
        fname = self.get_fname(what='cls', format='fits')
        s.save_fits(fname, overwrite=True)
        logger.debug("Saved to file %s.", fname)
        return fname

    def load_cls(self, return_dict: bool = False) -> sacc.Sacc | tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
        """Load saved angular power spectra.

        Parameters
        ----------
        return_dict : bool, optional
            If True, return a dictionary keyed by (i, j). Otherwise, return the
            SACC object.

        Returns
        -------
        sacc.Sacc or (ell, dict)
            If ``return_dict=False``, returns the loaded SACC object.
            If ``return_dict=True``, returns ``(ell, cls_dict)`` where ``ell`` is
            the ell array for each spectrum and ``cls_dict`` is keyed by (i, j).

        Raises
        ------
        FileNotFoundError
            If the SACC file does not exist.
        """
        fname = self.get_fname(what='cls', format='fits')
        s = sacc.Sacc.load_fits(fname)
        logger.debug("Found and loaded file at %s.", fname)
        if return_dict:
            tracers = list(s.tracers.keys())
            tc = s.get_tracer_combinations(self.sacc_type)
            cls_out = {}
            for i in range(self.nmaps):
                for j in range(self.nmaps):
                    if (tracers[i], tracers[j]) in tc:
                        ell, cls_out[i, j] = s.get_ell_cl(self.sacc_type, tracers[i], tracers[j])
            return ell, cls_out
        else:
            return s

    def _noise_cls(self, imap: int) -> float | np.ndarray:
        """Return the noise angular power spectrum for one map.

        Optional: only needed when calling :meth:`predict_cls` with
        ``error_bars=True``. Subclasses that don't model
        noise can leave this unimplemented; it raises ``NotImplementedError``
        if called.

        Parameters
        ----------
        imap : int
            Map index.

        Returns
        -------
        float or ndarray
            Noise power spectrum. If float, interpreted as white noise.

        Raises
        ------
        NotImplementedError
            If this builder does not implement a noise model.
        """
        raise NotImplementedError

    def predict_cls(
            self,
            imaps: int | Iterable[int] | None = None,
            *,
            auto_only: bool = False,
            pairs: list | None = None,
            binning: str = 'wsp',
            error_bars: bool = False,
            keep_cls: bool = False,
            stacked: bool = False,
            ell: np.ndarray | None = None) -> tuple:
        """Predict theory angular power spectra using CCL.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, uses all maps.
        auto_only : bool, optional
            If True, compute only auto-spectra.
        pairs : list of tuple of int or None, optional
            If given, only these ``(i, j)`` map pairs (overrides ``imaps`` and ``auto_only``).
        binning : {'wsp', 'average', 'unbinned'}, optional
            ``'wsp'`` (default): bandpowers binned with the bandpower windows of
            :attr:`wsp`, like the measured spectra. ``'average'``: bandpowers
            averaged over each bandpower. ``'unbinned'``: spectra at ``ell``.
        error_bars : bool, optional
            If True, compute Gaussian (diagonal-in-ell) error bars, with
            ``C_ell + N_ell`` for auto-spectra (noise from :meth:`_noise_cls`).
        keep_cls : bool, optional
            If True, cache the results on ``self.predicted_cls``.
        stacked : bool, optional
            If True, also return spectra stacked into a single vector (and, when
            ``error_bars=True``, a diagonal covariance matrix).
        ell : ndarray or None, optional
            Multipoles at which to evaluate CCL when ``binning='unbinned'``.
            If None, uses the effective bandpower ells from the binning scheme.

        Returns
        -------
        ell : ndarray
            Ell values (effective bandpower ells or explicit ``ell`` input).
        cl : dict
            Dictionary keyed by (i, j) with spectra arrays.
        dcl : dict
            Only with ``error_bars``: Gaussian error bars, keyed like ``cl``.
        cl_stacked : ndarray
            Only with ``stacked``: spectra stacked into one vector.
        cov : ndarray
            Only with ``stacked`` and ``error_bars``: covariance of ``cl_stacked``
            (diagonal in ell).

        Notes
        -----
        Wraps :meth:`pollux.cosmo.Theory.predict_cls` with the builder's tracers,
        matter power spectrum, workspace bandpower windows, pixel window
        (``f_ell``), noise model (:meth:`_noise_cls`) and sky fraction.
        """
        # cached prediction (see keep_cls)
        if self.predicted_cls is not None:
            return self.predicted_cls

        imaps, pairs = self._pairs(imaps, auto_only, pairs)

        if ell is not None:
            if binning != 'unbinned':
                raise ValueError("ell can only be given with binning='unbinned'")
            if error_bars:
                raise ValueError("ell cannot be given with error_bars")

        # bandpower windows of the workspace, and the field window (pixel window, or a given one above the first bandpower)
        bpws, f_ell = None, None
        if binning == 'wsp':
            bpws = self.wsp.get_bandpower_windows()
        if binning != 'unbinned':
            if self.f_ell is not None:
                if self.f_ell == 'pixwin':
                    f_ell = hp.pixwin(self.nside)[:self.b.lmax + 1]
                else:
                    f_ell = np.ones(self.b.lmax + 1)
                    f_ell[self.b.get_ell_list(0)[0]:] = self.f_ell

        # map tracers, with their noise for the error bars
        tracers = self.tracers
        noise = {tracers[i].name: self._noise_cls(i) for i in imaps} if error_bars else None
        res = self.theory.predict_cls({(i, j): (tracers[i], tracers[j]) for i, j in pairs},
                                      b=self.b,
                                      ell=ell,
                                      binning=binning,
                                      bpws=bpws,
                                      f_ell=f_ell,
                                      noise=noise,
                                      error_bars=error_bars,
                                      stacked=stacked,
                                      fsky=self.fsky,
                                      cross_covariance=not auto_only,
                                      progress=self.tqdm)
        if keep_cls:
            self.predicted_cls = res
        return res

    def reference_scales(self, smooth_Mpc: float | None = None) -> dict[str, np.ndarray]:
        """Return the angular scales of the simulation box, and of a smoothing, per map.

        Parameters
        ----------
        smooth_Mpc : float or None, optional
            Comoving Gaussian smoothing scale in Mpc.

        Returns
        -------
        dict
            ``theta_box`` (arcmin) and ``ell_box``, and ``theta_smooth``, ``ell_smooth``
            with ``smooth_Mpc``, one value per map, at the redshift where its signal comes
            from (:meth:`pollux.cosmo.Tracer.mean_z_kernel`, e.g. the lenses for lensing maps).
        """
        z_kernel = np.array([t.mean_z_kernel(self.cosmo) for t in self.tracers])
        d_a = pyccl.comoving_angular_distance(self.cosmo, 1. / (1. + z_kernel))
        rad2arcmin = 180. * 60. / np.pi
        ell_box = 2. * np.pi / self.reader.boxsize * d_a
        scales = dict(theta_box=rad2arcmin * np.pi / ell_box, ell_box=ell_box)
        if smooth_Mpc is not None:
            theta_smooth = smooth_Mpc / d_a
            scales.update(theta_smooth=rad2arcmin * theta_smooth, ell_smooth=np.pi / theta_smooth)
        return scales

    def plot_cls(
            self,
            plot_dir: str | None = None,
            *,
            modes: tuple[str, ...] = ('values', 'ratio'),
            ell_factor: int = 1,
            logy: bool = False,
            error_bars: bool = False,
            **predict_kwargs: Any) -> None:
        """Plot the saved angular power spectra against their prediction (:meth:`predict_cls`), one panel per map pair.

        Parameters
        ----------
        plot_dir : str or None, optional
            Output directory. Default is ``{save_dir}/plots``.
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
            Options of :meth:`predict_cls`, e.g. ``auto_only``, ``pairs`` or ``binning``: the
            predicted pairs are plotted.

        Notes
        -----
        Saves ``{base_fname}_cls``, with suffix ``_{mode}`` for modes other than ``'values'``.
        """
        from .plotting import plot_pairs
        plot_dir = plot_dir or os.path.join(self.save_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        need_errors = error_bars or 'diff_over_error' in modes
        ell, cl_th, *dcl = self.predict_cls(error_bars=need_errors, **predict_kwargs)
        dcl = dcl[0] if need_errors else None
        cl = self.load_cls(return_dict=True)[1]
        for mode in modes:
            plot_pairs(
                ell,
                data={ij: cl[ij] for ij in cl_th if ij in cl},
                theory=cl_th,
                theory_errors=dcl if (error_bars or mode == 'diff_over_error') else None,
                mode=mode,
                auto_only=all(i == j for i, j in cl_th),
                ell_factor=ell_factor,
                scales=self.reference_scales(),
                color_theory='orange',
                logy=logy,
                title=f'{self.base_fname} {{i}} {{j}}',
                sharey=False if mode == 'values' else 'all',
                fname=os.path.join(plot_dir, f'{self.base_fname}_cls' + ('' if mode == 'values' else f'_{mode}') + '.png'))

    def compute_map_moments(self, theta_smooth: np.ndarray | Iterable[float], nside_small: int = 512, save: bool = True) -> tuple[np.ndarray, dict, dict, dict]:
        """Compute 2nd/3rd/4th moments of smoothed maps.

        Parameters
        ----------
        theta_smooth : array_like of float
            Gaussian smoothing scales in arcmin.
        nside_small : int, optional
            Nside used for the smoothed maps and moment evaluation.
        save : bool, optional
            If True, saves results to ``what='2nd_3rd_4th_moments', format='npy'``.

        Returns
        -------
        theta_smooth : ndarray
            Smoothing scales (arcmin).
        second_moment : dict
            Covariance matrices keyed by smoothing scale (theta -> (nmaps,nmaps)).
        third_moment : dict
            Third-order moments keyed by smoothing scale
            (theta -> (nmaps,nmaps,nmaps)).
        fourth_moment : dict
            Fourth-order moments keyed by smoothing scale
            (theta -> (nmaps,nmaps,nmaps,nmaps)).

        Notes
        -----
        This method:
        1) loads a_lm for each map,
        2) resizes them to ``lmax_out = 2*nside_small``,
        3) makes smoothed maps for each ``theta``,
        4) computes raw moments on the masked pixels,
        5) saves results to ``what='2nd_3rd_4th_moments', format='npy'`` if ``save=True``.
        """
        if min(theta_smooth) <= hp.nside2resol(nside_small, arcmin=True):
            raise ValueError('Smoothing scale should be larger than pixel size to avoid pixel effects.')

        kappa = {}
        lmax_out = 2 * nside_small

        # Mask (downgraded from self.nside to nside_small for moment computation)
        mask_full = self._create_mask()
        mask = hp.ud_grade(mask_full, nside_small, order_in='NESTED', order_out='NESTED') > 0.5
        ipix = np.where(mask)[0]

        # First, let's compute kappa maps for all bins and all theta (smoothing scales)
        for imap in range(self.nmaps):
            kappa[imap] = {}

            # read the alm's
            alm = self.compute_alms(imaps=imap)[imap]
            lmax = hp.Alm.getlmax(len(alm))
            alm = hp.sphtfunc.resize_alm(alm, lmax=lmax, mmax=lmax, lmax_out=lmax_out, mmax_out=lmax_out)

            # smooth maps for all theta
            for theta in theta_smooth:
                kappa_map = hp.alm2map(alm, nside_small, sigma=theta * np.pi / (180. * 60), lmax=lmax_out)
                kappa_map = hp.pixelfunc.reorder(kappa_map, r2n=True)
                kappa_map[~mask] = hp.UNSEEN
                kappa_map[mask] -= np.mean(kappa_map[mask])  # Remove mean for moments calculation
                kappa[imap][theta] = kappa_map

        moments = {}
        moments['theta_smooth'] = theta_smooth
        moments['second_moment'] = {}
        moments['third_moment'] = {}
        moments['fourth_moment'] = {}

        # Now compute moments
        for theta in theta_smooth:
            # Stack data for all maps at given theta
            data = np.stack([kappa[imap][theta][ipix] for imap in range(self.nmaps)], axis=0)

            # 2nd moment
            moments['second_moment'][theta] = np.cov(data)

            # 3rd moment
            moments['third_moment'][theta] = np.zeros((self.nmaps, self.nmaps, self.nmaps))
            for i, j, k in product(range(self.nmaps), repeat=3):
                moments['third_moment'][theta][i, j, k] = np.mean(data[i] * data[j] * data[k])

            # 4th moment
            moments['fourth_moment'][theta] = np.zeros((self.nmaps, self.nmaps, self.nmaps, self.nmaps))
            for i, j, k, l in product(range(self.nmaps), repeat=4):
                moments['fourth_moment'][theta][i, j, k, l] = np.mean(data[i] * data[j] * data[k] * data[l])

        if save:
            fname = self.get_fname(what='2nd_3rd_4th_moments', format='npy')
            np.save(fname, moments)
            logger.info("Saved to file %s.", fname)

        return theta_smooth, moments['second_moment'], moments['third_moment'], moments['fourth_moment']

    def load_map_moments(self) -> tuple[np.ndarray, dict, dict, dict]:
        """Load previously saved 2nd/3rd/4th moments of smoothed maps.

        Returns
        -------
        theta_smooth : ndarray
            Smoothing scales (arcmin).
        second_moment : dict
            Covariance matrices keyed by smoothing scale (theta -> (nmaps,nmaps)).
        third_moment : dict
            Third-order moments keyed by smoothing scale
            (theta -> (nmaps,nmaps,nmaps)).
        fourth_moment : dict
            Fourth-order moments keyed by smoothing scale
            (theta -> (nmaps,nmaps,nmaps,nmaps)).

        Raises
        ------
        FileNotFoundError
            If the moments file does not exist.
        """
        fname = self.get_fname(what='2nd_3rd_4th_moments', format='npy')
        moments = np.load(fname, allow_pickle=True).item()
        logger.debug("Found and loaded file at %s.", fname)
        return moments['theta_smooth'], moments['second_moment'], moments['third_moment'], moments['fourth_moment']

    def generate_gaussian_maps(self, imaps: int | Iterable[int] | None = None, *, auto_only: bool = False, seed: int = -1) -> None:
        """Generate Gaussian random maps consistent with predicted spectra.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to generate. If None, generates all maps.
        auto_only : bool, optional
            If True, generate each map independently using its auto-spectrum.
            If False, generate a correlated set of maps from all cross-spectra.
        seed : int, optional
            Random seed passed to NaMaster synfast routines. Default is -1.

        Notes
        -----
        Maps are generated full-sky and then restricted to the padded footprint
        (like the maps they stand for) before writing as HealSparse maps.
        """
        imaps = list(self.iter(imaps))

        # Compute theory Cl's
        _, cls = self.predict_cls(binning='unbinned', auto_only=auto_only, ell=np.arange(3 * self.nside))

        _, ipix = self.get_mask(nest=False, return_ipix=True, padded=True)

        if auto_only:
            for imap in self.tqdm(imaps, desc="[generate_gaussian_maps]"):
                # Generate full-sky Gaussian map
                out = nmt.utils.synfast_spherical(self.nside, [cls[imap, imap]], spin_arr=[0], seed=seed)[0]

                # Store in HealSparse map
                hspmap = self._make_hspmap(ipix, out[ipix])

                # Save map
                fname = self.get_fname(imap)
                self._save_map(hspmap, fname)

        else:
            # Generate full-sky Gaussian map
            pairs = [(i1, i2) for i, i2 in enumerate(imaps) for i1 in imaps[i:]]
            outs = nmt.utils.synfast_spherical(self.nside, [cls[p] for p in pairs], spin_arr=[0] * len(imaps), seed=seed)

            for imap, out in zip(imaps, outs, strict=True):
                # Store in HealSparse map
                hspmap = self._make_hspmap(ipix, out[ipix])

                # Save map
                fname = self.get_fname(imap)
                self._save_map(hspmap, fname)


class MapSum:
    """Weighted sum of HealSparse maps over a footprint, accumulated in place.

    Maps are added one at a time into the output's own sparse array, so only one
    input map needs to be in memory at a time (``hsp.operations.sum_union`` needs
    all maps at once, or rebuilds the output for every pairwise sum).

    Parameters
    ----------
    mask : ndarray of bool
        Dense NESTED footprint mask at the maps' ``nside_sparse``.
    nside_coverage : int
        Coverage nside of the output map.
    dtype : data-type
        dtype of the output map.

    Notes
    -----
    Every footprint pixel of the output is valid, starting at zero. With coverage
    built from sorted coverage pixels, block ``k`` of the sparse array (after the
    leading sentinel block) holds the NESTED pixels of coverage pixel ``cov_pix[k]``.
    """

    def __init__(self, mask: np.ndarray, nside_coverage: int, dtype: np.dtype) -> None:
        self.nside = hp.npix2nside(len(mask))
        self.nfine = (self.nside // nside_coverage)**2
        mask_blocks = mask.reshape(-1, self.nfine)
        cov_pix = np.flatnonzero(mask_blocks.any(axis=1))
        self.cov = hsp.HealSparseCoverage.make_from_pixels(nside_coverage, self.nside, cov_pix)
        self.cov_index = self.cov[:]
        self.sentinel = hsp.utils.check_sentinel(dtype, None)
        self.sparse = np.full((len(cov_pix) + 1) * self.nfine, self.sentinel, dtype=dtype)
        self.sparse[self.nfine:].reshape(-1, self.nfine)[mask_blocks[cov_pix]] = 0

    def add(self, hspmap: hsp.HealSparseMap, scale: float = 1., offset: float = 0.) -> None:
        """Add ``scale * hspmap + offset`` on the valid pixels of ``hspmap``.

        Parameters
        ----------
        hspmap : hsp.HealSparseMap
            Map to add, with the same ``nside_sparse`` and ``nside_coverage`` and
            valid pixels inside the footprint.
        scale : float, optional
            Factor applied to the map values.
        offset : float, optional
            Constant added on each valid pixel of ``hspmap``.
        """
        src_cov = np.flatnonzero(hspmap.coverage_mask)
        src_cov = src_cov[self.cov.coverage_mask[src_cov]]  # inputs lie in the footprint, so this drops nothing
        _add_blocks(self.sparse, self.cov_index, hspmap._sparse_map, hspmap._cov_map[:], src_cov, self.nfine, hspmap._sentinel, scale, offset)

    def to_map(self) -> hsp.HealSparseMap:
        """Return the sum as a HealSparse map (sharing memory with this accumulator).

        Returns
        -------
        hsp.HealSparseMap
            Sum of the added maps, with every footprint pixel valid.
        """
        return hsp.HealSparseMap(cov_map=self.cov, sparse_map=self.sparse, nside_sparse=self.nside, sentinel=self.sentinel)


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
