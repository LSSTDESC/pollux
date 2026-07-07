from __future__ import annotations

import os
from typing import Iterable

import healpy as hp
import healsparse as hsp
import skyproj
import matplotlib.pyplot as plt
import numpy as np
import pyccl
import pymaster as nmt
import sacc
from itertools import product
from tqdm.auto import tqdm, trange

from .utils import *


class MapBuilder:
    """Base class to build, store, and analyze Healpix/HealSparse maps.

    This class centralizes:
    - file naming and I/O for HealSparse maps,
    - mask/footprint creation,
    - alm computation (Healpy),
    - pseudo-C_ell and decoupled bandpowers (NaMaster),
    - SACC export and plotting.

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
        Directory used to cache NaMaster workspaces. Defaults to ``save_dir``.
    sacc_type : str or None, optional
        SACC data type name used when saving spectra.
    b : pymaster.NmtBin, optional
        NaMaster binning scheme. If None, it is constructed lazily.
    nlb : int, optional
        Number of bandpowers when constructing bins lazily. Default is 20.
    remove_monopole : bool, optional
        If True, subtract the mean within the mask before harmonic transforms.
    nocompress : bool, optional
        If True, disable HealSparse compression. Default is True.
    f_ell : {'pixwin'} or ndarray or None, optional
        Multiplicative factor applied to theory spectra before binning (e.g.
        pixel window). If 'pixwin', uses the Healpix pixel window. Default is
        'pixwin'.
    
    Notes
    -----
    Healpix ordering: this codebase uses NESTED pixel ordering by default for
    internal Healpix/HealSparse operations and IO unless otherwise stated.
    """

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
    ) -> None:
        """Initialize derived attributes and set up directories."""
        if isinstance(mask, np.ndarray) and len(mask) != hp.nside2npix(nside):
            raise ValueError(
                f"mask has {len(mask)} pixels but nside={nside} requires {hp.nside2npix(nside)}.")
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
        self.b = b
        self.nlb = nlb
        self.remove_monopole = remove_monopole
        self.nocompress = nocompress
        self.f_ell = f_ell
        
        # Derived attributes
        self.npix = hp.nside2npix(self.nside)
        self._nlb_cached = self.nlb if self.b is None else None
        
        # Cached attributes (lazy-initialized)
        self.tracers = None
        self.tracers_sacc = None
        self.mask = None
        self.ipix = None
        self.mask_ring = None
        self.ipix_ring = None
        self.ipix_oct_fullsky = None
        self.wsp = None
        self.predicted_cls = None
        
        # Make sure folder exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"*** Map builder ({self.base_fname}) ***")
        print(f"  - Builder class = {self.__class__.__name__}")
        print(f"  - Number of maps = {self.nmaps}")
        print(f"  - Maps saved to", self.save_dir)

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
        match it:
            case None:
                return range(self.nmaps)
            case int():
                return [it]
            case _:
                return it
        
    def tqdm(self, it: Iterable[int] | int | None = None, **tqdm_kwargs) -> trange | tqdm | list[int]:
        """Wrap an iterable of ints (or default range) with tqdm progress bars.

        Parameters
        ----------
        it : Iterable[int] or int or None, optional
            - If None: iterate over ``range(self.nmaps)`` using :func:`trange`.
            - If Iterable[int]: wrap with :func:`tqdm`.
            - If int: return a singleton list ``[it]``.

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
        match it:
            case None:
                return trange(self.nmaps, **tqdm_kwargs)
            case int():
                return [it]
            case _:
                return tqdm(it, **tqdm_kwargs)

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

    def _create_tracers_ccl(self) -> list:
        """Create CCL tracer objects.

        Returns
        -------
        list
            List of tracer objects with length ``self.nmaps``.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_tracers_ccl(self) -> list:
        """Return (and cache) CCL tracer objects used for theory predictions.

        Returns
        -------
        list
            List of tracer objects with length ``self.nmaps``.
        """
        if self.tracers is None:
            self.tracers = self._create_tracers_ccl()
        return self.tracers

    def _create_tracers_sacc(self) -> list:
        """Create SACC tracer objects.

        Returns
        -------
        list
            List of SACC tracer objects with length ``self.nmaps``.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_tracers_sacc(self) -> list:
        """Return (and cache) SACC tracers used to store measured spectra.

        Returns
        -------
        list
            List of SACC tracer objects with length ``self.nmaps``.
        """
        if self.tracers_sacc is None:
            self.tracers_sacc = self._create_tracers_sacc()
        return self.tracers_sacc

    def construct_maps(
        self,
        imaps: int | Iterable[int] | None = None,
        **kwargs
    ) -> hsp.HealSparseMap | dict[int, hsp.HealSparseMap]:
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

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def check_maps(self, imaps: int | Iterable[int] | None = None, read: bool = False, fname: str | None = None, check_finite: bool = False) -> dict[int, bool | hsp.HealSparseMap | None] | bool | hsp.HealSparseMap | None:
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
        assert (imaps is None) or (fname is None)

        def check_one(filename):
            if os.path.exists(filename):
                if read:
                    try:
                        m = hsp.HealSparseMap.read(filename, nside_coverage=self.nside_c)
                        if check_finite:
                            a = m[m.valid_pixels]
                            if not (np.all(np.isfinite(a)) and np.any(a)):
                                return None
                        return m
                    except Exception as e:
                        print(f"Cannot read map at {filename}: {e}")
                        return None
                else:
                    return True
            else:
                if not read:
                    print(f"Cannot find map at {filename}.")
                return None if read else False

        if fname is not None:
            return check_one(fname)

        out = {}
        for imap in self.tqdm(imaps, desc="[check_maps]"):
            fname = self.get_fname(imap)
            out[imap] = check_one(fname)

        return out[imaps] if type(imaps) is int else out

    def load_maps(self, imaps: int | Iterable[int] | None = None, **fname_kwargs) -> hsp.HealSparseMap | dict[int, hsp.HealSparseMap]:
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
        maps = {}
        for imap in self.tqdm(imaps, desc="[load_maps]"):
            fname = self.get_fname(imap, **fname_kwargs)
            maps[imap] = hsp.HealSparseMap.read(fname, nside_coverage=self.nside_c)

        return maps[imaps] if type(imaps) is int else maps
    
    def plot_maps(self, maps_or_imaps: int | Iterable[int] | Iterable | None = None, nx: int | None = None, log: bool = False, title_prefix: str | None = None, cbar: bool = True, save: bool = True, backend: str = 'skyproj', **kwargs) -> None:
        """Plot one or multiple maps.

        Parameters
        ----------
        maps_or_imaps : int or iterable or None, optional
            - If None: loads and plots all maps.
            - If int: loads and plots that map index.
            - If iterable of int: loads and plots those indices.
            - If iterable of maps: plots provided maps directly.
        nx : int or None, optional
            Number of columns in the subplot grid. Defaults to ``nmaps``.
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
        assert backend in ('healpy', 'skyproj')

        match maps_or_imaps:
            case None:
                nmaps = self.nmaps
                imaps = range(nmaps)
                maps = None
            case int():
                nmaps = 1
                imaps = maps_or_imaps
                maps = None
            case _:
                maps_or_imaps = list(maps_or_imaps)
                nmaps = len(maps_or_imaps)
                assert nmaps > 0
                match maps_or_imaps[0]:
                    case int():
                        imaps = maps_or_imaps
                        maps = None
                    case _:
                        imaps = range(nmaps)
                        maps = maps_or_imaps

        if nx is None:
            nx = nmaps
        ny = int(np.ceil(nmaps / nx))

        if backend == 'skyproj':
            fig, axes = plt.subplots(ny, nx, figsize=(8 * nx, 5 * ny), squeeze=False)
            axes = axes.flat
        else:
            plt.figure(figsize=(4 * nx, 4 * ny))

        sub = 1
        for imap in self.tqdm(imaps):
            if maps is None:
                hp_map = self.load_maps(imap)
            else:
                hp_map = maps[imap]
            title = title_prefix + f" {imap}" if title_prefix is not None else None

            if backend == 'skyproj':
                ax = next(axes)
                if self.octant_only or self.footprint is None:
                    sp = skyproj.MollweideSkyproj(ax=ax)
                else:
                    sp = skyproj.McBrydeSkyproj(ax=ax)

                draw_kwargs = dict(kwargs)
                if log:
                    if isinstance(hp_map, hsp.HealSparseMap):
                        values = hp_map[hp_map.valid_pixels]
                    else:
                        values = hp_map[hp_map != hp.UNSEEN]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        log_finite = np.all(np.isfinite(np.log(values)))
                    draw_kwargs['norm'] = 'log' if log_finite else 'symlog'

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
            plt.savefig(fname, bbox_inches='tight', dpi=150, transparent=False)

    def _create_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """Create a footprint mask and list of valid pixels at ``self.nside``.

        Returns
        -------
        mask : ndarray of bool
            Dense mask array in NESTED ordering, length ``self.npix``.
        ipix : ndarray of int
            Pixel indices where mask is non-zero, in NESTED ordering.
        """
        if isinstance(self.footprint, str) and self.footprint == 'octant':
            mask = make_mask_radec_cuts(self.nside, 0, 90, 0, 90, eps=1e-10, nest=True, dtype=bool)
            ipix = np.where(mask)[0]
        elif isinstance(self.footprint, str):
            # path to a NESTED-ordering FITS mask
            mask, fits_header = hp.read_map(self.footprint, nest=True, h=True)
            ordering = dict(fits_header)['ORDERING']
            if ordering.upper() != 'NESTED':
                raise ValueError(f"mask at {self.footprint} has ORDERING={ordering!r} in its FITS header, expected 'NESTED'.")
            mask = mask.astype(bool)
            if len(mask) != self.npix:
                raise ValueError(
                    f"mask at {self.footprint} has {len(mask)} pixels but nside={self.nside} requires {self.npix}.")
            ipix = np.where(mask)[0]
        elif isinstance(self.footprint, np.ndarray):
            mask = self.footprint.astype(bool)
            ipix = np.where(mask)[0]
        else:
            mask = np.ones(self.npix, dtype=bool)
            ipix = np.arange(self.npix)

        return mask, ipix

    def get_mask(self, nest: bool = True, return_ipix: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return (and cache) the analysis mask.

        Parameters
        ----------
        nest : bool, optional
            If True, return mask in NESTED ordering. If False, return RING.
        return_ipix : bool, optional
            If True, also return the valid pixel indices.

        Returns
        -------
        mask : ndarray of bool
            Dense mask array at ``self.nside`` in the requested ordering.
        ipix : ndarray of int, optional
            Valid pixel indices (only if ``return_ipix=True``).

        Notes
        -----
        The mask is cached on first use as ``self.mask`` / ``self.ipix``
        (NESTED) and ``self.mask_ring`` / ``self.ipix_ring`` (RING). The two
        orderings are built independently and lazily: requesting ``nest=True``
        never triggers the (memory-heavy, at high nside) RING reorder, and
        vice versa. The mask is stored as ``bool`` rather than ``float`` to
        keep the cached footprint 8x smaller; callers that need a numeric
        mask (e.g. for NaMaster) should cast it explicitly.
        """
        if nest:
            if self.mask is None:
                self.mask, self.ipix = self._create_mask()
            if return_ipix:
                return self.mask, self.ipix
            else:
                return self.mask
        else:
            if self.mask_ring is None:
                if self.mask is None:
                    self.mask, self.ipix = self._create_mask()
                self.mask_ring = hp.reorder(self.mask, n2r=True)
                self.ipix_ring = hp.nest2ring(self.nside, self.ipix)
            if return_ipix:
                return self.mask_ring, self.ipix_ring
            else:
                return self.mask_ring

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

    def get_octant_to_fullsky_ipix(self, nest: bool = True) -> np.ndarray:
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
        AssertionError
            If ``self.octant_only`` is False.
        """
        assert self.octant_only

        if self.ipix_oct_fullsky is None:
            # geometric mapping: reflect southern hemisphere and reflect phi into first quadrant
            # (triangle wave, continuous across quadrant boundaries)
            theta, phi = hp.pix2ang(self.nside, np.arange(self.npix), nest=nest)
            ipix_mapped = hp.ang2pix(
                self.nside,
                np.where(theta <= np.pi / 2., theta, np.pi - theta),
                np.pi / 2. - np.abs((phi % np.pi) - np.pi / 2.),
                nest=nest)

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

            self.ipix_oct_fullsky = ipix_mapped

        return self.ipix_oct_fullsky

    def get_cls_b(self) -> 'nmt.NmtBin':
        """Return (and lazily create) the NaMaster binning scheme.

        Returns
        -------
        pymaster.NmtBin
            Bandpower binning used for :meth:`compute_cls` and :meth:`predict_cls`.

        Notes
        -----
        This is implemented as a lazy getter to keep the builder pickle/serialize
        friendly.
        """
        if self.b is None:
            self.b = make_nmtbin_logspaced(self.nside, 8, 2 * self.nside, self._nlb_cached or self.nlb, f_ell=self.f_ell)
        return self.b

    def get_cls_wsp(self) -> 'nmt.NmtWorkspace':
        """Return (and lazily create) the NaMaster workspace.

        The workspace stores the mode-coupling matrix and bandpower windows for
        the current mask and binning, and is cached on disk under ``self.wsp_dir``.

        Returns
        -------
        pymaster.NmtWorkspace
            Workspace containing the coupling matrix and bandpower windows.
        """

        if self.wsp is not None:
            return self.wsp
        else:
            fname = os.path.join(self.wsp_dir, f'wsp_nside{self.nside}.wsp')
            wsp = nmt.NmtWorkspace()
            if os.path.exists(fname):
                print("Loading workspace in", fname)
                wsp.read_from(fname)
            else:
                print("Computing and saving workspace in", fname)
                mask = self.get_mask(nest=False).astype(np.float64)
                b = self.get_cls_b()
                dummy_f = nmt.NmtField(mask, [np.zeros_like(mask)], lite=True)
                wsp.compute_coupling_matrix(dummy_f, dummy_f, b)
                wsp.write_to(fname)
            self.wsp = wsp
            return self.wsp

    def _field_map(self, imap: int, nest: bool = True) -> np.ndarray:
        """Load a map and return a dense Healpix array.

        Parameters
        ----------
        imap : int
            Map index to load.
        nest : bool, optional
            Output ordering.

        Returns
        -------
        ndarray
            Dense Healpix map with UNSEEN outside the HealSparse valid region.
        """
        return self.load_maps(imap).generate_healpix_map(nest=nest)

    def get_field_map(self, imap: int, nest: bool = True, replicate_octant: bool = False) -> np.ndarray:
        """Return a dense Healpix map suitable for harmonic transforms.

        Parameters
        ----------
        imap : int
            Map index.
        nest : bool, optional
            Ordering of the returned map.
        replicate_octant : bool, optional
            If True, replicate the octant footprint to a full-sky map before
            returning (useful for transforms assuming full sky). Requires
            ``footprint='octant'``.

        Returns
        -------
        ndarray
            Dense Healpix map of length ``hp.nside2npix(self.nside)``.

        Notes
        -----
        If ``self.remove_monopole`` is True, subtracts the mean of the map over
        the masked pixels prior to replication.
        """
        assert not replicate_octant or self.octant_only, "replicate_octant requires footprint='octant'"

        _map = self._field_map(imap, nest=nest)

        if self.remove_monopole:
            _, ipix = self.get_mask(nest=nest, return_ipix=True)
            _map[ipix] -= np.mean(_map[ipix])

        # Replicate if needed and apply mask
        if replicate_octant:
            ipix_o2f = self.get_octant_to_fullsky_ipix(nest=nest)
            _map = _map[ipix_o2f]

        return _map

    def compute_alms(self, imaps: int | Iterable[int] | None = None, save: bool = True, replicate_octant: bool = False, **map2am_kwargs) -> np.ndarray | dict[int, np.ndarray]:
        """Compute spherical harmonic coefficients a_lm from maps.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, uses all maps.
        save : bool, optional
            If True, saves a_lm to FITS files named by ``what='alms'``.
        replicate_octant : bool, optional
            If True, compute a_lm from the octant replicated to full sky.
            Requires ``footprint='octant'``. Cannot be combined with
            ``save=True`` in the current implementation.
        **map2am_kwargs
            Forwarded to :func:`healpy.map2alm`.

        Returns
        -------
        dict
            Mapping ``imap -> alm`` (complex ndarray).
        """
        assert not replicate_octant or self.octant_only, "replicate_octant requires footprint='octant'"
        assert not (save and replicate_octant)

        alms = {}
        for imap in self.tqdm(imaps, desc="[compute_alms]"):
            if save:
                fname = self.get_fname(imap, what='alms', format='fits')
                if os.path.exists(fname):
                    try:
                        alms[imap] = hp.read_alm(fname)
                        continue
                    except Exception as e:
                        print(f"Cannot read alm at {fname}: {e}")
                        pass

            mask = self.get_mask(nest=False)
            _map = self.get_field_map(imap, nest=False, replicate_octant=replicate_octant)

            alms[imap] = hp.map2alm(_map * mask, **map2am_kwargs)

            if save:
                hp.write_alm(fname, alms[imap], overwrite=True)

        return alms[imaps] if type(imaps) is int else alms

    def _smooth_alms(self, alm, sigma, ipix, lmax):
        """Smooth alm in-place and return a HealSparse map over the footprint.

        The pixel window function is deconvolved so that *sigma* is the sole
        effective smoothing in the output map — pixel smoothing is removed.
        ``alm`` is modified in-place and can be reused for further transforms
        (e.g. spin-2) after this call.

        Parameters
        ----------
        alm : array_like
            Spherical harmonic coefficients.  Modified in-place.
        sigma : float
            Gaussian smoothing width in radians (typically ``smooth_Mpc / chi``).
            This is the *only* smoothing present in the output.
        ipix : array_like of int
            Ring-ordered HEALPix pixel indices of the survey footprint.
        lmax : int
            Maximum multipole used for the back-transform ``alm2map``.

        Returns
        -------
        hsp.HealSparseMap
            Smoothed map restricted to the survey footprint.
        """
        ell = np.arange(lmax + 1)
        fl = np.exp(-0.5 * (ell * (ell + 1)) * sigma ** 2) / hp.pixwin(self.nside, pol=False, lmax=lmax)
        hp.almxfl(alm, fl, inplace=True)
        values = hp.alm2map(alm, self.nside, lmax=lmax)[ipix]
        hsmap = hsp.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
        hsmap.update_values_pix(ipix, values.astype(self.dtype), nest=False)
        return hsmap

    def compute_maps_from_alms(self, imaps: int | Iterable[int] | None = None, save: bool = True) -> hsp.HealSparseMap | dict[int, hsp.HealSparseMap]:
        """Reconstruct HealSparse maps from saved a_lm coefficients.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to reconstruct. If None, uses all maps.
        save : bool, optional
            If True, saves reconstructed maps to the default map filenames.

        Returns
        -------
        list of hsp.HealSparseMap
            Reconstructed maps, one per requested ``imap``.

        Notes
        -----
        Reconstruction is done in RING ordering and then written into a HealSparse
        map on the mask pixels.
        """
        _, ipix = self.get_mask(nest=False, return_ipix=True)

        maps = {}
        for imap in self.tqdm(imaps, desc="[compute_maps_from_alms]"):
            alm = hp.read_alm(self.get_fname(imap, what='alms', format='fits'))
            values = hp.alm2map(alm, nside=self.nside, pol=False)
            tmp = hsp.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            tmp.update_values_pix(ipix, values[ipix].astype(self.dtype), nest=False)
            if save:
                fname = self.get_fname(imap)
                tmp.write(fname, clobber=True)
            maps[imap] = tmp

        return maps[imaps] if type(imaps) is int else maps

    def _save_cls(self, ell: np.ndarray, cl: dict[tuple[int, int], np.ndarray]) -> None:
        """Save angular power spectra to a SACC FITS file.

        Parameters
        ----------
        ell : ndarray
            Effective multipoles (typically bandpower effective ells).
        cl : dict
            Dictionary keyed by (i, j) containing spectra arrays evaluated at
            ``ell``.
        """
        s = sacc.Sacc()
        tracers = self.get_tracers_sacc()
        for t in tracers:
            s.add_tracer_object(t)
        for i, j in cl.keys():
            for _ell, _cl in zip(ell, cl[i, j]):
                s.add_data_point(self.sacc_type, [tracers[i].name, tracers[j].name], _cl, ell=_ell)
        fname = self.get_fname(what='cls', format='fits')
        s.save_fits(fname, overwrite=True)

    def compute_cls(self, imaps: int | Iterable[int] | None = None, auto_only: bool = False, save: bool = True, use_precomputed_alms: bool = True, pair_job: int | None = None) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
        """Compute binned pseudo-C_ell bandpowers for requested map pairs.

        Parameters
        ----------
        imaps : array_like of int or None, optional
            Map indices. If None, uses all maps.
        auto_only : bool, optional
            If True, compute only auto-spectra.
        save : bool, optional
            If True, save spectra to SACC FITS (``what='cls'``).
        use_precomputed_alms : bool, optional
            If True, reads a_lm from disk and uses Healpy :func:`healpy.alm2cl`
            followed by NaMaster decoupling.
            If False, computes NaMaster fields from masked maps and runs full
            MASTER with :func:`pymaster.compute_full_master`.
        pair_job : int or None, optional
            If provided, compute only the pair at this index in the internal
            list of pairs (for embarrassingly-parallel workflows).

        Returns
        -------
        ell : ndarray
            Effective multipoles of the bandpowers.
        cl : dict
            Dictionary keyed by (i, j) containing decoupled bandpowers.
        """
        if imaps is None:
            imaps = range(self.nmaps)

        if auto_only:
            pairs = [(i, i) for i in imaps]
        else:
            # pairs = [(i1, i2) for i, i1 in enumerate(imaps) for i2 in imaps[:i + 1]]
            pairs = [(i1, i2) for i, i2 in enumerate(imaps) for i1 in imaps[i:]]  # with this order, i2 is preserved

        cl = {}
        b = self.get_cls_b()
        wsp = self.get_cls_wsp()

        if use_precomputed_alms:
            previous_pair = (-1, -1)
            for pair_count, (i, j) in enumerate(tqdm(pairs, desc="[compute_cls]")):
                if pair_job is not None:
                    if not pair_count == pair_job:
                        continue
                if not i == previous_pair[0]:
                    almi = hp.read_alm(self.get_fname(i, what='alms', format='fits'))
                if i == j:
                    almj = almi
                else:
                    if not j == previous_pair[1]:
                        almj = hp.read_alm(self.get_fname(j, what='alms', format='fits'))
                pcl = hp.alm2cl(almi, almj)
                cl[i, j] = wsp.decouple_cell([pcl])[0]
                previous_pair = (i, j)
        else:
            mask = self.get_mask(nest=False).astype(np.float64)
            fields = {i: None for i in imaps}
            for i, j in self.tqdm(pairs, desc="[compute_cls]"):
                if fields[i] is None:
                    fields[i] = nmt.NmtField(mask, [self.get_field_map(i, nest=False)])
                if fields[j] is None:
                    fields[j] = nmt.NmtField(mask, [self.get_field_map(j, nest=False)])
                cl[i, j] = nmt.compute_full_master(fields[i], fields[j], b, workspace=wsp)[0]

        ell = b.get_effective_ells()

        if save:
            self._save_cls(ell, cl)

        return ell, cl

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
            Must be implemented by subclasses if noise is needed.
        """
        raise NotImplementedError

    def predict_cls(self, imaps: int | Iterable[int] | None = None, auto_only: bool = False, approximate: bool = False, add_noise: bool = False, error_bars: bool = False, keep_cls: bool = False, stacked: bool = False, ell: np.ndarray | None = None) -> tuple:
        """Predict theory angular power spectra using CCL.

        Parameters
        ----------
        imaps : array_like of int or None, optional
            Map indices. If None, uses all maps.
        auto_only : bool, optional
            If True, compute only auto-spectra.
        approximate : bool, optional
            If True, return unbinned spectra evaluated at ``ell`` using CCL.
            If False, compute binned bandpowers using NaMaster bandpower windows.
        add_noise : bool, optional
            If True, add the map noise model for auto-spectra.
        error_bars : bool, optional
            If True, compute Gaussian (diagonal-in-ell) error bars.
        keep_cls : bool, optional
            If True, cache the results on ``self.predicted_cls``.
        stacked : bool, optional
            If True, also return spectra stacked into a single vector (and, when
            ``error_bars=True``, a diagonal covariance matrix).
        ell : ndarray or None, optional
            Multipoles at which to evaluate CCL when ``approximate=True``.
            If None, uses the effective bandpower ells from the binning scheme.

        Returns
        -------
        ell : ndarray
            Ell values (effective bandpower ells or explicit ``ell`` input).
        cl : dict
            Dictionary keyed by (i, j) with spectra arrays.

        Other Parameters
        ----------------
        error_bars : bool
            If True, also returns ``dcl`` (dict keyed by (i, j)).
        stacked : bool
            If True, also returns ``cl_stacked`` (1D array). If both
            ``stacked=True`` and ``error_bars=True``, also returns ``cov`` (2D
            array, diagonal by construction in this implementation).

        Notes
        -----
        When ``approximate=False``, this method evaluates raw C_ell up to
        ``b.lmax`` and bins them using NaMaster bandpower windows.
        """
        if self.predicted_cls is not None:
            return self.predicted_cls

        if imaps is None:
            imaps = range(self.nmaps)

        if auto_only:
            pairs = [(i, i) for i in imaps]
        else:
            # pairs = [(i1, i2) for i, i1 in enumerate(imaps) for i2 in imaps[:i + 1]]
            pairs = [(i1, i2) for i, i2 in enumerate(imaps) for i1 in imaps[i:]]  # with this order, i2 is preserved

        tracers = self.get_tracers_ccl()
        tracers_ = {imap: tracers[imap] for imap in imaps}

        b = self.get_cls_b()

        if ell is None:
            ell = b.get_effective_ells()
            nell = b.get_nell_list()
        else:
            assert approximate
            assert not error_bars

        order = lambda i, j: (max(i, j), min(i, j))

        if not approximate:
            wsp = self.get_cls_wsp()
            bpws = wsp.get_bandpower_windows()[0, :, 0, :]

        def angular_cl(*args):
            try:
                return pyccl.angular_cl(*args)
            except pyccl.CCLError:
                return pyccl.angular_cl(*args, limber_integration_method='spline')

        self.cosmo.compute_growth()

        cl = {}
        for i, j in self.tqdm(pairs, desc="[predict_cls]"):
            if approximate:
                cl[i, j] = angular_cl(self.cosmo, tracers_[i], tracers_[j], ell)
                if add_noise and i == j:
                    cl[i, j] += self._noise_cls(i)
            else:
                cl_raw = angular_cl(self.cosmo, tracers_[i], tracers_[j], range(b.lmax + 1))
                if self.f_ell is not None:
                    if self.f_ell == 'pixwin':
                        _f_ell = hp.pixwin(self.nside)[:b.lmax + 1]
                    else:
                        _f_ell = np.ones(b.lmax + 1)
                        _f_ell[b.get_ell_list(0)[0]:] = self.f_ell
                    cl_raw *= _f_ell**2
                cl[i, j] = np.dot(bpws, cl_raw)

                if add_noise and i == j:
                    nl_raw = self._noise_cls(i)
                    if self.f_ell is not None:
                        nl_raw *= _f_ell**2
                    cl[i, j] += np.dot(bpws, nl_raw)

        if error_bars:
            dcl = {}
            # Diagonal error bars
            for i, j in pairs:
                if i == j:
                    dcl2_num = 2 * (cl[i, i] + self._noise_cls(i))**2
                else:
                    dcl2_num = (cl[i, i] + self._noise_cls(i)) * (cl[j, j] + self._noise_cls(j)) + cl[order(i, j)]**2
                dcl[i, j] = np.sqrt(dcl2_num / ((2 * ell + 1.) * self.fsky * nell))

            if stacked:
                dcl_full = {}
                Dl = {}

                # Compute covariance blocks
                for i, j in pairs:
                    Dl[i, j] = np.copy(cl[i, j])
                    Dl[j, i] = np.copy(cl[i, j])
                    if i == j:
                        Dl[i, j] += self._noise_cls(i)

                # Stack Cl's and covariance blocks
                if auto_only:
                    for i, j in pairs:
                        dcl_full[i, j, i, j] = (Dl[i, j] * Dl[j, i] + Dl[i, i] * Dl[j, j]) / ((2 * ell + 1.) * self.fsky * nell)

                    cl_stacked = np.zeros(len(pairs) * len(ell))
                    cov = np.zeros([len(pairs) * len(ell), len(pairs) * len(ell)])
                    for nrow, (i, j) in enumerate(pairs):
                        cl_stacked[nrow * len(ell):(nrow + 1) * len(ell)] = np.copy(cl[i, j])
                        cov[nrow * len(ell):(nrow + 1) * len(ell), nrow * len(ell):(nrow + 1) * len(ell)] = np.diag(dcl_full[i, j, i, j])
                else:
                    for i, j in pairs:
                        for k, l in pairs:
                            dcl_full[i, j, k, l] = (Dl[i, l] * Dl[j, k] + Dl[i, k] * Dl[j, l]) / ((2 * ell + 1.) * self.fsky * nell)

                    cl_stacked = np.zeros(len(pairs) * len(ell))
                    cov = np.zeros([len(pairs) * len(ell), len(pairs) * len(ell)])
                    for nrow, (i, j) in enumerate(pairs):
                        cl_stacked[nrow * len(ell):(nrow + 1) * len(ell)] = np.copy(cl[i, j])
                        for ncol, (k, l) in enumerate(pairs):
                            cov[nrow * len(ell):(nrow + 1) * len(ell), ncol * len(ell):(ncol + 1) * len(ell)] = np.diag(dcl_full[i, j, k, l])

                if keep_cls:
                    self.predicted_cls = ell, cl, dcl, cl_stacked, cov
                return ell, cl, dcl, cl_stacked, cov
            else:
                if keep_cls:
                    self.predicted_cls = ell, cl, dcl
                return ell, cl, dcl
        else:
            if stacked:
                cl_stacked = np.zeros(len(pairs) * len(ell))
                for nrow, (i, j) in enumerate(pairs):
                    cl_stacked[nrow * len(ell):(nrow + 1) * len(ell)] = np.copy(cl[i, j])
                if keep_cls:
                    self.predicted_cls = ell, cl, cl_stacked
                return ell, cl, cl_stacked
            else:
                if keep_cls:
                    self.predicted_cls = ell, cl
                return ell, cl

    def triangle_plot_cls(
            self,
            auto_only: bool = False,
            logy: bool = False,
            theory_only: bool = False,
            data_only: bool = False,
            ratio_data_theory: bool = False,
            diff_data_theory_over_error: bool = False,
            save: bool = True,
            sharey: bool | str | None = None,
            figaxes: tuple | None = None,
            colortheory: str = 'orange',
            colordata: str = 'dodgerblue',
            approximate: bool = False,
            ell_factor: int = 0,
            error_bars: bool = False) -> None:
        """Plot a triangle matrix of theory and/or measured angular spectra.

        Parameters
        ----------
        auto_only : bool, optional
            If True, plot only auto-spectra (single row). Default is False.
        logy : bool, optional
            If True, use a log y-scale for standard spectra panels. Default is False.
        theory_only : bool, optional
            If True, plot theory curves only. Default is False.
        data_only : bool, optional
            If True, plot measured points only. Default is False.
        ratio_data_theory : bool, optional
            If True, plot ``C_ell(obs)/C_ell(th)``. Mutually exclusive with
            ``logy/theory_only/data_only``. Default is False.
        diff_data_theory_over_error : bool, optional
            If True, plot ``(C_ell(obs)-C_ell(th))/sigma(C_ell)``. Default is False.
        save : bool, optional
            If True, save figure as ``what='cls', format='png'``. Default is True.
        sharey : bool or {'row'} or None, optional
            Matplotlib sharey option passed to :func:`plt.subplots`.
        figaxes : tuple(fig, axes) or None, optional
            If provided, draw into the given (fig, axes).
        colortheory : str, optional
            Color for theory curves. Default is 'orange'.
        colordata : str, optional
            Color for measured points. Default is 'dodgerblue'.
        approximate : bool, optional
            Forwarded to :meth:`predict_cls`. Default is False.
        ell_factor : {0, 1, 2}, optional
            Vertical scaling:
            - 0: plot ``C_ell``
            - 1: plot ``ell*C_ell``
            - 2: plot ``ell(ell+1)C_ell/2pi``
        error_bars : bool, optional
            If True, draw theory uncertainty bands from Gaussian errors.
        """
        n = self.nmaps
        m = 1 if auto_only else n
        sharey = sharey or False  #False if auto_only else 'row'
        if figaxes is None:
            # fig, axes = plt.subplots(m, n, figsize=(4 * n, 3 * m), sharex=True, sharey=sharey, squeeze=False)
            fig, axes = plt.subplots(m, n, figsize=(3 * n, 3 * m), sharex=True, sharey=sharey, squeeze=False)
        else:
            fig, axes = figaxes

        if ratio_data_theory:
            assert not (logy or theory_only or data_only)

        ell, cl, dcl = self.predict_cls(auto_only=auto_only, approximate=approximate, error_bars=True)
        s = self.load_cls()

        tracers = list(s.tracers.keys())
        tc = s.get_tracer_combinations(self.sacc_type)

        if ell_factor == 0:
            ell_factor = 1.
            ylabel = r'$C_\ell$'
        elif ell_factor == 1:
            ell_factor = ell
            ylabel = r'$\ell\,C_\ell$'
        elif ell_factor == 2:
            ell_factor = ell * (ell + 1) / (2 * np.pi)
            ylabel = r'$\ell(\ell+1)C_\ell/2\pi$'
        else:
            raise ValueError

        for i in range(m):
            for j in range(n):
                if auto_only:
                    ax = axes[0, j]
                    i = j
                    title = f'{self.base_fname} {i}'
                else:
                    ax = axes[i, j]
                    title = f'{self.base_fname} {i} {j}'
                ax_used = False
                cl_obs = s.get_ell_cl(self.sacc_type, tracers[i], tracers[j])[1] if (tracers[i], tracers[j]) in tc else None
                cl_th = cl[i, j] if (i, j) in cl.keys() else None
                if ratio_data_theory:
                    if cl_obs is not None and cl_th is not None:
                        ax.plot(ell, cl_obs / cl_th, c=colordata)
                        if error_bars:
                            ax.fill_between(ell, 1. - dcl[i, j] / cl_th, 1. + dcl[i, j] / cl_th, color=colortheory, alpha=0.2, zorder=-99)
                        ax.axhline(y=1., c='.8', ls='--')
                        ax.set_ylabel(r'$C_\ell^{\rm obs} / C_\ell^{\rm th}$')
                        ax_used = True
                elif diff_data_theory_over_error:
                    if cl_obs is not None and cl_th is not None:
                        ax.plot(ell, (cl_obs - cl_th) / dcl[i, j], c=colordata)
                        if error_bars:
                            ax.axhspan(-1., +1., color=colortheory, alpha=0.2, zorder=-99)
                        ax.axhline(y=0., c='.8', ls='--')
                        ax.set_ylabel(r'$\Delta C_\ell / \sigma(C_\ell)$')
                        ax_used = True
                else:
                    if cl_th is not None and not data_only:
                        ax.plot(ell, ell_factor * cl_th, c=colortheory)
                        if error_bars:
                            ax.fill_between(ell, ell_factor * (cl_th - dcl[i, j]), ell_factor * (cl_th + dcl[i, j]), color=colortheory, alpha=0.2, zorder=-99)
                        ax_used = True
                    if cl_obs is not None and not theory_only:
                        ax.plot(ell, ell_factor * cl_obs, c=colordata, ls=' ', marker='o', markersize=5)
                        ax_used = True
                if ax_used:
                    ax.set_xlabel('Multipole $\ell$')
                    ax.set_ylabel(ylabel)
                    ax.set_xscale('log')
                    ax.set_title(title)
                    if logy:
                        ax.set_yscale('log')
                else:
                    ax.axis('off')

        fig.tight_layout()

        if save:
            fname = self.get_fname(what='cls', format='png')
            fig.savefig(fname, bbox_inches='tight')

    def compute_map_moments(self, theta_smooth: np.ndarray | Iterable[float], nside_small: int = 512) -> tuple[np.ndarray, dict, dict, dict]:
        """Compute 2nd/3rd/4th moments of smoothed maps.

        Parameters
        ----------
        theta_smooth : array_like of float
            Gaussian smoothing scales in arcmin.
        nside_small : int, optional
            Nside used for the smoothed maps and moment evaluation.

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
        5) caches results to ``what='2nd_3rd_4th_moments', format='npy'``.
        """
        assert min(theta_smooth) > hp.nside2resol(nside_small, arcmin=True), "Smoothing scale should be larger than pixel size to avoid pixel effects."
        
        fname = self.get_fname(what='2nd_3rd_4th_moments', format='npy')

        if os.path.exists(fname):
            print('Loading pre-existing file: ' + fname)
            moments = np.load(fname, allow_pickle=True).item()
            return moments['theta_smooth'], moments['second_moment'], moments['third_moment'], moments['fourth_moment']

        kappa = {}
        lmax_out = 2 * nside_small

        # Mask (downgraded from self.nside to nside_small for moment computation)
        mask_full, _ = self._create_mask()
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

        # Let's save them all
        print('Saving output as:' + fname)
        np.save(fname, moments)

        return theta_smooth, moments['second_moment'], moments['third_moment'], moments['fourth_moment']

    def generate_gaussian_maps(self, imaps: int | Iterable[int] | None = None, auto_only: bool = False, seed: int = -1) -> None:
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
        Maps are generated full-sky and then restricted to the analysis mask
        pixels before writing as HealSparse maps.
        """
        if imaps is None:
            imaps = range(self.nmaps)

        # Compute theory Cl's
        _, cls = self.predict_cls(approximate=True, auto_only=auto_only, ell=np.arange(3 * self.nside))

        _, ipix = self.get_mask(nest=False, return_ipix=True)

        if auto_only:
            for imap in self.tqdm(imaps, desc="[generate_gaussian_maps]"):
                # Generate full-sky Gaussian map
                out = nmt.utils.synfast_spherical(self.nside, [cls[imap, imap]], spin_arr=[0], seed=seed)[0]

                # Store in HealSparse map
                hsmap = hsp.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
                hsmap.update_values_pix(ipix, out[ipix].astype(self.dtype), nest=False)

                # Save map
                fname = self.get_fname(imap)
                hsmap.write(fname, clobber=True, format=self.save_format)

        else:
            # Generate full-sky Gaussian map
            pairs = [(i1, i2) for i, i2 in enumerate(imaps) for i1 in imaps[i:]]
            outs = nmt.utils.synfast_spherical(self.nside, [cls[p] for p in pairs], spin_arr=[0] * self.nmaps, seed=seed)

            for imap, out in zip(imaps, outs):
                # Store in HealSparse map
                hsmap = hsp.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
                hsmap.update_values_pix(ipix, out[ipix].astype(self.dtype), nest=False)

                # Save map
                fname = self.get_fname(imap)
                hsmap.write(fname, clobber=True, format=self.save_format)
