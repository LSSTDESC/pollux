"""Footprint of the maps (:class:`Footprint`): its mask, padded mask and weights of the harmonic transforms."""

from __future__ import annotations

import hashlib
import os
from functools import cached_property

import healpy as hp
import healsparse as hsp
import numpy as np
import pymaster as nmt

from .io import PolluxIO
from .utils.sky import make_mask_radec_cuts


class Footprint:
    """Footprint of the maps at one resolution: its mask, padded mask and weights of the harmonic transforms.

    Made once and shared by all builders. The maps are computed when first needed and kept in memory (not pickled,
    e.g. to pipeline tasks, which compute or read them again). The padded mask and the weights are saved once in
    ``{root_dir}/footprint/`` (:meth:`path`), next to the ``config.yaml`` of the footprint, written or checked when
    the footprint is made, like those of the other products. With padding, :attr:`alm_weights` computes (or reads)
    both, e.g. once before the builders of the maps.

    Parameters
    ----------
    nside : int
        Healpix resolution of the maps.
    io : pollux.io.files.PolluxIO
        Files of the root directory of the products.
    mask : ndarray or str or None, optional
        ``None`` for the full sky, ``'octant'`` for the positive octant (x, y, z > 0), a boolean Healpix map
        (NESTED, length ``hp.nside2npix(nside)``), or the path of a FITS file holding one. Default is None.
    padding_deg : float or None, optional
        With a mask, maps are built on the footprint dilated by this angle (the padded footprint), and their
        harmonic transforms weighted by the padded footprint apodized over the padding (:attr:`alm_weights`), so
        that they are not truncated at the edges of the footprint. Analyses (catalogs, sky fraction) use the
        footprint itself. Default is None (no padding).
    padding_nside : int, optional
        Resolution at which the padding and its apodization are computed. Default is 512.
    apotype : {'C1', 'C2', 'Smooth'} or None, optional
        Apodization of the padding (see :func:`pymaster.mask_apodization`), None for none. Default is 'C1'.

    Notes
    -----
    Maps and pixels are NESTED by default. In RING ordering, ``ipix(nest=False)[k]`` is the RING index of the k-th
    NESTED pixel (so it is not sorted).
    """

    def __init__(
            self,
            nside: int,
            io: PolluxIO,
            mask: np.ndarray | str | None = None,
            padding_deg: float | None = None,
            padding_nside: int = 512,
            apotype: str | None = 'C1') -> None:
        if isinstance(mask, np.ndarray) and len(mask) != hp.nside2npix(nside):
            raise ValueError(f"mask has {len(mask)} pixels but nside={nside} requires {hp.nside2npix(nside)}.")
        if isinstance(mask, str) and mask != 'octant' and not os.path.exists(mask):
            raise ValueError(f"mask path {mask} does not exist.")
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.io = io
        self.mask_input = mask
        self.fullsky = mask is None
        self.octant = isinstance(mask, str) and mask == 'octant'
        self.padding_deg = padding_deg if (padding_deg and mask is not None) else None
        self.padding_nside = min(padding_nside, nside)
        self.apotype = apotype
        self._mask, self._ipix = {}, {}  # masks (by nest, padded, dtype) and their pixels (by nest, padded) kept in memory
        self.io.write_config('footprint', self.config())

    def __getstate__(self) -> dict:
        """Return the state to pickle: without the maps in memory, computed or read again where needed.

        Returns
        -------
        dict
            The attributes, with empty :meth:`mask` and :meth:`ipix` memories and without :attr:`alm_weights`.
        """
        state = {k: v for k, v in self.__dict__.items() if k != 'alm_weights'}
        state['_mask'], state['_ipix'] = {}, {}
        return state

    def config(self) -> dict:
        """Return the configuration of the footprint, for its ``config.yaml`` (:func:`pollux.io.config.write_config`).

        Returns
        -------
        dict
            ``product`` (``'footprint'``), ``parameters`` (``nside``; ``mask``: ``'fullsky'``, ``'octant'``, or the
            first eight hexadecimal characters of the SHA-1 of the pixels of the mask, so that a footprint is checked
            by its pixels, not by the name of its file; padding and apodization) and ``inputs`` (the file of the mask).
        """
        match self.mask_input:
            case None:
                mask = 'fullsky'
            case str() if self.octant:
                mask = 'octant'
            case _:
                mask = hashlib.sha1(np.ascontiguousarray(self.ipix(), dtype=np.int64).tobytes()).hexdigest()[:8]
        padded = self.padding_deg is not None
        parameters = dict(
            nside=self.nside,
            mask=mask,
            padding_deg=self.padding_deg,
            padding_nside=self.padding_nside if padded else None,
            apotype=self.apotype if padded else None)
        inputs = [self.mask_input] if isinstance(self.mask_input, str) and not self.octant else []
        return dict(product='footprint', parameters=parameters, inputs=inputs)

    # ==================================================================================================================
    # region Masks

    def mask(self, nest: bool = True, padded: bool = False, dtype: type = bool) -> np.ndarray:
        """Return the footprint, or the padded footprint, as a dense map (boolean, or e.g. float for NaMaster).

        Computed once and kept in memory; the padded footprint is also saved once for all builders (:meth:`path`).

        Parameters
        ----------
        nest : bool, optional
            NESTED ordering if True (default), RING otherwise.
        padded : bool, optional
            The padded footprint (the footprint itself without ``padding_deg``). Default is False.
        dtype : data-type, optional
            Type of the values, 1 on the footprint and 0 elsewhere. Default is bool.

        Returns
        -------
        ndarray
            Length ``npix``.
        """
        padded = padded and self.padding_deg is not None
        key = (nest, padded, np.dtype(dtype))
        if key not in self._mask:
            if key[2] != bool:
                self._mask[key] = self.mask(nest=nest, padded=padded).astype(dtype)
            elif not nest:
                self._mask[key] = hp.reorder(self.mask(padded=padded), n2r=True)
            elif padded:
                self._mask[key] = self.io.cached(self.path('padded_mask'), self._pad, self._read_hspmap, self._write_hspmap)
            else:
                self._mask[key] = self._make_mask()
        return self._mask[key]

    def ipix(self, nest: bool = True, padded: bool = False) -> np.ndarray:
        """Return the pixels of the footprint, or of the padded footprint (:meth:`mask`).

        Parameters
        ----------
        nest : bool, optional
            NESTED indices if True (default), RING otherwise.
        padded : bool, optional
            The pixels of the padded footprint. Default is False.

        Returns
        -------
        ndarray of int
            In the order of the NESTED pixels: in RING ordering, ``ipix(nest=False)[k]`` is the RING index of the
            k-th NESTED pixel (so it is not sorted).
        """
        padded = padded and self.padding_deg is not None
        if (nest, padded) not in self._ipix:
            ipix = np.flatnonzero(self.mask(padded=padded))
            self._ipix[nest, padded] = ipix if nest else hp.nest2ring(self.nside, ipix)
        return self._ipix[nest, padded]

    @property
    def fsky(self) -> float:
        """Sky fraction of the footprint, from its number of pixels (so with its edge effects, e.g. at the octant boundary)."""
        return len(self.ipix()) / self.npix

    def _make_mask(self) -> np.ndarray:
        """Return the footprint, from the mask given (full sky, octant, array or file).

        Returns
        -------
        ndarray of bool
            Dense NESTED map.

        Raises
        ------
        ValueError
            If the mask file is not NESTED or not at ``nside``.
        """
        # the array case first: an array compares elementwise to 'octant'
        match self.mask_input:
            case None:
                return np.ones(self.npix, dtype=bool)
            case np.ndarray():
                return self.mask_input.astype(bool)
            case 'octant':
                return make_mask_radec_cuts(self.nside, 0, 90, 0, 90, eps=1e-10, nest=True, dtype=bool)
            case fname:
                mask, header = hp.read_map(fname, nest=True, h=True)
                ordering = dict(header)['ORDERING']
                if ordering.upper() != 'NESTED':
                    raise ValueError(f"mask at {fname} has ORDERING={ordering!r} in its FITS header, expected 'NESTED'.")
                if len(mask) != self.npix:
                    raise ValueError(f"mask at {fname} has {len(mask)} pixels but nside={self.nside} requires {self.npix}.")
                return mask.astype(bool)

    def _pad(self) -> np.ndarray:
        """Return the padded footprint: the footprint dilated by ``padding_deg``.

        The pixels within ``padding_deg`` of the footprint, where the C1 apodization of its complement at
        ``padding_nside`` is below 1 (whatever ``apotype``), upgraded to ``nside``: accurate to about a
        ``padding_nside`` pixel.

        Returns
        -------
        ndarray of bool
            Dense NESTED map.
        """
        mask = self.mask()
        outside_lowres = (hp.ud_grade((~mask).astype(np.float64), self.padding_nside, order_in='NESTED', order_out='RING') > 0.).astype(np.float64)
        apo_lowres = nmt.mask_apodization(outside_lowres, self.padding_deg, apotype='C1')
        padded_lowres = hp.reorder((apo_lowres < 1.).astype(np.float64), r2n=True) > 0.
        return mask | (hp.ud_grade(padded_lowres.astype(np.float64), self.nside, order_in='NESTED', order_out='NESTED') > 0.)

    # endregion

    # ==================================================================================================================
    # region Weights

    @cached_property
    def alm_weights(self) -> np.ndarray:
        """Weights of the maps in their harmonic transforms.

        The footprint, or with ``padding_deg`` the padded footprint apodized over the padding (:meth:`_apodize`),
        saved once for all builders (:meth:`path`, with the padded footprint it is computed from).

        Returns
        -------
        ndarray of float
            Dense RING map, 1 on the footprint.
        """
        if self.padding_deg is None:
            return self.mask(nest=False, dtype=np.float64)
        return self.io.cached(self.path('alm_weights'), self._apodize, self._read_hspmap, self._write_hspmap)

    def _apodize(self) -> np.ndarray:
        """Return the padded footprint apodized over the padding (``apotype``; unapodized if None).

        The apodization is computed at ``padding_nside`` and interpolated bilinearly at ``nside``, so that the weights
        have no steps at the scale of the ``padding_nside`` pixels.

        Returns
        -------
        ndarray of float
            Dense RING map, 1 on the footprint and between 0 and 1 in the padding.
        """
        if self.apotype is None:
            return self.mask(nest=False, padded=True).astype(np.float64)
        padded_lowres = (hp.ud_grade(self.mask(padded=True).astype(np.float64), self.padding_nside, order_in='NESTED', order_out='RING')
                         > 0.).astype(np.float64)
        apo_lowres = nmt.mask_apodization(padded_lowres, self.padding_deg, apotype=self.apotype)
        footprint = self.mask(nest=False)
        weights = footprint.astype(np.float64)
        ipix_padding = np.flatnonzero(self.mask(nest=False, padded=True) & ~footprint)
        for ipix in np.array_split(ipix_padding, max(1, len(ipix_padding) // 10_000_000)):
            theta, phi = hp.pix2ang(self.nside, ipix)
            weights[ipix] = np.clip(hp.get_interp_val(apo_lowres, theta, phi), 0., 1.)
        return weights

    # endregion

    # ==================================================================================================================
    # region Files

    def path(self, *name: str, ext: str = 'fits') -> str:
        """Return a file of the footprint, ``{root_dir}/footprint/{name}.{ext}``.

        E.g. the padded footprint (``'padded_mask'``), the weights (``'alm_weights'``), and the workspaces of the maps,
        which depend on the weights (:attr:`pollux.statistics.maps.MapStatistics.wsp`).

        Parameters
        ----------
        *name : str
            Parts of the name, joined by ``_`` (see :meth:`pollux.io.files.PolluxIO.path`).
        ext : str, optional
            Extension. Default is ``'fits'``.

        Returns
        -------
        str
        """
        return self.io.path('footprint', *name, ext=ext)

    def _read_hspmap(self, fname: str) -> np.ndarray:
        """Read the padded footprint or the weights saved by :meth:`_write_hspmap`.

        Parameters
        ----------
        fname : str
            File.

        Returns
        -------
        ndarray
            Dense map: the padded footprint (bool, NESTED) or the weights (float, RING).
        """
        hspmap = self.io.read_map(fname)
        ipix = hspmap.valid_pixels
        if hspmap.dtype == bool:
            dense = np.zeros(self.npix, dtype=bool)
            dense[ipix] = True
            return dense
        dense = np.zeros(self.npix)
        dense[hp.nest2ring(self.nside, ipix)] = hspmap[ipix]
        return dense

    def _write_hspmap(self, fname: str, dense: np.ndarray) -> None:
        """Write the padded footprint or the weights, as a HealSparse map of their values on the padded footprint.

        Parameters
        ----------
        fname : str
            File.
        dense : ndarray
            Dense map: the padded footprint (bool, NESTED) or the weights (float, RING).
        """
        if dense.dtype == bool:
            ipix = np.flatnonzero(dense)
            values = np.ones(len(ipix), dtype=bool)
        else:
            ipix = self.ipix(padded=True)
            values = dense[hp.nest2ring(self.nside, ipix)]
        hspmap = hsp.HealSparseMap.make_empty(32, self.nside, dtype=dense.dtype)
        hspmap.update_values_pix(ipix, values, nest=True)
        self.io.write_map(fname, hspmap, nocompress=True)

    # endregion
