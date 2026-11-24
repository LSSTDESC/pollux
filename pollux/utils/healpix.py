"""Dense HEALPix maps (healpy, RING ordering) and HealSparse maps (NESTED): exact and fast conversions between them
(:func:`reorder`, :func:`dense_to_hsp`, :func:`hsp_to_dense`), and values of either kind of map at sky positions
(:func:`values_pos`, :func:`interpolate_pos`).

The conversions reorder with ducc0's multithreaded ``ring2nest``/``nest2ring`` and build or read the HealSparse
arrays by coverage block, instead of :meth:`healsparse.HealSparseMap.update_values_pix` and
:meth:`~healsparse.HealSparseMap.generate_healpix_map`.

Dense maps mark the pixels without a value (outside a footprint) with ``hp.UNSEEN`` (the sentinel of float
HealSparse maps), as healpy does.
"""

from __future__ import annotations

import os

import ducc0
import healpy as hp
import healsparse as hsp
import hpgeom as hpg
import numba
import numpy as np

from .sky import radec2thetaphi

# pixels reordered per chunk, bounding the memory of the index arrays (8 bytes per pixel)
_CHUNK = 1 << 24

# ======================================================================================================================
# region Conversions


def reorder(values: np.ndarray, r2n: bool) -> np.ndarray:
    """Reorder a full-sky dense map between RING and NESTED orderings (as ``hp.reorder``, multithreaded).

    Parameters
    ----------
    values : ndarray
        Full-sky map (length ``12 * nside**2``).
    r2n : bool
        True from RING to NESTED, False from NESTED to RING.

    Returns
    -------
    ndarray
        The reordered map, of the dtype of ``values``.
    """
    npix = len(values)
    nside = hp.npix2nside(npix)
    base = ducc0.healpix.Healpix_Base(nside, 'RING' if r2n else 'NEST')
    n = _nthreads()
    out = np.empty_like(values)
    for i0 in range(0, npix, _CHUNK):
        index = np.arange(i0, min(i0 + _CHUNK, npix), dtype=np.int64)
        index = base.ring2nest(index, nthreads=n) if r2n else base.nest2ring(index, nthreads=n)
        _scatter(out, index, values[i0:i0 + len(index)])
    return out


def dense_to_hsp(
        values: np.ndarray, nside_coverage: int, nest: bool = False, valid: np.ndarray | None = None, dtype: np.dtype | None = None) -> hsp.HealSparseMap:
    """Return the HealSparse map of a full-sky dense map.

    The same map as ``make_empty(nside_coverage, nside, dtype)`` then ``update_values_pix`` on the valid pixels.

    Parameters
    ----------
    values : ndarray
        Full-sky dense map.
    nside_coverage : int
        Coverage nside of the HealSparse map.
    nest : bool, optional
        Ordering of ``values`` (and ``valid``). Default is False (RING).
    valid : ndarray of bool or None, optional
        Valid pixels of the map, in the ordering of ``values``. Default is those whose value (in ``dtype``) is not
        the sentinel (``hp.UNSEEN`` for floats).
    dtype : data-type or None, optional
        dtype of the map. Default is that of ``values``.

    Returns
    -------
    hsp.HealSparseMap
        With the coverage pixels holding at least one valid pixel.
    """
    dtype = np.dtype(dtype or values.dtype)
    sentinel = hsp.utils.check_sentinel(dtype.type, None)
    if valid is None:
        values = np.asarray(values, dtype=dtype)  # compared to the sentinel in the dtype of the map
    if not nest:
        values = reorder(values, r2n=True)
        if valid is not None:
            valid = reorder(valid, r2n=True)
    if valid is None:
        valid = values != sentinel
    nside = hp.npix2nside(len(values))
    nfine = (nside // nside_coverage)**2
    valid_blocks = valid.reshape(-1, nfine)
    all_valid = bool(valid.all())
    cov_pix = np.arange(len(valid_blocks)) if all_valid else np.flatnonzero(valid_blocks.any(axis=1))

    # block k of the sparse array, after the leading sentinel block, holds the pixels of coverage pixel cov_pix[k]
    sparse = np.empty((len(cov_pix) + 1) * nfine, dtype=dtype)
    sparse[:nfine] = sentinel
    blocks = sparse[nfine:].reshape(-1, nfine)
    if len(cov_pix) == len(valid_blocks):
        blocks[...] = values.reshape(-1, nfine)  # cast to dtype
    else:
        blocks[...] = values.reshape(-1, nfine)[cov_pix]
    if not all_valid:
        blocks[~valid_blocks[cov_pix]] = sentinel
    cov = hsp.HealSparseCoverage.make_from_pixels(nside_coverage, nside, cov_pix)
    return hsp.HealSparseMap(cov_map=cov, sparse_map=sparse, nside_sparse=nside, sentinel=sentinel)


def hsp_to_dense(hspmap: hsp.HealSparseMap, nest: bool = False) -> np.ndarray:
    """Return the full-sky dense map of a HealSparse map, with its sentinel on its invalid pixels.

    For float maps, the same as ``hspmap.generate_healpix_map(nest=nest)``.

    Parameters
    ----------
    hspmap : hsp.HealSparseMap
        Map, of a single numerical value per pixel.
    nest : bool, optional
        Ordering of the dense map. Default is False (RING).

    Returns
    -------
    ndarray
        Of the dtype of the HealSparse map.
    """
    nside = hspmap.nside_sparse
    nfine = (nside // hspmap.nside_coverage)**2
    sparse = hspmap._sparse_map
    cov_pix = np.flatnonzero(hspmap.coverage_mask)
    # block of each coverage pixel in the sparse array (its pixel p at index cov_map[c] + p)
    block = (hspmap._cov_map[:][cov_pix] + cov_pix * nfine) // nfine
    dense = np.empty(hp.nside2npix(nside), dtype=sparse.dtype)
    dense_blocks = dense.reshape(-1, nfine)
    if len(cov_pix) == len(dense_blocks) and np.array_equal(block, np.arange(1, len(block) + 1)):
        dense_blocks[...] = sparse[nfine:].reshape(-1, nfine)  # full sky, blocks in order
    else:
        dense[...] = hspmap._sentinel
        dense_blocks[cov_pix] = sparse.reshape(-1, nfine)[block]
    return dense if nest else reorder(dense, r2n=False)


def _nthreads() -> int:
    """Return the number of threads of the conversions.

    Returns
    -------
    int
        ``OMP_NUM_THREADS`` if set (e.g. 16 for the tasks of the default executor), else the cores available to the
        process.
    """
    n = os.environ.get('OMP_NUM_THREADS')
    return int(n) if n else len(os.sched_getaffinity(0))


@numba.njit(parallel=True)
def _scatter(out: np.ndarray, index: np.ndarray, values: np.ndarray) -> None:
    """Set ``out[index] = values`` in parallel, for distinct indices.

    Parameters
    ----------
    out : ndarray
        Array updated in place.
    index : ndarray of int
        Distinct indices of ``out`` (e.g. a chunk of a permutation of the pixels).
    values : ndarray
        Values, of the length of ``index``.
    """
    for i in numba.prange(len(index)):
        out[index[i]] = values[i]


# endregion

# ======================================================================================================================
# region Values of maps


def is_dense(m: np.ndarray | hsp.HealSparseMap) -> bool:
    """Return whether a map is a dense (RING) map rather than a HealSparse map.

    Parameters
    ----------
    m : ndarray or hsp.HealSparseMap
        Map.

    Returns
    -------
    bool
    """
    return not isinstance(m, hsp.HealSparseMap)


def values_pos(m: np.ndarray | hsp.HealSparseMap, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Return the values of a map (dense RING or HealSparse) in the pixels of sky positions.

    Parameters
    ----------
    m : ndarray or hsp.HealSparseMap
        Map.
    ra, dec : ndarray
        Positions in degrees.

    Returns
    -------
    ndarray
        The sentinel of the map where it has no value.
    """
    ipix = hp.ang2pix(_nside(m), *radec2thetaphi(ra, dec), nest=not is_dense(m))
    return m[ipix]


def interpolate_pos(m: np.ndarray | hsp.HealSparseMap, ra: np.ndarray, dec: np.ndarray, allow_partial: bool = True) -> np.ndarray:
    """Return the bilinear interpolation of a map (dense RING or HealSparse) at sky positions.

    The same as :meth:`healsparse.HealSparseMap.interpolate_pos` for both kinds of maps.

    Parameters
    ----------
    m : ndarray or hsp.HealSparseMap
        Map.
    ra, dec : ndarray
        Positions in degrees.
    allow_partial : bool, optional
        Interpolate from the neighbors with a value only, if some have none. Default is True.

    Returns
    -------
    ndarray
        ``hp.UNSEEN`` where no neighbor (or, without ``allow_partial``, not all) has a value.
    """
    if not is_dense(m):
        return m.interpolate_pos(ra, dec, allow_partial=allow_partial)
    # as healsparse, with the NESTED pixels of the weights reordered to RING
    nside = _nside(m)
    interp_pix, interp_wgt = hpg.get_interpolation_weights(nside, np.atleast_1d(ra), np.atleast_1d(dec), lonlat=True)
    aux = m[hpg.nest_to_ring(nside, interp_pix)]
    out_of_bounds = (aux == hsp.utils.check_sentinel(m.dtype.type, None))
    aux = aux.astype(np.float64)
    aux[out_of_bounds] = np.nan
    if not allow_partial:
        values = np.nansum(aux * interp_wgt, axis=1) / np.sum(interp_wgt, axis=1)
        values[~np.all(~out_of_bounds, axis=1)] = hpg.UNSEEN
    else:
        interp_wgt[out_of_bounds] = np.nan
        wgt_sum = np.nansum(interp_wgt, axis=1)
        values = np.nansum(aux * interp_wgt, axis=1)
        all_bad = (wgt_sum == 0.0)
        values[~all_bad] /= wgt_sum[~all_bad]
        values[all_bad] = hpg.UNSEEN
    return values


def _nside(m: np.ndarray | hsp.HealSparseMap) -> int:
    """Return the nside of a map (dense RING or HealSparse).

    Parameters
    ----------
    m : ndarray or hsp.HealSparseMap
        Map.

    Returns
    -------
    int
    """
    return hp.npix2nside(len(m)) if is_dense(m) else m.nside_sparse


# endregion
