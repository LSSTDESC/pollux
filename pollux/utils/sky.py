"""Sky helpers: coordinates, maps of point samples and masks."""
from __future__ import annotations

import healpy as hp
import numba
import numpy as np


def radec2thetaphi(ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (RA, Dec) in degrees to Healpix angles (theta, phi) in radians.

    Healpix conventions:
    - ``theta`` is colatitude in radians, in [0, pi]
    - ``phi`` is longitude in radians, in [0, 2pi)

    Parameters
    ----------
    ra : array_like
        Right ascension in degrees.
    dec : array_like
        Declination in degrees.

    Returns
    -------
    theta : ndarray
        Colatitude in radians.
    phi : ndarray
        Longitude in radians.
    """
    theta = np.pi / 2.0 - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    return theta, phi


def thetaphi2radec(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Healpix angles (theta, phi) in radians to (RA, Dec) in degrees.

    Healpix conventions:
    - ``theta`` is colatitude in radians
    - ``phi`` is longitude in radians

    Parameters
    ----------
    theta : array_like
        Colatitude in radians.
    phi : array_like
        Longitude in radians.

    Returns
    -------
    ra : ndarray
        Right ascension in degrees.
    dec : ndarray
        Declination in degrees.
    """
    ra = np.rad2deg(phi)
    dec = 90.0 - np.rad2deg(theta)
    return ra, dec


@numba.njit(parallel=False)  # this operation does NOT support parallelization!
def _add_at(a: np.ndarray, index: np.ndarray, b: np.ndarray) -> None:
    """Numba helper: add values ``b`` to ``a`` at given indices.

    Parameters
    ----------
    a : ndarray
        1D array modified in-place.
    index : ndarray of int
        Indices where to add.
    b : ndarray
        Values added elementwise: ``a[index[i]] += b[i]``.
    """
    n = len(b)
    for i in numba.prange(n):
        a[index[i]] += b[i]


def make_healpix_map(
        ra: np.ndarray,
        dec: np.ndarray,
        quantity: np.ndarray | None,
        nside: int,
        mask: np.ndarray | None = None,
        weight: np.ndarray | None = None,
        ipix: np.ndarray | None = None,
        fill_UNSEEN: bool = False,
        return_w_maps: bool = False,
        return_extra: bool = False,
        nest: bool = True,
        mode: str = 'mean') -> list:
    """Bin point samples into one or more dense Healpix maps.

    This function takes samples at sky coordinates (RA, Dec) and aggregates an
    associated quantity into Healpix pixels (mean or sum). It also returns a
    count map and a mask indicating which pixels are considered valid.

    Parameters
    ----------
    ra, dec : array_like
        Right ascension and declination in degrees. Must have same length ``N``.
    quantity : array_like or None
        Values associated with each object.
        - If 1D of length ``N``, returns one map.
        - If 2D of shape ``(nq, N)``, returns ``nq`` maps.
        - If None, only the count/mask maps are computed.
    nside : int
        Healpix resolution parameter.
    mask : array_like or None, optional
        If provided, a dense mask of length ``hp.nside2npix(nside)``. Pixels
        outside (mask==0) are set to 0 or UNSEEN depending on ``fill_UNSEEN``.
        If None, a mask is constructed from pixels with at least one object.
    weight : array_like or None, optional
        Per-object weights. If provided, must be broadcastable to ``quantity``:
        - shape ``(N,)`` for a single weight set shared across quantities, or
        - shape ``(nq, N)`` for one weight per quantity.
        All weights must be strictly positive.
    ipix : array_like of int or None, optional
        Precomputed pixel indices for each object. If None, computed with
        :func:`healpy.ang2pix` using ``nest`` ordering.
    fill_UNSEEN : bool, optional
        If True, set pixels outside the mask to ``hp.UNSEEN``. If False, set
        them to 0. Default is False.
    return_w_maps : bool, optional
        If True, also return maps of the sum of weights per pixel for each
        quantity component.
    return_extra : bool, optional
        If True, also return a dictionary containing summary counts and a masked
        version of ``ipix``.
    nest : bool, optional
        Pixel ordering used when computing ``ipix``. Default is True.
    mode : {'mean', 'sum'}, optional
        Aggregation mode:
        - 'sum': pixel value is sum(quantity*weight)
        - 'mean': pixel value is (sum(quantity*weight) / sum(weight)) on valid pixels

    Returns
    -------
    outmaps : list of ndarray
        List of dense maps, one per quantity component. Empty if
        ``quantity is None``.
    count : ndarray
        Dense count map (number of objects per pixel).
    returned_mask : ndarray
        Mask map (float array of 0/1 if created internally, otherwise the input
        mask is returned unchanged).
    sum_w_maps : list of ndarray, optional
        Only if ``return_w_maps=True``. Sum of weights per pixel, one per
        quantity component.
    extra : dict, optional
        Only if ``return_extra=True``. Contains:
        - ``count_tot_in_mask``
        - ``count_per_pixel_in_mask``
        - ``count_per_steradian_in_mask``
        - ``count_per_sqdegree_in_mask``
        - ``count_per_sqarcmin_in_mask``
        - ``ipix_masked`` : masked array of input pixel indices

    Notes
    -----
    This function uses small Numba kernels (``_add_at``) as a faster alternative
    to ``np.add.at`` for the accumulation steps.
    """
    if quantity is not None:
        quantity = np.atleast_2d(quantity)

        if weight is not None:
            w = np.atleast_2d(weight)
            if w.shape[0] > 1:
                if quantity.shape != w.shape:
                    raise ValueError("[make_healpix_map] quantity/weight arrays don't have the same length")
            else:
                w = np.tile(w[0], (quantity.shape[0], 1))

            if not np.all(w > 0.0):
                raise ValueError('[make_healpix_map] weight is not strictly positive')
        else:
            w = np.ones_like(quantity)

        if quantity.shape != w.shape:
            raise ValueError("[make_healpix_map] quantity/weight arrays don't have the same length")

    npix = hp.nside2npix(nside)

    if mask is not None:
        if len(mask) != npix:
            raise ValueError('[make_healpix_map] mask array does not have the right length')

    # Value to fill outside the mask
    x = hp.UNSEEN if fill_UNSEEN else 0.0

    # Make sure mode is correct
    if mode not in ['sum', 'mean']:
        raise ValueError("[make_healpix_map] mode should be 'mean' or 'sum'")

    count = np.zeros(npix, dtype=float)
    outmaps = []
    sum_w_maps = []

    # Getting pixels for each object
    if ipix is None:
        if len(ra) != len(dec):
            raise ValueError("[make_healpix_map] ra/dec arrays don't have the same length")
        if quantity is not None:
            if len(ra) != quantity.shape[1]:
                raise ValueError("[make_healpix_map] ra and quantity arrays don't have the same length")
        ipix = hp.ang2pix(nside, (90.0 - dec) / 180.0 * np.pi, ra / 180.0 * np.pi, nest=nest)
    else:
        if quantity is not None:
            if len(ipix) != quantity.shape[1]:
                raise ValueError('[make_healpix_map] ipix has wrong size')

    # Counting objects in pixels
    count += np.bincount(ipix, minlength=npix)

    # Creating the mask if it does not exist
    if mask is None:
        bool_mask = (count > 0)
    else:
        bool_mask = mask.astype(bool)

    # Create the maps
    if quantity is not None:
        for i in range(quantity.shape[0]):
            sum_w = np.zeros(npix, dtype=float)
            _add_at(sum_w, ipix, w[i, :])

            outmap = np.zeros(npix, dtype=float)
            _add_at(outmap, ipix, quantity[i, :] * w[i, :])

            if mode == 'mean':
                outmap[bool_mask] /= sum_w[bool_mask]

            outmap[np.logical_not(bool_mask)] = x

            outmaps.append(outmap)
            if return_w_maps:
                sum_w_maps.append(sum_w)

    returned_mask = bool_mask.astype(float) if mask is None else mask

    res = [outmaps, count, returned_mask]

    if return_w_maps:
        res += [sum_w_maps]

    if return_extra:
        extra = {}
        extra['count_tot_in_mask'] = np.sum(count[bool_mask])
        extra['count_per_pixel_in_mask'] = extra['count_tot_in_mask'] * 1.0 / np.sum(bool_mask.astype(int))
        extra['count_per_steradian_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=False)
        extra['count_per_sqdegree_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=True)
        extra['count_per_sqarcmin_in_mask'] = extra['count_per_sqdegree_in_mask'] / 60.0**2
        extra['ipix_masked'] = np.ma.array(ipix, mask=bool_mask[ipix])

        res += [extra]

    return res


def make_mask_radec_cuts(
        nside: int,
        ramin: float,
        ramax: float,
        decmin: float,
        decmax: float,
        eps: float = 0.0,
        dtype: type = float,
        nest: bool = True,
        coord: str = 'C') -> np.ndarray:
    """Generate a rectangular mask from longitude/latitude cuts.

    Parameters
    ----------
    nside : int
        Healpix nside.
    ramin, ramax : float
        Longitude bounds in degrees (right ascension for ``coord='C'``), in
        [0, 360] with ``ramin < ramax``.
    decmin, decmax : float
        Latitude bounds in degrees (declination for ``coord='C'``), in [-90, 90]
        with ``decmin < decmax``.
    eps : float, optional
        Margin (in radians in theta/phi space) excluded from each boundary.
        Default is 0.
    dtype : data-type, optional
        dtype used for the returned mask values (typically float or int).
    nest : bool, optional
        If True, use NESTED pixel ordering. Default is True.
    coord : {'C', 'E', 'G'}, optional
        Frame of the bounds: equatorial (default), ecliptic or galactic. The
        mask itself is in equatorial coordinates, e.g. ``coord='E'`` with
        ``decmin=-90, decmax=30`` (and ``ramin=0, ramax=360``) masks ecliptic
        latitudes above 30 degrees.

    Returns
    -------
    ndarray
        Dense mask array of shape ``(hp.nside2npix(nside),)`` with values 0/1.

    Notes
    -----
    The rectangle is implemented by selecting pixels whose (theta, phi) in the
    frame ``coord`` lie within the converted ranges.
    """
    if not ((0.0 <= ramin <= 360.0) & (0.0 <= ramax <= 360.0) & (ramin < ramax)):
        raise ValueError(f"Need 0 <= ramin < ramax <= 360, not ramin={ramin}, ramax={ramax}")
    if not ((-90.0 <= decmin <= 90.0) & (-90.0 <= decmax <= 90.0) & (decmin < decmax)):
        raise ValueError(f"Need -90 <= decmin < decmax <= 90, not decmin={decmin}, decmax={decmax}")

    mask = np.zeros(hp.nside2npix(nside), dtype=dtype)
    th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=nest)
    if coord != 'C':
        # pixel centers in the frame of the bounds, with longitudes in [0, 2 pi)
        th, ph = hp.Rotator(coord=['C', coord])(th, ph)
        ph = np.mod(ph, 2. * np.pi)
    ths, phs = radec2thetaphi([ramin, ramax], [decmin, decmax])
    thmin, thmax = min(ths), max(ths)
    phmin, phmax = min(phs), max(phs)
    mask[np.where((th < thmax - eps) & (th > thmin + eps) & (ph > phmin + eps) & (ph < phmax - eps))[0]] = dtype(1)
    return mask
