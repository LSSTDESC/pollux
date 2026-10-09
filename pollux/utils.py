import warnings
from functools import partial

import healpy as hp
import numba
import numpy as np
import pymaster as nmt
from astropy.io import fits
from pynverse import inversefunc
from scipy.integrate import quad as scipy_quad
from scipy.spatial import KDTree
from scipy.stats import truncnorm

QUAD_EPS_REL = 1e-6
QUAD_EPS_ABS = 0.0
QUAD_LIMIT = 1000


def quad(*args, **kwargs):
    """Integrate a function using SciPy quad with project-wide default tolerances.

    This is a thin wrapper around :func:`scipy.integrate.quad` that enforces
    consistent absolute/relative tolerances and recursion limit across the codebase.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :func:`scipy.integrate.quad`.
    **kwargs
        Keyword arguments forwarded to :func:`scipy.integrate.quad`.
        User-provided ``epsabs``, ``epsrel`` and ``limit`` are overridden by the
        module constants ``QUAD_EPS_ABS``, ``QUAD_EPS_REL`` and ``QUAD_LIMIT``.

    Returns
    -------
    float
        The value of the integral (the first element returned by SciPy quad).

    Notes
    -----
    SciPy returns ``(value, error_estimate)``; this wrapper returns only ``value``.
    """
    return scipy_quad(*args, epsabs=QUAD_EPS_ABS, epsrel=QUAD_EPS_REL, limit=QUAD_LIMIT, **kwargs)[0]


def powspace(start, stop, num, power):
    """Generate a power-law spaced grid between two positive endpoints.

    This is similar to :func:`numpy.linspace` and :func:`numpy.geomspace`, but uses
    a power mapping:
    - transform endpoints as ``x -> x**power``,
    - take a linear grid in transformed space,
    - map back with exponent ``1/power``.

    Parameters
    ----------
    start, stop : float
        Endpoints of the interval. For non-integer ``power``, these should be
        non-negative.
    num : int
        Number of samples to generate.
    power : float
        Power-law exponent controlling the spacing. ``power=1`` gives linear
        spacing. ``power>1`` emphasizes samples near ``start``; ``power<1``
        emphasizes samples near ``stop``.

    Returns
    -------
    ndarray
        Array of shape ``(num,)`` containing the grid.

    See Also
    --------
    numpy.linspace, numpy.geomspace
    """
    start = np.power(start, float(power))
    stop = np.power(stop, float(power))
    return np.power(np.linspace(start, stop, num=num), 1.0 / power)


def radec2thetaphi(ra, dec):
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


def thetaphi2radec(theta, phi):
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
def _add_at_cst(a, index, b):
    """Numba helper: add a constant value to ``a`` at given indices.

    Parameters
    ----------
    a : ndarray
        1D array modified in-place.
    index : ndarray of int
        Indices where to add.
    b : float
        Constant added to each ``a[index[i]]``.
    """
    n = len(index)
    for i in numba.prange(n):
        a[index[i]] += b


@numba.njit(parallel=False)
def _add_at(a, index, b):
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
        ra, dec, quantity, nside, mask=None, weight=None, ipix=None, fill_UNSEEN=False, return_w_maps=False, return_extra=False, nest=True, mode='mean'):
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
                assert quantity.shape == w.shape, "[make_healpix_map] quantity/weight arrays don't have the same length"
            else:
                w = np.tile(w[0], (quantity.shape[0], 1))

            assert np.all(w > 0.0), "[make_healpix_map] weight is not strictly positive"
        else:
            w = np.ones_like(quantity)

        assert quantity.shape == w.shape, "[make_healpix_map] quantity/weight arrays don't have the same length"

    npix = hp.nside2npix(nside)

    if mask is not None:
        assert len(mask) == npix, "[make_healpix_map] mask array does not have the right length"

    # Value to fill outside the mask
    x = hp.UNSEEN if fill_UNSEEN else 0.0

    # Make sure mode is correct
    assert (mode in ['sum', 'mean']), "[make_healpix_map] mode should be 'mean' or 'sum'"

    count = np.zeros(npix, dtype=float)
    outmaps = []
    sum_w_maps = []

    # Getting pixels for each object
    if ipix is None:
        assert len(ra) == len(dec), "[make_healpix_map] ra/dec arrays don't have the same length"
        if quantity is not None:
            assert len(ra) == quantity.shape[1]
        ipix = hp.ang2pix(nside, (90.0 - dec) / 180.0 * np.pi, ra / 180.0 * np.pi, nest=nest)
    else:
        if quantity is not None:
            assert len(ipix) == quantity.shape[1], "[make_healpix_map] ipix has wrong size"

    # Counting objects in pixels
    _add_at_cst(count, ipix, 1.0)

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


def make_mask_radec_cuts(nside, ramin, ramax, decmin, decmax, eps=0.0, dtype=float, nest=True, coord='C'):
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
    assert (0.0 <= ramin <= 360.0) & (0.0 <= ramax <= 360.0) & (ramin < ramax)
    assert (-90.0 <= decmin <= 90.0) & (-90.0 <= decmax <= 90.0) & (decmin < decmax)

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


def make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=np.linspace, f_ell=None, verbose=False, b_lmax=None, bins_ell=None):
    """Create a NaMaster bandpower binning object (:class:`pymaster.NmtBin`).

    Parameters
    ----------
    nside : int
        Healpix nside used to decide a default ``b_lmax = 3*nside - 1`` if not provided.
    lmin, lmax : int
        Minimum and maximum multipoles included in the binning.
    n_ell_bins : int
        Number of bandpower bins.
    bin_func : callable, optional
        Function used to generate bin edges as ``bin_func(lmin, lmax, n_ell_bins+1)``.
        Examples: :func:`numpy.linspace`, :func:`numpy.geomspace`.
    f_ell : {'pixwin'} or ndarray or None, optional
        Optional multiplicative factor applied to each ell (forwarded to NaMaster).
        - If 'pixwin', uses ``1/pixwin(nside)^2`` for deconvolving pixel window.
        - If ndarray, must have length ``lmax-lmin+1`` giving values for each ell.
    verbose : bool, optional
        If True, print bin ranges.
    b_lmax : int or None, optional
        Maximum multipole supported by the bin object. If None, uses ``3*nside-1``.
    bins_ell : ndarray or None, optional
        Directly specify bin edges (length ``n_ell_bins+1``). Overrides ``bin_func``.

    Returns
    -------
    pymaster.NmtBin
        NaMaster binning scheme.

    Warns
    -----
    UserWarning
        If ``lmax > 3*nside`` (multipoles above 3*nside are generally not meaningful
        for healpix maps and may be removed elsewhere).
    """
    ells = np.arange(lmin, lmax + 1)

    if lmax > 3 * nside:
        warnings.warn("Warning: lmax>3*nside so multipoles will be removed after 3*nside, but lmax will be used for binning.")

    # Make bin limits
    if bins_ell is not None:
        warnings.warn("Warning: using provided `bins_ell` instead of computing it from `bin_func`.")
        assert bins_ell[0] == lmin and bins_ell[-1] == lmax
    else:
        bins_ell = bin_func(lmin, lmax, n_ell_bins + 1).astype(float)

    # Get bands
    bpws = np.digitize(ells.astype(float), bins_ell) - 1
    bpws[0] = 0
    bpws[-1] = n_ell_bins - 1

    # Make sure each multipole is assigned to one of the n_ell_bins bins
    assert np.all(np.unique(bpws) == np.arange(n_ell_bins))

    # Multiplicative factor
    if f_ell is not None:
        if f_ell == 'pixwin':
            f_ell = 1.0 / hp.pixwin(nside)[lmin:lmax + 1]**2
        else:
            assert type(f_ell) == np.ndarray
            assert len(f_ell) == len(ells)

    if b_lmax is None:
        b_lmax = 3 * nside - 1

    b = nmt.NmtBin(ells=ells, bpws=bpws, weights=None, f_ell=f_ell, lmax=b_lmax)

    if b.get_n_bands() != n_ell_bins:
        warnings.warn("Warning: for some reason, b.get_n_bands() != n_ell_bins")

    if verbose:
        for i in range(b.get_n_bands()):
            print("Bin {:4d} = [{:5d} - {:5d}]".format(i, b.get_ell_list(i)[0], b.get_ell_list(i)[-1]))

    return b


def make_nmtbin_logspaced(nside, lmin, lmax, n_ell_bins, f_ell=None, **kwargs):
    """Convenience wrapper to create log-spaced NaMaster bins.

    Parameters
    ----------
    nside, lmin, lmax, n_ell_bins : int
        See :func:`make_nmtbin`.
    f_ell : {'pixwin'} or ndarray or None, optional
        See :func:`make_nmtbin`.
    **kwargs
        Forwarded to :func:`make_nmtbin`.

    Returns
    -------
    pymaster.NmtBin
        NaMaster bin object.
    """
    return make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=np.geomspace, f_ell=f_ell, **kwargs)


def make_nmtbin_linspaced(nside, lmin, lmax, n_ell_bins, f_ell=None, **kwargs):
    """Convenience wrapper to create linearly spaced NaMaster bins.

    Parameters
    ----------
    nside, lmin, lmax, n_ell_bins : int
        See :func:`make_nmtbin`.
    f_ell : {'pixwin'} or ndarray or None, optional
        See :func:`make_nmtbin`.
    **kwargs
        Forwarded to :func:`make_nmtbin`.

    Returns
    -------
    pymaster.NmtBin
        NaMaster bin object.
    """
    return make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=np.linspace, f_ell=f_ell, **kwargs)


def make_nmtbin_powspaced(nside, lmin, lmax, n_ell_bins, power, f_ell=None, **kwargs):
    """Convenience wrapper to create power-law spaced NaMaster bins.

    Parameters
    ----------
    nside, lmin, lmax, n_ell_bins : int
        See :func:`make_nmtbin`.
    power : float
        Exponent passed to :func:`powspace`.
    f_ell : {'pixwin'} or ndarray or None, optional
        See :func:`make_nmtbin`.
    **kwargs
        Forwarded to :func:`make_nmtbin`.

    Returns
    -------
    pymaster.NmtBin
        NaMaster bin object.
    """
    return make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=partial(powspace, power=power), f_ell=f_ell, **kwargs)


def split_bins(x, func=None, ifunc=None, zmax=None):
    """Compute bin edges from sample centers using a midpoint rule in transformed space.

    Given an array of bin centers ``x`` (monotonic), this constructs edges such that
    edges are midpoints between adjacent centers in the space defined by ``func``
    (or equivalently via its inverse ``ifunc``):

        edge[i] = ifunc( 0.5 * (func(x[i-1]) + func(x[i])) )

    The last edge is extrapolated symmetrically, unless ``zmax`` is provided
    to clamp it.

    Parameters
    ----------
    x : array_like
        Monotonically increasing bin centers.
    func : callable or None, optional
        Monotonic transform function. If None, it is inferred as the inverse of
        ``ifunc``.
    ifunc : callable or None, optional
        Inverse transform. If None, inferred as the inverse of ``func``.
    zmax : float or None, optional
        If provided, use this value as the final/high edge.

    Returns
    -------
    ndarray
        Array of length ``len(x)+1`` containing bin edges.

    Raises
    ------
    AssertionError
        If both ``func`` and ``ifunc`` are None.
    """
    assert func or ifunc

    if func is None:
        func = inversefunc(ifunc, domain=[0.0, None], image=[0.0, None])
    if ifunc is None:
        ifunc = inversefunc(func, domain=[0.0, None], image=[0.0, None])

    n = len(x)
    bins = np.zeros(n + 1)
    bins[1:n] = ifunc(0.5 * (func(x[:-1]) + func(x[1:])))

    if zmax is None:
        bins[n] = ifunc(2 * func(x[-1]) - func(bins[n - 1]))
    else:
        bins[n] = zmax

    return bins


def read_params(fname):
    """Read a HACC-style parameter file into a dict.

    The expected format is one entry per line::

        KEY value...

    Comment lines starting with ``#`` and blank lines are ignored.

    Parameters
    ----------
    fname : str
        File path to the parameter file.

    Returns
    -------
    dict
        Mapping from parameter names to raw string values.

    Notes
    -----
    Values are not parsed to numeric types here; callers are expected to cast
    them to float/int as needed.
    """
    with open(fname, 'r') as f:
        params = {}
        for x in f:
            if not (x.startswith('#') or x.startswith('\n')):
                try:
                    k, v = x.rstrip().split(' ', 1)
                    params[k] = v
                except ValueError:
                    pass
    return params


def save_dict_fits_file(fname, cat):
    """Write a dictionary of 1D float arrays to a FITS binary table.

    Parameters
    ----------
    fname : str
        Output file path. Existing files are overwritten.
    cat : dict
        Dictionary mapping column name to 1D numpy array.

    Notes
    -----
    Columns are written with FITS format='E' (32-bit float). If you need other
    dtypes, extend this helper accordingly.
    """
    hdul = [fits.PrimaryHDU()]
    header = fits.Header()

    cols = []
    for name, array in cat.items():
        cols.append(fits.Column(name=name, format='E', array=array))
    hdul.append(fits.BinTableHDU.from_columns(cols, header))

    fits.HDUList(hdul).writeto(fname, overwrite=True)


def trim(a, mindiff, return_indices=False):
    """Downsample a monotonically increasing array with a minimum spacing constraint.

    The algorithm keeps elements such that adjacent kept values are separated by
    at least ``mindiff`` whenever possible, while always keeping the first and
    last elements.

    Parameters
    ----------
    a : array_like
        Monotonically increasing 1D array.
    mindiff : float
        Minimum desired spacing between retained points.
    return_indices : bool, optional
        If True, also return the indices of retained elements in the input array.

    Returns
    -------
    trimmed : ndarray
        Trimmed array.
    indices : list of int, optional
        Only if ``return_indices=True``.

    Notes
    -----
    This is a greedy procedure; it does not guarantee a globally optimal subset
    under all possible criteria, but is fast and deterministic.
    """
    x = [a[0]]
    idx = [0]
    for i in range(1, len(a) - 1):
        if a[i + 1] - x[-1] > mindiff:
            x.append(a[i])
            idx.append(i)
    x.append(a[i + 1])
    idx.append(i + 1)
    if return_indices:
        return np.array(x), idx
    else:
        return np.array(x)


@numba.njit(parallel=True)
def inter_diff(idx1, idx2):
    """Return intersection and differences between two arrays of unique integers.

    Parameters
    ----------
    idx1, idx2 : ndarray of int
        1D arrays containing unique integer values.

    Returns
    -------
    idx1_int : ndarray of int
        Indices into ``idx1`` for elements also present in ``idx2``.
    idx2_int : ndarray of int
        Indices into ``idx2`` for elements also present in ``idx1``.
    idx1_diff : ndarray of int
        Indices into ``idx1`` for elements not present in ``idx2``.
    idx2_diff : ndarray of int
        Indices into ``idx2`` for elements not present in ``idx1``.

    Notes
    -----
    This Numba implementation concatenates and stable-sorts both arrays, then
    identifies adjacent matches for the intersection and non-matched entries for
    the differences.
    """
    idx = np.concatenate((idx1, idx2))

    order = np.argsort(idx, kind='mergesort')
    sidx = idx[order]

    # intersection
    mask = sidx[1:] == sidx[:-1]
    idx1_int = order[:-1][mask]
    idx2_int = order[1:][mask] - idx1.size

    # difference
    flag = np.concatenate((np.array([True]), ~mask, np.array([True])))
    z = order[flag[1:] & flag[:-1]]
    w = z >= idx1.size
    idx1_diff = z[~w]
    idx2_diff = z[w] - idx1.size

    return idx1_int, idx2_int, idx1_diff, idx2_diff


def arange_endpoint(start, end, step):
    """Like :func:`numpy.arange`, but force the last element to equal ``end``.

    Parameters
    ----------
    start, end, step : float
        Same meaning as in :func:`numpy.arange`.

    Returns
    -------
    ndarray
        Array starting at ``start`` with steps of ``step``, whose last value is
        set exactly to ``end``.

    Notes
    -----
    This is useful when floating rounding could otherwise miss the endpoint.
    """
    out = np.arange(start, end, step)
    out[-1] = end
    return out


def trunc_norm(myclip_a, myclip_b, loc, scale):
    """Construct a SciPy truncated normal distribution.

    Parameters
    ----------
    myclip_a, myclip_b : float
        Lower and upper bounds (in the same units as the distribution variable).
    loc : float
        Mean of the underlying normal distribution.
    scale : float
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    scipy.stats.truncnorm
        Frozen truncated normal distribution object.
    """
    a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
    return truncnorm(loc=loc, scale=scale, a=a, b=b)


def subsample_ppf(z, n, ppf, keep_first_last=True):
    """Subsample points to match a target distribution via its percent point function.

    Given a sorted grid of candidate points ``z`` and a target percent point
    function (inverse CDF) ``ppf``, this function finds ``n`` indices in ``z``
    closest to ``ppf(u)`` for a uniform grid ``u`` in [0, 1].

    It iteratively refines the ``u`` grid (increasing its resolution) until the
    nearest-neighbor selection yields exactly ``n`` unique indices.

    Parameters
    ----------
    z : ndarray
        1D array of candidate points (typically redshifts or scale factors).
    n : int
        Number of points to keep.
    ppf : callable
        Percent point function (inverse CDF) mapping u in [0,1] to values in the
        domain of ``z``.
    keep_first_last : bool, optional
        If True, include u=0 and u=1 in the trial grid. If False, use midpoints
        of equal-width bins in u.

    Returns
    -------
    unique_idx : ndarray of int
        Indices in ``z`` of the chosen points.
    z_sub : ndarray
        The selected subset ``z[unique_idx]``.

    Notes
    -----
    Uses a 1D :class:`scipy.spatial.KDTree` nearest-neighbor query. The returned
    indices are unique but not necessarily evenly spaced in ``z``.
    """
    tree = KDTree(z[:, None])
    for usize in range(n, len(z) + 1):
        if keep_first_last:
            u = np.linspace(0, 1, usize)
        else:
            uedge = np.linspace(0, 1, usize + 1)
            u = 0.5 * (uedge[1:] + uedge[:-1])
        best_idx = tree.query(ppf(u)[:, None])[1]
        unique_idx = np.unique(best_idx)
        if len(unique_idx) == n:
            break
    if usize > n:
        print(f"[subsample_ppf] had to try finer grid {usize}>{n}")
    return unique_idx, z[unique_idx]


def sample_pdf(x, pdf, n, rng):
    """Sample from a 1D distribution specified by samples of its PDF.

    The PDF is assumed piecewise-linear between points in ``x``. Sampling is
    performed by building a trapezoidal CDF and inverting it via interpolation.

    Parameters
    ----------
    x : ndarray
        1D grid (must be increasing).
    pdf : ndarray
        PDF values on the grid (non-negative). Does not need to be normalized.
    n : int
        Number of random draws.
    rng : numpy.random.Generator
        Random number generator used to draw uniform variates.

    Returns
    -------
    ndarray
        Array of ``n`` samples distributed approximately according to the input PDF.

    Notes
    -----
    This routine uses trapezoidal integration to approximate the CDF:
    ``CDF[i] = \int_{x0}^{xi} pdf(x) dx``.
    """
    dx = np.diff(x)
    trapz = 0.5 * (pdf[1:] + pdf[:-1]) * dx
    cdf = np.cumsum(trapz) / np.sum(trapz)
    cdf = np.insert(cdf, 0, 0.0)
    u = rng.uniform(size=n)
    return np.interp(u, cdf, x)
