"""Numerical helpers: integration, grids, bin edges and sampling."""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from pynverse import inversefunc
from scipy.integrate import quad as scipy_quad
from scipy.spatial import KDTree
from scipy.stats import truncnorm

logger = logging.getLogger(__name__)

QUAD_EPS_REL = 1e-6
QUAD_EPS_ABS = 0.0
QUAD_LIMIT = 1000


def quad(*args: Any, **kwargs: Any) -> float:
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


def powspace(start: float, stop: float, num: int, power: float) -> np.ndarray:
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


def arange_endpoint(start: float, end: float, step: float) -> np.ndarray:
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


def trim(a: np.ndarray, mindiff: float, return_indices: bool = False) -> np.ndarray | tuple[np.ndarray, list[int]]:
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


def split_bins(x: np.ndarray, func: Callable | None = None, ifunc: Callable | None = None, zmax: float | None = None) -> np.ndarray:
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
    ValueError
        If both ``func`` and ``ifunc`` are None.
    """
    if not (func or ifunc):
        raise ValueError("func or ifunc must be given")

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


def trunc_norm(myclip_a: float, myclip_b: float, loc: float, scale: float) -> Any:
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


def subsample_ppf(z: np.ndarray, n: int, ppf: Callable, keep_first_last: bool = True) -> tuple[np.ndarray, np.ndarray]:
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
        logger.debug("[subsample_ppf] had to try finer grid %d>%d", usize, n)
    return unique_idx, z[unique_idx]


def sample_pdf(x: np.ndarray, pdf: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
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
    ``CDF[i] = int_{x0}^{xi} pdf(x) dx``.
    """
    dx = np.diff(x)
    trapz = 0.5 * (pdf[1:] + pdf[:-1]) * dx
    cdf = np.cumsum(trapz) / np.sum(trapz)
    cdf = np.insert(cdf, 0, 0.0)
    u = rng.uniform(size=n)
    return np.interp(u, cdf, x)
