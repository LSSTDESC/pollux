"""NaMaster bandpower binnings."""
from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Callable

import healpy as hp
import numpy as np
import pymaster as nmt

from .numerics import powspace


def make_nmtbin(
        nside: int,
        lmin: int,
        lmax: int,
        n_ell_bins: int,
        bin_func: Callable = np.linspace,
        f_ell: str | np.ndarray | None = None,
        verbose: bool = False,
        b_lmax: int | None = None,
        bins_ell: np.ndarray | None = None) -> nmt.NmtBin:
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
            f_ell = np.asarray(f_ell)
            if len(f_ell) != len(ells):
                raise ValueError(f"f_ell has {len(f_ell)} values, but there are {len(ells)} multipoles")

    if b_lmax is None:
        b_lmax = 3 * nside - 1

    b = nmt.NmtBin(ells=ells, bpws=bpws, weights=None, f_ell=f_ell, lmax=b_lmax)

    if b.get_n_bands() != n_ell_bins:
        warnings.warn("Warning: for some reason, b.get_n_bands() != n_ell_bins")

    if verbose:
        for i in range(b.get_n_bands()):
            print("Bin {:4d} = [{:5d} - {:5d}]".format(i, b.get_ell_list(i)[0], b.get_ell_list(i)[-1]))

    return b


def make_nmtbin_logspaced(nside: int, lmin: int, lmax: int, n_ell_bins: int, f_ell: str | np.ndarray | None = None, **kwargs: Any) -> nmt.NmtBin:
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


def make_nmtbin_linspaced(nside: int, lmin: int, lmax: int, n_ell_bins: int, f_ell: str | np.ndarray | None = None, **kwargs: Any) -> nmt.NmtBin:
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


def make_nmtbin_powspaced(nside: int, lmin: int, lmax: int, n_ell_bins: int, power: float, f_ell: str | np.ndarray | None = None, **kwargs: Any) -> nmt.NmtBin:
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
