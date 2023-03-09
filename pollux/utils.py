import warnings
from functools import partial

import healpy as hp
import numpy as np
import pymaster as nmt
from pynverse import inversefunc
from scipy.integrate import quad as scipy_quad
from scipy.spatial import KDTree
from scipy.stats import truncnorm

QUAD_EPS_REL = 1e-6
QUAD_EPS_ABS = 0.
QUAD_LIMIT = 1000


def quad(*args, **kwargs):
    """
    Simple wrapper to uniformize quad options
    """
    return scipy_quad(*args, epsabs=QUAD_EPS_ABS, epsrel=QUAD_EPS_REL, limit=QUAD_LIMIT, **kwargs)[0]


def powspace(start, stop, num, power):
    """
    Like linspace or geomspace but with a power law scaling.
    """
    start = np.power(start, float(power))
    stop = np.power(stop, float(power))
    return np.power(np.linspace(start, stop, num=num), 1. / power)


def radec2thetaphi(ra, dec):
    """
    Convert ra and dec (in degrees) to theta and phi (Healpix conventions, in radians)

    """
    theta = np.pi / 2. - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    return theta, phi


def thetaphi2radec(theta, phi):
    """
    Convert theta and phi (Healpix conventions, in radians) to ra and dec (in degrees)

    """
    ra = np.rad2deg(phi)
    dec = 90. - np.rad2deg(theta)
    return ra, dec


def make_mask_radec_cuts(nside, ramin, ramax, decmin, decmax, dtype=float):
    """Generate a square mask limited by RA/DEC

    Parameters
    ----------
    nside : int
        `nside` parameter for healpix.
    ramin, ramax : float
        Should be in degrees in [0., 360.]
    decmin, decmax: [type]
        Should be in degrees in [-90., +90.]

    Returns
    -------
    array
        Returns the square mask.
    """
    # Make sure arguments are correct
    assert (0. <= ramin <= 360.) & (0. <= ramax <= 360.) & (ramin < ramax)
    assert (-90. <= decmin <= 90.) & (-90. <= decmax <= 90.) & (decmin < decmax)

    mask = np.zeros(hp.nside2npix(nside))
    th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    ths, phs = radec2thetaphi([ramin, ramax], [decmin, decmax])
    thmin, thmax = min(ths), max(ths)
    phmin, phmax = min(phs), max(phs)
    mask[np.where((th < thmax) & (th > thmin) & (ph > phmin) & (ph < phmax))[0]] = dtype(1)
    return mask


def make_nmtbin(
        nside,
        lmin,
        lmax,
        n_ell_bins,
        bin_func=np.linspace,
        f_ell=None,
        verbose=False,
        b_lmax=None,
        bins_ell=None,
        **kwargs):
    """Create a NaMaster binning object `NmtBin`

    Parameters
    ----------
    nside
        `nside` parameter for healpix.
    lmin
        Minimum multipole
    lmax
        Maximum multipole
    n_ell_bins
        Number of bins
    bin_func, optional
        Binning function, by default np.linspace
    f_ell, optional
        Function of ell to include, can be `"pix"` to correct for the Healpix pixel window function, by default None
    verbose, optional
        Feedback, by default False
    b_lmax, optional
        Maximum multipole to include, by default None
    bins_ell, optional
        Directly provide bins in multipole, by default None

    Returns
    -------
        A `NmtBin` object
    """
    # Define ell range
    ells = np.arange(lmin, lmax + 1)

    if lmax > 3 * nside:
        warnings.warn(
            "Warning: lmax>3*nside so multipoles will be removed after 3*snide, but lmax will be used for binning.")

    # Make bin limits
    if bins_ell is not None:
        warnings.warn("Warning: using provided `bins_ell` instead of computing it from `bin_func`.")
        assert bins_ell[0] == lmin and bins_ell[-1] == lmax
    else:
        bins_ell = bin_func(lmin, lmax, n_ell_bins + 1).astype(float)

    # Get bands
    bpws = np.digitize(ells.astype(float), bins_ell) - 1
    bpws[0] = 0  # fixes bin edge issues...
    bpws[-1] = n_ell_bins - 1

    # Make sure each multipole is assigned to one of the n_ell_bins bins
    assert np.all(np.unique(bpws) == np.arange(n_ell_bins))

    # Multiplicative factor
    if f_ell is not None:
        if f_ell == 'pixwin':
            f_ell = np.ones(len(ells))
            f_ell[:min(lmax + 1, 3 * nside) - lmin] = 1. / hp.pixwin(nside)[lmin:min(lmax + 1, 3 * nside)]**2
        else:
            assert type(f_ell) == np.ndarray
            assert len(f_ell) == len(ells)

    b = nmt.NmtBin(nside, ells=ells, bpws=bpws, weights=np.ones(len(ells)), f_ell=f_ell, lmax=b_lmax, **kwargs)

    if b.get_n_bands() != n_ell_bins:
        warnings.warn("Warning: for some reason, b.get_n_bands() != n_ell_bins")

    if verbose:
        for i in range(b.get_n_bands()):
            print("Bin {:4d} = [{:5d} - {:5d}]".format(i, b.get_ell_list(i)[0], b.get_ell_list(i)[-1]))

    return b


def make_nmtbin_logspaced(nside, lmin, lmax, n_ell_bins, f_ell=None, **kwargs):
    """
    Helper to generate logarithmically spaced `NmtBin`
    """
    return make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=np.geomspace, f_ell=f_ell, **kwargs)


def make_nmtbin_linspaced(nside, lmin, lmax, n_ell_bins, f_ell=None, **kwargs):
    """
    Helper to generate linearly spaced `NmtBin`
    """
    return make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=np.linspace, f_ell=f_ell, **kwargs)


def make_nmtbin_powspaced(nside, lmin, lmax, n_ell_bins, power, f_ell=None, **kwargs):
    """
    Helper to generate power-law spaced `NmtBin`
    """
    return make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=partial(powspace, power=power), f_ell=f_ell, **kwargs)


def trim_array_min_diff(a, mindiff, return_indices=False):
    """
    Trim a monotonously increasing array such that the difference between adjacent values is the largest possible below `mindiff`
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


def read_params(fname):
    """
    Read cosmology from HACC run files
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


def get_z_shell_fz(zsnapshots, func=None, ifunc=None, zmax=None, return_fzbins=False):
    """
    Compute the redshift of the shell edges, set at the center in some space defined by `func` or its inverse `ifunc`
    """
    assert func or ifunc
    if func is None:
        func = inversefunc(ifunc, domain=[0., None], image=[0., None])
    if ifunc is None:
        ifunc = inversefunc(func, domain=[0., None], image=[0., None])
    nshell = len(zsnapshots)
    z_shell_bins = np.zeros(nshell + 1)
    z_shell_bins[1:nshell] = ifunc(0.5 * (func(zsnapshots[:-1]) + func(zsnapshots[1:])))
    if zmax is None:
        z_shell_bins[nshell] = ifunc(2 * func(zsnapshots[-1]) - func(z_shell_bins[nshell - 1]))
    else:
        z_shell_bins[nshell] = zmax
    if return_fzbins:
        return z_shell_bins, func(z_shell_bins)
    else:
        return z_shell_bins


def get_trunc_norm(myclip_a, myclip_b, loc, scale):
    """
    Create a truncated normal distribution
    """
    a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
    return truncnorm(loc=loc, scale=scale, a=a, b=b)


def subsample_ppf(z, n, ppf, keep_first_last=True):
    """
    Compute the redshifts `z` that are the closest to points spread uniformly for the distribution defined by its percent point function `ppf`
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