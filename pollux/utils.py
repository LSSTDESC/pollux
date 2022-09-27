import itertools
import warnings
from functools import partial

import healpy as hp
import numba
import numpy as np
import pymaster as nmt
from scipy.spatial.transform import Rotation


@numba.jit(nopython=True, parallel=False)
def _add_at_one(a, index):
    n = len(index)
    for i in numba.prange(n):
        a[index[i]] += 1


@numba.jit(nopython=True, parallel=True)
def xyz2r2(xyz):
    return xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2


@numba.jit(nopython=True, parallel=True)
def xyz2r(xyz):
    return np.sqrt(xyz2r2(xyz))


@numba.jit(nopython=True, parallel=True)
def xyzr2tp(xyz, r):
    assert len(xyz) == len(r)
    n = len(xyz)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    theta = np.arccos(z / r)
    phi = np.arctan2(x, y)
    return theta, phi


@numba.jit(nopython=True, parallel=True)
def Axpb(A, b, xyz):
    return np.dot(xyz + b, A)


@numba.jit(nopython=True)
def Ax(A, xyz):
    return np.dot(xyz, A)


@numba.jit(nopython=True, parallel=True)
def Axpb_inv(A, b, xyz):
    return np.dot(xyz, A.T) - b


def random_isometry(rng, boxsize):
    pcomb = list(itertools.product([1, -1], repeat=3))  # all permuted orders
    r = Rotation.random(random_state=rng).as_matrix()
    p = pcomb[rng.choice(len(pcomb))]
    A = np.dot(r, np.diag(np.array(p)))
    b = rng.random(3) * boxsize
    return A, b


def AABB_sphere_intersect(box_edges, s, r2):
    # see https://stackoverflow.com/a/4579069
    c1, c2 = box_edges.min(axis=0), box_edges.max(axis=0)
    d2 = r2
    for i in range(3):
        if s[i] < c1[i]:
            d2 -= (s[i] - c1[i])**2
        elif s[i] > c2[i]:
            d2 -= (s[i] - c2[i])**2
    return d2 > 0.


def flatten(xss):
    return [x for xs in xss for x in xs]


def powspace(start, stop, num, power):
    """
    Like linspace or geomspace but with a power law scaling.
    """
    start = np.power(start, float(power))
    stop = np.power(stop, float(power))
    return np.power(np.linspace(start, stop, num=num), 1. / power)


def radec2thetaphi(ra, dec):
    """
    Converts ra and dec (in degrees) to theta and phi (Healpix conventions, in radians)

    """
    theta = np.pi / 2. - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    return theta, phi


def thetaphi2radec(theta, phi):
    """
    Converts theta and phi (Healpix conventions, in radians) to ra and dec (in degrees)

    """
    ra = np.rad2deg(phi)
    dec = 90. - np.rad2deg(theta)
    return ra, dec


def make_mask_radec_cuts(nside, ramin, ramax, decmin, decmax):
    """
    Generates a square mask limited by RA/DEC.

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
    mask[np.where((th < thmax) & (th > thmin) & (ph > phmin) & (ph < phmax))[0]] = 1.
    return mask


def make_nmtbin(nside,
                lmin,
                lmax,
                n_ell_bins,
                bin_func=np.linspace,
                f_ell=None,
                verbose=False,
                b_lmax=None,
                bins_ell=None,
                **kwargs):
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
    return make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=np.geomspace, f_ell=f_ell, **kwargs)


def make_nmtbin_linspaced(nside, lmin, lmax, n_ell_bins, f_ell=None, **kwargs):
    return make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=np.linspace, f_ell=f_ell, **kwargs)


def make_nmtbin_powspaced(nside, lmin, lmax, n_ell_bins, power, f_ell=None, **kwargs):
    return make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=partial(powspace, power=power), f_ell=f_ell, **kwargs)


# hp_UNSEEN = hp.UNSEEN
# @numba.jit(nopython=True, parallel=False)
# def remove_UNSEEN_in_mask(hpmap, mask, value=0.):
#     out = np.zeros_like(hpmap)
#     for i in range(len(out)):
#         if mask[i]:
#             if hpmap[i] == hp.UNSEEN:
#                 out[i] = value
#         else:
#             out[i] = hpmap[i]

#     return out
#     # x = hpmap[mask.astype(bool)]
#     # w = np.argwhere(x == )
#     # x[w] = value


def trim_array_min_diff(a, mindiff, return_indices=False):
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
