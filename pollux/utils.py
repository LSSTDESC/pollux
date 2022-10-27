import itertools
from re import A
import warnings
from functools import partial

import healpy as hp
import numba
import numpy as np
import pymaster as nmt
from scipy.spatial.transform import Rotation


# @numba.njit(parallel=False)
# def _add_at_one(a, index):
#     n = len(index)
#     for i in numba.prange(n):
#         a[index[i]] += 1


@numba.njit(parallel=True)
def xyz2r2(xyz):
    return xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2


# @numba.njit(parallel=True)
# def xyz2r(xyz):
#     return np.sqrt(xyz2r2(xyz))


# @numba.njit(parallel=True)
# def xyzr2tp(xyz, r):
#     assert len(xyz) == len(r)
#     n = len(xyz)
#     x = xyz[:, 0]
#     y = xyz[:, 1]
#     z = xyz[:, 2]
#     theta = np.arccos(z / r)
#     phi = np.arctan2(x, y)
#     return theta, phi


@numba.njit(parallel=True)
def Axpb(A, b, xyz):
    return np.dot(xyz + b, A)


@numba.njit
def Ax(A, xyz):
    return np.dot(xyz, A)


# @numba.njit(parallel=True)
# def Axpb_inv(A, b, xyz):
#     return np.dot(xyz, A.T) - b


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


@numba.njit
def _AABB_sphere_intersect(box_edges, s, r2):
    # see https://stackoverflow.com/a/4579069
    # c1, c2 = box_edges.min(axis=0), box_edges.max(axis=0)
    d2 = r2
    for i in range(3):
        c1 = np.min(box_edges[:,i])
        c2 = np.max(box_edges[:,i])
        if s[i] < c1:
            d2 -= (s[i] - c1)**2
        elif s[i] > c2:
            d2 -= (s[i] - c2)**2
    return d2 > 0.


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


inv_halfpi = 2./np.pi

@numba.njit
def _loc2pix(nside, order, ncap, npix, z, phi, sth):
    """"
    Healpix function loc2pix
    """
    za = np.abs(z)
    tt = phi * inv_halfpi % 4.
    if za <= 2./3.:
        nl4 = 4*nside
        temp1 = nside*(0.5+tt)
        temp2 = nside*z*0.75
        jp = int(temp1-temp2) # index of  ascending edge line
        jm = int(temp1+temp2) # index of descending edge line

        # ring number counted from z=2/3
        ir = nside + 1 + jp - jm # in {1,2n+1}
        kshift = 1-(ir%2) # kshift=1 if ir even, 0 otherwise

        t1 = jp+jm-nside+kshift+1+nl4+nl4
        if order>0:
            ip = (t1>>1)&(nl4-1)
        else:
            ip = ((t1>>1)%nl4) # in {0,4n-1}

        return ncap + (ir-1)*nl4 + ip
    
    else : # Equatorial region
        tp = tt-np.floor(tt)
        if za<0.99:
            tmp = nside*np.sqrt(3*(1-za))
        else:
            tmp = nside*sth/np.sqrt((1.+za)/3.)

        jp = int(tp*tmp) # increasing edge line index
        jm = int((1.-tp)*tmp) # decreasing edge line index

        ir = jp+jm+1 # ring number counted from the closest pole
        ip = int(tt*ir) # in {0,4*ir-1}

        if z>0:
            return 2*ir*(ir-1) + ip
        else:
            return npix - 2*ir*(ir+1) + ip

@numba.njit
def _vec2pix_sub(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """
    Healpix function vec2pix
    """
    # apply shift
    tmpx = x + shiftx
    tmpy = y + shifty
    tmpz = z + shiftz
    
    # get radius and remove particles out of shel
    r2 = tmpx**2 + tmpy**2 + tmpz**2
    if r2 < rmin2 or r2 >= rmax2:
        return -1
    
    # remove particles out of octant if octant_only
    if octant_only: # check if particle out of octant
        if (tmpx<0.) or (tmpy<0.) or (tmpz<0.):
            return -1
        
    # get longitude (phi), altitude (nz) and distance to z-axis (sth) of particle projected on the unit sphere
    phi = np.arctan2(tmpy, tmpx)
    xl = 1./np.sqrt(r2)
    nz = tmpz*xl
    sth = np.sqrt(tmpx**2+tmpy**2)*xl
    
    return _loc2pix(nside, order, ncap, npix, nz, phi, sth)

@numba.njit(parallel=True)
def _vec2pix(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """"
    Parallelized Healpix function vec2pix to go from x,y,z to healpix pixel
    """
    n = len(x)
    ipix = np.zeros(n, dtype=np.int_)
    for i in numba.prange(n):
        ipix[i] = _vec2pix_sub(nside, order, ncap, npix, x[i], y[i], z[i], shiftx, shifty, shiftz, rmin2, rmax2, octant_only)
    return ipix

@numba.njit
def unique_pix_pos(ipix, npix):
    n = len(ipix)
    c = np.zeros(npix, dtype=np.int_)
    for i in range(n):
        pos = ipix[i]
        if pos>=0:
            c[pos] += 1
    u = np.argwhere(c)[:,0]
    return u, c[u] 

@numba.njit
def _vec2pix_unique(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    ipix = _vec2pix(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only)
    return unique_pix_pos(ipix, npix)

def vec2pix_unique(nside, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    order = hp.nside2order(nside)
    ncap = (nside*nside-nside)<<1
    npix = hp.nside2npix(nside)
    return _vec2pix_unique(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only)


@numba.njit
def _construct_shell_sub(xyz, box_edges_original, A, b, nrepeat, boxsize, rmin2, rmax2, apply_random_isometry, octant_only, nside, order, ncap, npix):
    n = len(xyz)
    
    # precompute isometry on particles (since repeats are then only a shift away)
    if apply_random_isometry:
        xyz = Axpb(A, b, xyz)      
        box_edges = Axpb(A, b, box_edges_original)      
    else:
        box_edges = box_edges_original
    
    count = np.zeros(npix, dtype=np.int_)

    repeats = []

    # repeat the box
    for ix in range(-nrepeat, nrepeat + 1):
        for iy in range(-nrepeat, nrepeat + 1):
            for iz in range(-nrepeat, nrepeat + 1):
                # compute shift
                shift_original = np.expand_dims(np.array([ix, iy, iz]) * boxsize, 0)
                if apply_random_isometry:
                    shift = Ax(A, shift_original)
                else:
                    shift = shift_original
                    
                # apply isometry to box edges
                tmp_box_edges = shift + box_edges

                # check if intersects octant, otherwise skip
                if octant_only:
                    found_edge_in_octant = False
                    for e in tmp_box_edges:
                        if np.all(e>0.):
                            found_edge_in_octant = True
                    if not found_edge_in_octant:
                        continue

                ## check if box intersects with shell, otherwise skip
                # first, check if box fully inside inner sphere
                if np.all(xyz2r2(tmp_box_edges) < rmin2):
                    continue

                # second, check if box intersects with outer sphere
                sphere_center = -b if apply_random_isometry else np.zeros(3)
                if not _AABB_sphere_intersect(box_edges_original+shift_original, sphere_center, rmax2):
                    continue

                # if we got this far, then this repeat intersects with the shell
                repeats.append([ix, iy, iz])
                
                ## add particles to shell
                # compute corresponding healpy pixels of all particles (particles out of shell/octant quickly set to -1) with vec2pix and loc2pix (translated into python from Healpix code)
                ipix = _vec2pix(nside, order, ncap, npix,
                                xyz[:,0], xyz[:,1], xyz[:,2],
                                shift[0,0], shift[0,1], shift[0,2],
                                rmin2, rmax2, octant_only)

                for i in range(n):
                    pos = ipix[i]
                    if pos>=0:
                        count[pos] += 1
                    
    u = np.argwhere(count)[:,0]
    return u, count[u] 
