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


@numba.njit(parallel=False)  # this operation does NOT support parallelization!
def _add_at_cst(a, index, b):
    n = len(index)
    for i in numba.prange(n):
        a[index[i]] += b


@numba.njit(parallel=False)
def _add_at(a, index, b):
    n = len(b)
    for i in numba.prange(n):
        a[index[i]] += b[i]


def make_healpix_map(
        ra,
        dec,
        quantity,
        nside,
        mask=None,
        weight=None,
        ipix=None,
        fill_UNSEEN=False,
        return_w_maps=False,
        return_extra=False,
        nest=True,
        mode='mean'):
    """
    Creates healpix maps of quantity observed at ra, dec (in degrees) by taking
    the mean or sum of quantity in each pixel.

    Parameters
    ----------
    ra : array
        Right ascension.
    dec : array
        Declination.
    quantity : array
        `quantity` can be 2D, in which case several maps are created.
    nside : int
        `nside` parameter for healpix.
    mask : array
        If None, the mask is created and has value 1 in pixels that contain at
        least one object, 0 elsewhere.
    weight : type
        Weights of objects (the default is None, in which case all objects have
        weight 1). Must be the same size as `quantity`.
    ipix : array
        `ipix` should be the array of healpix pixel indices corresponding to the
        input `ra` and `dec`. By default it is None and will be computed.

    fill_UNSEEN : boolean
        If `fill_UNSEEN` is True, pixels outside the mask are filled with
        hp.UNSEEN, 0 otherwise (the default is False).
    return_extra : boolean
        If True, a dictionnary is returned that contains count statistics and
        the masked `ipix` array to allow for statistics on the quantities to be
        computed.
    mode : string
        Whether to return the 'mean' or 'sum' of quantity in each pixel.

    Returns
    -------
    List of outmaps, the count map and the mask map.

    """

    if quantity is not None:
        quantity = np.atleast_2d(quantity)

        if weight is not None:
            w = np.atleast_2d(weight)
            # Weights can also be the same for all quantities
            # assert quantity.shape==weight.shape, "[make_healpix_map] quantity and weight must have the same shape"
            if w.shape[0] > 1:
                assert quantity.shape == w.shape, "[make_healpix_map] quantity/weight arrays don't have the same length"
            else:
                w = np.tile(w[0], (quantity.shape[0], 1))

            assert np.all(w > 0.), "[make_healpix_map] weight is not strictly positive"
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
    # np.add.at(count, ipix, 1.)
    _add_at_cst(count, ipix, 1.)

    # Creating the mask if it does not exist
    if mask is None:
        bool_mask = (count > 0)
    else:
        bool_mask = mask.astype(bool)

    # # Masking the count in the masked area
    # count[np.logical_not(bool_mask)] = x
    # if mask is None:
    #     assert np.all(count[bool_mask] > 0), "[make_healpix_map] count[bool_mask] is not positive on the provided mask !"

    # Create the maps
    if quantity is not None:
        for i in range(quantity.shape[0]):
            sum_w = np.zeros(npix, dtype=float)
            # np.add.at(sum_w, ipix, w[i,:])
            _add_at(sum_w, ipix, w[i, :])

            outmap = np.zeros(npix, dtype=float)
            # np.add.at(outmap, ipix, quantity[i,:]*w[i,:])
            _add_at(outmap, ipix, quantity[i, :] * w[i, :])

            if mode == 'mean':
                outmap[bool_mask] /= sum_w[bool_mask]

            outmap[np.logical_not(bool_mask)] = x

            outmaps.append(outmap)
            if return_w_maps:
                sum_w_maps.append(sum_w)

    if mask is None:
        returned_mask = bool_mask.astype(float)
    else:
        returned_mask = mask

    res = [outmaps, count, returned_mask]

    if return_w_maps:
        res += [sum_w_maps]

    if return_extra:
        extra = {}
        extra['count_tot_in_mask'] = np.sum(count[bool_mask])
        extra['count_per_pixel_in_mask'] = extra['count_tot_in_mask'] * 1. / np.sum(bool_mask.astype(int))
        extra['count_per_steradian_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=False)
        extra['count_per_sqdegree_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=True)
        extra['count_per_sqarcmin_in_mask'] = extra['count_per_sqdegree_in_mask'] / 60.**2
        extra['ipix_masked'] = np.ma.array(ipix, mask=bool_mask[ipix])

        res += [extra]

    return res


def make_mask_radec_cuts(nside, ramin, ramax, decmin, decmax, dtype=float, nest=True):
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
    th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=nest)
    ths, phs = radec2thetaphi([ramin, ramax], [decmin, decmax])
    thmin, thmax = min(ths), max(ths)
    phmin, phmax = min(phs), max(phs)
    mask[np.where((th <= thmax) & (th > thmin) & (ph >= phmin) & (ph < phmax))[0]] = dtype(1)
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


def trim_array(a, mindiff, return_indices=False):
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


def compute_shell_edges(zsnapshots, func=None, ifunc=None, zmax=None, return_fzbins=False):
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


def trunc_norm(myclip_a, myclip_b, loc, scale):
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


def sample_pdf(x, pdf, n, rng):
    """
    Sample a distribution given its pdf.
    """
    dx = np.diff(x)
    trapz = 0.5 * (pdf[1:] + pdf[:-1]) * dx
    cdf = np.cumsum(trapz) / np.sum(trapz)
    cdf = np.insert(cdf, 0, 0.)
    u = rng.uniform(size=n)
    return np.interp(u, cdf, x)


def save_dict_fits_file(fname, cat):
    hdul = [fits.PrimaryHDU()]
        
    header = fits.Header()
    cols = []
    for name, array in cat.items():
        cols.append(fits.Column(name=name, format='E', array=array))

    hdul.append(fits.BinTableHDU.from_columns(cols, header))

    fits.HDUList(hdul).writeto(fname, overwrite=True)