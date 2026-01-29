import numpy as np
import healpy as hp
import healsparse
from astropy.io import fits

from .utils import radec2thetaphi, quad, sample_pdf


class CatalogSourceGenerator(object):
    """Generic generator for mock source-galaxy catalogs for lensing analyses.

    This class generates positions and redshifts for galaxies in each tomographic
    bin and interpolates lensing quantities (e.g. kappa/gamma) from maps produced
    by a :class:`~pollux.lensing.LensingBuilder`.

    Parameters
    ----------
    lensing : pollux.lensing.LensingBuilder
        Lensing map builder used to define tomographic bins and provide maps to
        interpolate.
    name : str
        Catalog name used for book-keeping and file naming. This argument is
        required and should be provided first when constructing catalog
        generators.
    nbar : float or array_like or None
        Galaxy surface density in gal/arcmin^2. If a float, the same surface
        density is used for all tomographic bins; if an array_like, it must
        have length ``lensing.nmaps`` and provide a per-bin value. If ``None``,
        the values are taken from ``lensing.nbar`` (API units: gal/arcmin^2).
    interp_list : list of str, optional
        Names of fields to interpolate from maps. Typical values are
        ``['kappa', 'gamma1', 'gamma2']``.
    interp_pixels : bool, optional
        If True, use pixel interpolation at continuous (RA, DEC) coordinates.
        If False, evaluate maps at the nearest pixel center (via ``ang2pix``).

    Attributes
    ----------
    Ngal : ndarray of int
        Number of galaxies drawn per tomographic bin (deterministic, derived from
        ``nbar`` and footprint area).
    nbar_per_pix : ndarray
        Mean expected counts per pixel per tomographic bin (using pixel area at
        this nside).
    header_dict : dict
        FITS header keyword/value pairs stored in the output catalogs.
    """

    def __init__(self, lensing, name, nbar=None, interp_list=['kappa', 'gamma1', 'gamma2'], interp_pixels=True):
        self.lensing = lensing
        self.snap_lensing = self.lensing.shell_builder.get_snapshot_lensing_builder()
        self.nside = self.lensing.nside
        self.ncats = self.lensing.nmaps
        self.nshells = self.lensing.shell_builder.nshells
        self.interp_list = interp_list
        self.interp_pixels = interp_pixels
        self.name = name

        # Use lensing nbar by default, otherwise accept scalar or per-bin array
        if nbar is None:
            nbar_arr = np.atleast_1d(self.lensing.nbar).astype(float)
        else:
            nbar_arr = np.atleast_1d(nbar).astype(float)
            if nbar_arr.size == 1:
                nbar_arr = np.full(self.lensing.nmaps, float(nbar_arr[0]))
            elif nbar_arr.size != self.lensing.nmaps:
                raise ValueError(f"nbar must be a scalar or array of length {self.lensing.nmaps}, got {nbar_arr.size}")
        self.nbar = nbar_arr

        # number of galaxies to draw (or mean for Poisson sampling)
        area_std = np.pi / 2. if self.lensing.octant_only else 4. * np.pi
        area_arcmin = area_std * (180. * 60 / np.pi)**2
        self.Ngal = (self.nbar * area_arcmin).astype(int)
        self.area_pix = hp.nside2pixarea(self.nside, degrees=False) * (180. * 60. / np.pi)**2
        self.nbar_per_pix = self.nbar * self.area_pix

        self.header_dict = {f'NBAR_{i}': n for i, n in enumerate(self.nbar)}

    def generate(self, ikappas=None):
        """Generate mock catalogs and interpolate requested map quantities.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate. If None, generate for all bins.

        Notes
        -----
        The generation proceeds in three steps:
        1) draw galaxies (positions/redshifts and shell membership),
        2) interpolate requested fields from maps,
        3) save one FITS file per tomographic bin.
        """
        catalog = [{} for _ in range(self.lensing.nmaps)]

        # Generate ra, dec, z, etc.
        self._generate_galaxies(catalog, ikappas=ikappas)

        # Interpolate kappa/shear values
        self._interpolate_maps(catalog, ikappas=ikappas)

        # Save it!
        self._save_catalogs(catalog, ikappas=ikappas)

    def _generate_galaxies(self, catalog, ikappas=None):
        """Populate a catalog container with galaxy properties.

        Parameters
        ----------
        catalog : list of dict
            List of per-bin dictionaries that will be filled in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _interpolate_maps(self, catalog, ikappas=None):
        """Interpolate map quantities at galaxy positions.

        Parameters
        ----------
        catalog : list of dict
            Per-bin catalog dictionaries containing at least ``'ra'``, ``'dec'``,
            ``'chi'`` and ``'idx_shell'``.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to process.

        Notes
        -----
        This performs linear interpolation between consecutive shells in comoving
        distance and optionally interpolates across pixels (via HealSparse
        ``interpolate_pos``). For ``k='delta'``, a density contrast map is built
        from snapshot-lensing density shells.
        """
        builder = self.lensing.shell_builder
        chi_bins_edges = builder.shell_edges_comoving_distance()

        # Interpolate shear values, looping over shells for speed
        for k in self.interp_list:
            map_high = None
            for ishell in self.lensing.shell_builder.tqdm(None, desc=f"[generate] interpolating maps {k}"):
                # Load maps
                map_low = map_high
                if any(s in k for s in ['t11', 't12', 't22']):
                    map_high = self.lensing.shell_builder.load_maps(ishell, what=k)
                elif k == 'delta':
                    _, ipix_mask = self.snap_lensing.shell_builder.get_mask(return_ipix=True)
                    density_masked = self.snap_lensing.shell_builder.get_field_map(ishell, nest=True)[ipix_mask]
                    map_high = healsparse.HealSparseMap.make_empty(self.lensing.nside_c, self.nside, self.lensing.dtype)
                    map_high.update_values_pix(ipix_mask, density_masked.astype(self.lensing.dtype))
                    print([density_masked.min(), density_masked.max()])
                else:
                    map_high = self.snap_lensing.load_maps(ishell, what=None if k == 'kappa' else k)

                chi_low, chi_high = chi_bins_edges[[ishell, ishell + 1]]

                # Loop over tomographic bins
                for ikappa in self.lensing.tqdm(ikappas, disable=True):
                    # select galaxies in current shell
                    w = (catalog[ikappa]['idx_shell'] == ishell)

                    if np.any(w):
                        if k == 'delta':
                            if self.interp_pixels:
                                catalog[ikappa][k][w] = map_high.interpolate_pos(
                                    catalog[ikappa]['ra'][w],
                                    catalog[ikappa]['dec'][w],
                                    allow_partial=True,
                                )
                            else:
                                idx = hp.ang2pix(
                                    self.nside,
                                    *radec2thetaphi(catalog[ikappa]['ra'][w], catalog[ikappa]['dec'][w]),
                                    nest=True,
                                )
                                catalog[ikappa][k][w] = map_high[idx]
                        else:
                            # prepare for interpolation
                            dchi = (catalog[ikappa]['chi'][w] - chi_low)  # distance

                            # interpolation between shells and (optionnally) accross pixels
                            if self.interp_pixels:
                                if ishell == 0:
                                    catalog[ikappa][k][w] = dchi * map_high.interpolate_pos(
                                        catalog[ikappa]['ra'][w], catalog[ikappa]['dec'][w], allow_partial=True) / (chi_high - chi_low)
                                else:
                                    map_low_interp = map_low.interpolate_pos(catalog[ikappa]['ra'][w], catalog[ikappa]['dec'][w], allow_partial=True)
                                    map_high_interp = map_high.interpolate_pos(catalog[ikappa]['ra'][w], catalog[ikappa]['dec'][w], allow_partial=True)
                                    catalog[ikappa][k][w] = map_low_interp + dchi * (map_high_interp - map_low_interp) / (chi_high - chi_low)
                            else:
                                idx = hp.ang2pix(self.nside, *radec2thetaphi(catalog[ikappa]['ra'][w], catalog[ikappa]['dec'][w]), nest=True)
                                if ishell == 0:
                                    catalog[ikappa][k][w] = dchi * map_high[idx] / (chi_high - chi_low)
                                else:
                                    catalog[ikappa][k][w] = map_low[idx] + dchi * (map_high[idx] - map_low[idx]) / (chi_high - chi_low)

    def _save_catalogs(self, catalog, ikappas=None):
        """Write per-bin catalogs to FITS files.

        Parameters
        ----------
        catalog : list of dict
            Per-bin outputs containing ``'ra'``, ``'dec'``, ``'z'``, ``'chi'`` and
            any fields listed in ``self.interp_list``.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to save.

        Returns
        -------
        list of dict
            The input ``catalog`` object (returned for convenience).
        """
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] saving", leave=False):

            ra = fits.Column(name='ra', format='E', array=catalog[ikappa]['ra'])
            dec = fits.Column(name='dec', format='E', array=catalog[ikappa]['dec'])
            z = fits.Column(name='z', format='E', array=catalog[ikappa]['z'])
            chi = fits.Column(name='chi', format='E', array=catalog[ikappa]['chi'])

            interp_colums = [fits.Column(name=k, format='E', array=catalog[ikappa][k]) for k in self.interp_list]

            header = fits.Header()
            for k, v in self.header_dict.items():
                header[k] = v

            hdul = [fits.PrimaryHDU(), fits.BinTableHDU.from_columns([ra, dec, z, chi] + interp_colums, header)]

            fname = self.lensing.get_fname(ikappa, what=self.name, format='fits')
            fits.HDUList(hdul).writeto(fname, overwrite=True)

        return catalog

    def load_catalog(self, ikappa):
        """Load a saved FITS catalog for one tomographic bin.

        Parameters
        ----------
        ikappa : int
            Tomographic-bin index.

        Returns
        -------
        astropy.io.fits.HDUList
            Opened FITS HDU list. The caller is responsible for closing it.
        """
        fname = self.lensing.get_fname(ikappa, what=self.name, format='fits')
        return fits.open(fname)


class CatalogSourceUniform(CatalogSourceGenerator):
    """Generate source galaxies uniformly over the footprint.

    Redshifts are drawn from the tomographic tracer distributions. Angular
    positions are drawn uniformly over the observed sky region (octant or full
    sky depending on ``lensing.octant_only``).
    """

    def __init__(self, lensing, name='catuniform', nbar=None, interp_list=['kappa', 'gamma1', 'gamma2'], interp_pixels=True):
        super().__init__(lensing, name, nbar, interp_list=interp_list, interp_pixels=interp_pixels)

    def _generate_galaxies(self, catalog, ikappas=None):
        """Generate redshifts and positions for all selected tomographic bins.

        Parameters
        ----------
        catalog : list of dict
            Per-bin dictionaries updated in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        """
        # Generate ra, dec and redshift
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] generating ra, dec, z"):
            self._generate_z(catalog, ikappa)
            self._generate_pos(catalog, ikappa)

            for k in self.interp_list:
                catalog[ikappa][k] = np.zeros_like(catalog[ikappa]['z'])

    def _generate_pos(self, catalog, ikappa):
        """Draw uniform (RA, DEC) positions over the footprint.

        Parameters
        ----------
        catalog : list of dict
            Per-bin output container.
        ikappa : int
            Tomographic-bin index.
        """
        rng = np.random.default_rng(self.lensing.shell_builder.seed + ikappa * self.nshells)

        # Generate random positions with a uniform distribution over the footprint
        if self.lensing.octant_only:
            ra = rng.uniform(0., 90., self.Ngal[ikappa])
            dec = np.rad2deg(np.arcsin(np.random.uniform(0., 1., self.Ngal[ikappa])))
        else:
            ra = rng.uniform(0., 360., self.Ngal[ikappa])
            dec = np.rad2deg(np.arcsin(np.random.uniform(-1., 1., self.Ngal[ikappa])))

        catalog[ikappa]['ra'] = ra
        catalog[ikappa]['dec'] = dec

    def _generate_z(self, catalog, ikappa):
        """Draw redshifts from the tracer n(z) via inverse-CDF sampling.

        Parameters
        ----------
        catalog : list of dict
            Per-bin output container.
        ikappa : int
            Tomographic-bin index.

        Notes
        -----
        Shell membership is set by digitizing redshifts onto the shell edges
        returned by ``shell_builder.shell_edges_redshift()``.
        """
        builder = self.lensing.shell_builder
        z_bins_edges = builder.shell_edges_redshift()

        rng = np.random.default_rng(self.lensing.shell_builder.seed + ikappa * self.nshells)

        # Generate redshifts according to lensing object redshift distributions (using cdf sampling)
        t = self.lensing.get_tracers_ccl()[ikappa]
        zz = np.linspace(0, z_bins_edges[-1], 1000)
        nz = t.get_dndz(zz)

        z = sample_pdf(zz, nz, self.Ngal[ikappa], rng)
        chi = builder.cosmo.comoving_radial_distance(1. / (1. + z))

        catalog[ikappa]['z'] = z
        catalog[ikappa]['chi'] = chi
        catalog[ikappa]['idx_shell'] = np.digitize(z, z_bins_edges) - 1


class CatalogSourceClusteredMultinomial(CatalogSourceUniform):
    """Generate clustered source galaxies using multinomial sampling.

    The total number of galaxies per tomographic bin is fixed, and positions are
    drawn per shell from a multinomial distribution over pixels, proportional to
    ``1 + b * delta`` where ``delta`` is the density contrast map.

    Parameters
    ----------
    biases : array_like
        Linear galaxy bias per tomographic bin (length ``lensing.nmaps``).
    """

    def __init__(self, lensing, name='catmultinom', nbar=None, *, biases, interp_list=['kappa', 'gamma1', 'gamma2'], interp_pixels=True):
        super().__init__(lensing, name, nbar, interp_list=interp_list, interp_pixels=interp_pixels)

        if biases is None:
            raise ValueError('biases must be provided for CatalogSourceClusteredMultinomial')
        assert len(biases) == self.lensing.nmaps
        self.biases = biases
        for i, b in enumerate(self.biases):
            self.header_dict[f'BIAS_{i}'] = b

    def _generate_galaxies(self, catalog, ikappas=None):
        """Generate redshifts first, then positions conditioned on density maps.

        Parameters
        ----------
        catalog : list of dict
            Per-bin dictionaries updated in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        """
        # Generate redshifts first, needed to know what shells galaxies are in
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] generating z"):
            self._generate_z(catalog, ikappa)
            for k in ['ra', 'dec'] + self.interp_list:
                catalog[ikappa][k] = np.zeros_like(catalog[ikappa]['z'])

        # Then generate positions, looping over shells for speed
        for ishell in self.lensing.shell_builder.tqdm(None, desc=f"[generate] generating ra, dec from density"):
            self._generate_pos(catalog, ishell, ikappas=ikappas)

    def _generate_pos(self, catalog, ishell, ikappas=None):
        """Draw positions in a given shell using multinomial pixel sampling.

        Parameters
        ----------
        catalog : list of dict
            Per-bin dictionaries updated in-place.
        ishell : int
            Shell index.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to process.
        """
        _, ipix_mask = self.snap_lensing.shell_builder.get_mask(return_ipix=True)
        density_masked = self.snap_lensing.shell_builder.get_field_map(ishell, nest=True)[ipix_mask]

        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, leave=False):
            # select galaxies in current shell
            w = (catalog[ikappa]['idx_shell'] == ishell)

            if np.any(w):
                # Multinomial sampling of number of galaxies per pixels
                n = np.sum(w)
                p = np.clip(1. + self.biases[ikappa] * density_masked, 0., None)
                p /= np.sum(p)

                ipix = np.random.choice(ipix_mask, size=n, replace=True, p=p)

                # Sampling ra, dec within pixel
                ra, dec = generate_radec_in_pix(self.nside, ipix)
                catalog[ikappa]['ra'][w] = ra
                catalog[ikappa]['dec'][w] = dec


class CatalogSourceClusteredPoisson(CatalogSourceGenerator):
    """Generate clustered source galaxies using Poisson sampling.

    In each shell and tomographic bin, the number of galaxies per pixel is drawn
    as a Poisson random variable with mean proportional to
    ``(1 + b * delta) * <nbar>``. The total number of galaxies therefore
    fluctuates.

    Parameters
    ----------
    biases : array_like
        Linear galaxy bias per tomographic bin (length ``lensing.nmaps``).
    """

    def __init__(self, lensing, name='catpoisson', nbar=None, *, biases, interp_list=['kappa', 'gamma1', 'gamma2'], interp_pixels=True):
        super().__init__(lensing, name, nbar, interp_list=interp_list, interp_pixels=interp_pixels)

        if biases is None:
            raise ValueError('biases must be provided for CatalogSourceClusteredPoisson')
        assert len(biases) == self.lensing.nmaps
        self.biases = biases
        for i, b in enumerate(self.biases):
            self.header_dict[f'BIAS_{i}'] = b

    def _generate_galaxies(self, catalog, ikappas=None):
        """Generate a clustered catalog by shell-wise Poisson sampling.

        Parameters
        ----------
        catalog : list of dict
            Per-bin dictionaries updated in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.

        Notes
        -----
        This method:
        1) computes the expected mean counts per pixel per shell from each
           tracer n(z),
        2) samples positions conditioned on density maps,
        3) samples redshifts within each shell for each bin.
        """
        builder = self.lensing.shell_builder
        z_bins_edges = builder.shell_edges_redshift()

        # First compute mean density per shell for each tracer's n(z)
        nbar_per_pix_per_bin = []
        for ikappa, t in enumerate(self.lensing.get_tracers_ccl()):
            nz_int = np.array([quad(lambda z: t.get_dndz(z), z_bins_edges[ishell], z_bins_edges[ishell + 1]) for ishell in range(self.snap_lensing.nmaps)])
            nbar_per_pix_per_bin.append(nz_int / np.sum(nz_int) * self.nbar_per_pix[ikappa])
            catalog[ikappa]['ra'] = []
            catalog[ikappa]['dec'] = []
            catalog[ikappa]['z'] = []
            catalog[ikappa]['idx_shell'] = []
        self.nbar_per_pix_per_bin = np.array(nbar_per_pix_per_bin)

        # Second generate positions, looping over shells
        for ishell in self.lensing.shell_builder.tqdm(None, desc=f"[generate] generating ra, dec from density"):
            self._generate_pos(catalog, ishell, ikappas=ikappas)

        # Third sample redshifts within shells
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] generating z"):
            self._generate_z(catalog, ikappa)

        for ikappa in self.lensing.tqdm(ikappas, disable=True):
            catalog[ikappa]['ra'] = np.concatenate(catalog[ikappa]['ra'])
            catalog[ikappa]['dec'] = np.concatenate(catalog[ikappa]['dec'])
            catalog[ikappa]['z'] = np.concatenate(catalog[ikappa]['z'])
            catalog[ikappa]['idx_shell'] = np.concatenate(catalog[ikappa]['idx_shell'])
            catalog[ikappa]['chi'] = builder.cosmo.comoving_radial_distance(1. / (1. + catalog[ikappa]['z']))
            for k in self.interp_list:
                catalog[ikappa][k] = np.zeros_like(catalog[ikappa]['z'])

    def _generate_pos(self, catalog, ishell, ikappas=None):
        """Sample galaxy positions in one shell using pixel-wise Poisson draws.

        Parameters
        ----------
        catalog : list of dict
            Per-bin dictionaries updated in-place.
        ishell : int
            Shell index.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to process.
        """
        _, ipix_mask = self.snap_lensing.shell_builder.get_mask(return_ipix=True)
        density_masked = self.snap_lensing.shell_builder.get_field_map(ishell, nest=True)[ipix_mask]

        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, leave=False):
            rng = np.random.default_rng(self.lensing.shell_builder.seed + ishell + ikappa * self.nshells)

            # Poisson sampling
            p = np.clip(1. + self.biases[ikappa] * density_masked, 0., None)
            lam = p * self.nbar_per_pix_per_bin[ikappa][ishell]
            n_per_pix = rng.poisson(lam=lam)
            ipix = np.repeat(ipix_mask, n_per_pix)

            # Sampling ra, dec within pixel
            ra, dec = generate_radec_in_pix(self.nside, ipix, rng)

            catalog[ikappa]['ra'].append(ra)
            catalog[ikappa]['dec'].append(dec)

    def _generate_z(self, catalog, ikappa):
        """Sample galaxy redshifts within each shell for one tomographic bin.

        Parameters
        ----------
        catalog : list of dict
            Per-bin dictionaries updated in-place.
        ikappa : int
            Tomographic-bin index.

        Notes
        -----
        For each shell, redshifts are sampled from the tracer n(z) restricted to
        that shell's redshift interval.
        """
        builder = self.lensing.shell_builder
        z_bins_edges = builder.shell_edges_redshift()

        t = self.lensing.get_tracers_ccl()[ikappa]
        for ishell in self.lensing.shell_builder.tqdm(None, disable=True):
            rng = np.random.default_rng(self.lensing.shell_builder.seed + ishell + ikappa * self.nshells)

            zz = np.linspace(z_bins_edges[ishell], z_bins_edges[ishell + 1], 100)
            nz = t.get_dndz(zz)
            ngal = len(catalog[ikappa]['ra'][ishell])

            catalog[ikappa]['z'].append(sample_pdf(zz, nz, ngal, rng))
            catalog[ikappa]['idx_shell'].append(np.full(ngal, ishell))


def generate_radec_in_pix(nside, ipix, rng, log2nside_high=29):
    """Sample random (RA, DEC) coordinates uniformly within Healpix pixels.

    This uses an approximation: pixels are upsampled to a very high nside
    (``2**log2nside_high``) and one sub-pixel is selected uniformly at random.
    See https://astro.pages.rwth-aachen.de/astrotools/_modules/healpytools.html#rand_vec_from_map.

    Parameters
    ----------
    nside : int
        Target Healpix nside of the input pixel indices.
    ipix : array_like of int
        Pixel indices at resolution ``nside`` (assumed NESTED).
    rng : numpy.random.Generator
        Random number generator.
    log2nside_high : int, optional
        Log2 of the high resolution nside used internally. Default is 29.

    Returns
    -------
    ra, dec : ndarray
        Longitudes and latitudes in degrees (Healpy ``lonlat=True`` convention).
    """
    order = hp.nside2order(nside)
    nside_high = 2**log2nside_high
    n_up = log2nside_high - order

    ipix_high = ipix * (4**n_up)
    ipix_high += rng.integers(0, 4**n_up, size=np.size(ipix))

    return hp.pix2ang(nside_high, ipix_high, nest=True, lonlat=True)  # ra, dec
