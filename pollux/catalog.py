import numpy as np
import healpy as hp
from astropy.io import fits

from .utils import radec2thetaphi, quad, sample_pdf


class CatalogGenerator:

    def __init__(self):
        pass


class CatalogSourceUniform(CatalogGenerator):

    def __init__(self, lensing, nbar, interp_list=['kappa', 'gamma1', 'gamma2'], interp_pixels=True, name='catuniform'):
        self.lensing = lensing
        self.snap_lensing = self.lensing.shell_builder.get_snapshot_lensing_builder()
        self.nside = self.lensing.nside
        self.ncats = self.lensing.nmaps
        self.nshells = self.lensing.shell_builder.nshells

        assert len(nbar) == self.lensing.nmaps
        self.nbar = nbar

        self.interp_list = interp_list
        self.interp_pixels = interp_pixels

        self.name = name

        # number of galaxies to draw (or mean for Poisson sampling)
        area_std = np.pi / 2. if self.lensing.octant_only else 4. * np.pi
        area_arcmin = area_std * (180. * 60 / np.pi)**2
        self.Ngal = (nbar * area_arcmin).astype(int)
        self.area_pix = hp.nside2pixarea(self.nside, degrees=False) * (180. * 60. / np.pi)**2
        self.nbar_per_pix = nbar * self.area_pix

    def generate(self, ikappas=None):
        catalog = [{} for _ in range(self.lensing.nmaps)]

        # Generate ra, dec, z, etc.
        self._generate_galaxies(catalog, ikappas=ikappas)

        # Interpolate kappa/shear values
        self._interpolate_maps(catalog, ikappas=ikappas)

        # Save it!
        self._save_catalogs(catalog, ikappas=ikappas)

    def _generate_galaxies(self, catalog, ikappas=None):
        # Generate ra, dec and redshift
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] generating ra, dec, z"):
            self._generate_z(catalog, ikappa)
            self._generate_pos(catalog, ikappa)

            for k in self.interp_list:
                catalog[ikappa][k] = np.zeros_like(z)

    def _generate_z(self, catalog, ikappa):
        reader = self.lensing.reader
        z_bins_edges = reader.shell_edges_redshift()

        # Generate redshifts according to lensing object redshift distributions (using cdf sampling)
        t = self.lensing.get_tracers()[ikappa]
        zz = np.linspace(0, z_bins_edges[-1], 1000)
        nz = t.get_dndz(zz)

        z = sample_pdf(zz, nz, self.Ngal[ikappa])
        chi = reader.cosmo.comoving_radial_distance(1. / (1. + z))

        catalog[ikappa]['z'] = z
        catalog[ikappa]['chi'] = chi
        catalog[ikappa]['idx_shell'] = np.digitize(z, z_bins_edges) - 1

    def _generate_pos(self, catalog, ikappa):
        # Generate random positions with a uniform distribution over the footprint
        if self.lensing.octant_only:
            ra = np.random.uniform(0., 90., self.Ngal[ikappa])
            dec = np.rad2deg(np.arcsin(np.random.uniform(0., 1., self.Ngal[ikappa])))
        else:
            ra = np.random.uniform(0., 360., self.Ngal[ikappa])
            dec = np.rad2deg(np.arcsin(np.random.uniform(-1., 1., self.Ngal[ikappa])))

        catalog[ikappa]['ra'] = ra
        catalog[ikappa]['dec'] = dec

    def _interpolate_maps(self, catalog, ikappas=None):
        reader = self.lensing.reader
        chi_bins_edges = reader.shell_edges_comoving_distance()

        # Interpolate shear values, looping over shells for speed
        for k in self.interp_list:
            for ishell in self.lensing.shell_builder.tqdm(None, desc=f"[generate] interpolating maps {k}"):
                # Load maps
                if ishell == 0:
                    if k in ['t11', 't22', 't12']:
                        map_high = self.lensing.shell_builder.load_maps(0, what=k)
                    else:
                        map_high = self.snap_lensing.load_maps(0, what=None if k == 'kappa' else k)
                else:
                    map_low = map_high
                    if k in ['t11', 't22', 't12']:
                        map_high = self.lensing.shell_builder.load_maps(ishell, what=k)
                    else:
                        map_high = self.snap_lensing.load_maps(ishell, what=None if k == 'kappa' else k)

                chi_low, chi_high = chi_bins_edges[[ishell, ishell + 1]]

                # Loop over tomographic bins
                for ikappa in self.lensing.tqdm(ikappas, disable=True):
                    # select galaxies in current shell
                    w = (catalog[ikappa]['idx_shell'] == ishell)

                    if np.any(w):
                        # prepare for interpolation
                        dchi = (catalog[ikappa]['chi'][w] - chi_low)  # distance

                        # interpolation between shells and (optionnally) accross pixels
                        if self.interp_pixels:
                            if ishell == 0:
                                catalog[ikappa][k][w] = dchi * map_high.interpolate_pos(
                                    catalog[ikappa]['ra'][w], catalog[ikappa]['dec'][w],
                                    allow_partial=True) / (chi_high - chi_low)
                            else:
                                map_low_interp = map_low.interpolate_pos(
                                    catalog[ikappa]['ra'][w], catalog[ikappa]['dec'][w], allow_partial=True)
                                map_high_interp = map_high.interpolate_pos(
                                    catalog[ikappa]['ra'][w], catalog[ikappa]['dec'][w], allow_partial=True)
                                catalog[ikappa][k][w] = map_low_interp + dchi * (map_high_interp -
                                                                                 map_low_interp) / (chi_high - chi_low)
                        else:
                            idx = hp.ang2pix(
                                self.nside,
                                *radec2thetaphi(catalog[ikappa]['ra'][w], catalog[ikappa]['dec'][w]),
                                nest=True)  # pixels
                            if ishell == 0:
                                catalog[ikappa][k][w] = dchi * map_high[idx] / (chi_high - chi_low)
                            else:
                                catalog[ikappa][k][w] = map_low[idx] + dchi * (map_high[idx] -
                                                                               map_low[idx]) / (chi_high - chi_low)

    def _save_catalogs(self, catalog, ikappas=None):
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] saving", leave=False):

            hdul = []

            ra = fits.Column(name='ra', format='E', array=catalog[ikappa]['ra'])
            dec = fits.Column(name='dec', format='E', array=catalog[ikappa]['dec'])
            z = fits.Column(name='z', format='E', array=catalog[ikappa]['z'])
            chi = fits.Column(name='chi', format='E', array=catalog[ikappa]['chi'])

            interp_colums = [fits.Column(name=k, format='E', array=catalog[ikappa][k]) for k in self.interp_list]

            header = fits.Header()
            hdul = [fits.PrimaryHDU(), fits.BinTableHDU.from_columns([ra, dec, z, chi] + interp_colums, header)]

            fname = self.lensing.get_fname(ikappa, what=self.name, force_fits=True)
            fits.HDUList(hdul).writeto(fname, overwrite=True)

        return catalog
    
    def load_catalog(self, ikappa):
        fname = self.lensing.get_fname(ikappa, what=self.name, force_fits=True)
        return fits.open(fname)


def generate_radec_in_pix(nside, ipix, log2nside_high=29):
    # Approximated by sampling at very high nside (2**29)
    # From https://astro.pages.rwth-aachen.de/astrotools/_modules/healpytools.html#rand_vec_from_map
    order = hp.nside2order(nside)
    nside_high = 2**log2nside_high
    n_up = log2nside_high - order

    ipix_high = ipix * (4**n_up)
    ipix_high += np.random.randint(0, 4**n_up, size=np.size(ipix))

    return hp.pix2ang(nside_high, ipix_high, nest=True, lonlat=True)  # ra, dec


class CatalogSourceClusteredMultinomial(CatalogSourceUniform):

    def __init__(
            self,
            lensing_builder,
            nbar,
            biases,
            interp_list=['kappa', 'gamma1', 'gamma2'],
            interp_pixels=True,
            name='catmultinom'):
        super().__init__(lensing_builder, nbar, interp_list, interp_pixels, name)

        assert len(biases) == self.lensing.nmaps
        self.biases = biases

    def _generate_galaxies(self, catalog, ikappas=None):
        # Generate redshifts first, needed to know what shells galaxies are in
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] generating z"):
            self._generate_z(catalog, ikappa)
            for k in ['ra', 'dec'] + self.interp_list:
                catalog[ikappa][k] = np.zeros_like(catalog[ikappa]['z'])

        # Then generate positions, looping over shells for speed
        for ishell in self.lensing.shell_builder.tqdm(None, desc=f"[generate] generating ra, dec from density"):
            self._generate_pos(catalog, ishell, ikappas=ikappas)

    def _generate_pos(self, catalog, ishell, ikappas=None):
        # Load density map
        _, ipix_mask = self.snap_lensing.shell_builder.get_mask(return_ipix=True)
        density_masked = self.snap_lensing.shell_builder.get_field_map(ishell)[ipix_mask]

        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, leave=False):
            # select galaxies in current shell
            w = (catalog[ikappa]['idx_shell'] == ishell)

            if np.any(w):
                # Multinomial sampling of number of galaxies per pixels
                n = np.sum(w)
                p = np.clip(1. + self.biases[ikappa] * density_masked, 0., None)
                p /= np.sum(p)

                # n_per_pix = np.random.multinomial(n, pvals=p)
                # ipix = np.repeat(np.arange(self.lensing.npix), n_per_pix)

                ipix = np.random.choice(ipix_mask, size=n, replace=True, p=p)

                # Sampling ra, dec within pixel
                ra, dec = generate_radec_in_pix(self.nside, ipix)

                catalog[ikappa]['ra'][w] = ra
                catalog[ikappa]['dec'][w] = dec


class CatalogSourceClusteredPoisson(CatalogSourceUniform):

    def __init__(
            self,
            lensing_builder,
            nbar,
            biases,
            interp_list=['kappa', 'gamma1', 'gamma2'],
            interp_pixels=True,
            name='catpoisson'):
        super().__init__(lensing_builder, nbar, interp_list, interp_pixels, name)

        assert len(biases) == self.lensing.nmaps
        self.biases = biases

    def _generate_galaxies(self, catalog, ikappas=None):
        reader = self.lensing.reader
        z_bins_edges = reader.shell_edges_redshift()

        # First, we compute the mean density per shell for each tracer's n(z)
        nbar_per_pix_per_bin = []
        for ikappa, t in enumerate(self.lensing.get_tracers()):
            nz_int = np.array([
                quad(lambda z: t.get_dndz(z), z_bins_edges[ishell], z_bins_edges[ishell + 1])
                for ishell in range(self.snap_lensing.nmaps)
            ])
            nbar_per_pix_per_bin.append(nz_int / np.sum(nz_int) * self.nbar_per_pix[ikappa])
            catalog[ikappa]['ra'] = []
            catalog[ikappa]['dec'] = []
            catalog[ikappa]['z'] = []
            catalog[ikappa]['idx_shell'] = []
        self.nbar_per_pix_per_bin = np.array(nbar_per_pix_per_bin)

        # Second, generate positions, looping over shells for speed
        for ishell in self.lensing.shell_builder.tqdm(None, desc=f"[generate] generating ra, dec from density"):
            self._generate_pos(catalog, ishell, ikappas=ikappas)

        # Third, sample redshifts within shells
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] generating z"):
            self._generate_z(catalog, ikappa)

        for ikappa in self.lensing.tqdm(ikappas, disable=True):
            catalog[ikappa]['ra'] = np.concatenate(catalog[ikappa]['ra'])
            catalog[ikappa]['dec'] = np.concatenate(catalog[ikappa]['dec'])
            catalog[ikappa]['z'] = np.concatenate(catalog[ikappa]['z'])
            catalog[ikappa]['idx_shell'] = np.concatenate(catalog[ikappa]['idx_shell'])
            catalog[ikappa]['chi'] = reader.cosmo.comoving_radial_distance(1. / (1. + catalog[ikappa]['z']))
            for k in self.interp_list:
                catalog[ikappa][k] = np.zeros_like(catalog[ikappa]['z'])

    def _generate_pos(self, catalog, ishell, ikappas=None):
        # Load density map
        _, ipix_mask = self.snap_lensing.shell_builder.get_mask(return_ipix=True)
        density_masked = self.snap_lensing.shell_builder.get_field_map(ishell)[ipix_mask]

        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, leave=False):
            # Poisson sampling
            p = np.clip(1. + self.biases[ikappa] * density_masked, 0., None)
            lam = p * self.nbar_per_pix_per_bin[ikappa][ishell]
            n_per_pix = np.random.poisson(lam=lam)
            ipix = np.repeat(ipix_mask, n_per_pix)

            # Sampling ra, dec within pixel
            ra, dec = generate_radec_in_pix(self.nside, ipix)

            catalog[ikappa]['ra'].append(ra)
            catalog[ikappa]['dec'].append(dec)

    def _generate_z(self, catalog, ikappa):
        reader = self.lensing.reader
        z_bins_edges = reader.shell_edges_redshift()

        t = self.lensing.get_tracers()[ikappa]
        for ishell in self.lensing.shell_builder.tqdm(None, disable=True):
            zz = np.linspace(z_bins_edges[ishell], z_bins_edges[ishell + 1], 100)
            nz = t.get_dndz(zz)
            ngal = len(catalog[ikappa]['ra'][ishell])
            catalog[ikappa]['z'].append(sample_pdf(zz, nz, ngal))
            catalog[ikappa]['idx_shell'].append(np.full(ngal, ishell))
