import os
import numpy as np
from typing import Any
import healpy as hp
import healsparse
from astropy.io import fits

from .utils import radec2thetaphi, quad, sample_pdf


class CatalogSourceGenerator:
    """Generic generator for mock source-galaxy catalogs for lensing analyses.

    This class generates positions and redshifts for galaxies in each tomographic
    bin and interpolates lensing quantities (e.g. kappa/gamma) from maps produced
    by a lensing builder.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder used to define tomographic bins and provide maps to
        interpolate.
    tag : str or None, optional
        Tag used to name the output directory ``catalogs_{tag}/``. Default is None.
    nbar : float, array-like, or None, optional
        Galaxy surface density in gal/arcmin^2. If a float, the same surface
        density is used for all tomographic bins; if an array-like, it must
        have length ``lensing.nmaps`` and provide a per-bin value. If ``None``,
        the values are taken from ``lensing.nbar``. Default is None.
    smooth_Mpc : float or None, optional
        Physical Gaussian smoothing scale in Mpc applied to density and shear
        maps before sampling and interpolation.  ``None`` means no smoothing.
    interp_list : list of str, optional
        Names of fields to interpolate from maps. Default is
        ``['kappa', 'gamma1', 'gamma2']``.
    interp_pixels : bool, optional
        If True, use pixel interpolation at continuous (RA, DEC) coordinates.
        If False, evaluate maps at the nearest pixel center. Default is True.
    remove_mean_list : list of str, optional
        Names of fields for which to remove the mean value over the footprint
        after interpolation. Default is ``['kappa']``.
    augmentors : list, optional
        List of catalog augmentors. Default is empty list.

    Attributes
    ----------
    Ngal : ndarray of int
        Number of galaxies drawn per tomographic bin.
    nbar_per_pix : ndarray
        Mean expected counts per pixel per tomographic bin.
    header_dict : dict
        FITS header keyword/value pairs stored in the output catalogs.
    """

    def __init__(
        self,
        lensing: Any,
        tag: str | None = None,
        nbar_input: float | np.ndarray | None = None,
        smooth_Mpc: float | None = None,
        interp_list: list[str] | None = None,
        interp_pixels: bool = True,
        remove_mean_list: list[str] | None = None,
        augmentors: list | None = None,
    ) -> None:
        """Initialize catalog generator and compute derived attributes."""
        self.lensing = lensing
        self.tag = tag
        self.nbar_input = nbar_input
        self.interp_pixels = interp_pixels
        self.smooth_Mpc = smooth_Mpc

        # Set defaults for list parameters
        self.interp_list = interp_list if interp_list is not None else ['kappa', 'gamma1', 'gamma2']
        self.remove_mean_list = remove_mean_list if remove_mean_list is not None else ['kappa']
        self.augmentors = augmentors if augmentors is not None else []

        # Catalogs live alongside the lensing maps; tag distinguishes parallel runs
        cat_dir = f'catalogs_{tag}' if tag is not None else 'catalogs'
        self._catalog_dir = os.path.join(lensing.save_dir, cat_dir)
        os.makedirs(self._catalog_dir, exist_ok=True)

        # Compute derived attributes
        self.cosmo = self.lensing.cosmo
        self.snap_lensing = self.lensing.shell_builder.snapshot_lensing_builder()
        self.nside = self.lensing.nside
        self.ncats = self.lensing.nmaps
        self.nshells = self.lensing.shell_builder.nshells

        # Handle augmentor fields
        for augmentor in self.augmentors:
            for field_name in augmentor.required_interp_fields:
                if field_name not in self.interp_list:
                    print(f"Warning: adding field '{field_name}' to interp_list for augmentor '{augmentor.name}'")
                    self.interp_list.append(field_name)
        self.augmentor_kwargs = {ikappa: {} for ikappa in self.lensing.iter()}

        # Handle nbar
        if self.nbar_input is None:
            nbar_arr = np.atleast_1d(self.lensing.nbar).astype(float)
        else:
            nbar_arr = np.atleast_1d(self.nbar_input).astype(float)
            if nbar_arr.size == 1:
                nbar_arr = np.full(self.lensing.nmaps, float(nbar_arr[0]))
            elif nbar_arr.size != self.lensing.nmaps:
                raise ValueError(f"nbar must be a scalar or array of length {self.lensing.nmaps}, got {nbar_arr.size}")
        self.nbar = nbar_arr

        # Compute galaxy counts and areas
        area_std = np.pi / 2. if self.lensing.octant_only else 4. * np.pi
        area_arcmin = area_std * (180. * 60 / np.pi)**2
        self.Ngal = (self.nbar * area_arcmin).astype(int)
        self.area_pix = hp.nside2pixarea(self.nside, degrees=False) * (180. * 60. / np.pi)**2
        self.nbar_per_pix = self.nbar * self.area_pix
        self.header_dict = {f'NBAR_{i}': n for i, n in enumerate(self.nbar)}

        print(f"*** Catalog ({self.tag if self.tag is not None else 'default'}) ***")
        print(f"  - Catalog class = {self.__class__.__name__}")
        print(f"  - Lensing class = {self.lensing.__class__.__name__}")
        print(f"  - Shell builder class = {self.lensing.shell_builder.__class__.__name__}")
        print(f"  - Number of catalogs = {self.ncats}")
        print(f"  - Catalogs saved to {self._catalog_dir}")

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
        catalog = {ikappa: {} for ikappa in self.lensing.iter(ikappas)}

        # Generate ra, dec, z, etc.
        self._generate_galaxies(catalog, ikappas=ikappas)

        # Interpolate kappa/shear values
        self._interpolate_maps(catalog, ikappas=ikappas)

        # Augment catalogs if requested
        for augmentor in self.augmentors:
            for ikappa in self.lensing.tqdm(ikappas, desc=f"[generate] augmenting catalogs with {augmentor.name}", leave=False):
                augmentor.augment_catalog(self.cosmo, catalog[ikappa], **self.augmentor_kwargs[ikappa])

        # Save it!
        self._save_catalogs(catalog, ikappas=ikappas)

    def _generate_galaxies(self, catalog, ikappas=None):
        """Populate a catalog container with galaxy properties.

        Parameters
        ----------
        catalog : dict
            Dictionary containing per-bin catalog data that will be filled in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

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

        # Fields that live in the shell builder (density/tidal), vs the snapshot lensing builder
        _SHELL_FIELDS = frozenset({'density', 't11', 't22', 't12'})

        # Interpolate shear values, looping over shells for speed
        for k in self.interp_list:
            map_high = None
            for ishell in self.lensing.shell_builder.tqdm(None, desc=f"[generate] interpolating maps {k}"):
                # Load maps
                map_low = map_high
                if k in _SHELL_FIELDS:
                    if k == 'density' and self.smooth_Mpc is None:
                        # Shell builder stores count maps: use get_field_map to normalize.
                        _, ipix_mask = self.snap_lensing.shell_builder.get_mask(return_ipix=True)
                        density_masked = self.snap_lensing.shell_builder.get_field_map(ishell, nest=True)[ipix_mask]
                        map_high = healsparse.HealSparseMap.make_empty(self.lensing.nside_c, self.nside, self.lensing.dtype)
                        map_high.update_values_pix(ipix_mask, density_masked.astype(self.lensing.dtype))
                    else:
                        map_high = self.lensing.shell_builder.load_maps(ishell, what=k, smooth_Mpc=self.smooth_Mpc)
                else:
                    # kappa, gamma1, gamma2: live in the snapshot lensing builder.
                    # kappa is the default product (what=None); gamma* use what=k.
                    what = None if k == 'kappa' else k
                    map_high = self.snap_lensing.load_maps(ishell, what=what, smooth_Mpc=self.smooth_Mpc)

                # Remove mean if requested
                if k in self.remove_mean_list:
                    valid_pixels = map_high.valid_pixels
                    values = map_high.get_values_pix(valid_pixels)
                    values -= values.mean()
                    map_high.update_values_pix(valid_pixels, values)

                chi_low, chi_high = chi_bins_edges[[ishell, ishell + 1]]

                # Loop over tomographic bins
                for ikappa in self.lensing.tqdm(ikappas, disable=True):
                    # select galaxies in current shell
                    w = (catalog[ikappa]['idx_shell'] == ishell)

                    if np.any(w):
                        ra = catalog[ikappa]['ra'][w]
                        dec = catalog[ikappa]['dec'][w]
                        if 'delta' in k:
                            # no interpolation between shells for delta (integrated quantity)
                            if self.interp_pixels:
                                catalog[ikappa][k][w] = map_high.interpolate_pos(
                                    ra,
                                    dec,
                                    allow_partial=True,
                                )
                            else:
                                idx = hp.ang2pix(
                                    self.nside,
                                    *radec2thetaphi(ra, dec),
                                    nest=True,
                                )
                                catalog[ikappa][k][w] = map_high[idx]
                        else:
                            # prepare for interpolation
                            dchi = (catalog[ikappa]['chi'][w] - chi_low)  # distance

                            # interpolation between shells and (optionnally) accross pixels
                            if self.interp_pixels:
                                if ishell == 0:
                                    catalog[ikappa][k][w] = dchi * map_high.interpolate_pos(ra, dec, allow_partial=True) / (chi_high - chi_low)
                                else:
                                    map_low_interp = map_low.interpolate_pos(ra, dec, allow_partial=True)
                                    map_high_interp = map_high.interpolate_pos(ra, dec, allow_partial=True)
                                    catalog[ikappa][k][w] = map_low_interp + dchi * (map_high_interp - map_low_interp) / (chi_high - chi_low)
                            else:
                                idx = hp.ang2pix(self.nside, *radec2thetaphi(ra, dec), nest=True)
                                if ishell == 0:
                                    catalog[ikappa][k][w] = dchi * map_high[idx] / (chi_high - chi_low)
                                else:
                                    catalog[ikappa][k][w] = map_low[idx] + dchi * (map_high[idx] - map_low[idx]) / (chi_high - chi_low)

    def get_fname(self, ikappa, format='fits'):
        """Return the path for the saved catalog of one tomographic bin.

        Parameters
        ----------
        ikappa : int
            Tomographic-bin index.
        format : str, optional
            File extension without the leading dot. Default is ``'fits'``.

        Returns
        -------
        str
            ``{lensing.save_dir}/catalogs_{tag}/cat_imap{ikappa}.{format}``
            (or ``catalogs/`` when tag is None).
        """
        return os.path.join(self._catalog_dir, f'cat_imap{ikappa}.{format}')

    def _save_catalogs(self, catalog, ikappas=None):
        """Write per-bin catalogs to FITS files.

        Parameters
        ----------
        catalog : list of dict
            Per-bin outputs containing ``'ra'``, ``'dec'``, ``'z'``, ``'chi'`` and
            any fields listed in ``self.interp_list``.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to save.
        """
        columns = self.interp_list
        for augmentor in self.augmentors:
            columns += augmentor.new_columns

        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] saving", leave=False):

            ra = fits.Column(name='ra', format='E', array=catalog[ikappa]['ra'])
            dec = fits.Column(name='dec', format='E', array=catalog[ikappa]['dec'])
            z = fits.Column(name='z', format='E', array=catalog[ikappa]['z'])
            chi = fits.Column(name='chi', format='E', array=catalog[ikappa]['chi'])

            interp_colums = [fits.Column(name=k, format='E', array=catalog[ikappa][k]) for k in columns]

            header = fits.Header()
            for k, v in self.header_dict.items():
                header[k] = v

            hdul = [fits.PrimaryHDU(), fits.BinTableHDU.from_columns([ra, dec, z, chi] + interp_colums, header)]

            fname = self.get_fname(ikappa, format='fits')
            fits.HDUList(hdul).writeto(fname, overwrite=True)

    def load_catalog(self, ikappa, return_header=False):
        """Load a saved FITS catalog for one tomographic bin.

        Parameters
        ----------
        ikappa : int
            Tomographic-bin index.

        Returns
        -------
        header : astropy.io.fits.Header
            FITS header from the binary table HDU.
        table : numpy.ndarray
            Table data from the binary table HDU.
        """
        fname = self.get_fname(ikappa, format='fits')
        hdul = fits.open(fname)
        try:
            header = hdul[1].header
            table = hdul[1].data
        finally:
            hdul.close()
        if return_header:
            return header, table
        else:
            return table


class CatalogSourceUniform(CatalogSourceGenerator):
    """Generate source galaxies uniformly over the footprint.

    Redshifts are drawn from the tomographic tracer distributions. Angular
    positions are drawn uniformly over the observed sky region (octant or full
    sky depending on ``lensing.octant_only``).

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag used to name the output directory ``catalogs_{tag}/``. Default is None.
    nbar : float, array-like, or None, optional
        Galaxy surface density. Default is None.
    smooth_Mpc : float or None, optional
        Physical Gaussian smoothing scale in Mpc. Default is None.
    interp_list : list of str, optional
        Fields to interpolate. Default is ``['kappa', 'gamma1', 'gamma2']``.
    interp_pixels : bool, optional
        Use pixel interpolation. Default is True.
    remove_mean_list : list of str, optional
        Fields to remove mean. Default is ``['kappa']``.
    augmentors : list, optional
        List of augmentors. Default is empty list.
    """

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
            dec = np.rad2deg(np.arcsin(rng.uniform(0., 1., self.Ngal[ikappa])))
        else:
            ra = rng.uniform(0., 360., self.Ngal[ikappa])
            dec = np.rad2deg(np.arcsin(rng.uniform(-1., 1., self.Ngal[ikappa])))

        catalog[ikappa]['ra'] = ra
        catalog[ikappa]['dec'] = dec


class CatalogSourceClustered(CatalogSourceGenerator):
    """Base class for clustered source galaxy catalogs.

    This class provides common functionality for generating clustered catalogs
    where galaxy positions are biased by a density field.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag used to name the output directory ``catalogs_{tag}/``. Default is None.
    nbar_input : float, array-like, or None, optional
        Galaxy surface density in gal/arcmin^2. Default is None.
    bias : float or array-like
        Linear galaxy bias per tomographic bin. If a float, the same bias is used
        for all tomographic bins; if an array-like, it must have length
        ``lensing.nmaps`` and provide a per-bin value.
    smooth_Mpc : float or None, optional
        Physical Gaussian smoothing scale in Mpc applied to the density and shear
        fields before sampling and interpolation.  When set, galaxy positions are
        drawn from ``density_{smooth_Mpc}Mpc`` maps and all fields in
        ``interp_list`` are loaded from their smoothed counterparts.  When None,
        the raw density contrast is used via ``get_field_map``.
    interp_list : list of str, optional
        Base field names to interpolate at galaxy positions (e.g.
        ``['kappa', 'gamma1', 'gamma2', 't11', 't22', 't12']``).  Smoothed
        filenames are resolved automatically from ``smooth_Mpc``.  Default is
        ``['kappa', 'gamma1', 'gamma2']``.
    interp_pixels : bool, optional
        Use pixel interpolation. Default is True.
    remove_mean_list : list of str, optional
        Fields whose mean is subtracted before interpolation. Default is ``['kappa']``.
    augmentors : list, optional
        List of augmentors applied after map interpolation. Default is empty list.
    """

    def __init__(
        self,
        lensing: Any,
        tag: str | None = None,
        nbar_input: float | np.ndarray | None = None,
        bias: float | np.ndarray | None = None,
        smooth_Mpc: float | None = None,
        interp_list: list[str] | None = None,
        interp_pixels: bool = True,
        remove_mean_list: list[str] | None = None,
        augmentors: list | None = None,
    ) -> None:
        """Initialize clustered catalog generator and validate bias parameter."""
        super().__init__(
            lensing=lensing,
            tag=tag,
            nbar_input=nbar_input,
            smooth_Mpc=smooth_Mpc,
            interp_list=interp_list,
            interp_pixels=interp_pixels,
            remove_mean_list=remove_mean_list,
            augmentors=augmentors,
        )

        if bias is None:
            raise ValueError(f'bias must be provided for {self.__class__.__name__}')

        # Normalize bias: accept scalar or per-bin array
        bias_arr = np.atleast_1d(bias).astype(float)
        if bias_arr.size == 1:
            bias_arr = np.full(self.lensing.nmaps, float(bias_arr[0]))
        elif bias_arr.size != self.lensing.nmaps:
            raise ValueError(f"bias must be a scalar or array of length {self.lensing.nmaps}, got {bias_arr.size}")
        self.bias = bias_arr
        
        # Store bias values in augmentor kwargs
        for ikappa in self.lensing.iter():
            self.augmentor_kwargs[ikappa]['bias'] = self.bias[ikappa]

        # Store bias values in header dict
        for i, b in enumerate(self.bias):
            self.header_dict[f'BIAS_{i}'] = b


class CatalogSourceClusteredMultinomial(CatalogSourceClustered):
    """Generate clustered source galaxies using multinomial sampling.

    The total number of galaxies per tomographic bin is fixed, and positions are
    drawn per shell from a multinomial distribution over pixels, proportional to
    ``1 + b * delta`` where ``delta`` is the density contrast field.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag used to name the output directory ``catalogs_{tag}/``. Default is None.
    nbar_input : float, array-like, or None, optional
        Galaxy surface density in gal/arcmin^2. Default is None.
    bias : float or array-like
        Linear galaxy bias per tomographic bin.
    smooth_Mpc : float or None, optional
        Physical Gaussian smoothing scale in Mpc.  See ``CatalogSourceClustered``.
    interp_list : list of str, optional
        Base field names to interpolate. Default is ``['kappa', 'gamma1', 'gamma2']``.
    interp_pixels : bool, optional
        Use pixel interpolation. Default is True.
    remove_mean_list : list of str, optional
        Fields whose mean is subtracted before interpolation. Default is ``['kappa']``.
    augmentors : list, optional
        List of augmentors. Default is empty list.
    """

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
        if self.smooth_Mpc is None:
            density_masked = self.snap_lensing.shell_builder.get_field_map(ishell, nest=True)[ipix_mask]
        else:
            density_masked = self.snap_lensing.shell_builder.load_maps(ishell, what='density', smooth_Mpc=self.smooth_Mpc)[ipix_mask]
        
        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, leave=False):
            rng = np.random.default_rng(self.lensing.shell_builder.seed + ishell + ikappa * self.nshells)

            # select galaxies in current shell
            w = (catalog[ikappa]['idx_shell'] == ishell)

            if np.any(w):
                # Multinomial sampling of number of galaxies per pixels
                n = np.sum(w)
                p = np.clip(1. + self.bias[ikappa] * density_masked, 0., None)
                p /= np.sum(p)

                ipix = rng.choice(ipix_mask, size=n, replace=True, p=p)

                # Sampling ra, dec within pixel
                ra, dec = generate_radec_in_pix(self.nside, ipix, rng)
                catalog[ikappa]['ra'][w] = ra
                catalog[ikappa]['dec'][w] = dec


class CatalogSourceClusteredPoisson(CatalogSourceClustered):
    """Generate clustered source galaxies using Poisson sampling.

    In each shell and tomographic bin, the number of galaxies per pixel is drawn
    as a Poisson random variable with mean proportional to
    ``(1 + b * delta) * <nbar>``. The total number of galaxies therefore
    fluctuates.

    Parameters
    ----------
    lensing : LensingBuilder
        Lensing map builder.
    tag : str or None, optional
        Tag used to name the output directory ``catalogs_{tag}/``. Default is None.
    nbar_input : float, array-like, or None, optional
        Galaxy surface density in gal/arcmin^2. Default is None.
    bias : float or array-like
        Linear galaxy bias per tomographic bin.
    smooth_Mpc : float or None, optional
        Physical Gaussian smoothing scale in Mpc.  See ``CatalogSourceClustered``.
    interp_list : list of str, optional
        Base field names to interpolate. Default is ``['kappa', 'gamma1', 'gamma2']``.
    interp_pixels : bool, optional
        Use pixel interpolation. Default is True.
    remove_mean_list : list of str, optional
        Fields whose mean is subtracted before interpolation. Default is ``['kappa']``.
    augmentors : list, optional
        List of augmentors. Default is empty list.
    """

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

        # First compute mean density per shell for each tracer's n(z) (all bins needed for nbar)
        nbar_per_pix_per_bin = []
        for ikappa, t in enumerate(self.lensing.get_tracers_ccl()):
            nz_int = np.array([quad(lambda z: t.get_dndz(z), z_bins_edges[ishell], z_bins_edges[ishell + 1]) for ishell in range(self.snap_lensing.nmaps)])
            nbar_per_pix_per_bin.append(nz_int / np.sum(nz_int) * self.nbar_per_pix[ikappa])
        self.nbar_per_pix_per_bin = np.array(nbar_per_pix_per_bin)

        # Initialize catalog entries only for the requested bins
        for ikappa in self.lensing.iter(ikappas):
            catalog[ikappa]['ra'] = []
            catalog[ikappa]['dec'] = []
            catalog[ikappa]['z'] = []
            catalog[ikappa]['idx_shell'] = []

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
        if self.smooth_Mpc is None:
            density_masked = self.snap_lensing.shell_builder.get_field_map(ishell, nest=True)[ipix_mask]
        else:
            density_masked = self.snap_lensing.shell_builder.load_maps(ishell, what='density', smooth_Mpc=self.smooth_Mpc)[ipix_mask]
            
        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, leave=False):
            rng = np.random.default_rng(self.lensing.shell_builder.seed + ishell + ikappa * self.nshells)

            # Poisson sampling
            p = np.clip(1. + self.bias[ikappa] * density_masked, 0., None)
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


class CatalogAugmentor:
    """Base class for catalog augmentors that add fields to existing catalogs.

    Subclasses must implement the ``augment_catalog`` method and set
    ``self.name``, ``self.required_interp_fields``, and ``self.new_columns``
    in their ``__init__``.
    """

    def augment_catalog(self, cosmo, catalog, **kwargs):
        raise NotImplementedError