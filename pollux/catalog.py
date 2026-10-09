import logging
import os
import warnings
import numpy as np
from typing import Any
import healpy as hp
import pymaster as nmt
import sacc
import treecorr
import fitsio
from astropy.io import fits

from .cosmo import Tracer
from .utils import radec2thetaphi, quad, sample_pdf

logger = logging.getLogger(__name__)


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
        Physical Gaussian smoothing scale in Mpc of the fields used for intrinsic
        alignments (``SMOOTHED_FIELDS``: density and tidal tensor), interpolated from
        their smoothed maps. Convergence, shear and the density used to place
        galaxies are not smoothed. ``None`` means no smoothing.
    interp_list : list of str, optional
        Names of fields to interpolate from maps. Default is
        ``['kappa', 'gamma1', 'gamma2']``.
    interp_pixels : bool, optional
        If True, use pixel interpolation at continuous (RA, DEC) coordinates.
        If False, evaluate maps at the nearest pixel center. Default is True.
    augmentors : list, optional
        List of catalog augmentors. Default is empty list.
    cls_b : pymaster.NmtBin or None, optional
        Binning of :meth:`compute_cls` and :meth:`predict_cls`, whose ``lmax`` is the
        maximum multipole of the catalog fields and workspaces. Default is the shell
        builder's binning (see :attr:`b`).
    cls_lmax_mask : int or None, optional
        Maximum multipole of the fields' masks (at least the ``lmax`` of :attr:`b`). Default is that ``lmax``.

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
        augmentors: list | None = None,
        cls_b: Any = None,
        cls_lmax_mask: int | None = None,
    ) -> None:
        """Initialize catalog generator and compute derived attributes."""
        self.lensing = lensing
        self.tag = tag
        self.nbar_input = nbar_input
        self.interp_pixels = interp_pixels
        self.smooth_Mpc = smooth_Mpc

        # Set defaults for list parameters
        self.interp_list = interp_list if interp_list is not None else ['kappa', 'gamma1', 'gamma2']
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
                    warnings.warn(f"Adding field '{field_name}' to interp_list for augmentor '{augmentor.name}'.")
                    self.interp_list.append(field_name)
        self.augmentor_kwargs = {ikappa: {'ikappa': ikappa} for ikappa in self.lensing.iter()}
        self.cls_b = cls_b
        self.cls_lmax_mask = cls_lmax_mask
        self._b = None
        self._footprint = None
        self.predicted_cls = None
        self._wsp = {}

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

        logger.info("*** Catalog (%s) ***", self.tag if self.tag is not None else 'default')
        logger.info("  - Catalog class = %s.", self.__class__.__name__)
        logger.info("  - Lensing class = %s.", self.lensing.__class__.__name__)
        logger.info("  - Shell builder class = %s.", self.lensing.shell_builder.__class__.__name__)
        logger.info("  - Number of catalogs = %d.", self.ncats)
        logger.info("  - Catalogs saved to %s.", self._catalog_dir)

    def generate(self, ikappas=None, ishell=None):
        """Generate mock catalogs and interpolate requested map quantities.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate. If None, generate for all bins.
        ishell : int or None, optional
            If given, only generate the galaxies of shell ``ishell`` and save them
            as partial catalogs (one per bin), to be combined with
            :meth:`combine_shells`. Only supported by subclasses drawing galaxies
            shell by shell (:class:`CatalogSourceClusteredPoisson`).

        Notes
        -----
        The generation proceeds in four steps:
        1) draw galaxies (positions/redshifts and shell membership),
        2) interpolate requested fields from maps,
        3) apply the augmentors,
        4) save one FITS file per tomographic bin.
        """
        catalog = {ikappa: {} for ikappa in self.lensing.iter(ikappas)}
        ishells = None if ishell is None else [ishell]

        # Generate ra, dec, z, etc.
        self._generate_galaxies(catalog, ikappas=ikappas, ishells=ishells)

        # Interpolate kappa/shear values
        self._interpolate_maps(catalog, ikappas=ikappas, ishells=ishells)

        # Augment catalogs if requested
        for augmentor in self.augmentors:
            for ikappa in self.lensing.tqdm(ikappas, desc=f"[generate] augmenting catalogs with {augmentor.name}", leave=False):
                augmentor.augment_catalog(self.cosmo, catalog[ikappa], **self.augmentor_kwargs[ikappa])

        # Save it!
        self._save_catalogs(catalog, ikappas=ikappas, ishell=ishell)

    def combine_shells(self, ikappas=None):
        """Concatenate the per-shell partial catalogs into one catalog per bin.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to merge. If None, merge all bins.

        Notes
        -----
        Partial catalogs are produced by :meth:`generate` with ``ishell`` and are
        concatenated in shell order, which gives the same catalog as generating
        all shells at once.
        """
        for ikappa in self.lensing.tqdm(ikappas, desc="[combine_shells]", leave=False):
            parts = [fits.getdata(self.get_fname(ikappa, ishell=ishell), 1) for ishell in range(self.nshells)]
            header = fits.Header()
            for k, v in self.header_dict.items():
                header[k] = v
            hdul = [fits.PrimaryHDU(), fits.BinTableHDU(data=np.concatenate(parts), header=header)]
            fname = self.get_fname(ikappa)
            fits.HDUList(hdul).writeto(fname, overwrite=True)
            logger.debug("Saved to file %s.", fname)

    def _generate_galaxies(self, catalog, ikappas=None, ishells=None):
        """Populate a catalog container with galaxy properties.

        Parameters
        ----------
        catalog : dict
            Dictionary containing per-bin catalog data that will be filled in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        ishells : list of int or None, optional
            Shells whose galaxies are generated. If None, all shells.

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
        returned by ``shell_builder.shell_edges_redshift``.
        """
        builder = self.lensing.shell_builder
        z_edges = builder.shell_edges_redshift

        rng = np.random.default_rng(self.lensing.shell_builder.seed + ikappa * self.nshells)

        # Generate redshifts according to lensing object redshift distributions (using cdf sampling)
        t = self.lensing.tracers_ccl[ikappa]
        zz = np.linspace(0, z_edges[-1], 1000)
        nz = t.get_dndz(zz)

        z = sample_pdf(zz, nz, self.Ngal[ikappa], rng)
        chi = builder.cosmo.comoving_radial_distance(1. / (1. + z))

        catalog[ikappa]['z'] = z
        catalog[ikappa]['chi'] = chi
        catalog[ikappa]['idx_shell'] = np.digitize(z, z_edges) - 1

    def _interpolate_maps(self, catalog, ikappas=None, ishells=None):
        """Interpolate map quantities at galaxy positions.

        Parameters
        ----------
        catalog : list of dict
            Per-bin catalog dictionaries containing at least ``'ra'``, ``'dec'``,
            ``'chi'`` and ``'idx_shell'``.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to process.
        ishells : list of int or None, optional
            Shells whose galaxies are interpolated. If None, all shells.

        Notes
        -----
        This performs linear interpolation between consecutive shells in comoving
        distance and optionally interpolates across pixels (via HealSparse
        ``interpolate_pos``). For ``k='delta'``, a density contrast map is built
        from snapshot-lensing density shells.
        """
        for k in self.interp_list:
            if ishells is None:
                # all shells: each map is read once and reused as the lower map of the next shell
                map_high = None
                for ishell in self.lensing.shell_builder.tqdm(None, desc=f"[generate] interpolating maps {k}"):
                    map_low = map_high
                    map_high = self._load_interp_map(k, ishell)
                    self._interpolate_shell(catalog, k, ishell, map_low, map_high, ikappas=ikappas)
            else:
                for ishell in ishells:
                    map_low = self._load_interp_map(k, ishell - 1) if ishell > 0 and 'delta' not in k else None
                    map_high = self._load_interp_map(k, ishell)
                    self._interpolate_shell(catalog, k, ishell, map_low, map_high, ikappas=ikappas)

    # fields used for intrinsic alignments, interpolated from maps smoothed by smooth_Mpc
    SMOOTHED_FIELDS = ('density', 't11', 't22', 't12')

    def _load_interp_map(self, k, ishell):
        """Load the map of field ``k`` for one shell.

        Parameters
        ----------
        k : str
            Field name (e.g. ``'kappa'``, ``'gamma1'``, ``'density'``, ``'t11'``).
        ishell : int
            Shell index.

        Returns
        -------
        hsp.HealSparseMap
            Map of field ``k``, smoothed for the fields in ``SMOOTHED_FIELDS`` if
            ``self.smooth_Mpc`` is set.
        """
        smooth_Mpc = self.smooth_Mpc if k in self.SMOOTHED_FIELDS else None
        # Fields that live in the shell builder (density/tidal), vs the snapshot lensing builder
        if k in ('density', 't11', 't22', 't12'):
            if k == 'density' and smooth_Mpc is None:
                hsp_map = self.snap_lensing.shell_builder.get_field_hspmap(ishell)
            else:
                hsp_map = self.lensing.shell_builder.load_maps(ishell, what=k, smooth_Mpc=smooth_Mpc)
        else:
            # kappa, gamma1, gamma2: live in the snapshot lensing builder.
            # kappa is the default product (what=None); gamma* use what=k.
            what = None if k == 'kappa' else k
            hsp_map = self.snap_lensing.load_maps(ishell, what=what, smooth_Mpc=smooth_Mpc)

        return hsp_map

    def _interpolate_shell(self, catalog, k, ishell, map_low, map_high, ikappas=None):
        """Interpolate field ``k`` at the positions of the galaxies of one shell.

        Parameters
        ----------
        catalog : list of dict
            Per-bin catalog dictionaries, updated in place.
        k : str
            Field name.
        ishell : int
            Shell index.
        map_low, map_high : hsp.HealSparseMap or None
            Maps of field ``k`` for shells ``ishell - 1`` and ``ishell``; ``map_low``
            is unused for ``ishell == 0`` and for ``'delta'`` fields.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to process.
        """
        chi_low, chi_high = self.lensing.shell_builder.shell_edges_comoving_distance[[ishell, ishell + 1]]

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

    def get_fname(self, ikappa, format='fits', ishell=None):
        """Return the path for the saved catalog of one tomographic bin.

        Parameters
        ----------
        ikappa : int
            Tomographic-bin index.
        format : str, optional
            File extension without the leading dot. Default is ``'fits'``.
        ishell : int or None, optional
            If given, path of the partial catalog of shell ``ishell``.

        Returns
        -------
        str
            ``{lensing.save_dir}/catalogs_{tag}/cat_imap{ikappa}.{format}``
            (or ``catalogs/`` when tag is None), or
            ``.../shells/cat_imap{ikappa}_shell{ishell}.{format}`` for a partial catalog.
        """
        if ishell is None:
            return os.path.join(self._catalog_dir, f'cat_imap{ikappa}.{format}')
        return os.path.join(self._catalog_dir, 'shells', f'cat_imap{ikappa}_shell{ishell}.{format}')

    def _save_catalogs(self, catalog, ikappas=None, ishell=None):
        """Write per-bin catalogs to FITS files.

        Parameters
        ----------
        catalog : list of dict
            Per-bin outputs containing ``'ra'``, ``'dec'``, ``'z'``, ``'chi'`` and
            any fields listed in ``self.interp_list``.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to save.
        ishell : int or None, optional
            If given, save as the partial catalogs of shell ``ishell``.
        """
        columns = list(self.interp_list)
        for augmentor in self.augmentors:
            columns += augmentor.new_columns
        if ishell is not None:
            os.makedirs(os.path.dirname(self.get_fname(0, ishell=ishell)), exist_ok=True)

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

            fname = self.get_fname(ikappa, format='fits', ishell=ishell)
            fits.HDUList(hdul).writeto(fname, overwrite=True)
            logger.debug("Saved to file %s.", fname)

    def load_catalogs(self, ikappas=None, return_header=False, subsample=None, seed=None, columns=None, remove_mean=None):
        """Load saved FITS catalog(s) for one or more tomographic bins.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to load. If None, loads all bins.
        return_header : bool, optional
            If True, also return the FITS header(s).
        subsample : float or None, optional
            If not None, randomly draw (without replacement, seeded by
            ``seed + ikappa``) this fraction of galaxies from each loaded
            catalog, with ``0 < subsample < 1``. A random draw is used rather
            than a fixed stride, since galaxies are stored in a fixed
            (non-random) order and a stride would sample a specific,
            non-representative pattern. Default is None (no subsampling).
        seed : int or None, optional
            Seed for the subsampling RNG. Only used when ``subsample`` is not
            None. Defaults to the shell builder's seed.
        columns : list of str or None, optional
            Columns to read. If None (default), all columns.
        remove_mean : list of str or None, optional
            Columns whose mean over each loaded catalog is subtracted.

        Returns
        -------
        header : astropy.io.fits.Header or dict, optional
            Only if ``return_header=True``. FITS header from the binary table
            HDU if ``ikappas`` is a single int, otherwise a dict keyed by bin index.
        table : numpy.ndarray or dict
            Structured array (native byte order) if ``ikappas`` is a single int,
            otherwise a dict keyed by bin index.

        Notes
        -----
        Only the requested rows and columns are read (with ``fitsio``).
        """
        rng_seed = self.lensing.shell_builder.seed if seed is None else seed

        headers = {}
        tables = {}
        for ikappa in self.lensing.tqdm(ikappas, desc="[load_catalogs]"):
            fname = self.get_fname(ikappa, format='fits')
            headers[ikappa] = fits.getheader(fname, ext=1)
            rows = None
            if subsample is not None:
                rows = self._subsample_rows(headers[ikappa]['NAXIS2'], subsample, np.random.default_rng(rng_seed + ikappa))
            table = fitsio.read(fname, ext=1, columns=columns, rows=rows)
            table = table.astype(table.dtype.newbyteorder('='))
            for col in remove_mean or []:
                table[col] -= table[col].mean()
            tables[ikappa] = table
            logger.debug("Found and loaded catalog at %s.", fname)

        if type(ikappas) is int:
            header, table = headers[ikappas], tables[ikappas]
        else:
            header, table = headers, tables

        if return_header:
            return header, table
        else:
            return table

    def _random_radec(self, ipix_mask, n, rng):
        """Draw ``n`` random (RA, DEC) positions uniformly within the footprint.

        Parameters
        ----------
        ipix_mask : ndarray of int
            Valid pixel indices of the footprint (NESTED, at ``self.nside``).
        n : int
            Number of random positions to draw.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        ra, dec : ndarray
            Longitudes and latitudes in degrees.
        """
        ipix_rand = rng.choice(ipix_mask, size=n, replace=True)
        return generate_radec_in_pix(self.nside, ipix_rand, rng)

    @staticmethod
    def _subsample_rows(nrows, subsample, rng):
        """Draw the rows of a random subsample of a catalog, without replacement.

        Galaxies are stored in a fixed order (e.g. by shell/generation order),
        so a fixed stride would sample a specific, non-representative pattern.

        Parameters
        ----------
        nrows : int
            Number of rows of the catalog.
        subsample : float
            Fraction of rows to keep, with ``0 < subsample < 1``.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        numpy.ndarray
            Sorted row indices.
        """
        assert 0. < subsample < 1., "subsample must satisfy 0 < subsample < 1"
        return np.sort(rng.choice(nrows, size=int(subsample * nrows), replace=False))

    @property
    def b(self):
        """Return (and lazily create) the NaMaster binning of :meth:`compute_cls` and :meth:`predict_cls`.

        Returns
        -------
        pymaster.NmtBin
            ``cls_b``, or else the shell builder's binning (:attr:`MapBuilder.b`)
            without its pixel-window deconvolution, since catalog fields are sampled
            at exact positions.
        """
        if self._b is None:
            if self.cls_b is not None:
                self._b = self.cls_b
            else:
                b_ref = self.lensing.shell_builder.b
                n_bands = b_ref.get_n_bands()
                ells = np.concatenate([b_ref.get_ell_list(i) for i in range(n_bands)])
                weights = np.concatenate([b_ref.get_weight_list(i) for i in range(n_bands)])
                bpws = np.concatenate([np.full(len(b_ref.get_ell_list(i)), i) for i in range(n_bands)])
                self._b = nmt.NmtBin(ells=ells, bpws=bpws, weights=weights, f_ell=None, lmax=b_ref.lmax)
        return self._b

    @property
    def lmax_mask(self):
        """Maximum multipole of the masks of the catalog fields and of the footprint (``cls_lmax_mask``)."""
        lmax_mask = self.cls_lmax_mask if self.cls_lmax_mask is not None else self.b.lmax
        if lmax_mask < self.b.lmax:
            raise ValueError(f"cls_lmax_mask={lmax_mask} is below the fields' lmax={self.b.lmax}")
        return lmax_mask

    @property
    def footprint(self):
        """Footprint mask (RING, float, at the shell builder's ``nside``), the mask of map-based fields."""
        if self._footprint is None:
            self._footprint = self.lensing.shell_builder.get_mask(nest=False).astype(np.float64)
        return self._footprint

    def _cls_fields(self, ikappas, what, shear_cols=('gamma1', 'gamma2'), randoms=False, nran_factor=5, seed=None, subsample=None,
                    shared_randoms=True, remove_mean=True):
        """Build the NaMaster catalog fields used by :meth:`compute_cls`.

        Parameters
        ----------
        ikappas : list of int
            Tomographic bins.
        what : list of str
            Statistics to compute (see :meth:`compute_cls`), which set the fields needed.
        shear_cols, randoms, nran_factor, seed, subsample, shared_randoms, remove_mean
            See :meth:`compute_cls`.

        Returns
        -------
        f_dens, f_shear : dict
            Density (``NmtFieldCatalogClustering``) and spin-2 shear
            (``NmtFieldCatalog``) fields keyed by bin, if needed by ``what``.
            Each field has a ``mask_amplitude`` attribute, the ratio of its mask
            to :attr:`footprint` (its weighted number of galaxies or randoms per
            steradian of footprint, or 1 for density fields with the footprint mask).
        """
        builder = self.lensing.shell_builder
        lmax, lmax_mask = self.b.lmax, self.lmax_mask
        rng_seed = builder.seed if seed is None else seed
        _, ipix_mask = builder.get_mask(nest=True, return_ipix=True)
        area = ipix_mask.size * hp.nside2pixarea(builder.nside)
        need_density = 'clustering' in what or 'shear_clustering' in what
        need_shear = 'shear' in what or 'shear_clustering' in what
        col1, col2 = shear_cols

        # Build each bin's catalog field(s) once, reused across all pairs
        f_dens, f_shear = {}, {}
        shared_radec = None
        for ikappa in self.lensing.tqdm(ikappas, desc="[compute_cls] building fields"):
            cat = self.load_catalogs(ikappa, subsample=subsample, seed=rng_seed, columns=['ra', 'dec', *shear_cols],
                                     remove_mean=list(shear_cols) if remove_mean else None)
            positions = [cat['ra'], cat['dec']]
            weights_gal = np.ones(cat['ra'].size)

            if need_density:
                if not randoms:
                    f = nmt.NmtFieldCatalogClustering(positions, weights_gal, None, None, lmax=lmax, lmax_mask=lmax_mask, lonlat=True,
                                                      mask=self.footprint)
                    f.mask_amplitude = 1.
                else:
                    if shared_randoms:
                        if shared_radec is None:
                            rng = np.random.default_rng(rng_seed)
                            nran = int(nran_factor * cat['ra'].size)
                            shared_radec = self._random_radec(ipix_mask, nran, rng)
                        ra_rand, dec_rand = shared_radec
                    else:
                        rng = np.random.default_rng(rng_seed + ikappa)
                        nran = int(nran_factor * cat['ra'].size)
                        ra_rand, dec_rand = self._random_radec(ipix_mask, nran, rng)
                    f = nmt.NmtFieldCatalogClustering(positions, weights_gal, [ra_rand, dec_rand], np.ones(len(ra_rand)), lmax=lmax,
                                                      lmax_mask=lmax_mask, lonlat=True)
                    f.mask_amplitude = len(ra_rand) / area
                f_dens[ikappa] = f

            if need_shear:
                f = nmt.NmtFieldCatalog(positions, weights_gal, [cat[col1], cat[col2]], spin=2, lmax=lmax, lmax_mask=lmax_mask, lonlat=True)
                f.mask_amplitude = weights_gal.sum() / area
                f_shear[ikappa] = f

        return f_dens, f_shear

    def _wsp_fname(self, stat, pair=None, randoms=False):
        """Return the path of a workspace (see :meth:`get_wsp`).

        Parameters
        ----------
        stat : {'clustering', 'shear', 'shear_clustering'}
            Statistic.
        pair : tuple of int or None, optional
            Bin pair, or None for the workspace shared by all pairs.
        randoms : bool, optional
            For a bin pair, whether density fields are masked by randoms (see :meth:`compute_cls`).

        Returns
        -------
        str
            ``{catalog_dir}/wsp_{stat}_lmax{lmax}.wsp``, or
            ``{catalog_dir}/wsp_{stat}_{i}_{j}[_randoms]_lmax{lmax}.wsp`` (with ``i >= j`` for
            the symmetric clustering and shear statistics). Bandpower windows are saved
            next to it, with suffix ``_bpws.npy`` (see :meth:`get_bpws`).
        """
        if pair is None:
            name = stat
        else:
            i, j = pair
            if stat != 'shear_clustering':
                i, j = max(i, j), min(i, j)
            name = f'{stat}_{i}_{j}' + ('_randoms' if randoms and stat != 'shear' else '')
        return os.path.join(self._catalog_dir, f'wsp_{name}_lmax{self.b.lmax}.wsp')

    def get_wsp(self, stat, pair=None, fields=None, read_unbinned_MCM=True, **field_kwargs):
        """Return (and lazily create) the workspace of one statistic and bin pair.

        The workspace stores the mode-coupling matrix and bandpower windows. It is
        either specific to a bin pair, computed from the masks of the pair's catalog
        fields (the weighted galaxy positions, and the footprint or randoms for
        density), or shared by all bin pairs (``pair=None``), computed from
        :attr:`footprint`. The shared workspace is exact in expectation for galaxies
        uniformly distributed over the footprint once pseudo-spectra are divided by
        the fields' ``mask_amplitude`` (see :meth:`compute_cls`); with clustered
        sources, it misses the coupling from the clustering of galaxy positions.

        Workspaces are cached on disk (:meth:`_wsp_fname`), with their bandpower
        windows when computed; shared workspaces are also cached in memory.

        Parameters
        ----------
        stat : {'clustering', 'shear', 'shear_clustering'}
            Statistic.
        pair : tuple of int or None, optional
            Bin pair, or None (default) for the shared workspace.
        fields : tuple or None, optional
            For a bin pair, the two fields to compute the workspace from, if not
            on disk. By default, the fields of the pair.
        read_unbinned_MCM : bool, optional
            When reading from disk, also read the unbinned mode-coupling matrix,
            needed for bandpower windows but not to decouple spectra, and large
            (~77 GB for spin-2 fields at lmax 24575). Default is True.
        **field_kwargs
            Field options for a bin pair (``shear_cols``, ``randoms``, ``nran_factor``,
            ``seed``, ``subsample``, ``shared_randoms``, ``remove_mean``), see :meth:`compute_cls`.

        Returns
        -------
        pymaster.NmtWorkspace
        """
        if pair is None and stat in self._wsp and (self._wsp[stat][1] or not read_unbinned_MCM):
            return self._wsp[stat][0]
        fname = self._wsp_fname(stat, pair, field_kwargs.get('randoms', False))
        wsp = nmt.NmtWorkspace()
        if os.path.exists(fname):
            wsp.read_from(fname, read_unbinned_MCM=read_unbinned_MCM)
            logger.info("Found and loaded workspace at %s.", fname)
        else:
            if pair is None:
                spins = {'clustering': (0, 0), 'shear': (2, 2), 'shear_clustering': (0, 2)}[stat]
                f_spin = {spin: nmt.NmtField(self.footprint, None, spin=spin, lmax=self.b.lmax, lmax_mask=self.lmax_mask)
                          for spin in set(spins)}
                fields = (f_spin[spins[0]], f_spin[spins[1]])
            elif fields is None:
                i, j = pair
                f_dens, f_shear = self._cls_fields(sorted({i, j}), [stat], **field_kwargs)
                f1, f2 = {'clustering': (f_dens, f_dens), 'shear': (f_shear, f_shear), 'shear_clustering': (f_dens, f_shear)}[stat]
                fields = (f1[i], f2[j])
            wsp.compute_coupling_matrix(*fields, self.b)
            wsp.write_to(fname)
            np.save(self._bpws_fname(fname), wsp.get_bandpower_windows())
            logger.info("Computed and saved workspace to file %s.", fname)
            read_unbinned_MCM = True
        if pair is None:
            self._wsp[stat] = (wsp, read_unbinned_MCM)
        return wsp

    @staticmethod
    def _bpws_fname(wsp_fname):
        """Return the path of the bandpower windows saved next to a workspace."""
        return wsp_fname[:-len('.wsp')] + '_bpws.npy'

    def get_bpws(self, stat, pair=None, randoms=False):
        """Return the bandpower windows of a workspace (see :meth:`get_wsp`).

        Read from the file saved with the workspace, else from the workspace
        (and saved). The shared workspace is computed if missing.

        Parameters
        ----------
        stat : {'clustering', 'shear', 'shear_clustering'}
            Statistic.
        pair : tuple of int or None, optional
            Bin pair, or None (default) for the shared workspace.
        randoms : bool, optional
            For a bin pair, whether density fields are masked by randoms (see :meth:`compute_cls`).

        Returns
        -------
        ndarray
            Shape ``(n_cl, n_bands, n_cl, lmax + 1)``.

        Raises
        ------
        FileNotFoundError
            If the workspace of a bin pair has not been computed (see :meth:`compute_cls`).
        """
        fname = self._wsp_fname(stat, pair, randoms)
        fname_bpws = self._bpws_fname(fname)
        if os.path.exists(fname_bpws):
            return np.load(fname_bpws)
        if pair is not None and not os.path.exists(fname):
            raise FileNotFoundError(f"No workspace at {fname}, compute it with compute_cls or get_wsp.")
        bpws = self.get_wsp(stat, pair, randoms=randoms).get_bandpower_windows()
        np.save(fname_bpws, bpws)
        return bpws

    def compute_cls(
            self,
            ikappas=None,
            what='all',
            auto_only=False,
            shear_cols=('gamma1', 'gamma2'),
            randoms=False,
            nran_factor=5,
            seed=None,
            shared_wsp=False,
            subsample=None,
            shared_randoms=True,
            remove_mean=True,
            pairs=None,
            save=True):
        """Measure angular power spectra directly from the generated catalog(s),
        including all cross-tomographic-bin combinations.

        Uses NaMaster's catalog-based pseudo-C_ell estimators (``NmtFieldCatalog``
        for shear, ``NmtFieldCatalogClustering`` for clustering), binned with the
        shell builder's own NmtBin (:attr:`ShellBuilder.b`) so that the
        resulting ``ell`` matches other C_ell products in the codebase for direct
        comparability. The pixel-window deconvolution folded into the shell
        builder's binning (appropriate for map-based estimators) is stripped out,
        since catalog fields are sampled at exact positions and have no pixel
        window to correct for.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic bin indices to process. If None, uses all bins.
        what : {'shear', 'clustering', 'shear_clustering', 'all'} or list of str, optional
            Which spectra to compute:
            - 'clustering': galaxy density auto/cross-spectra (dd), symmetric i<=j pairs.
            - 'shear': shear E/B-mode auto/cross-spectra (ee, bb), symmetric i<=j pairs.
            - 'shear_clustering': density-shear cross-spectrum (de, db), all (i, j)
              ordered pairs since lens-bin i / source-bin j is not symmetric.
            - 'all': compute all three. Default.
        auto_only : bool, optional
            If True, restrict to same-bin pairs (i, i) only, skipping cross-bin
            spectra. Default is False (compute all cross-bin combinations).
        shear_cols : tuple of str, optional
            Names of the two catalog columns to use as the spin-2 shear field,
            e.g. ``('gamma1', 'gamma2')`` (lensing shear, default), ``('g1', 'g2')``
            (reduced shear), ``('e1_obs', 'e2_obs')`` (observed ellipticity), or
            any IA-related pair such as ``('e1_ia_ta', 'e2_ia_ta')``. Must be
            present in the loaded catalog (see the augmentors used at generation
            time, e.g. :class:`~pollux.ia.CatalogAugmentorShear`, :class:`~pollux.ia.IACalculator`).
        randoms : bool, optional
            If False (default), the mask of density fields is the :attr:`footprint`.
            If True, it is a random catalog (see ``nran_factor`` and ``shared_randoms``).
        nran_factor : int, optional
            With ``randoms``, ratio of random points to galaxies for each bin's
            clustering random catalog. Default is 5.
        seed : int or None, optional
            Seed for the random catalogs' RNG. Defaults to the shell builder's seed.
        shared_wsp : bool, optional
            If True, use one mode-coupling matrix per statistic (dd, shear-shear,
            density-shear) for all bin pairs, computed from the footprint, with
            pseudo-spectra divided by the fields' mask amplitudes (see :meth:`get_wsp`).
            This is exact in expectation for galaxies uniformly distributed over the
            footprint, but misses the coupling from the clustering of galaxy
            positions. If False (default), use one workspace per pair, computed from
            the masks of its fields (exact, slower). Workspaces are cached on disk
            and reused by later calls and by :meth:`predict_cls`; delete them if the
            catalogs or settings change.
        subsample : float or None, optional
            If not None, randomly draw (without replacement, seeded by
            ``seed + ikappa``) this fraction of galaxies from each loaded
            catalog before building fields, with ``0 < subsample < 1``. A
            random draw is used rather than a fixed stride, since galaxies
            are stored in a fixed (non-random) order and a stride would
            sample a specific, non-representative pattern. Default is None
            (no subsampling).
        shared_randoms : bool, optional
            With ``randoms``, if True (default), draw a single random-point realization (sized
            from the first processed bin) and reuse it for every bin's
            density field, rather than drawing independent randoms per bin.
            This matches the fact that all bins share the same survey
            footprint, and is cheaper since randoms are only drawn once. If
            False, each bin gets its own randoms, seeded by ``seed + ikappa``.
        remove_mean : bool, optional
            If True (default), subtract the mean of the shear columns over each bin
            before the measurement.
        pairs : list of tuple of int or None, optional
            If given, only compute these ``(i, j)`` bin pairs, for every
            statistic in ``what`` (overrides ``ikappas`` and ``auto_only``).
            Only the catalogs of the bins involved are loaded.
        save : bool, optional
            If True (default), save the spectra to a SACC FITS file via
            :meth:`_save_cls`, at ``{self._catalog_dir}/cls.fits``.

        Returns
        -------
        ell : ndarray
            Effective multipoles of the bandpowers (shared with other C_ell
            products via the same underlying binning, minus pixel-window
            deconvolution).
        cl : dict
            Keyed by SACC data type (e.g. ``'galaxy_density_cl'``,
            ``'galaxy_shear_cl_ee'``, ``'galaxy_shear_cl_bb'``,
            ``'galaxy_shearDensity_cl_e'``, ``'galaxy_shearDensity_cl_b'``,
            depending on ``what``), each mapping to a dict keyed by ``(i, j)``.
        """
        if isinstance(what, str):
            what = ['clustering', 'shear', 'shear_clustering'] if what == 'all' else [what]

        ikappas = sorted({k for pair in pairs for k in pair}) if pairs is not None else list(self.lensing.iter(ikappas))
        b = self.b
        ell = b.get_effective_ells()
        f_dens, f_shear = self._cls_fields(ikappas, what, shear_cols, randoms, nran_factor, seed, subsample, shared_randoms, remove_mean)

        cl = {}

        def decoupled(f1, f2, name, pair):
            # the shared workspace is that of the footprint, to which pseudo-spectra are rescaled (see get_wsp)
            pcl = nmt.compute_coupled_cell(f1, f2)
            if shared_wsp:
                return self.get_wsp(name, read_unbinned_MCM=False).decouple_cell(pcl / (f1.mask_amplitude * f2.mask_amplitude))
            return self.get_wsp(name, pair=pair, fields=(f1, f2), read_unbinned_MCM=False, randoms=randoms).decouple_cell(pcl)

        # symmetric pairs (i <= j): valid for same-type correlations (dd, shear-shear)
        if pairs is not None:
            sym_pairs = list(pairs)
        elif auto_only:
            sym_pairs = [(i, i) for i in ikappas]
        else:
            sym_pairs = [(i1, i2) for i, i2 in enumerate(ikappas) for i1 in ikappas[i:]]

        if 'clustering' in what:
            cl_dd = cl[sacc.standard_types.galaxy_density_cl] = {}
            for i, j in self.lensing.tqdm(sym_pairs, desc="[compute_cls] clustering"):
                cl_dd[i, j] = decoupled(f_dens[i], f_dens[j], 'clustering', (i, j)).squeeze()

        if 'shear' in what:
            cl_ee = cl[sacc.standard_types.galaxy_shear_cl_ee] = {}
            cl_bb = cl[sacc.standard_types.galaxy_shear_cl_bb] = {}
            for i, j in self.lensing.tqdm(sym_pairs, desc="[compute_cls] shear"):
                cl_ee_bb = decoupled(f_shear[i], f_shear[j], 'shear', (i, j))
                cl_ee[i, j] = cl_ee_bb[0]
                cl_bb[i, j] = cl_ee_bb[3]

        if 'shear_clustering' in what:
            # not symmetric under bin exchange (lens bin i, source bin j) -> all ordered pairs
            if pairs is not None:
                de_pairs = list(pairs)
            else:
                de_pairs = [(i, i) for i in ikappas] if auto_only else [(i, j) for i in ikappas for j in ikappas]
            cl_de = cl[sacc.standard_types.galaxy_shearDensity_cl_e] = {}
            cl_db = cl[sacc.standard_types.galaxy_shearDensity_cl_b] = {}
            for i, j in self.lensing.tqdm(de_pairs, desc="[compute_cls] shear x clustering"):
                cl_de_db = decoupled(f_dens[i], f_shear[j], 'shear_clustering', (i, j))
                cl_de[i, j] = cl_de_db[0]
                cl_db[i, j] = cl_de_db[1]

        if save:
            self._save_cls(ell, cl)

        return ell, cl

    # which tracer role (density or shear) each SACC data type's (i, j) legs refer to
    _SACC_TYPE_LEGS = {
        sacc.standard_types.galaxy_density_cl: ('density', 'density'),
        sacc.standard_types.galaxy_shear_cl_ee: ('shear', 'shear'),
        sacc.standard_types.galaxy_shear_cl_bb: ('shear', 'shear'),
        sacc.standard_types.galaxy_shearDensity_cl_e: ('density', 'shear'),
        sacc.standard_types.galaxy_shearDensity_cl_b: ('density', 'shear'),
    }

    def _save_cls(self, ell, cl):
        """Save catalog-based angular power spectra to a SACC FITS file.

        The catalog's density and shear fields share the same underlying n(z)
        per bin (galaxy redshifts are drawn from the lensing tracers' own
        dndz, see :meth:`_generate_z`), but are registered under distinct
        tracer names (``density_{i}`` vs. the lensing builder's ``kappa_{i}``)
        so that clustering data points don't reference a lensing-named tracer.

        Parameters
        ----------
        ell : ndarray
            Effective multipoles (as returned by :meth:`compute_cls`).
        cl : dict
            Dictionary keyed by SACC data type, each mapping to a dict keyed
            by ``(i, j)`` (as returned by :meth:`compute_cls`).

        Returns
        -------
        str
            Path to the saved SACC FITS file.
        """
        s = sacc.Sacc()
        shear_tracers = self.lensing.tracers_sacc
        density_tracers = [t.sacc() for t in self.density_tracers]
        for t in shear_tracers + density_tracers:
            s.add_tracer_object(t)

        tracers_by_role = {'density': density_tracers, 'shear': shear_tracers}

        for data_type, pairs in cl.items():
            role_i, role_j = self._SACC_TYPE_LEGS[data_type]
            tracers_i, tracers_j = tracers_by_role[role_i], tracers_by_role[role_j]
            for (i, j), _cl in pairs.items():
                for _ell, _cl_ell in zip(ell, _cl):
                    s.add_data_point(data_type, [tracers_i[i].name, tracers_j[j].name], _cl_ell, ell=_ell)
        fname = os.path.join(self._catalog_dir, 'cat_cls.fits')
        s.save_fits(fname, overwrite=True)
        logger.debug("Saved to file %s.", fname)
        return fname

    def load_cls(self, return_dict: bool = False) -> sacc.Sacc | tuple[np.ndarray, dict[str, dict[tuple[int, int], np.ndarray]]]:
        """Load catalog-based angular power spectra previously saved by :meth:`_save_cls`.

        Parameters
        ----------
        return_dict : bool, optional
            If True, return ``(ell, cl)``. Otherwise, return the SACC object.

        Returns
        -------
        sacc.Sacc or (ell, dict)
            If ``return_dict=False``, returns the loaded SACC object.
            If ``return_dict=True``, returns ``(ell, cl)`` where ``ell`` is the
            multipole array and ``cl`` is a dict keyed by SACC data type, each
            mapping to a dict keyed by ``(i, j)`` (same structure as returned
            by :meth:`compute_cls`). Reads only the data types and tracer
            pairs actually present in the file (:meth:`compute_cls` may not
            have computed all of them), with no dependency on ``self.lensing``
            beyond the file path.
        """
        fname = os.path.join(self._catalog_dir, 'cat_cls.fits')
        s = sacc.Sacc.load_fits(fname)
        logger.debug("Found and loaded file at %s.", fname)
        if not return_dict:
            return s

        def tracer_index(name):
            return int(name.rsplit('_', 1)[-1])

        ell = None
        cl = {}
        for data_type in s.get_data_types():
            pairs = {}
            for name_i, name_j in s.get_tracer_combinations(data_type):
                ell, _cl = s.get_ell_cl(data_type, name_i, name_j)
                pairs[tracer_index(name_i), tracer_index(name_j)] = _cl
            cl[data_type] = pairs

        return ell, cl

    def compute_2pt(
            self,
            ikappas=None,
            what='all',
            auto_only=False,
            shear_cols=('gamma1', 'gamma2'),
            theta_min=1.,
            theta_max=100.,
            ntheta=20,
            sep_units='arcmin',
            bin_slop=None,
            nran_factor=5,
            seed=None,
            subsample=None,
            shared_randoms=True,
            remove_mean=True,
            error_bars=False,
            theta_per_pair=False,
            pairs=None,
            save=True):
        """Measure real-space two-point correlation functions directly from the
        generated catalog(s), including all cross-tomographic-bin combinations.

        Uses TreeCorr's angular correlation estimators (``NNCorrelation`` for
        clustering, ``GGCorrelation`` for shear, ``NGCorrelation`` for the
        shear-density cross-correlation), binned in angular separation between
        ``theta_min`` and ``theta_max``. This is the real-space analog of
        :meth:`compute_cls`; ``ikappas``, ``what``, ``auto_only``,
        ``shear_cols``, ``nran_factor``, ``seed`` and ``subsample`` have the
        same meaning there.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic bin indices to process. If None, uses all bins.
        what : {'shear', 'clustering', 'shear_clustering', 'all'} or list of str, optional
            Which correlations to compute:
            - 'clustering': galaxy density autocorrelation ``w(theta)``, symmetric i<=j pairs.
            - 'shear': shear autocorrelation ``xi_+/-(theta)``, symmetric i<=j pairs.
            - 'shear_clustering': density-shear cross-correlation ``gamma_t/x(theta)``,
              all (i, j) ordered pairs since lens-bin i / source-bin j is not symmetric.
            - 'all': compute all three. Default.
        auto_only : bool, optional
            If True, restrict to same-bin pairs (i, i) only, skipping cross-bin
            correlations. Default is False.
        shear_cols : tuple of str, optional
            Names of the two catalog columns to use as the spin-2 shear field.
            See :meth:`compute_cls`.
        theta_min, theta_max : float, optional
            Angular separation range, in ``sep_units``. Defaults are 1 and 100 arcmin.
        ntheta : int, optional
            Number of log-spaced angular bins. Default is 20.
        sep_units : str, optional
            Units of ``theta_min``/``theta_max`` and of the returned ``theta``.
            Default is 'arcmin'.
        bin_slop : float or None, optional
            TreeCorr ``bin_slop`` (binning approximation tolerance). If None,
            uses TreeCorr's own default.
        nran_factor : int, optional
            Ratio of random points to galaxies for each bin's random catalog,
            used for the Landy-Szalay ``w(theta)`` estimator and for
            subtracting additive bias in ``gamma_t/x(theta)``. Default is 5.
        seed : int or None, optional
            Seed for the random catalogs' RNG. Defaults to the shell builder's seed.
        subsample : float or None, optional
            If not None, randomly draw (without replacement, seeded by
            ``seed + ikappa``) this fraction of galaxies from each loaded
            catalog before building fields, with ``0 < subsample < 1``. A
            random draw is used rather than a fixed stride, since galaxies
            are stored in a fixed (non-random) order and a stride would
            sample a specific, non-representative pattern. Default is None
            (no subsampling).
        shared_randoms : bool, optional
            If True (default), draw a single random-point realization (sized
            from the first processed bin) and reuse it for every bin's
            random catalog (used both for the ``w(theta)`` estimator and for
            the ``gamma_t/x(theta)`` random-lens subtraction), rather than
            drawing independent randoms per bin. This matches the fact that
            all bins share the same survey footprint, and is cheaper since
            randoms (and their TreeCorr catalog/tree) are only built once.
            If False, each bin gets its own randoms, seeded by ``seed + ikappa``.
        remove_mean : bool, optional
            If True (default), subtract the mean of the shear columns over each bin
            before the measurement.
        error_bars : bool, optional
            If True, also return ``dxi``: standard deviations (shot-noise
            error bars from TreeCorr), with the same structure as ``xi``.
            Default is False.
        theta_per_pair : bool, optional
            If True, return ``theta`` as a dict with the same structure as
            ``xi`` (each pair's own mean separation from its correlation
            object's ``meanlogr``) instead of a single shared array. The
            exact mean separation is catalog-dependent and can differ
            slightly from one pair to another; this is always what gets
            written to the SACC file regardless of this flag. Default is
            False (return a single array, taken from the first pair computed).
        pairs : list of tuple of int or None, optional
            If given, only compute these ``(i, j)`` bin pairs, for every
            statistic in ``what`` (overrides ``ikappas`` and ``auto_only``).
            Only the catalogs of the bins involved are loaded.
        save : bool, optional
            If True (default), save the correlation functions to a SACC FITS
            file via :meth:`_save_2pt`, at ``{self._catalog_dir}/cat_2pt.fits``.

        Returns
        -------
        theta : ndarray or dict
            Mean angular separation, in ``sep_units``. A single shared array,
            or a dict keyed like ``xi`` if ``theta_per_pair=True``.
        xi : dict
            Keyed by SACC data type (e.g. ``'galaxy_density_xi'``,
            ``'galaxy_shear_xi_plus'``, ``'galaxy_shear_xi_minus'``,
            ``'galaxy_shearDensity_xi_t'``, ``'galaxy_shearDensity_xi_x'``,
            depending on ``what``), each mapping to a dict keyed by ``(i, j)``.

        Other Parameters
        ----------------
        error_bars : bool
            If True, also returns ``dxi`` (dict keyed like ``xi``).
        """
        if isinstance(what, str):
            what = ['clustering', 'shear', 'shear_clustering'] if what == 'all' else [what]

        builder = self.lensing.shell_builder
        rng_seed = builder.seed if seed is None else seed

        _, ipix_mask = builder.get_mask(nest=True, return_ipix=True)

        ikappas = sorted({k for pair in pairs for k in pair}) if pairs is not None else list(self.lensing.iter(ikappas))

        need_density = 'clustering' in what or 'shear_clustering' in what
        need_shear = 'shear' in what or 'shear_clustering' in what
        col1, col2 = shear_cols

        corr_kwargs = dict(min_sep=theta_min, max_sep=theta_max, nbins=ntheta, sep_units=sep_units, bin_slop=bin_slop)

        # Build each bin's catalog(s) once, reused across all pairs below.
        # A single galaxy catalog serves as both the density tracer (for NN)
        # and the shear tracer (for GG/NG): NNCorrelation only looks at
        # positions and simply ignores the g1/g2 fields when present.
        cat_gal, cat_rand = {}, {}
        shared_cat_rand = None
        for ikappa in self.lensing.tqdm(ikappas, desc="[compute_2pt] building catalogs"):
            cat = self.load_catalogs(ikappa, subsample=subsample, seed=rng_seed, columns=['ra', 'dec', *shear_cols],
                                     remove_mean=list(shear_cols) if remove_mean else None)

            gal_kwargs = dict(ra=cat['ra'], dec=cat['dec'], ra_units='deg', dec_units='deg')
            if need_shear:
                gal_kwargs.update(g1=cat[col1], g2=cat[col2], flip_g2=True)
            cat_gal[ikappa] = treecorr.Catalog(**gal_kwargs)

            if need_density:
                if shared_randoms:
                    if shared_cat_rand is None:
                        rng = np.random.default_rng(rng_seed)
                        nran = int(nran_factor * cat['ra'].size)
                        ra_rand, dec_rand = self._random_radec(ipix_mask, nran, rng)
                        shared_cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units='deg', dec_units='deg')
                    cat_rand[ikappa] = shared_cat_rand
                else:
                    rng = np.random.default_rng(rng_seed + ikappa)
                    nran = int(nran_factor * cat['ra'].size)
                    ra_rand, dec_rand = self._random_radec(ipix_mask, nran, rng)
                    cat_rand[ikappa] = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units='deg', dec_units='deg')

        xi = {}
        dxi = {}
        theta_pairs = {}
        theta = None

        # symmetric pairs (i <= j): valid for same-type correlations (NN, GG)
        if pairs is not None:
            sym_pairs = list(pairs)
        elif auto_only:
            sym_pairs = [(i, i) for i in ikappas]
        else:
            sym_pairs = [(i1, i2) for i, i2 in enumerate(ikappas) for i1 in ikappas[i:]]

        if 'clustering' in what:
            xi_nn = xi[sacc.standard_types.galaxy_density_xi] = {}
            dxi_nn = dxi[sacc.standard_types.galaxy_density_xi] = {}
            theta_nn = theta_pairs[sacc.standard_types.galaxy_density_xi] = {}
            for i, j in self.lensing.tqdm(sym_pairs, desc="[compute_2pt] clustering"):
                dd = treecorr.NNCorrelation(**corr_kwargs)
                rr = treecorr.NNCorrelation(**corr_kwargs)
                dr = treecorr.NNCorrelation(**corr_kwargs)
                if i == j:
                    dd.process(cat_gal[i])
                    rr.process(cat_rand[i])
                    dr.process(cat_gal[i], cat_rand[i])
                    xi_nn[i, j], var_nn = dd.calculateXi(rr=rr, dr=dr)
                else:
                    rd = treecorr.NNCorrelation(**corr_kwargs)
                    dd.process(cat_gal[i], cat_gal[j])
                    rr.process(cat_rand[i], cat_rand[j])
                    dr.process(cat_gal[i], cat_rand[j])
                    rd.process(cat_rand[i], cat_gal[j])
                    xi_nn[i, j], var_nn = dd.calculateXi(rr=rr, dr=dr, rd=rd)
                dxi_nn[i, j] = np.sqrt(var_nn)
                theta_nn[i, j] = np.exp(dd.meanlogr)
                if theta is None:
                    theta = theta_nn[i, j]

        if 'shear' in what:
            xi_p = xi[sacc.standard_types.galaxy_shear_xi_plus] = {}
            xi_m = xi[sacc.standard_types.galaxy_shear_xi_minus] = {}
            dxi_p = dxi[sacc.standard_types.galaxy_shear_xi_plus] = {}
            dxi_m = dxi[sacc.standard_types.galaxy_shear_xi_minus] = {}
            theta_p = theta_pairs[sacc.standard_types.galaxy_shear_xi_plus] = {}
            theta_m = theta_pairs[sacc.standard_types.galaxy_shear_xi_minus] = {}
            for i, j in self.lensing.tqdm(sym_pairs, desc="[compute_2pt] shear"):
                gg = treecorr.GGCorrelation(**corr_kwargs)
                if i == j:
                    gg.process(cat_gal[i])
                else:
                    gg.process(cat_gal[i], cat_gal[j])
                xi_p[i, j] = gg.xip
                xi_m[i, j] = gg.xim
                dxi_p[i, j] = np.sqrt(gg.varxip)
                dxi_m[i, j] = np.sqrt(gg.varxim)
                theta_p[i, j] = theta_m[i, j] = np.exp(gg.meanlogr)
                if theta is None:
                    theta = theta_p[i, j]

        if 'shear_clustering' in what:
            # not symmetric under bin exchange (lens bin i, source bin j) -> all ordered pairs
            if pairs is not None:
                de_pairs = list(pairs)
            else:
                de_pairs = [(i, i) for i in ikappas] if auto_only else [(i, j) for i in ikappas for j in ikappas]
            xi_t = xi[sacc.standard_types.galaxy_shearDensity_xi_t] = {}
            xi_x = xi[sacc.standard_types.galaxy_shearDensity_xi_x] = {}
            dxi_t = dxi[sacc.standard_types.galaxy_shearDensity_xi_t] = {}
            dxi_x = dxi[sacc.standard_types.galaxy_shearDensity_xi_x] = {}
            theta_t = theta_pairs[sacc.standard_types.galaxy_shearDensity_xi_t] = {}
            theta_x = theta_pairs[sacc.standard_types.galaxy_shearDensity_xi_x] = {}
            for i, j in self.lensing.tqdm(de_pairs, desc="[compute_2pt] shear x clustering"):
                ng = treecorr.NGCorrelation(**corr_kwargs)
                rg = treecorr.NGCorrelation(**corr_kwargs)
                ng.process(cat_gal[i], cat_gal[j])
                rg.process(cat_rand[i], cat_gal[j])
                xi_t[i, j], xi_x[i, j], var_ng = ng.calculateXi(rg=rg)
                dxi_t[i, j] = dxi_x[i, j] = np.sqrt(var_ng)
                theta_t[i, j] = theta_x[i, j] = np.exp(ng.meanlogr)
                if theta is None:
                    theta = theta_t[i, j]

        if save:
            self._save_2pt(theta_pairs, xi, dxi=dxi)

        theta_out = theta_pairs if theta_per_pair else theta

        if error_bars:
            return theta_out, xi, dxi
        else:
            return theta_out, xi

    # which tracer role (density or shear) each SACC data type's (i, j) legs refer to
    _SACC_TYPE_LEGS_XI = {
        sacc.standard_types.galaxy_density_xi: ('density', 'density'),
        sacc.standard_types.galaxy_shear_xi_plus: ('shear', 'shear'),
        sacc.standard_types.galaxy_shear_xi_minus: ('shear', 'shear'),
        sacc.standard_types.galaxy_shearDensity_xi_t: ('density', 'shear'),
        sacc.standard_types.galaxy_shearDensity_xi_x: ('density', 'shear'),
    }

    def _save_2pt(self, theta, xi, dxi=None):
        """Save catalog-based real-space two-point functions to a SACC FITS file.

        Mirrors :meth:`_save_cls`: the catalog's density and shear fields
        share the same underlying n(z) per bin but are registered under
        distinct tracer names (``density_{i}`` vs. the lensing builder's
        ``kappa_{i}``).

        Parameters
        ----------
        theta : dict
            Angular separations, keyed by SACC data type then by ``(i, j)``
            (the ``theta_pairs`` structure computed internally by
            :meth:`compute_2pt`, i.e. each pair's own mean separation, since
            it is catalog-dependent and can differ slightly between pairs).
        xi : dict
            Dictionary keyed by SACC data type, each mapping to a dict keyed
            by ``(i, j)`` (as returned by :meth:`compute_2pt`).
        dxi : dict or None, optional
            Error bars with the same structure as ``xi``; if given, saved as the
            diagonal of the SACC covariance.

        Returns
        -------
        str
            Path to the saved SACC FITS file.
        """
        s = sacc.Sacc()
        shear_tracers = self.lensing.tracers_sacc
        density_tracers = [t.sacc() for t in self.density_tracers]
        for t in shear_tracers + density_tracers:
            s.add_tracer_object(t)

        tracers_by_role = {'density': density_tracers, 'shear': shear_tracers}

        for data_type, pairs in xi.items():
            role_i, role_j = self._SACC_TYPE_LEGS_XI[data_type]
            tracers_i, tracers_j = tracers_by_role[role_i], tracers_by_role[role_j]
            for (i, j), _xi in pairs.items():
                _theta = theta[data_type][i, j]
                for _theta_k, _xi_k in zip(_theta, _xi):
                    s.add_data_point(data_type, [tracers_i[i].name, tracers_j[j].name], _xi_k, theta=_theta_k)
        if dxi is not None:
            # same order as the data points added above
            var = np.concatenate([dxi[data_type][ij]**2 for data_type, pairs in xi.items() for ij in pairs])
            s.add_covariance(np.diag(var))
        fname = os.path.join(self._catalog_dir, 'cat_2pt.fits')
        s.save_fits(fname, overwrite=True)
        logger.debug("Saved to file %s.", fname)
        return fname

    def load_2pt(self, return_dict: bool = False, return_errors: bool = False) -> sacc.Sacc | tuple:
        """Load catalog-based two-point functions previously saved by :meth:`_save_2pt`.

        Parameters
        ----------
        return_dict : bool, optional
            If True, return ``(theta, xi)``. Otherwise, return the SACC object.
        return_errors : bool, optional
            With ``return_dict``, also return the error bars (square root of the
            diagonal of the saved covariance), with the same structure as ``xi``.

        Returns
        -------
        sacc.Sacc or (dict, dict)
            If ``return_dict=False``, returns the loaded SACC object.
            If ``return_dict=True``, returns ``(theta, xi)``, both dicts keyed
            by SACC data type then by ``(i, j)`` (same structure as
            ``theta_pairs``/``xi`` in :meth:`compute_2pt`; each pair's own
            mean separation is preserved, since it is catalog-dependent).
            Reads only the data types and tracer pairs actually present in
            the file (:meth:`compute_2pt` may not have computed all of them),
            with no dependency on ``self.lensing`` beyond the file path.
        """
        fname = os.path.join(self._catalog_dir, 'cat_2pt.fits')
        s = sacc.Sacc.load_fits(fname)
        logger.debug("Found and loaded file at %s.", fname)
        if not return_dict:
            return s

        def tracer_index(name):
            return int(name.rsplit('_', 1)[-1])

        theta = {}
        xi = {}
        dxi = {}
        sigma = np.sqrt(np.diag(s.covariance.covmat)) if return_errors and s.covariance is not None else None
        for data_type in s.get_data_types():
            theta_pairs, xi_pairs, dxi_pairs = {}, {}, {}
            for name_i, name_j in s.get_tracer_combinations(data_type):
                key = tracer_index(name_i), tracer_index(name_j)
                theta_pairs[key], xi_pairs[key] = s.get_theta_xi(data_type, name_i, name_j)
                if sigma is not None:
                    dxi_pairs[key] = sigma[s.indices(data_type, (name_i, name_j))]
            theta[data_type] = theta_pairs
            xi[data_type] = xi_pairs
            dxi[data_type] = dxi_pairs

        if return_errors:
            return theta, xi, dxi
        return theta, xi


    @property
    def lensing_tracers(self):
        """Lensing tracers of the catalog, those of the lensing builder (:attr:`MapBuilder.tracers`).

        Returns
        -------
        list of pollux.cosmo.Tracer
            Shear tracers (one per bin) with the n(z) of the catalog's galaxies.
        """
        return self.lensing.tracers

    @property
    def density_tracers(self):
        """Density tracers of the catalog: each bin's n(z) and linear bias.

        Returns
        -------
        list of pollux.cosmo.Tracer
            ``'density'`` tracers named ``density_{i}``, with the n(z) of the lensing
            tracers (the galaxies are both lenses and sources).
        """
        bias = getattr(self, 'bias', np.ones(self.ncats))
        return [Tracer(f"density_{i}", 'density', z=t.z, nz=t.nz, bias=bias[i]) for i, t in enumerate(self.lensing_tracers)]

    def _tracers(self):
        """Return the tracers of the density and shear legs of the statistics."""
        return {'density': self.density_tracers, 'shear': self.lensing_tracers}

    @staticmethod
    def _stat_pairs(stat, ikappas):
        """Return the bin pairs of one statistic: ``i >= j`` for auto-type statistics, all ordered pairs otherwise."""
        if stat == 'shear_clustering':
            return [(i, j) for i in ikappas for j in ikappas]
        return [(i1, i2) for k, i2 in enumerate(ikappas) for i1 in ikappas[k:]]

    def _noise_cls(self, tracer, shear_cols=('gamma1', 'gamma2'), subsample=None):
        """Return the white noise level of one catalog tracer.

        Parameters
        ----------
        tracer : pollux.cosmo.Tracer
            Density (``density_{i}``) or shear (``kappa_{i}``) tracer of bin ``i``.
        shear_cols : tuple of str, optional
            Shear columns used for the measurement (see :meth:`compute_cls`): shape
            noise only for ellipticities (``e1_*``), none for the lensing shear.
        subsample : float or None, optional
            Fraction of galaxies used for the measurement.

        Returns
        -------
        float
            Shot noise ``1 / n`` (density) or shape noise ``sigma_e**2 / n`` (shear),
            with ``n`` the bin's galaxy density per steradian.
        """
        i = int(tracer.name.rsplit('_', 1)[-1])
        n_sr = self.nbar[i] * (180. * 60. / np.pi)**2 * (subsample or 1.)
        if tracer.kind == 'density':
            return 1. / n_sr
        if not shear_cols[0].startswith('e'):
            return 0.
        sigma_e = next((a.sigma_e for a in self.augmentors if hasattr(a, 'sigma_e')), 0.)
        return sigma_e**2 / n_sr

    def predict_cls(self, ikappas=None, what='all', auto_only=False, pairs=None, binning='wsp', shared_wsp=False, randoms=False, ell=None,
                    shear_cols=('gamma1', 'gamma2'), subsample=None, error_bars=False, keep_cls=False, stacked=False):
        """Predict the catalog angular power spectra.

        Predictions use the lensing builder's :attr:`~pollux.map_builder.MapBuilder.theory`,
        i.e. the matter power spectrum of the catalog's shells (CCL's nonlinear P(k)
        for dark-matter-only shells, S19-corrected for baryonified shells). Predicted
        spectra are noise-free, like the measured ones (NaMaster subtracts the noise
        bias of catalog auto-spectra).

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Tomographic bins. If None, all bins.
        what : {'shear', 'clustering', 'shear_clustering', 'all'} or list of str, optional
            Statistics to predict (see :meth:`compute_cls`).
        auto_only : bool, optional
            If True, only same-bin pairs ``(i, i)``.
        pairs : list of tuple of int or None, optional
            If given, only these ``(i, j)`` bin pairs, for every statistic in ``what``
            (overrides ``ikappas`` and ``auto_only``).
        binning : {'wsp', 'average', 'unbinned'}, optional
            ``'wsp'`` (default): bandpowers binned with the bandpower windows of the
            workspaces of :meth:`compute_cls` (see ``shared_wsp`` and :meth:`get_bpws`). ``'average'``: bandpowers averaged over each bandpower.
            ``'unbinned'``: spectra at ``ell``.
        shared_wsp : bool, optional
            With ``binning='wsp'``, as in :meth:`compute_cls`: use the workspace shared
            by all pairs of each statistic if True, else that of each pair (default).
        randoms : bool, optional
            With per-pair workspaces, as in :meth:`compute_cls`.
        ell : ndarray or None, optional
            Multipoles for ``binning='unbinned'``. Default is the effective bandpower multipoles.
        shear_cols, subsample
            As in :meth:`compute_cls`, setting the noise (see :meth:`_noise_cls`).
        error_bars : bool, optional
            If True, also return Gaussian error bars (E-mode and density spectra),
            with ``C_ell + N_ell`` for auto-spectra (:meth:`_noise_cls`) and the
            lensing builder's sky fraction.
        keep_cls : bool, optional
            If True, cache the results on ``self.predicted_cls``.
        stacked : bool, optional
            If True, also return the E-mode and density spectra of all statistics
            stacked into one vector (in the order of ``what``, then of the pairs), and
            with ``error_bars`` their joint covariance.

        Returns
        -------
        ell : ndarray
            Multipoles.
        cl : dict
            Keyed by SACC data type, then by ``(i, j)`` (same structure as :meth:`compute_cls`).
        dcl : dict
            Only with ``error_bars``, keyed like ``cl`` (E-mode and density types).
        cl_stacked, cov : ndarray
            Only with ``stacked`` (``cov`` only with ``error_bars`` too).

        Notes
        -----
        Wraps :meth:`pollux.cosmo.Theory.predict_cls` (one call for all statistics,
        so that the covariance includes cross-statistic blocks).
        """
        if self.predicted_cls is not None:
            return self.predicted_cls
        if isinstance(what, str):
            what = ['clustering', 'shear', 'shear_clustering'] if what == 'all' else [what]
        tracers = self._tracers()
        st = sacc.standard_types
        # data types of each statistic, with (tracer roles, index of the coupled spectrum in the bandpower windows)
        types = {
            'clustering': (('density', 'density'), [(st.galaxy_density_cl, 0)]),
            'shear': (('shear', 'shear'), [(st.galaxy_shear_cl_ee, 0), (st.galaxy_shear_cl_bb, 3)]),
            'shear_clustering': (('density', 'shear'), [(st.galaxy_shearDensity_cl_e, 0), (st.galaxy_shearDensity_cl_b, 1)]),
        }
        ikappas = list(self.lensing.iter(ikappas))

        tracer_pairs, bpws = {}, {}
        for stat in what:
            (role_i, role_j), _ = types[stat]
            if pairs is not None:
                stat_pairs = [tuple(pair) for pair in pairs]
            elif auto_only:
                stat_pairs = [(i, i) for i in ikappas]
            else:
                stat_pairs = self._stat_pairs(stat, ikappas)
            windows = {}  # bandpower windows by workspace (bin pair, or None if shared)
            for i, j in stat_pairs:
                tracer_pairs[stat, i, j] = (tracers[role_i][i], tracers[role_j][j])
                if binning == 'wsp':
                    pair = None if shared_wsp else (i, j)
                    if pair not in windows:
                        windows[pair] = self.get_bpws(stat, pair, randoms)
                    bpws[stat, i, j] = windows[pair]

        noise = None
        if error_bars:
            noise = {t.name: self._noise_cls(t, shear_cols, subsample) for t1, t2 in tracer_pairs.values() for t in (t1, t2)}
        res = self.lensing.theory.predict_cls(
            tracer_pairs, b=self.b, ell=ell, binning=binning, bpws=bpws, noise=noise, error_bars=error_bars,
            stacked=stacked, fsky=self.lensing.fsky, all_components=True)

        cl = {}
        for (stat, i, j), c in res[1].items():
            c = np.atleast_2d(c)  # components (a single one without bandpower windows)
            for data_type, k in types[stat][1]:
                cl.setdefault(data_type, {})[i, j] = c[k] if k < len(c) else np.zeros_like(c[0])
        out = [res[0], cl]
        if error_bars:
            dcl = {}
            for (stat, i, j), d in res[2].items():
                dcl.setdefault(types[stat][1][0][0], {})[i, j] = d
            out.append(dcl)
        out += list(res[2 + error_bars:])
        out = tuple(out)
        if keep_cls:
            self.predicted_cls = out
        return out

    # tracer roles of each real-space SACC data type, and component of the correlation (xi+ or xi-)
    _XI_THEORY = {
        sacc.standard_types.galaxy_shear_xi_plus: ('shear', 'shear', 0),
        sacc.standard_types.galaxy_shear_xi_minus: ('shear', 'shear', 1),
        sacc.standard_types.galaxy_shearDensity_xi_t: ('density', 'shear', None),
        sacc.standard_types.galaxy_density_xi: ('density', 'density', None),
    }

    def predict_2pt(self, theta, nell=500):
        """Predict the catalog two-point correlation functions.

        Predictions use the matter power spectrum of the catalog's shells (see :meth:`predict_cls`).

        Parameters
        ----------
        theta : dict
            Angular separations in arcmin keyed by SACC data type, then by ``(i, j)``
            (as returned by :meth:`load_2pt` or :meth:`compute_2pt` with ``theta_per_pair=True``).
        nell : int, optional
            Number of log-spaced multipoles up to the bandpower ``lmax`` used for the Hankel transforms.

        Returns
        -------
        dict
            Same structure as ``theta``; ``galaxy_shearDensity_xi_x`` is zero.
        """
        theory = self.lensing.theory
        tracers = self._tracers()
        xi = {}
        for data_type, type_pairs in theta.items():
            xi[data_type] = {}
            for (i, j), tt in type_pairs.items():
                if data_type not in self._XI_THEORY:
                    xi[data_type][i, j] = np.zeros(len(tt))
                    continue
                role_i, role_j, component = self._XI_THEORY[data_type]
                res = theory.correlation(tracers[role_i][i], tracers[role_j][j], tt, self.b.lmax, nell=nell)
                xi[data_type][i, j] = res if component is None else res[component]
        return xi

    def plot_2pt(self, reference=None, plot_dir=None):
        """Plot the saved two-point functions against their prediction.

        Parameters
        ----------
        reference : CatalogSourceGenerator or None, optional
            Catalog with saved two-point functions (e.g. dark-matter-only, for a
            baryonified catalog): the ratio of the two catalogs' functions is also
            plotted against the ratio of their predictions.
        plot_dir : str or None, optional
            Output directory. Default is ``{catalog_dir}/plots``.

        Notes
        -----
        Saves ``xi_shear``, ``xi_shear_density`` and ``xi_density``, and
        ``*_ratio_to_reference`` with ``reference``.
        """
        from .plotting import plot_pairs
        plot_dir = plot_dir or os.path.join(self._catalog_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        theta, xi, dxi = self.load_2pt(return_dict=True, return_errors=True)
        xi_th = self.predict_2pt(theta)
        if reference is not None:
            _, xi_ref = reference.load_2pt(return_dict=True)
            xi_th_ref = reference.predict_2pt(theta)
        scales = self.lensing.reference_scales()
        st = sacc.standard_types
        for data_types, name, ylabel, ratio_label in (
                ({st.galaxy_shear_xi_plus: 'dodgerblue', st.galaxy_shear_xi_minus: 'orange'}, 'xi_shear',
                 r'$\theta \xi_\pm(\theta)$', r'$\xi_\pm / \xi_\pm^{\rm ref} - 1$'),
                ({st.galaxy_shearDensity_xi_t: 'red'}, 'xi_shear_density',
                 r'$\theta \gamma_{\rm t}(\theta)$', r'$\gamma_{\rm t} / \gamma_{\rm t}^{\rm ref} - 1$'),
                ({st.galaxy_density_xi: 'green'}, 'xi_density', r'$\theta w(\theta)$', r'$w / w^{\rm ref} - 1$')):
            data_types = {t: c for t, c in data_types.items() if t in xi}
            modes = [('values', ylabel, '')] + ([('ratio_to_reference', ratio_label, '_ratio_to_reference')] if reference is not None else [])
            for mode, label, suffix in modes:
                figaxes = None
                for k, (data_type, color) in enumerate(data_types.items()):
                    last = k == len(data_types) - 1
                    figaxes = plot_pairs(
                        theta[data_type], data=xi[data_type], theory=xi_th[data_type], data_errors=dxi.get(data_type) if dxi else None,
                        data_ref=xi_ref[data_type] if reference is not None else None,
                        theory_ref=xi_th_ref[data_type] if reference is not None else None, mode=mode, nbins=self.ncats, y_factor=1,
                        scales=scales if mode == 'values' else None, color=color, ylim=(-.5, .2) if mode != 'values' else None,
                        xlabel=r'$\theta$ (arcmin)', ylabel=label, figaxes=figaxes,
                        fname=os.path.join(plot_dir, f'{name}{suffix}.png') if last else None)

    def plot_cls(self, plot_dir=None, modes=('values', 'ratio'), ell_factor=1, error_bars=False, **predict_kwargs):
        """Plot the saved angular power spectra against their prediction (:meth:`predict_cls`).

        Parameters
        ----------
        plot_dir : str or None, optional
            Output directory. Default is ``{catalog_dir}/plots``.
        modes : tuple of str, optional
            Plots to make, see :func:`pollux.plotting.plot_pairs` (``'values'``, ``'ratio'``,
            ``'diff_over_error'``). Default is ``('values', 'ratio')``.
        ell_factor : {0, 1, 2}, optional
            In ``'values'`` mode, plot ``C_ell`` (0), ``ell C_ell`` (1, default) or
            ``ell (ell + 1) C_ell / 2 pi`` (2).
        error_bars : bool, optional
            If True, draw Gaussian errors of the prediction as bands (always used by
            ``'diff_over_error'``).
        **predict_kwargs
            Options of :meth:`predict_cls`, e.g. ``shared_wsp`` and ``randoms`` as used by :meth:`compute_cls`.

        Notes
        -----
        Saves ``cl_shear_ee``, ``cl_shear_density`` and ``cl_density``, with suffix
        ``_{mode}`` for modes other than ``'values'``.
        """
        from .plotting import plot_pairs
        plot_dir = plot_dir or os.path.join(self._catalog_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        ell, cl = self.load_cls(return_dict=True)
        st = sacc.standard_types
        stat_of_type = {st.galaxy_density_cl: 'clustering', st.galaxy_shear_cl_ee: 'shear', st.galaxy_shearDensity_cl_e: 'shear_clustering'}
        what = sorted({stat_of_type[t] for t in cl if t in stat_of_type})
        need_errors = error_bars or 'diff_over_error' in modes
        _, cl_th, *dcl = self.predict_cls(what=what, error_bars=need_errors, **predict_kwargs)
        dcl = dcl[0] if need_errors else {}
        scales = self.lensing.reference_scales()
        ylabel = {0: r'$C_\ell$', 1: r'$\ell C_\ell$', 2: r'$\ell(\ell+1)C_\ell/2\pi$'}[ell_factor]
        for data_type, name, color in ((st.galaxy_shear_cl_ee, 'cl_shear_ee', 'orange'),
                                       (st.galaxy_shearDensity_cl_e, 'cl_shear_density', 'red'),
                                       (st.galaxy_density_cl, 'cl_density', 'green')):
            if data_type not in cl:
                continue
            for mode in modes:
                plot_pairs(ell, data=cl[data_type], theory=cl_th[data_type],
                           theory_errors=dcl.get(data_type) if (error_bars or mode == 'diff_over_error') else None, mode=mode,
                           nbins=self.ncats, y_factor=ell_factor, scales=scales, color=color, ylim=(-1, 1) if mode == 'ratio' else None,
                           ylabel=ylabel if mode == 'values' else None,
                           fname=os.path.join(plot_dir, name + ('' if mode == 'values' else f'_{mode}') + '.png'))


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
    augmentors : list, optional
        List of augmentors. Default is empty list.
    """

    def _generate_galaxies(self, catalog, ikappas=None, ishells=None):
        """Generate redshifts and positions for all selected tomographic bins.

        Parameters
        ----------
        catalog : list of dict
            Per-bin dictionaries updated in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        ishells : None
            Per-shell generation is not supported.
        """
        if ishells is not None:
            raise NotImplementedError(f"Per-shell generation is not supported by {self.__class__.__name__}.")

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
        Physical Gaussian smoothing scale in Mpc of the intrinsic-alignment fields,
        see :class:`CatalogSourceGenerator`. Galaxy positions are drawn from the
        unsmoothed density contrast (``get_field_hspmap``).
    interp_list : list of str, optional
        Base field names to interpolate at galaxy positions (e.g.
        ``['kappa', 'gamma1', 'gamma2', 't11', 't22', 't12']``).  Smoothed
        filenames of the intrinsic-alignment fields are resolved from ``smooth_Mpc``.  Default is
        ``['kappa', 'gamma1', 'gamma2']``.
    interp_pixels : bool, optional
        Use pixel interpolation. Default is True.
    augmentors : list, optional
        List of augmentors applied after map interpolation. Default is empty list.
    cls_b, cls_lmax_mask
        Binning of the catalog angular power spectra, see :class:`CatalogSourceGenerator`.
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
        augmentors: list | None = None,
        cls_b: Any = None,
        cls_lmax_mask: int | None = None,
    ) -> None:
        """Initialize clustered catalog generator and validate bias parameter."""
        super().__init__(
            lensing=lensing,
            tag=tag,
            nbar_input=nbar_input,
            smooth_Mpc=smooth_Mpc,
            interp_list=interp_list,
            interp_pixels=interp_pixels,
            augmentors=augmentors,
            cls_b=cls_b,
            cls_lmax_mask=cls_lmax_mask,
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
    augmentors : list, optional
        List of augmentors. Default is empty list.
    """

    def _generate_galaxies(self, catalog, ikappas=None, ishells=None):
        """Generate redshifts first, then positions conditioned on density maps.

        Parameters
        ----------
        catalog : list of dict
            Per-bin dictionaries updated in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        ishells : None
            Per-shell generation is not supported (the number of galaxies per bin
            is fixed over all shells).
        """
        if ishells is not None:
            raise NotImplementedError(f"Per-shell generation is not supported by {self.__class__.__name__}.")

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
        # unsmoothed density contrast, relative to the cosmic mean
        _, ipix_mask = self.snap_lensing.shell_builder.get_mask(return_ipix=True)
        density_masked = self.snap_lensing.shell_builder.get_field_hspmap(ishell).get_values_pix(ipix_mask)

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
    augmentors : list, optional
        List of augmentors. Default is empty list.
    """

    def _generate_galaxies(self, catalog, ikappas=None, ishells=None):
        """Generate a clustered catalog by shell-wise Poisson sampling.

        Parameters
        ----------
        catalog : list of dict
            Per-bin dictionaries updated in-place.
        ikappas : int or array_like of int or None, optional
            Tomographic-bin indices to generate.
        ishells : list of int or None, optional
            Shells whose galaxies are generated. If None, all shells.

        Notes
        -----
        This method:
        1) computes the expected mean counts per pixel per shell from each
           tracer n(z),
        2) samples positions conditioned on density maps,
        3) samples redshifts within each shell for each bin.
        """
        builder = self.lensing.shell_builder
        z_edges = builder.shell_edges_redshift

        # First compute mean density per shell for each tracer's n(z) (all bins needed for nbar)
        nbar_per_pix_per_bin = []
        for ikappa, t in enumerate(self.lensing.tracers_ccl):
            nz_int = np.array([quad(lambda z: t.get_dndz(z), z_edges[ishell], z_edges[ishell + 1]) for ishell in range(self.snap_lensing.nmaps)])
            nbar_per_pix_per_bin.append(nz_int / np.sum(nz_int) * self.nbar_per_pix[ikappa])
        self.nbar_per_pix_per_bin = np.array(nbar_per_pix_per_bin)

        # Initialize catalog entries only for the requested bins
        for ikappa in self.lensing.iter(ikappas):
            catalog[ikappa]['ra'] = []
            catalog[ikappa]['dec'] = []
            catalog[ikappa]['z'] = []
            catalog[ikappa]['idx_shell'] = []

        # Second generate positions, looping over shells
        ishells = list(range(self.nshells)) if ishells is None else ishells
        for ishell in self.lensing.shell_builder.tqdm(ishells, desc=f"[generate] generating ra, dec from density"):
            self._generate_pos(catalog, ishell, ikappas=ikappas)

        # Third sample redshifts within shells
        for ikappa in self.lensing.tqdm(ikappas, desc="[generate] generating z"):
            self._generate_z(catalog, ikappa, ishells)

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
        # unsmoothed density contrast, relative to the cosmic mean
        _, ipix_mask = self.snap_lensing.shell_builder.get_mask(return_ipix=True)
        density_masked = self.snap_lensing.shell_builder.get_field_hspmap(ishell).get_values_pix(ipix_mask)

        # Loop over tomographic bins
        for ikappa in self.lensing.tqdm(ikappas, leave=False):
            rng = np.random.default_rng(self.lensing.shell_builder.seed + ishell + ikappa * self.nshells)

            # no galaxy of this bin in this shell
            if self.nbar_per_pix_per_bin[ikappa][ishell] == 0.:
                catalog[ikappa]['ra'].append(np.zeros(0))
                catalog[ikappa]['dec'].append(np.zeros(0))
                continue

            # Poisson sampling
            p = np.clip(1. + self.bias[ikappa] * density_masked, 0., None)
            lam = p * self.nbar_per_pix_per_bin[ikappa][ishell]
            n_per_pix = rng.poisson(lam=lam)
            ipix = np.repeat(ipix_mask, n_per_pix)

            # Sampling ra, dec within pixel
            ra, dec = generate_radec_in_pix(self.nside, ipix, rng)

            catalog[ikappa]['ra'].append(ra)
            catalog[ikappa]['dec'].append(dec)

    def _generate_z(self, catalog, ikappa, ishells):
        """Sample galaxy redshifts within each shell for one tomographic bin.

        Parameters
        ----------
        catalog : list of dict
            Per-bin dictionaries updated in-place, with one entry of
            ``catalog[ikappa]['ra']`` per shell of ``ishells``.
        ikappa : int
            Tomographic-bin index.
        ishells : list of int
            Shells whose galaxies were generated.

        Notes
        -----
        For each shell, redshifts are sampled from the tracer n(z) restricted to
        that shell's redshift interval.
        """
        builder = self.lensing.shell_builder
        z_edges = builder.shell_edges_redshift

        t = self.lensing.tracers_ccl[ikappa]
        for i, ishell in enumerate(ishells):
            rng = np.random.default_rng(self.lensing.shell_builder.seed + ishell + ikappa * self.nshells)

            zz = np.linspace(z_edges[ishell], z_edges[ishell + 1], 100)
            nz = t.get_dndz(zz)
            ngal = len(catalog[ikappa]['ra'][i])

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
