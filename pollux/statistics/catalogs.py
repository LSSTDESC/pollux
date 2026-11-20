"""Two-point statistics of the catalogs (:class:`CatalogStatistics`, inherited by :class:`~pollux.catalog.CatalogGenerator`).

Cls with NaMaster (catalog fields), two-point functions with TreeCorr, their predictions with the same tracers, binning,
pixel window and smoothing, and plots against them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, Iterable, NamedTuple

import healpy as hp
import numpy as np
import pyccl
import pymaster as nmt
import sacc
import treecorr
from astropy.io import fits

from ..cosmo import Tracer, TracerSum
from ..utils.binning import binning_id
from ..utils.sky import generate_radec_in_pix
from .base import TwoPointStatistics

if TYPE_CHECKING:
    from ..catalog import CatalogGenerator

logger = logging.getLogger(__name__)


class Shapes(NamedTuple):
    """Shapes of the shear legs of the statistics of a catalog: what they measure and predict (:meth:`of`)."""

    name: str  # 'shear', 'ia', 'shear_plus_ia' or 'observed'
    terms: tuple[str, ...]  # fields summed: lensing shear ('shear') and intrinsic alignments ('ia')
    columns: tuple[tuple[str, ...], tuple[str, ...]]  # catalog columns summed into each component
    tracers: list  # tracer of each bin (a TracerSum for several fields)
    shape_noise: bool  # whether the shapes have the shape noise of the galaxies

    # fields summed by each kind of shapes ('observed' includes the intrinsic alignments if the shear augmentor does)
    TERMS = {'shear': ('shear',), 'ia': ('ia',), 'shear_plus_ia': ('shear', 'ia'), 'observed': ('shear', 'ia')}

    @property
    def column_key(self) -> tuple[str, str]:
        """The columns of each component joined by ``'+'``, as in file names and options.

        Returns
        -------
        tuple of str
            E.g. ``('gamma1+e1_ia_ta', 'gamma2+e2_ia_ta')``.
        """
        return tuple('+'.join(cols) for cols in self.columns)

    @classmethod
    def of(cls, catalog: CatalogStatistics, shapes: str = 'shear', shear_cols: tuple[str, str] = ('gamma1', 'gamma2')) -> Shapes:
        """Return what the shapes ``shapes`` of the shear legs measure and predict.

        Parameters
        ----------
        catalog : CatalogStatistics
            Catalog (its augmentors and tracers).
        shapes : {'shear', 'ia', 'shear_plus_ia', 'observed'}, optional
            Shapes: the lensing shear (``shear_cols``), the intrinsic alignments (``e1_ia_*``,
            ``e2_ia_*`` of the :class:`~pollux.ia.CatalogAugmentorIA`), their sum without shape noise,
            or the observed ellipticities (``e1_obs``, ``e2_obs`` of the
            :class:`~pollux.catalog.CatalogAugmentorShear`), with shape noise, predicted as the sum of the
            lensing shear and of the intrinsic alignments they include.
        shear_cols : tuple of str, optional
            Columns of the lensing shear.

        Returns
        -------
        Shapes
            Fields summed, catalog columns summed into each component, tracers (sums of tracers
            for several fields, :class:`~pollux.cosmo.TracerSum`) and shape noise.
        """
        if shapes not in cls.TERMS:
            raise ValueError(f"shapes must be one of {sorted(cls.TERMS)}, got {shapes!r}")
        if shapes == 'observed':
            if catalog.shear_augmentor is None or not catalog.shear_augmentor.compute_observed:
                raise ValueError("shapes='observed' needs a CatalogAugmentorShear with compute_observed=True")
            terms = ('shear', 'ia') if catalog.shear_augmentor.include_ia and catalog.ia_augmentor is not None else ('shear',)
            columns = (('e1_obs',), ('e2_obs',))
        else:
            terms = cls.TERMS[shapes]
            if 'ia' in terms and catalog.ia_augmentor is None:
                raise ValueError(f"shapes={shapes!r} needs an intrinsic-alignment augmentor")
            # e1_ia_*, e2_ia_* of each IA model (TA, TT)
            ia_cols = catalog.ia_augmentor.new_columns if 'ia' in terms else []
            columns = tuple(tuple(([shear_cols[k]] if 'shear' in terms else []) + ia_cols[k::2]) for k in (0, 1))

        # tracers of each bin: that of the field, or the sum of those of the fields
        tracers = {'shear': catalog.lensing_tracers, 'ia': catalog.ia_tracers if 'ia' in terms else None}
        if len(terms) == 1:
            bin_tracers = tracers[terms[0]]
        else:
            bin_tracers = [TracerSum(f'{shapes}_{i}', [tracers[term][i] for term in terms]) for i in range(catalog.ncats)]
        return cls(shapes, terms, columns, bin_tracers, shape_noise=shapes == 'observed')


class CatalogStatistics(TwoPointStatistics):
    """Two-point statistics of a catalog: measurements, predictions and plots.

    Mixin of :class:`~pollux.catalog.CatalogGenerator`, using its lensing builder (``lensing``), tomographic bins,
    galaxy densities (``nbar``), augmentors and directories.
    """

    # tracer roles (density or shear) of the (i, j) legs of each harmonic-space SACC data type
    _SACC_TYPE_LEGS_CLS = {
        sacc.standard_types.galaxy_density_cl: ('density', 'density'),
        sacc.standard_types.galaxy_shear_cl_ee: ('shear', 'shear'),
        sacc.standard_types.galaxy_shear_cl_bb: ('shear', 'shear'),
        sacc.standard_types.galaxy_shearDensity_cl_e: ('density', 'shear'),
        sacc.standard_types.galaxy_shearDensity_cl_b: ('density', 'shear'),
    }

    # index of each harmonic-space SACC data type among the coupled spectra of its statistic
    _SACC_TYPE_COMPONENT_CLS = {
        sacc.standard_types.galaxy_density_cl: 0,
        sacc.standard_types.galaxy_shear_cl_ee: 0,
        sacc.standard_types.galaxy_shear_cl_bb: 3,
        sacc.standard_types.galaxy_shearDensity_cl_e: 0,
        sacc.standard_types.galaxy_shearDensity_cl_b: 1,
    }

    # harmonic-space SACC data types of each statistic
    _STAT_TYPES_CLS = {
        'clustering': (sacc.standard_types.galaxy_density_cl,),
        'shear': (sacc.standard_types.galaxy_shear_cl_ee, sacc.standard_types.galaxy_shear_cl_bb),
        'shear_clustering': (sacc.standard_types.galaxy_shearDensity_cl_e, sacc.standard_types.galaxy_shearDensity_cl_b),
    }

    # tracer roles (density or shear) of the (i, j) legs of each real-space SACC data type
    _SACC_TYPE_LEGS_2PT = {
        sacc.standard_types.galaxy_density_xi: ('density', 'density'),
        sacc.standard_types.galaxy_shear_xi_plus: ('shear', 'shear'),
        sacc.standard_types.galaxy_shear_xi_minus: ('shear', 'shear'),
        sacc.standard_types.galaxy_shearDensity_xi_t: ('density', 'shear'),
        sacc.standard_types.galaxy_shearDensity_xi_x: ('density', 'shear'),
    }

    # component of the predicted correlation (xi+ or xi-) of each real-space SACC data type (None if a single one; zero if absent)
    _SACC_TYPE_COMPONENT_2PT = {
        sacc.standard_types.galaxy_density_xi: None,
        sacc.standard_types.galaxy_shear_xi_plus: 0,
        sacc.standard_types.galaxy_shear_xi_minus: 1,
        sacc.standard_types.galaxy_shearDensity_xi_t: None,
    }

    # real-space SACC data types of each statistic, in the order of the correlations of TreeCorr
    _STAT_TYPES_2PT = {
        'clustering': (sacc.standard_types.galaxy_density_xi,),
        'shear': (sacc.standard_types.galaxy_shear_xi_plus, sacc.standard_types.galaxy_shear_xi_minus),
        'shear_clustering': (sacc.standard_types.galaxy_shearDensity_xi_t, sacc.standard_types.galaxy_shearDensity_xi_x),
    }

    # shape fields of the shear legs of the predictions: lensing shear (G) and intrinsic alignments (I)

    def _path_stat(self, stat: str, shapes: str = 'shear') -> str:
        """Return the file of the statistics measured with the shapes ``shapes``, ``cat_{stat}[_{shapes}].fits``.

        Parameters
        ----------
        stat : {'cls', '2pt'}
            Statistics.
        shapes : str, optional
            Shapes of the shear legs. Default is ``'shear'``.

        Returns
        -------
        str
        """
        return self.io.path('catalogs', *self.io.catalog_stat_name(stat, shapes))

    # ==================================================================================================================
    # region Tracers

    @property
    def density_tracers(self) -> list[Tracer]:
        """Density tracers of the catalog: each bin's n(z) and linear bias.

        Returns
        -------
        list of pollux.cosmo.Tracer
            ``'density'`` tracers named ``density_{i}``, with the n(z) of the lensing
            tracers (the galaxies are both lenses and sources), and the smoothing of
            the density used to place them (``'positions'`` smoothing, at least the pixel size).
        """
        # bias placing the galaxies, none for uniform catalogs
        bias = getattr(self, 'bias', np.zeros(self.ncats))
        # one scale per shell, between the shells' comoving distances
        scales = self.smoothing['positions']
        smooth_Mpc = None if scales[0] is None else (tuple(self.lensing.shell_builder.shell_edges_comoving_distance), scales)
        return [
            Tracer(f"density_{i}", 'density', z=t.z, nz=t.nz, bias=bias[i], smooth_Mpc=smooth_Mpc, smooth_nside=self.lensing.nside)
            for i, t in enumerate(self.lensing_tracers)
        ]

    @property
    def lensing_tracers(self) -> list[Tracer]:
        """Lensing tracers of the catalog, those of the lensing builder (:attr:`~pollux.statistics.maps.MapStatistics.tracers`).

        Returns
        -------
        list of pollux.cosmo.Tracer
            Shear tracers (one per bin) with the n(z) of the catalog's galaxies.
        """
        return self.lensing.tracers

    @property
    def ia_tracers(self) -> list[Tracer] | None:
        """Intrinsic-alignment tracers of the catalog, from its :class:`~pollux.ia.CatalogAugmentorIA` augmentor.

        Returns
        -------
        list of pollux.cosmo.Tracer or None
            ``'ia'`` tracers named ``ia_{i}``, with the n(z) of the lensing tracers, the amplitudes
            of the augmentor (:meth:`~pollux.ia.CatalogAugmentorIA.tracer_amplitudes`, with the bias of
            each bin) and the smoothing of the tidal field; None without intrinsic alignments.
        """
        calculator = self.ia_augmentor
        if calculator is None:
            return None
        # bias placing the galaxies, none for uniform catalogs (as in the augmentor's density weighting)
        bias = getattr(self, 'bias', np.zeros(self.ncats))
        # smoothing of the tidal field, one scale per shell
        scales = self.smoothing['t11']
        smooth_Mpc = None if scales[0] is None else (tuple(self.lensing.shell_builder.shell_edges_comoving_distance), scales)
        tracers = []
        for i, t in enumerate(self.lensing_tracers):
            amplitudes = calculator.tracer_amplitudes(float(bias[i]))
            tracers.append(Tracer(f"ia_{i}", 'ia', z=t.z, nz=t.nz, smooth_Mpc=smooth_Mpc, smooth_nside=self.lensing.nside, ia=amplitudes))
        return tracers

    def _sacc_legs(self, legs: dict[str, tuple[str, str]]) -> dict[str, tuple[list, list]]:
        """Return the SACC tracers of the legs of each data type, from their roles (for :meth:`_save_sacc`).

        Parameters
        ----------
        legs : dict
            Roles (``'shear'``, ``'density'``) of the two legs of each data type (e.g. ``_SACC_TYPE_LEGS_CLS``).

        Returns
        -------
        dict
            ``{data_type: (tracers_i, tracers_j)}``, lists of SACC tracers by bin.
        """
        tracers = {'shear': [t.sacc() for t in self.lensing.tracers], 'density': [t.sacc() for t in self.density_tracers]}
        return {data_type: (tracers[role_i], tracers[role_j]) for data_type, (role_i, role_j) in legs.items()}

    def reference_scales(self, roles: tuple[str, str], shapes: str = 'shear') -> dict:
        """Return the reference scales of a statistic: of the lensing maps, and the smoothing of the fields of its legs.

        Parameters
        ----------
        roles : tuple of str
            Roles of the two legs (``'density'`` or ``'shear'``), as in ``_SACC_TYPE_LEGS_2PT``.
        shapes : str, optional
            Shapes of the shear legs (see :meth:`compute_2pt`): for ``'ia'``, their smoothing is
            that of the tidal field (the lensing shear is not smoothed).

        Returns
        -------
        dict
            :meth:`~pollux.statistics.maps.MapStatistics.reference_scales` of the lensing builder, with,
            if a leg is smoothed, ``theta_smooth`` (arcmin) and ``ell_smooth`` per bin pair: the
            smoothing of the density placing the galaxies of a density leg, or of the tidal field
            of an IA leg (``shapes='ia'``) (:meth:`~pollux.cosmo.Tracer.effective_smooth_Mpc`, at the mean redshift
            of the bin's n(z)), the geometric mean of the two legs if both are smoothed.
        """
        scales = self.lensing.reference_scales()

        # tracers carrying the smoothing of each leg (none for the lensing shear)
        leg_tracers = {'density': self.density_tracers, 'shear': self.ia_tracers if shapes == 'ia' else None}
        smoothed = [role for role in dict.fromkeys(roles) if leg_tracers[role] is not None and leg_tracers[role][0].smooth_Mpc is not None]
        if not smoothed:
            return scales

        # angle of the smoothing of each bin and smoothed leg, at the mean redshift of its galaxies
        theta = {}
        for role in smoothed:
            angles = []
            for t in leg_tracers[role]:
                chi = pyccl.comoving_radial_distance(self.lensing.cosmo, 1. / (1. + t.mean_redshift_nz()))
                angles.append(float(Tracer.effective_smooth_Mpc(t.smooth_Mpc, chi, t.smooth_nside)) / chi)
            theta[role] = np.array(angles)

        # per pair: geometric mean over the smoothed legs
        nbins = len(self.density_tracers)
        theta_pair = {}
        for i in range(nbins):
            for j in range(nbins):
                legs = [theta[role][k] for role, k in zip(roles, (i, j), strict=True) if role in theta]
                theta_pair[i, j] = float(np.exp(np.mean(np.log(legs))))
        rad2arcmin = 180. * 60. / np.pi
        scales['theta_smooth'] = {ij: rad2arcmin * t for ij, t in theta_pair.items()}
        scales['ell_smooth'] = {ij: np.pi / t for ij, t in theta_pair.items()}
        return scales

    # endregion

    # ==================================================================================================================
    # region Fields

    def compute_alms(
            self,
            ibins: int | Iterable[int] | None = None,
            what: str | list[str] = 'all',
            *,
            shapes: str = 'shear',
            shear_cols: tuple[str, str] = ('gamma1', 'gamma2'),
            subsample: float | None = None,
            remove_mean: bool = True,
            randoms: bool = False,
            randoms_per_galaxy: int = 5,
            randoms_shared: bool = True,
            seed: int | None = None,
            save: bool = True) -> tuple[dict[int, nmt.NmtFieldCatalogClustering], dict[int, nmt.NmtFieldCatalog]]:
        """Compute the NaMaster catalog fields of tomographic bins, used by :meth:`compute_cls`.

        Each field (its a_lm and those of its mask) is the costly part of the catalog Cls: it is
        computed once per bin and saved as an intermediate file, in a folder named by an identifier of what
        the field depends on (the multipoles of the field and mask, and the options that apply to its kind),
        read by later calls instead of loading the catalog again.

        Parameters
        ----------
        ibins : int or array_like of int or None, optional
            Tomographic bins. If None, all bins.
        what : {'shear', 'clustering', 'shear_clustering', 'all'} or list of str, optional
            Statistics to compute (see :meth:`compute_cls`), which set the fields needed: density
            fields for ``'clustering'`` and ``'shear_clustering'``, shear fields for ``'shear'`` and
            ``'shear_clustering'``.
        shapes, shear_cols, subsample, remove_mean, randoms, randoms_per_galaxy, randoms_shared, seed : optional
            See :meth:`compute_cls`.
        save : bool, optional
            If True (default), read the fields saved, and save those computed. If False, compute
            them without saving.

        Returns
        -------
        f_dens, f_shear : dict
            Density (``NmtFieldCatalogClustering``) and spin-2 shear
            (``NmtFieldCatalog``) fields keyed by bin, if needed by ``what``.
            Each field has a ``mask_amplitude`` attribute, the ratio of its mask
            to footprint of the maps (:meth:`pollux.footprint.Footprint.mask`) (its weighted number of galaxies or randoms per
            steradian of footprint, or 1 for density fields with the footprint mask).
            Fields read from disk have their a_lm, not their catalog values or mask map.
        """
        if isinstance(what, str):
            what = list(self._STAT_TYPES_CLS) if what == 'all' else [what]
        ibins = list(self.lensing.iter(ibins))
        seed = self.lensing.shell_builder.seed if seed is None else seed
        # density fields for clustering and galaxy-galaxy lensing, shear fields for cosmic shear and galaxy-galaxy lensing
        kinds = [kind for kind, stats in (('density', ('clustering', 'shear_clustering')), ('shear', ('shear', 'shear_clustering'))) if set(stats) & set(what)]
        # columns of the shear fields, summed ('+') for shapes of several fields
        if 'shear' in kinds:
            shear_cols = Shapes.of(self, shapes, shear_cols).column_key
        options = dict(
            shear_cols=tuple(shear_cols),
            subsample=subsample,
            remove_mean=remove_mean,
            randoms=randoms,
            randoms_per_galaxy=randoms_per_galaxy,
            randoms_shared=randoms_shared,
            seed=seed)

        # folder of the fields of each kind, named by an identifier of what they depend on
        folders = {}
        for kind in kinds:
            keys = ('subsample', 'seed') + (('randoms', 'randoms_per_galaxy', 'randoms_shared') if kind == 'density' else ('shear_cols', 'remove_mean'))
            params = dict(lmax=self.b.lmax, lmax_mask=self.lmax_mask, **{k: options[k] for k in keys})
            folders[kind] = 'cls_fields_' + hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]

        fields = {'density': {}, 'shear': {}}
        for ibin in self.lensing.tqdm(ibins, desc="[compute_alms]"):
            fnames = {kind: self.io.path('catalogs', kind, self.io.BIN_INDEX.format(ibin), ext='pkl', intermediate=True, folder=folders[kind]) for kind in kinds}
            missing = [kind for kind in kinds if not (save and os.path.exists(fnames[kind]))]
            if missing:
                built = self._compute_cls_fields(ibin, missing, **options)
            for kind in kinds:
                if kind in missing:
                    fields[kind][ibin] = built[kind]
                    if save:
                        # without its catalog values and mask map (the largest arrays), which pseudo-Cls and workspaces
                        # don't use once the a_lm of the field and of its mask are computed
                        field = built[kind]
                        if field.alm_mask is None:
                            field.alm_mask = field.get_mask_alms()  # not kept by "lite" fields
                        for attr in ('field', 'mask', 'maps'):
                            if hasattr(field, attr):
                                setattr(field, attr, None)
                        self.io.write_pickle(fnames[kind], field)
                else:
                    fields[kind][ibin] = self.io.read_pickle(fnames[kind])
                    logger.debug("Found and loaded field at %s.", fnames[kind])
        return fields['density'], fields['shear']

    def _compute_cls_fields(
            self, ibin: int, kinds: list[str], shear_cols: tuple[str, str], subsample: float | None, remove_mean: bool, randoms: bool, randoms_per_galaxy: int,
            randoms_shared: bool, seed: int) -> dict[str, nmt.NmtField]:
        """Build the catalog fields ``kinds`` of one bin (see :meth:`compute_alms`).

        Parameters
        ----------
        ibin : int
            Tomographic bin.
        kinds : list of str
            ``'density'`` and/or ``'shear'``.
        shear_cols, subsample, remove_mean, randoms, randoms_per_galaxy, randoms_shared, seed
            See :meth:`compute_cls`.

        Returns
        -------
        dict
            ``{kind: field}``, each field with its ``mask_amplitude``.
        """
        builder = self.lensing.shell_builder
        lmax, lmax_mask = self.b.lmax, self.lmax_mask
        # footprint pixels (to draw randoms) and area (to normalize the mask amplitudes)
        ipix_mask = builder.footprint.ipix(nest=True)
        area = ipix_mask.size * hp.nside2pixarea(builder.nside)
        # columns of the two shear components, each a sum of columns ('+', see compute_alms)
        cols1, cols2 = (c.split('+') for c in shear_cols)
        cat = self.load_catalogs(
            ibin, subsample=subsample, seed=seed, columns=['ra', 'dec', *cols1, *cols2], remove_mean=cols1 + cols2 if remove_mean else None)
        positions = [cat['ra'], cat['dec']]
        weights_gal = np.ones(cat['ra'].size)
        fields = {}

        if 'density' in kinds:
            # density: footprint mask, or randoms drawn over the footprint
            if not randoms:
                f = nmt.NmtFieldCatalogClustering(
                    positions,
                    weights_gal,
                    None,
                    None,
                    lmax=lmax,
                    lmax_mask=lmax_mask,
                    lonlat=True,
                    mask=self.lensing.footprint.mask(nest=False, dtype=np.float64))
                f.mask_amplitude = 1.
            else:
                # one set of randoms for all bins (sized from the first bin, the same whatever the bins computed),
                # or one per bin
                if randoms_shared:
                    rng = np.random.default_rng(seed)
                    nran = int(randoms_per_galaxy * self._ngal_used(0, subsample))
                else:
                    rng = np.random.default_rng(seed + ibin)
                    nran = int(randoms_per_galaxy * cat['ra'].size)
                ra_rand, dec_rand = generate_radec_in_pix(self.nside, rng.choice(ipix_mask, size=nran), rng)
                f = nmt.NmtFieldCatalogClustering(
                    positions, weights_gal, [ra_rand, dec_rand], np.ones(len(ra_rand)), lmax=lmax, lmax_mask=lmax_mask, lonlat=True)
                # randoms per steradian of footprint
                f.mask_amplitude = len(ra_rand) / area
            fields['density'] = f

        if 'shear' in kinds:
            # shear: spin-2 field of the galaxies, whose mask amplitude is their number per steradian
            shear = [sum(cat[col] for col in cols) for cols in (cols1, cols2)]
            f = nmt.NmtFieldCatalog(positions, weights_gal, shear, spin=2, lmax=lmax, lmax_mask=lmax_mask, lonlat=True)
            f.mask_amplitude = weights_gal.sum() / area
            fields['shear'] = f
        return fields

    def _ngal_used(self, ibin: int, subsample: float | None) -> int:
        """Return the number of galaxies of a bin used by the statistics (all, or the subsample read by :meth:`load_catalogs`).

        Parameters
        ----------
        ibin : int
            Tomographic bin.
        subsample : float or None
            Fraction of the galaxies read.

        Returns
        -------
        int
        """
        nrows = fits.getheader(self.path(ibin, ext='fits'), ext=1)['NAXIS2']
        return nrows if subsample is None else int(subsample * nrows)

    # endregion

    # ==================================================================================================================
    # region Workspaces

    @cached_property
    def b(self) -> nmt.NmtBin:
        """Return (and lazily create) the NaMaster binning of :meth:`compute_cls` and :meth:`predict_cls`.

        Returns
        -------
        pymaster.NmtBin
            ``cls_b``, or else the shell builder's binning (:attr:`MapBuilder.b`)
            without its pixel-window deconvolution: the pixel window is in the predictions
            instead (see :meth:`predict_cls`).
        """
        if self.cls_b is not None:
            return self.cls_b
        b_ref = self.lensing.shell_builder.b
        n_bands = b_ref.get_n_bands()
        ells = np.concatenate([b_ref.get_ell_list(i) for i in range(n_bands)])
        weights = np.concatenate([b_ref.get_weight_list(i) for i in range(n_bands)])
        bpws = np.concatenate([np.full(len(b_ref.get_ell_list(i)), i) for i in range(n_bands)])
        return nmt.NmtBin(ells=ells, bpws=bpws, weights=weights, f_ell=None, lmax=b_ref.lmax)

    @property
    def lmax_mask(self) -> int:
        """Maximum multipole of the masks of the catalog fields and of the footprint (``cls_lmax_mask``, at least that of the binning).

        Returns
        -------
        int
        """
        lmax_mask = self.cls_lmax_mask if self.cls_lmax_mask is not None else self.b.lmax
        if lmax_mask < self.b.lmax:
            raise ValueError(f"cls_lmax_mask={lmax_mask} is below the fields' lmax={self.b.lmax}")
        return lmax_mask

    def wsp(
            self,
            stat: str,
            pair: tuple[int, int] | None = None,
            fields: tuple | None = None,
            read_unbinned_MCM: bool = True,
            **field_kwargs: Any) -> nmt.NmtWorkspace:
        """Return (and lazily create) the workspace of one statistic and bin pair.

        The workspace stores the mode-coupling matrix and bandpower windows. It is
        either specific to a bin pair, computed from the masks of the pair's catalog
        fields (the weighted galaxy positions, and the footprint or randoms for
        density), or shared by all bin pairs (``pair=None``), computed from
        footprint of the maps (:meth:`pollux.footprint.Footprint.mask`). The shared workspace is exact in expectation for galaxies
        uniformly distributed over the footprint once pseudo-spectra are divided by
        the fields' ``mask_amplitude`` (see :meth:`compute_cls`); with clustered
        sources, it misses the coupling from the clustering of galaxy positions.

        Workspaces are cached on disk (:meth:`_path_wsp`), with their bandpower
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
            Field options for a bin pair (``shapes``, ``shear_cols``, ``randoms``, ``randoms_per_galaxy``,
            ``seed``, ``subsample``, ``randoms_shared``, ``remove_mean``), see :meth:`compute_cls`.

        Returns
        -------
        pymaster.NmtWorkspace
        """
        # shared workspace already in memory (with the unbinned coupling matrix, if needed)
        if pair is None and stat in self._wsp and (self._wsp[stat][1] or not read_unbinned_MCM):
            return self._wsp[stat][0]

        # otherwise read it from disk, or compute and save it (with its bandpower windows)
        wsp = self.io.cached(
            self._path_wsp(stat, pair, field_kwargs.get('randoms', False)), lambda: self._compute_wsp(stat, pair, fields, **field_kwargs),
            lambda fname: self.io.read_wsp(fname, read_unbinned_MCM=read_unbinned_MCM), self._write_wsp)

        # keep the shared workspace in memory
        if pair is None:
            self._wsp[stat] = (wsp, read_unbinned_MCM)
        return wsp

    def _path_wsp(self, stat: str, pair: tuple[int, int] | None = None, randoms: bool = False) -> str:
        """Return the path of a workspace (see :meth:`wsp`).

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
            ``{catalog_dir}/wsp_{stat}_lmax{lmax}_b{id}.wsp``, or
            ``{catalog_dir}/wsp_{stat}_{i}_{j}[_randoms]_lmax{lmax}_b{id}.wsp`` (with ``i >= j`` for
            the symmetric clustering and shear statistics), with ``id`` the identifier of the
            binning and of ``cls_lmax_mask`` (:func:`pollux.utils.binning_id`). Bandpower windows are saved
            next to it, with suffix ``_bpws.npy`` (see :meth:`bpws`).
        """
        if pair is None:
            name = stat
        else:
            i, j = pair
            if stat != 'shear_clustering':
                i, j = max(i, j), min(i, j)
            name = f'{stat}_{i}_{j}' + ('_randoms' if randoms and stat != 'shear' else '')
        # binning of the bandpowers and of the masks
        return self.io.path('catalogs', f'wsp_{name}_lmax{self.b.lmax}_b{binning_id(self.b, extra=(self.lmax_mask,))}', ext='wsp', intermediate=True)

    def _compute_wsp(self, stat: str, pair: tuple[int, int] | None, fields: tuple | None, **field_kwargs: Any) -> nmt.NmtWorkspace:
        """Compute the workspace of one statistic: shared (``pair`` None), from the footprint, or of a bin pair, from its fields (see :meth:`wsp`).

        Parameters
        ----------
        stat : str
            Statistics of the catalog (e.g. ``'shear'``).
        pair : tuple of int or None
            Bin pair.
        fields : tuple or None
            The two fields of the pair (default: computed).
        **field_kwargs
            Options of the fields of the pair (see :meth:`compute_cls`).

        Returns
        -------
        pymaster.NmtWorkspace
        """
        if pair is None:
            # shared: fields of the footprint mask, with the spins of the statistic
            spins = {'clustering': (0, 0), 'shear': (2, 2), 'shear_clustering': (0, 2)}[stat]
            f_spin = {
                spin: nmt.NmtField(self.lensing.footprint.mask(nest=False, dtype=np.float64), None, spin=spin, lmax=self.b.lmax, lmax_mask=self.lmax_mask)
                for spin in set(spins)
            }
            fields = (f_spin[spins[0]], f_spin[spins[1]])
        elif fields is None:
            # per pair: the catalog fields of the two bins
            i, j = pair
            f_dens, f_shear = self.compute_alms(sorted({i, j}), [stat], **field_kwargs)
            f1, f2 = {'clustering': (f_dens, f_dens), 'shear': (f_shear, f_shear), 'shear_clustering': (f_dens, f_shear)}[stat]
            fields = (f1[i], f2[j])
        wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(*fields, self.b)
        return wsp

    def bpws(self, stat: str, pair: tuple[int, int] | None = None, randoms: bool = False) -> np.ndarray:
        """Return the bandpower windows of a workspace (see :meth:`wsp`).

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
        fname = self._path_wsp(stat, pair, randoms)
        if pair is not None and not os.path.exists(fname) and not os.path.exists(self._path_bpws(fname)):
            raise FileNotFoundError(f"No workspace at {fname}, compute it with compute_cls or wsp.")
        return self._saved_bpws(fname, lambda: self.wsp(stat, pair, randoms=randoms))

    # endregion

    # ==================================================================================================================
    # region Power spectra

    def compute_cls(
            self,
            ibins: int | Iterable[int] | None = None,
            what: str | list[str] = 'all',
            *,
            auto_only: bool = False,
            pairs: list[tuple[int, int]] | None = None,
            shared_wsp: bool = False,
            randoms: bool = False,
            shapes: str = 'shear',
            shear_cols: tuple[str, str] = ('gamma1', 'gamma2'),
            subsample: float | None = None,
            remove_mean: bool = True,
            randoms_per_galaxy: int = 5,
            randoms_shared: bool = True,
            seed: int | None = None,
            save: bool = True,
            use_precomputed_alms: bool = True) -> tuple[np.ndarray, dict]:
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
        ibins : int or array_like of int or None, optional
            Tomographic bin indices to process. If None, uses all bins.
        what : {'shear', 'clustering', 'shear_clustering', 'all'} or list of str, optional
            Which spectra to compute:

            - 'clustering': galaxy density auto/cross-spectra (dd), symmetric i>=j pairs.
            - 'shear': shear E/B-mode auto/cross-spectra (ee, bb), symmetric i>=j pairs.
            - 'shear_clustering': density-shear cross-spectrum (de, db), all (i, j)
              ordered pairs since lens-bin i / source-bin j is not symmetric.
            - 'all': compute all three. Default.
        auto_only : bool, optional
            If True, restrict to same-bin pairs (i, i) only, skipping cross-bin
            spectra. Default is False (compute all cross-bin combinations).
        pairs : list of tuple of int or None, optional
            If given, only compute these ``(i, j)`` bin pairs, for every
            statistic in ``what`` (overrides ``ibins`` and ``auto_only``).
            Only the catalogs of the bins involved are loaded.
        shared_wsp : bool, optional
            If True, use one mode-coupling matrix per statistic (dd, shear-shear,
            density-shear) for all bin pairs, computed from the footprint, with
            pseudo-spectra divided by the fields' mask amplitudes (see :meth:`wsp`).
            This is exact in expectation for galaxies uniformly distributed over the
            footprint, but misses the coupling from the clustering of galaxy
            positions. If False (default), use one workspace per pair, computed from
            the masks of its fields (exact, slower). Workspaces are cached on disk
            and reused by later calls and by :meth:`predict_cls`; delete them if the
            catalogs or settings change.
        randoms : bool, optional
            If False (default), the mask of density fields is the footprint of the maps (:meth:`pollux.footprint.Footprint.mask`).
            If True, it is a random catalog (see ``randoms_per_galaxy`` and ``randoms_shared``).
        shapes : {'shear', 'ia', 'shear_plus_ia', 'observed'}, optional
            Shapes of the shear legs, as in :meth:`compute_2pt`: the lensing shear (``shear_cols``,
            default), the intrinsic alignments, their sum without shape noise, or the observed
            ellipticities, with shape noise. The fields sum the columns of the shapes
            (:meth:`Shapes.of`).
        shear_cols : tuple of str, optional
            Names of the two catalog columns of the lensing shear in the shapes,
            e.g. ``('gamma1', 'gamma2')`` (default) or ``('g1', 'g2')`` (reduced shear).
            Must be present in the loaded catalog (see the augmentors used at generation
            time, e.g. :class:`~pollux.catalog.CatalogAugmentorShear`, :class:`~pollux.ia.CatalogAugmentorIA`).
        subsample : float or None, optional
            If not None, randomly draw (without replacement, seeded by
            ``seed + ibin``) this fraction of galaxies from each loaded
            catalog before building fields, with ``0 < subsample < 1``. A
            random draw is used rather than a fixed stride, since galaxies
            are stored in a fixed (non-random) order and a stride would
            sample a specific, non-representative pattern. Default is None
            (no subsampling).
        remove_mean : bool, optional
            If True (default), subtract the mean of the shear columns over each bin
            before the measurement.
        randoms_per_galaxy : int, optional
            With ``randoms``, ratio of random points to galaxies for each bin's
            clustering random catalog. Default is 5.
        randoms_shared : bool, optional
            With ``randoms``, if True (default), draw a single random-point realization (sized
            from the galaxies of the first bin, whatever the bins computed) and reuse it for every bin's
            density field, rather than drawing independent randoms per bin.
            This matches the fact that all bins share the same survey
            footprint, and is cheaper since randoms are only drawn once. If
            False, each bin gets its own randoms, seeded by ``seed + ibin``.
        seed : int or None, optional
            Seed for the random catalogs' RNG. Default is the shell builder's seed.
        save : bool, optional
            If True (default), save the spectra to a SACC FITS file via
            :meth:`_save_cls`, at ``cat_cls[_{shapes}].fits`` of the catalogs.
        use_precomputed_alms : bool, optional
            If True (default), the fields of each bin come from :meth:`compute_alms`: read from their
            intermediate files, or computed and saved. If False, they are computed and not saved.

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
            what = list(self._STAT_TYPES_CLS) if what == 'all' else [what]

        # bins, and symmetric pairs (i >= j) for same-type correlations
        ibins, sym_pairs = self.lensing._pairs(ibins, auto_only, pairs)
        # bandpowers, and the catalog fields of each bin (built once, reused by all pairs)
        b = self.b
        ell = b.get_effective_ells()
        f_dens, f_shear = self.compute_alms(
            ibins,
            what,
            shapes=shapes,
            shear_cols=shear_cols,
            subsample=subsample,
            remove_mean=remove_mean,
            randoms=randoms,
            randoms_per_galaxy=randoms_per_galaxy,
            randoms_shared=randoms_shared,
            seed=seed,
            save=use_precomputed_alms)

        # decoupled spectra, keyed by SACC data type then bin pair
        cl = {}

        def decoupled(f1: nmt.NmtField, f2: nmt.NmtField, name: str, pair: tuple[int, int]) -> np.ndarray:
            # the shared workspace is that of the footprint, to which pseudo-spectra are rescaled (see wsp)
            pcl = nmt.compute_coupled_cell(f1, f2)
            if shared_wsp:
                return self.wsp(name, read_unbinned_MCM=False).decouple_cell(pcl / (f1.mask_amplitude * f2.mask_amplitude))
            return self.wsp(name, pair=pair, fields=(f1, f2), read_unbinned_MCM=False, randoms=randoms).decouple_cell(pcl)

        if 'clustering' in what:
            for i, j in self.lensing.tqdm(sym_pairs, desc="[compute_cls] clustering"):
                c = decoupled(f_dens[i], f_dens[j], 'clustering', (i, j))
                for data_type in self._STAT_TYPES_CLS['clustering']:
                    cl.setdefault(data_type, {})[i, j] = c[self._SACC_TYPE_COMPONENT_CLS[data_type]]

        if 'shear' in what:
            for i, j in self.lensing.tqdm(sym_pairs, desc="[compute_cls] shear"):
                c = decoupled(f_shear[i], f_shear[j], 'shear', (i, j))
                for data_type in self._STAT_TYPES_CLS['shear']:
                    cl.setdefault(data_type, {})[i, j] = c[self._SACC_TYPE_COMPONENT_CLS[data_type]]

        if 'shear_clustering' in what:
            # not symmetric under bin exchange (lens bin i, source bin j) -> all ordered pairs
            de_pairs = sym_pairs if (pairs is not None or auto_only) else [(i, j) for i in ibins for j in ibins]
            for i, j in self.lensing.tqdm(de_pairs, desc="[compute_cls] shear x clustering"):
                c = decoupled(f_dens[i], f_shear[j], 'shear_clustering', (i, j))
                for data_type in self._STAT_TYPES_CLS['shear_clustering']:
                    cl.setdefault(data_type, {})[i, j] = c[self._SACC_TYPE_COMPONENT_CLS[data_type]]

        if save:
            self._save_cls(ell, cl, shapes=shapes)

        return ell, cl

    def _save_cls(self, ell: np.ndarray, cl: dict, shapes: str = 'shear', metadata: dict | None = None) -> str:
        """Save catalog-based angular power spectra to a SACC FITS file (:meth:`_path_stat`).

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
        shapes : str, optional
            Shapes of the shear legs (see :meth:`compute_cls`), which set the file name.
        metadata : dict or None, optional
            Metadata of the SACC file (e.g. the identifier of the options of the measurement, see
            :mod:`pollux.pipeline.tasks`).

        Returns
        -------
        str
            Path to the saved SACC FITS file.
        """
        return self._save_sacc(self._path_stat('cls', shapes), cl, ell, self._sacc_legs(self._SACC_TYPE_LEGS_CLS), metadata=metadata)

    def load_cls(self, return_dict: bool = False, return_errors: bool = False, shapes: str = 'shear') -> sacc.Sacc | tuple:
        """Load catalog-based angular power spectra previously saved by :meth:`_save_cls`.

        Parameters
        ----------
        return_dict : bool, optional
            If True, return ``(ell, cl)``. Otherwise, return the SACC object.
        return_errors : bool, optional
            With ``return_dict``, also return the error bars of the saved covariance, with the same
            structure as ``cl`` (empty without covariance).
        shapes : str, optional
            Shapes of the shear legs of the measurement (see :meth:`compute_cls`).

        Returns
        -------
        sacc.Sacc or (ell, dict[, dict])
            If ``return_dict=False``, returns the loaded SACC object.
            If ``return_dict=True``, returns ``(ell, cl)`` where ``ell`` is the
            multipole array and ``cl`` is a dict keyed by SACC data type, each
            mapping to a dict keyed by ``(i, j)`` (same structure as returned
            by :meth:`compute_cls`). Reads only the data types and tracer
            pairs actually present in the file (:meth:`compute_cls` may not
            have computed all of them), with no dependency on ``self.lensing``
            beyond the file path.
        """
        res = self._load_sacc(self._path_stat('cls', shapes), return_dict=return_dict, return_errors=return_errors)
        if not return_dict:
            return res
        ell = next(e for pairs in res[0].values() for e in pairs.values())
        return (ell, *res[1:])

    def predict_cls(
            self,
            ibins: int | Iterable[int] | None = None,
            what: str | list[str] = 'all',
            *,
            auto_only: bool = False,
            pairs: list[tuple[int, int]] | None = None,
            binning: str = 'wsp',
            shared_wsp: bool = False,
            randoms: bool = False,
            shapes: str = 'shear',
            shear_cols: tuple[str, str] = ('gamma1', 'gamma2'),
            subsample: float | None = None,
            error_bars: bool = False,
            stacked: bool = False,
            ell: np.ndarray | None = None,
            pixel_window: bool = True) -> tuple:
        """Predict the catalog angular power spectra.

        Predictions use the lensing builder's :attr:`~pollux.statistics.maps.MapStatistics.theory`,
        i.e. the matter power spectrum of the catalog's shells (CCL's nonlinear P(k)
        for dark-matter-only shells, S19-corrected for baryonified shells). Predicted
        spectra are noise-free, like the measured ones (NaMaster subtracts the noise
        bias of catalog auto-spectra).

        Parameters
        ----------
        ibins : int or array_like of int or None, optional
            Tomographic bins. If None, all bins.
        what : {'shear', 'clustering', 'shear_clustering', 'all'} or list of str, optional
            Statistics to predict (see :meth:`compute_cls`).
        auto_only : bool, optional
            If True, only same-bin pairs ``(i, i)``.
        pairs : list of tuple of int or None, optional
            If given, only these ``(i, j)`` bin pairs, for every statistic in ``what``
            (overrides ``ibins`` and ``auto_only``).
        binning : {'wsp', 'average', 'unbinned'}, optional
            ``'wsp'`` (default): bandpowers binned with the bandpower windows of the
            workspaces of :meth:`compute_cls` (see ``shared_wsp`` and :meth:`bpws`). ``'average'``: bandpowers averaged over each bandpower.
            ``'unbinned'``: spectra at ``ell``.
        shared_wsp : bool, optional
            With ``binning='wsp'``, as in :meth:`compute_cls`: use the workspace shared
            by all pairs of each statistic if True, else that of each pair (default).
        randoms : bool, optional
            With per-pair workspaces, as in :meth:`compute_cls`.
        shapes : {'shear', 'ia', 'shear_plus_ia', 'observed'}, optional
            Shapes of the shear legs (see :meth:`compute_cls`): their tracers are sums of those of
            their terms (:meth:`Shapes.of`), e.g. GG + GI + IG + II for ``'shear_plus_ia'``, so
            that the error bars are those of the summed field.
        shear_cols, subsample : optional
            As in :meth:`compute_cls`, setting the noise (see :meth:`_noise_cls`).
        error_bars : bool, optional
            If True, also return Gaussian error bars (E-mode and density spectra),
            with ``C_ell + N_ell`` for auto-spectra (:meth:`_noise_cls`) and the
            lensing builder's sky fraction.
        stacked : bool, optional
            If True, also return the E-mode and density spectra of all statistics
            stacked into one vector (in the order of ``what``, then of the pairs), and
            with ``error_bars`` their joint covariance.
        ell : ndarray or None, optional
            Multipoles for ``binning='unbinned'``. Default is the effective bandpower multipoles.
        pixel_window : bool, optional
            If True (default), multiply the spectra by the pixel window of the maps (one per
            field): galaxies are placed in the pixels of the density maps, and the fields are
            interpolated from pixelized maps.

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
        if isinstance(what, str):
            what = list(self._STAT_TYPES_CLS) if what == 'all' else [what]
        # tracers of the legs of each statistic
        shape = Shapes.of(self, shapes, shear_cols)
        tracers = {'density': self.density_tracers, 'shear': shape.tracers}
        # bins, and symmetric pairs (i >= j) for same-type correlations
        ibins, sym_pairs = self.lensing._pairs(ibins, auto_only, pairs)

        # tracer pairs of all statistics, with the bandpower windows of their workspaces
        tracer_pairs, bpws = {}, {}
        for stat in what:
            role_i, role_j = self._SACC_TYPE_LEGS_CLS[self._STAT_TYPES_CLS[stat][0]]
            # shear x clustering is not symmetric under bin exchange (lens bin i, source bin j) -> all ordered pairs
            stat_pairs = sym_pairs
            if stat == 'shear_clustering' and pairs is None and not auto_only:
                stat_pairs = [(i, j) for i in ibins for j in ibins]
            windows = {}  # bandpower windows by workspace (bin pair, or None if shared)
            for i, j in stat_pairs:
                tracer_pairs[stat, i, j] = (tracers[role_i][i], tracers[role_j][j])
                if binning == 'wsp':
                    pair = None if shared_wsp else (i, j)
                    if pair not in windows:
                        windows[pair] = self.bpws(stat, pair, randoms)
                    bpws[stat, i, j] = windows[pair]

        # noise of each tracer, for the error bars
        noise = None
        if error_bars:
            noise = {t.name: self._noise_cls(t, shape.shape_noise, subsample) for t1, t2 in tracer_pairs.values() for t in (t1, t2)}

        # predict all statistics at once (for their joint covariance)
        res = self.lensing.theory.predict_cls(
            tracer_pairs,
            b=self.b,
            ell=ell,
            binning=binning,
            bpws=bpws,
            f_ell=hp.pixwin(self.lensing.nside, lmax=self.b.lmax) if pixel_window else None,
            noise=noise,
            error_bars=error_bars,
            stacked=stacked,
            fsky=self.lensing.footprint.fsky,
            all_components=True)

        # split the coupled components into SACC data types, as in compute_cls
        cl = {}
        for (stat, i, j), c in res[1].items():
            c = np.atleast_2d(c)  # components (a single one without bandpower windows)
            for data_type in self._STAT_TYPES_CLS[stat]:
                k = self._SACC_TYPE_COMPONENT_CLS[data_type]
                cl.setdefault(data_type, {})[i, j] = c[k] if k < len(c) else np.zeros_like(c[0])

        # outputs: ell, cl, error bars of the first data type of each statistic, then stacked spectra and covariance
        out = [res[0], cl]
        if error_bars:
            dcl = {}
            for (stat, i, j), d in res[2].items():
                dcl.setdefault(self._STAT_TYPES_CLS[stat][0], {})[i, j] = d
            out.append(dcl)
        out += list(res[2 + error_bars:])
        return tuple(out)

    def _noise_cls(self, tracer: Tracer | TracerSum, shape_noise: bool = False, subsample: float | None = None) -> float:
        """Return the white noise level of one catalog tracer.

        Parameters
        ----------
        tracer : pollux.cosmo.Tracer or pollux.cosmo.TracerSum
            Density (``density_{i}``) or shape tracer of bin ``i`` (see :meth:`Shapes.of`).
        shape_noise : bool, optional
            Whether the shapes have shape noise (``shape_noise`` of :meth:`Shapes.of`): only the observed
            ellipticities, not the lensing shear or the intrinsic alignments.
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
        if not shape_noise or self.shear_augmentor is None:
            return 0.
        return self.shear_augmentor.sigma_e**2 / n_sr

    def plot_cls(
            self,
            plot_dir: str | None = None,
            *,
            modes: tuple[str, ...] = ('values', 'ratio'),
            ell_factor: int = 1,
            logy: bool = False,
            error_bars: bool = False,
            shapes: str = 'shear',
            **predict_kwargs: Any) -> None:
        """Plot the saved angular power spectra against their prediction (:meth:`predict_cls`), one panel per bin pair.

        Parameters
        ----------
        plot_dir : str or None, optional
            Output directory. Default is ``{catalog_dir}/plots``.
        modes : tuple of str, optional
            Plots to make, see :func:`pollux.statistics.plotting.plot_pairs` (``'values'``, ``'ratio'``,
            ``'diff_over_error'``). Default is ``('values', 'ratio')``.
        ell_factor : {0, 1, 2}, optional
            In ``'values'`` mode, plot ``C_ell`` (0), ``ell C_ell`` (1, default) or
            ``ell (ell + 1) C_ell / 2 pi`` (2).
        logy : bool, optional
            If True, use a log y-scale in ``'values'`` mode.
        error_bars : bool, optional
            If True, draw Gaussian errors of the prediction as bands (always used by
            ``'diff_over_error'``).
        shapes : {'shear', 'ia', 'shear_plus_ia', 'observed'}, optional
            Shapes of the shear legs of the measurement (see :meth:`compute_cls`).
        **predict_kwargs
            Options of :meth:`predict_cls`, e.g. ``auto_only``, ``pairs``, or ``shared_wsp`` and
            ``randoms`` as used by :meth:`compute_cls`: the predicted pairs are plotted.

        Notes
        -----
        Saves ``cl_{shapes}_ee``, ``cl_{shapes}_x_density`` and, for the lensing shear,
        ``cl_density``, with suffix ``_{mode}`` for modes other than ``'values'``.
        """
        from .plotting import plot_pairs
        plot_dir = plot_dir or self.io.dir('catalogs', folder='plots')
        os.makedirs(plot_dir, exist_ok=True)

        # measured spectra, and their prediction for the statistics found
        ell, cl = self.load_cls(return_dict=True, shapes=shapes)
        what = [stat for stat, types in self._STAT_TYPES_CLS.items() if types[0] in cl]
        need_errors = error_bars or 'diff_over_error' in modes
        _, cl_th, *dcl = self.predict_cls(what=what, error_bars=need_errors, shapes=shapes, **predict_kwargs)
        dcl = dcl[0] if need_errors else {}

        # one figure per E-mode/density data type and mode
        st = sacc.standard_types
        for data_type, name, color in ((st.galaxy_shear_cl_ee, f'cl_{shapes}_ee', 'orange'), (st.galaxy_shearDensity_cl_e, f'cl_{shapes}_x_density', 'red'),
                                       (st.galaxy_density_cl, 'cl_density', 'green')):
            if data_type not in cl or (data_type == st.galaxy_density_cl and shapes != 'shear'):
                continue
            for mode in modes:
                plot_pairs(
                    ell,
                    data={ij: cl[data_type][ij] for ij in cl_th[data_type] if ij in cl[data_type]},
                    theory=cl_th[data_type],
                    theory_errors=dcl.get(data_type) if (error_bars or mode == 'diff_over_error') else None,
                    mode=mode,
                    auto_only=all(i == j for i, j in cl_th[data_type]),
                    ell_factor=ell_factor,
                    scales=self.reference_scales(self._SACC_TYPE_LEGS_CLS[data_type], shapes=shapes),
                    color=color,
                    logy=logy,
                    xlim=(self.b.get_ell_min(0), self.b.get_ell_max(self.b.get_n_bands() - 1)),
                    sharey=False,
                    fname=os.path.join(plot_dir, name + ('' if mode == 'values' else f'_{mode}') + '.png'))

    # endregion

    # ==================================================================================================================
    # region Correlations

    def compute_2pt(
            self,
            ibins: int | Iterable[int] | None = None,
            what: str | list[str] = 'all',
            *,
            auto_only: bool = False,
            pairs: list[tuple[int, int]] | None = None,
            theta_min: float = 1.,
            theta_max: float = 100.,
            ntheta: int = 20,
            sep_units: str = 'arcmin',
            bin_slop: float | None = None,
            shear_cols: tuple[str, str] = ('gamma1', 'gamma2'),
            shapes: str = 'shear',
            subsample: float | None = None,
            remove_mean: bool = True,
            randoms_per_galaxy: int = 5,
            randoms_shared: bool = True,
            seed: int | None = None,
            error_bars: bool = False,
            theta_per_pair: bool = False,
            save: bool = True) -> tuple:
        """Measure real-space two-point correlation functions directly from the
        generated catalog(s), including all cross-tomographic-bin combinations.

        Uses TreeCorr's angular correlation estimators (``NNCorrelation`` for
        clustering, ``GGCorrelation`` for shear, ``NGCorrelation`` for the
        shear-density cross-correlation), binned in angular separation between
        ``theta_min`` and ``theta_max``. This is the real-space analog of
        :meth:`compute_cls`; ``ibins``, ``what``, ``auto_only``,
        ``shear_cols``, ``randoms_per_galaxy``, ``seed`` and ``subsample`` have the
        same meaning there.

        Parameters
        ----------
        ibins : int or array_like of int or None, optional
            Tomographic bin indices to process. If None, uses all bins.
        what : {'shear', 'clustering', 'shear_clustering', 'all'} or list of str, optional
            Which correlations to compute:

            - 'clustering': galaxy density autocorrelation ``w(theta)``, symmetric i>=j pairs.
            - 'shear': shear autocorrelation ``xi_+/-(theta)``, symmetric i>=j pairs.
            - 'shear_clustering': density-shear cross-correlation ``gamma_t/x(theta)``,
              all (i, j) ordered pairs since lens-bin i / source-bin j is not symmetric.
            - 'all': compute all three. Default.
        auto_only : bool, optional
            If True, restrict to same-bin pairs (i, i) only, skipping cross-bin
            correlations. Default is False.
        pairs : list of tuple of int or None, optional
            If given, only compute these ``(i, j)`` bin pairs, for every
            statistic in ``what`` (overrides ``ibins`` and ``auto_only``).
            Only the catalogs of the bins involved are loaded.
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
        shear_cols : tuple of str, optional
            Names of the two catalog columns to use as the spin-2 shear field.
            See :meth:`compute_cls`.
        shapes : {'shear', 'ia', 'shear_plus_ia', 'observed'}, optional
            Shapes of the shear legs, as in :meth:`predict_2pt`: the lensing shear
            (``shear_cols``, default), the intrinsic alignments (II and gI), their
            sum without shape noise (GG + GI + IG + II and gG + gI), or the observed
            ellipticities, with shape noise. The same galaxies
            are used for all (same ``subsample`` and ``seed``), so that GI is exactly the
            difference of the first three.
        subsample : float or None, optional
            If not None, randomly draw (without replacement, seeded by
            ``seed + ibin``) this fraction of galaxies from each loaded
            catalog before building fields, with ``0 < subsample < 1``. A
            random draw is used rather than a fixed stride, since galaxies
            are stored in a fixed (non-random) order and a stride would
            sample a specific, non-representative pattern. Default is None
            (no subsampling).
        remove_mean : bool, optional
            If True (default), subtract the mean of the shear columns over each bin
            before the measurement.
        randoms_per_galaxy : int, optional
            Ratio of random points to galaxies for each bin's random catalog,
            used for the Landy-Szalay ``w(theta)`` estimator and for
            subtracting additive bias in ``gamma_t/x(theta)``. Default is 5.
        randoms_shared : bool, optional
            If True (default), draw a single random-point realization (sized
            from the galaxies of the first bin, whatever the bins computed) and reuse it for every bin's
            random catalog (used both for the ``w(theta)`` estimator and for
            the ``gamma_t/x(theta)`` random-lens subtraction), rather than
            drawing independent randoms per bin. This matches the fact that
            all bins share the same survey footprint, and is cheaper since
            randoms (and their TreeCorr catalog/tree) are only built once.
            If False, each bin gets its own randoms, seeded by ``seed + ibin``.
        seed : int or None, optional
            Seed for the random catalogs' RNG. Default is the shell builder's seed.
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
        save : bool, optional
            If True (default), save the correlation functions to a SACC FITS
            file via :meth:`_save_2pt`, at ``cat_2pt[_{shapes}].fits`` of the catalogs
            (``cat_2pt_{shapes}.fits`` for shapes other than ``'shear'``).

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
        dxi : dict
            Only with ``error_bars``: shot-noise error bars, keyed like ``xi``.
        """
        if isinstance(what, str):
            what = list(self._STAT_TYPES_2PT) if what == 'all' else [what]

        builder = self.lensing.shell_builder
        seed = builder.seed if seed is None else seed

        # footprint pixels, to draw randoms
        ipix_mask = builder.footprint.ipix(nest=True)

        # bins, and symmetric pairs (i >= j) for same-type correlations
        ibins, sym_pairs = self.lensing._pairs(ibins, auto_only, pairs)

        # catalogs needed by the statistics, and the angular binning
        need_density = 'clustering' in what or 'shear_clustering' in what
        need_shear = 'shear' in what or 'shear_clustering' in what
        cols1, cols2 = (list(cols) for cols in Shapes.of(self, shapes, shear_cols).columns)

        corr_kwargs = dict(min_sep=theta_min, max_sep=theta_max, nbins=ntheta, sep_units=sep_units, bin_slop=bin_slop)

        # Build each bin's catalog(s) once, reused across all pairs below.
        # A single galaxy catalog serves as both the density tracer (for NN)
        # and the shear tracer (for GG/NG): NNCorrelation only looks at
        # positions and simply ignores the g1/g2 fields when present.
        cat_gal, cat_rand = {}, {}
        shared_cat_rand = None
        for ibin in self.lensing.tqdm(ibins, desc="[compute_2pt] building catalogs"):
            columns = cols1 + cols2
            cat = self.load_catalogs(ibin, subsample=subsample, seed=seed, columns=['ra', 'dec', *columns], remove_mean=columns if remove_mean else None)

            # galaxies, with their shapes if needed (sum of the columns; TreeCorr's g2 convention is flipped)
            gal_kwargs = dict(ra=cat['ra'], dec=cat['dec'], ra_units='deg', dec_units='deg')
            if need_shear:
                gal_kwargs.update(g1=sum(cat[c] for c in cols1), g2=sum(cat[c] for c in cols2), flip_g2=True)
            cat_gal[ibin] = treecorr.Catalog(**gal_kwargs)

            if need_density:
                # randoms over the footprint (for w(theta) and the random-lens subtraction of gamma_t): one set for all bins, or one per bin
                if randoms_shared:
                    if shared_cat_rand is None:
                        # sized from the galaxies of the first bin, the same whatever the bins of the call
                        rng = np.random.default_rng(seed)
                        nran = int(randoms_per_galaxy * self._ngal_used(0, subsample))
                        ra_rand, dec_rand = generate_radec_in_pix(self.nside, rng.choice(ipix_mask, size=nran), rng)
                        shared_cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units='deg', dec_units='deg')
                    cat_rand[ibin] = shared_cat_rand
                else:
                    rng = np.random.default_rng(seed + ibin)
                    nran = int(randoms_per_galaxy * cat['ra'].size)
                    ra_rand, dec_rand = generate_radec_in_pix(self.nside, rng.choice(ipix_mask, size=nran), rng)
                    cat_rand[ibin] = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units='deg', dec_units='deg')

        # correlations, errors and mean separations, keyed by SACC data type then bin pair
        xi = {}
        dxi = {}
        theta_pairs = {}
        theta = None

        if 'clustering' in what:
            (xi_nn, dxi_nn, theta_nn), = ([d.setdefault(t, {}) for d in (xi, dxi, theta_pairs)] for t in self._STAT_TYPES_2PT['clustering'])
            for i, j in self.lensing.tqdm(sym_pairs, desc="[compute_2pt] clustering"):
                dd = treecorr.NNCorrelation(**corr_kwargs)
                rr = treecorr.NNCorrelation(**corr_kwargs)
                dr = treecorr.NNCorrelation(**corr_kwargs)
                # Landy-Szalay estimator
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
            (xi_p, dxi_p, theta_p), (xi_m, dxi_m, theta_m) = ([d.setdefault(t, {}) for d in (xi, dxi, theta_pairs)] for t in self._STAT_TYPES_2PT['shear'])
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
            de_pairs = sym_pairs if (pairs is not None or auto_only) else [(i, j) for i in ibins for j in ibins]
            (xi_t, dxi_t, theta_t), (xi_x, dxi_x,
                                     theta_x) = ([d.setdefault(t, {}) for d in (xi, dxi, theta_pairs)] for t in self._STAT_TYPES_2PT['shear_clustering'])
            for i, j in self.lensing.tqdm(de_pairs, desc="[compute_2pt] shear x clustering"):
                ng = treecorr.NGCorrelation(**corr_kwargs)
                rg = treecorr.NGCorrelation(**corr_kwargs)
                # lens bin i, source bin j, with the signal around randoms subtracted
                ng.process(cat_gal[i], cat_gal[j])
                rg.process(cat_rand[i], cat_gal[j])
                xi_t[i, j], xi_x[i, j], var_ng = ng.calculateXi(rg=rg)
                dxi_t[i, j] = dxi_x[i, j] = np.sqrt(var_ng)
                theta_t[i, j] = theta_x[i, j] = np.exp(ng.meanlogr)
                if theta is None:
                    theta = theta_t[i, j]

        if save:
            self._save_2pt(theta_pairs, xi, dxi=dxi, shapes=shapes)

        theta_out = theta_pairs if theta_per_pair else theta

        if error_bars:
            return theta_out, xi, dxi
        else:
            return theta_out, xi

    def _save_2pt(self, theta: dict, xi: dict, dxi: dict | None = None, shapes: str = 'shear', metadata: dict | None = None) -> str:
        """Save catalog-based real-space two-point functions to a SACC FITS file (:meth:`_path_stat`).

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
        shapes : str, optional
            Shapes of the shear legs (see :meth:`compute_2pt`), which set the file name.
        metadata : dict or None, optional
            Metadata of the SACC file (e.g. the identifier of the options of the measurement, see
            :mod:`pollux.pipeline.tasks`).

        Returns
        -------
        str
            Path to the saved SACC FITS file.
        """
        return self._save_sacc(
            self._path_stat('2pt', shapes), xi, theta, self._sacc_legs(self._SACC_TYPE_LEGS_2PT), coord='theta', errors=dxi, metadata=metadata)

    def load_2pt(self, return_dict: bool = False, return_errors: bool = False, shapes: str = 'shear') -> sacc.Sacc | tuple:
        """Load catalog-based two-point functions previously saved by :meth:`_save_2pt`.

        Parameters
        ----------
        return_dict : bool, optional
            If True, return ``(theta, xi)``. Otherwise, return the SACC object.
        return_errors : bool, optional
            With ``return_dict``, also return the error bars (square root of the
            diagonal of the saved covariance), with the same structure as ``xi``.
        shapes : str, optional
            Shapes of the shear legs of the measurement (see :meth:`compute_2pt`).

        Returns
        -------
        sacc.Sacc or (dict, dict[, dict])
            If ``return_dict=False``, returns the loaded SACC object.
            If ``return_dict=True``, returns ``(theta, xi)``, both dicts keyed
            by SACC data type then by ``(i, j)`` (same structure as
            ``theta_pairs``/``xi`` in :meth:`compute_2pt`; each pair's own
            mean separation is preserved, since it is catalog-dependent).
            Reads only the data types and tracer pairs actually present in
            the file (:meth:`compute_2pt` may not have computed all of them),
            with no dependency on ``self.lensing`` beyond the file path.
        """
        return self._load_sacc(self._path_stat('2pt', shapes), return_dict=return_dict, return_errors=return_errors)

    def predict_2pt(self, theta: dict, nell: int = 500, *, shapes: str = 'shear', pixel_window: bool = True) -> dict:
        """Predict the catalog two-point correlation functions.

        Predictions use the matter power spectrum of the catalog's shells (see :meth:`predict_cls`).

        Parameters
        ----------
        theta : dict
            Angular separations in arcmin keyed by SACC data type, then by ``(i, j)``
            (as returned by :meth:`load_2pt` or :meth:`compute_2pt` with ``theta_per_pair=True``).
        nell : int, optional
            Number of log-spaced multipoles up to the bandpower ``lmax`` used for the Hankel transforms.
        shapes : {'shear', 'ia', 'shear_plus_ia', 'observed'}, optional
            Shapes of the shear legs: ``'shear'`` (default), the lensing shear (columns ``gamma1``,
            ``gamma2``); ``'ia'``, the intrinsic alignments alone (II, gI; columns ``e1_ia_*``, ``e2_ia_*``);
            ``'shear_plus_ia'``, their sum without shape noise (GG + GI + IG + II for cosmic shear,
            gG + gI for galaxy-galaxy lensing); ``'observed'``, the observed ellipticities ``e1_obs``,
            ``e2_obs`` (with shape noise, of zero mean), predicted as ``'shear_plus_ia'``, or as
            ``'shear'`` without intrinsic alignments (``include_ia`` of :class:`~pollux.catalog.CatalogAugmentorShear`).
            See :attr:`ia_tracers`.
        pixel_window : bool, optional
            If True (default), multiply the spectra by the pixel window of the maps (one per field),
            as :meth:`predict_cls`: galaxies are placed in the pixels of the density maps, and the
            fields are interpolated from pixelized maps.

        Returns
        -------
        dict
            Same structure as ``theta``; ``galaxy_shearDensity_xi_x`` is zero.
        """
        # shear legs of the shapes (sums of tracers, e.g. GG + GI + IG + II for shear_plus_ia)
        tracers = {'density': self.density_tracers, 'shear': Shapes.of(self, shapes).tracers}
        f_ell = hp.pixwin(self.lensing.nside, lmax=self.b.lmax) if pixel_window else None
        xi = {}
        for data_type, type_pairs in theta.items():
            if data_type not in self._SACC_TYPE_COMPONENT_2PT:
                xi[data_type] = {ij: np.zeros(len(tt)) for ij, tt in type_pairs.items()}
                continue
            role_i, role_j = self._SACC_TYPE_LEGS_2PT[data_type]
            component = self._SACC_TYPE_COMPONENT_2PT[data_type]
            pairs = {(i, j): (tracers[role_i][i], tracers[role_j][j]) for i, j in type_pairs}
            res = self.lensing.theory.predict_2pt(pairs, type_pairs, self.b.lmax, nell=nell, f_ell=f_ell)
            xi[data_type] = {ij: r if component is None else r[component] for ij, r in res.items()}
        return xi

    def plot_2pt(
            self,
            plot_dir: str | None = None,
            *,
            modes: tuple[str, ...] = ('values', 'ratio'),
            reference: CatalogGenerator | None = None,
            xlim: tuple[float, float] | None = None,
            shapes: str = 'shear') -> None:
        """Plot the saved two-point functions against their prediction (:meth:`predict_2pt`), one panel per bin pair.

        Parameters
        ----------
        plot_dir : str or None, optional
            Output directory. Default is ``{catalog_dir}/plots``.
        modes : tuple of str, optional
            Plots to make, see :func:`pollux.statistics.plotting.plot_pairs` (``'values'``, ``'ratio'``,
            ``'diff_over_error'``, with the measurement errors). Default is ``('values', 'ratio')``.
            Values are multiplied by ``theta``.
        reference : CatalogGenerator or None, optional
            Catalog with saved two-point functions (e.g. dark-matter-only, for a
            baryonified catalog): adds the ``'ratio_to_reference'`` mode, the ratio of the
            two catalogs' functions against the ratio of their predictions.
        xlim : tuple of float or None, optional
            Limits of the x axis, e.g. the edges of the binning, ``(theta_min, theta_max)`` of
            :meth:`compute_2pt`. Default is the range of the measured ``theta``.
        shapes : {'shear', 'ia', 'shear_plus_ia', 'observed'}, optional
            Shapes of the shear legs of the measurement (see :meth:`compute_2pt`).

        Notes
        -----
        Saves, with suffix ``_{mode}`` for modes other than ``'values'``, auto-correlations named
        after their field and cross-correlations ``{field}_x_{field}``: ``xi_shear``,
        ``xi_shear_x_density`` and ``xi_density`` for the lensing shear; ``xi_ia`` (II) and
        ``xi_ia_x_density`` (gI) for the intrinsic alignments, and ``xi_shear_x_ia`` (GI: shear plus
        IA, minus shear, minus IA, when the three are measured); ``xi_shear_plus_ia`` and
        ``xi_shear_plus_ia_x_density`` for the sum of the shear and IA (without shape noise);
        ``xi_observed`` and ``xi_observed_x_density`` for the observed ellipticities (with shape noise).
        """
        from .plotting import plot_pairs
        plot_dir = plot_dir or self.io.dir('catalogs', folder='plots')
        os.makedirs(plot_dir, exist_ok=True)
        st = sacc.standard_types
        shear_types = (st.galaxy_shear_xi_plus, st.galaxy_shear_xi_minus)
        colors = {st.galaxy_shear_xi_plus: 'dodgerblue', st.galaxy_shear_xi_minus: 'orange', st.galaxy_shearDensity_xi_t: 'red', st.galaxy_density_xi: 'green'}

        # measurements and predictions (and those of the reference catalog)
        theta, xi, dxi = self.load_2pt(return_dict=True, return_errors=True, shapes=shapes)
        xi_th = self.predict_2pt(theta, shapes=shapes)
        xi_ref, xi_th_ref = {}, {}
        if reference is not None:
            _, xi_ref = reference.load_2pt(return_dict=True, shapes=shapes)
            xi_th_ref = reference.predict_2pt(theta, shapes=shapes)
            modes = (*modes, 'ratio_to_reference')

        # figures: name, y label, and the values of the data types overplotted (theta, data, theory,
        # data errors, reference data and theory)
        label, label_t = {
            'shear': ('', ''),
            'ia': (r'^{\rm II}', r'^{\rm gI}'),
            'shear_plus_ia': (r'^{\rm G+I}', r'^{\rm G+I}'),
            'observed': (r'^{\rm obs}', r'^{\rm obs}')
        }[shapes]
        prefix = f'xi_{shapes}'
        groups = [(prefix, rf'$\theta \xi{label}_\pm(\theta)$', shear_types),
                  (f'{prefix}_x_density', rf'$\theta \gamma{label_t}_{{\rm t}}(\theta)$', (st.galaxy_shearDensity_xi_t,))]
        if shapes == 'shear':
            groups.append(('xi_density', r'$\theta w(\theta)$', (st.galaxy_density_xi,)))
        figures = []
        for fig_name, ylabel, data_types in groups:
            figures.append((fig_name, ylabel, {t: (theta[t], xi[t], xi_th[t], dxi.get(t), xi_ref.get(t), xi_th_ref.get(t)) for t in data_types if t in xi}))

        # GI = (shear + IA) - shear - IA, from the same galaxies (data) and from the predictions at the same theta
        if shapes == 'ia' and all(os.path.exists(self._path_stat('2pt', s)) for s in ('shear', 'shear_plus_ia')):
            _, xi_tot = self.load_2pt(return_dict=True, shapes='shear_plus_ia')
            _, xi_g = self.load_2pt(return_dict=True, shapes='shear')
            th_tot, th_g = self.predict_2pt(theta, shapes='shear_plus_ia'), self.predict_2pt(theta, shapes='shear')
            gi = {}
            for t in shear_types:
                if t in xi and t in xi_tot and t in xi_g:
                    pairs = [ij for ij in xi[t] if ij in xi_tot[t] and ij in xi_g[t]]
                    data = {ij: xi_tot[t][ij] - xi_g[t][ij] - xi[t][ij] for ij in pairs}
                    pred = {ij: th_tot[t][ij] - th_g[t][ij] - xi_th[t][ij] for ij in pairs}
                    gi[t] = ({ij: theta[t][ij] for ij in pairs}, data, pred, None, None, None)
            figures.append(('xi_shear_x_ia', r'$\theta \xi^{\rm GI}_\pm(\theta)$', gi))

        # one figure per group of data types and mode, the data types of a group overplotted (and named in the legend)
        pm_labels = {st.galaxy_shear_xi_plus: r'$\xi_+$', st.galaxy_shear_xi_minus: r'$\xi_-$'}
        for fig_name, ylabel, data_types in figures:
            for mode in modes:
                if mode == 'ratio_to_reference' and fig_name == 'xi_shear_x_ia':
                    continue
                # the last data type of the group saves the figure
                figaxes = None
                for k, (data_type, (th, data, theory, errors, data_ref, theory_ref)) in enumerate(data_types.items()):
                    last = k == len(data_types) - 1
                    figaxes = plot_pairs(
                        th,
                        data=data,
                        theory=theory,
                        data_errors=errors,
                        data_ref=data_ref,
                        theory_ref=theory_ref,
                        mode=mode,
                        auto_only=all(i == j for i, j in data),
                        ell_factor=1,
                        scales=self.reference_scales(self._SACC_TYPE_LEGS_2PT[data_type], shapes='ia' if fig_name == 'xi_shear_x_ia' else shapes),
                        color=colors[data_type],
                        label=pm_labels.get(data_type) if len(data_types) > 1 else None,
                        xlabel=r'$\theta$ (arcmin)',
                        ylabel=ylabel if mode == 'values' else None,
                        xlim=xlim,
                        sharey=False,
                        figaxes=figaxes,
                        fname=os.path.join(plot_dir, fig_name + ('' if mode == 'values' else f'_{mode}') + '.png') if last else None)

    # endregion
