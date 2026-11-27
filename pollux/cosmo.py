"""Cosmology helpers: tracers and theory predictions, and reference cosmologies.

:class:`Tracer` describes an observable (a density, shear or convergence field
with its redshift distribution), from which CCL and SACC tracers are derived.
:class:`Theory` predicts angular power spectra and correlation functions between
tracers, for a given cosmology and matter power spectrum. Both are shared by the
map builders and the catalogs.

The module-level cosmologies are convenience definitions for examples/tests and
are not used by the main pipeline, which reads cosmology from simulation metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

import healpy as hp
import numpy as np
import pyccl
import sacc
from scipy.special import ndtr
from scipy.integrate import trapezoid
import warnings

if TYPE_CHECKING:
    import pymaster

# ======================================================================================================================
# region Cosmologies

cosmo_slics = pyccl.Cosmology(
    Omega_c=0.2905 - 0.0447,
    Omega_b=0.0447,
    h=0.6898,
    n_s=0.969,
    sigma8=0.826,
    Omega_k=0.,
)

cosmo_dc2 = pyccl.Cosmology(
    Omega_c=0.220,
    Omega_b=0.02258 / (0.71**2),
    h=0.71,
    n_s=0.963,
    sigma8=0.8,
    Omega_k=0.,
)

cosmo_hacc = pyccl.Cosmology(
    Omega_c=0.220,
    Omega_b=0.02258 / 0.71**2,
    h=0.71,
    n_s=0.963,
    sigma8=0.8,
    Omega_k=0.,
)
# endregion

# ======================================================================================================================
# region Tracers


class Tracer:
    """An observable field, from which CCL and SACC tracers are derived.

    Parameters
    ----------
    name : str
        Name, used for SACC (e.g. ``'kappa_0'``, ``'density_3'``).
    kind : {'density', 'shear', 'ia', 'convergence'}
        - ``'density'``: number counts (or matter density with ``bias=1``) with
          redshift distribution ``(z, nz)``, without RSD.
        - ``'shear'``: weak lensing of sources with redshift distribution ``(z, nz)``
          (spin 2; also used for the convergence of the same sources).
        - ``'ia'``: intrinsic alignments of galaxies with redshift distribution ``(z, nz)``
          (spin 2), with the amplitudes ``ia``. The shapes of sources are ``'shear'`` plus ``'ia'``.
        - ``'convergence'``: lensing convergence of a single source plane at ``z_source``.
    z, nz : ndarray or None, optional
        Redshift distribution (need not be normalized), for ``'density'``, ``'shear'`` and ``'ia'``.
    bias : float, optional
        Linear bias, constant in redshift, for ``'density'``. Default is 1.
    z_source : float or None, optional
        Source redshift, for ``'convergence'``.
    smooth_Mpc : float, tuple or None, optional
        Comoving scale of a Gaussian smoothing of the field (e.g. galaxies placed
        from a smoothed density), for ``'density'``: a scale, or ``(chi_edges, scales)``,
        one scale per shell of comoving distances ``chi_edges`` (Mpc, one more than
        ``scales``). Predictions multiply the matter
        power spectrum by ``exp(-k**2 s**2 / 2)`` per smoothed tracer, with ``s`` the
        effective scale (:meth:`effective_smooth_Mpc`), which is exact in Limber for a
        transverse smoothing. Default is None.
    smooth_nside : int or None, optional
        Nside of the smoothed maps, whose pixel size bounds the smoothing from below
        (see :meth:`effective_smooth_Mpc`). Default is None (no bound).
    ia : dict or None, optional
        Amplitudes of the intrinsic alignments, for ``'ia'``, in the conventions of
        :func:`pyccl.nl_pt.translate_IA_norm`: ``a1`` (tidal alignment, NLA), ``a2`` (tidal
        torquing) and ``a1delta`` (density weighting of the tidal alignment, e.g. ``b_TA a1``
        for galaxies counted with ``1 + b_TA delta``), constant in redshift. With ``a1`` only
        (default 1), the NLA model; else TATT (perturbation theory with FAST-PT, see
        :meth:`Theory.angular_cl`). The smoothing ``smooth_Mpc`` is that of the tidal field.
    """

    KINDS = ('density', 'shear', 'ia', 'convergence')

    def __init__(
            self,
            name: str,
            kind: str,
            z: np.ndarray | None = None,
            nz: np.ndarray | None = None,
            bias: float = 1.,
            z_source: float | None = None,
            smooth_Mpc: float | None = None,
            smooth_nside: int | None = None,
            ia: dict | None = None) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        if kind == 'convergence' and z_source is None:
            raise ValueError("z_source is required for kind='convergence'")
        if kind != 'convergence' and (z is None or nz is None):
            raise ValueError(f"z and nz are required for kind={kind!r}")
        if smooth_Mpc and kind not in ('density', 'ia'):
            raise ValueError("smooth_Mpc is only supported for kind='density' and 'ia'")
        if ia is not None and kind != 'ia':
            raise ValueError("ia is only supported for kind='ia'")
        ia = {'a1': 1., **(ia or {})} if kind == 'ia' else None
        if ia is not None and set(ia) - {'a1', 'a2', 'a1delta'}:
            raise ValueError(f"ia takes a1, a2 and a1delta, got {sorted(ia)}")
        self.name = name
        self.kind = kind
        self.z = None if z is None else np.asarray(z)
        self.nz = None if nz is None else np.asarray(nz)
        self.bias = bias
        self.z_source = z_source
        if isinstance(smooth_Mpc, tuple):
            smooth_Mpc = (tuple(map(float, smooth_Mpc[0])), tuple(map(float, smooth_Mpc[1])))
            if len(smooth_Mpc[0]) != len(smooth_Mpc[1]) + 1:
                raise ValueError("smooth_Mpc=(chi_edges, scales) needs one more edge than scales")
        self.smooth_Mpc = smooth_Mpc or None
        self.smooth_nside = smooth_nside
        self.ia = ia
        self._ccl = None

    def __getstate__(self) -> dict:
        """Return the state for pickling, without the cached CCL tracer (rebuilt on demand).

        Returns
        -------
        dict
        """
        state = self.__dict__.copy()
        state['_ccl'] = None
        return state

    @property
    def spin(self) -> int:
        """Spin of the field (2 for shear and intrinsic alignments, 0 otherwise)."""
        return 2 if self.kind in ('shear', 'ia') else 0

    @property
    def _needs_pt(self) -> bool:
        """Whether the predictions of the intrinsic alignments need perturbation theory: terms beyond NLA (``a2`` or ``a1delta``)."""
        return self.kind == 'ia' and any(self.ia.get(k) for k in ('a2', 'a1delta'))

    # ==================================================================================================================
    # region CCL tracers

    def ccl(self, cosmo: pyccl.Cosmology) -> pyccl.Tracer:
        """Return the CCL tracer for this field (cached for a given cosmology).

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology.

        Returns
        -------
        pyccl.Tracer
            ``NumberCountsTracer``, ``WeakLensingTracer`` (for ``'ia'``, without shear and with
            the NLA amplitude ``a1``) or ``CMBLensingTracer`` (the convergence of a single source
            plane, at any redshift).
        """
        if self._ccl is None or self._ccl[0] is not cosmo:
            if self.kind == 'density':
                tracer = pyccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(self.z, self.nz), bias=(self.z, np.full_like(self.z, self.bias)))
            elif self.kind == 'shear':
                tracer = pyccl.WeakLensingTracer(cosmo, dndz=(self.z, self.nz))
            elif self.kind == 'ia':
                tracer = pyccl.WeakLensingTracer(cosmo, dndz=(self.z, self.nz), has_shear=False, ia_bias=(self.z, np.full_like(self.z, self.ia['a1'])))
            else:
                tracer = pyccl.CMBLensingTracer(cosmo, z_source=self.z_source)
            self._ccl = (cosmo, tracer)
        return self._ccl[1]

    def _ccl_pt_kernel(self, cosmo: pyccl.Cosmology) -> pyccl.Tracer:
        """Return the CCL tracer of the radial kernel of the intrinsic alignments (n(z), unit amplitude), for spectra of perturbation theory.

        The amplitudes are in the power spectra of perturbation theory (:meth:`_pt_tracer`), not in the kernel.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology.

        Returns
        -------
        pyccl.Tracer
            ``WeakLensingTracer`` without shear, with ``ia_bias=1`` and ``use_A_ia=False``.
        """
        return pyccl.WeakLensingTracer(cosmo, dndz=(self.z, self.nz), has_shear=False, ia_bias=(self.z, np.ones_like(self.z)), use_A_ia=False)

    def _pt_tracer(self, cosmo: pyccl.Cosmology, nla_only: bool = False) -> pyccl.nl_pt.PTIntrinsicAlignmentTracer:
        """Return the perturbation-theory tracer of the intrinsic alignments, with the coefficients ``c1``, ``c2``, ``cdelta`` of :attr:`ia`.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology.
        nla_only : bool, optional
            If True, only the tidal alignment term ``c1``.

        Returns
        -------
        pyccl.nl_pt.PTIntrinsicAlignmentTracer
        """
        z = np.linspace(0., max(4., float(np.max(self.z))), 200)
        ia = {'a1': self.ia['a1']} if nla_only else self.ia
        c1, c1delta, c2 = pyccl.nl_pt.translate_IA_norm(cosmo, z=z, **ia)
        coeffs = {name: (z, c * np.ones_like(z)) for name, c in (('c1', c1), ('c2', c2), ('cdelta', c1delta)) if c is not None}
        return pyccl.nl_pt.PTIntrinsicAlignmentTracer(**coeffs)

    # endregion

    # ==================================================================================================================
    # region SACC

    def sacc(self) -> sacc.tracers.BaseTracer:
        """Return the SACC tracer for this field.

        Returns
        -------
        sacc.tracers.BaseTracer
            ``NZTracer`` for fields with a redshift distribution, ``MiscTracer`` otherwise.
        """
        if self.kind == 'convergence':
            return sacc.tracers.MiscTracer(self.name)
        return sacc.tracers.NZTracer(self.name, self.z, self.nz)

    # endregion

    # ==================================================================================================================
    # region Redshifts

    def mean_redshift_nz(self) -> float:
        """Mean redshift of the n(z) of the field (the source redshift for ``'convergence'``).

        For lensing fields this is the redshift of the sources, not of the structures
        they probe: see :meth:`mean_redshift_kernel`.

        Returns
        -------
        float
        """
        if self.kind == 'convergence':
            return self.z_source
        return trapezoid(self.nz * self.z, self.z) / trapezoid(self.nz, self.z)

    def mean_redshift_kernel(self, cosmo: pyccl.Cosmology, nchi: int = 1000) -> float:
        """Mean redshift weighted by the radial kernel of the field, i.e. where its signal comes from.

        Equal to :meth:`mean_redshift_nz` for density fields (whose kernel is their n(z)); for
        lensing fields, the mean redshift of the lensing efficiency, between the
        observer and the sources.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology.
        nchi : int, optional
            Number of comoving distances sampled up to the farthest source. Default is 1000.

        Returns
        -------
        float
        """
        zmax = self.z_source if self.kind == 'convergence' else np.max(self.z)
        chi = np.linspace(0., pyccl.comoving_radial_distance(cosmo, 1. / (1. + zmax)), nchi + 1)[1:]
        kernel = np.sum(self.ccl(cosmo).get_kernel(chi), axis=0)
        z = 1. / pyccl.scale_factor_of_chi(cosmo, chi) - 1.
        return trapezoid(z * kernel, chi) / trapezoid(kernel, chi)

    # endregion

    # ==================================================================================================================
    # region Smoothing

    @staticmethod
    def effective_smooth_Mpc(smooth_Mpc: float | tuple, chi: float | np.ndarray, nside: int | None) -> float | np.ndarray:
        r"""Return the comoving smoothing scale actually applied at a distance: at least the pixel size.

        Smoothing below the pixel size is not possible (the pixel window cannot be
        deconvolved beyond it, and maps would ring), so maps at ``nside`` are smoothed
        by the larger of ``smooth_Mpc`` and the pixel size ``hp.nside2resol(nside)``
        at the comoving distance ``chi``. Used by the maps and their predictions.

        Parameters
        ----------
        smooth_Mpc : float or tuple
            Requested comoving Gaussian smoothing scale in Mpc, or ``(chi_edges, scales)``,
            one scale per shell of comoving distances ``chi_edges`` (the nearest shell
            outside them). Shell maps are smoothed by one angle, that of the scale at
            their far edge, so within a shell the comoving scale is ``scale * chi / chi_far``.
        chi : float or ndarray
            Comoving distance in Mpc.
        nside : int or None
            Nside of the maps. If None, the requested scale is returned.

        Returns
        -------
        float or ndarray
            Comoving smoothing scale in Mpc, the shape of ``chi``.

        Notes
        -----
        .. math::

            s_{\rm eff}(\chi) = \max\left(s(\chi),\, \theta_{\rm pix}\, \chi\right), \qquad
            s(\chi) = s_i \frac{\chi}{\chi_{i+1}} \quad \text{for } \chi_i \le \chi < \chi_{i+1},

        with :math:`\theta_{\rm pix}` the pixel size (``hp.nside2resol``) and, for one scale per shell, :math:`s_i` that of
        shell :math:`i` (a constant :math:`s` otherwise).
        """
        chi = np.asarray(chi, dtype=float)
        if isinstance(smooth_Mpc, tuple):
            # angle of each shell's scale at its far edge, as the maps (MapBuilder._smooth_sigma), at the distance chi
            chi_edges, scales = np.asarray(smooth_Mpc[0]), np.asarray(smooth_Mpc[1])
            ishell = np.clip(np.searchsorted(chi_edges, chi, side='right') - 1, 0, len(scales) - 1)
            smooth_Mpc = scales[ishell] * chi / chi_edges[ishell + 1]
        if nside is None:
            return smooth_Mpc + 0. * chi
        return np.maximum(smooth_Mpc, hp.nside2resol(nside) * chi)

    # endregion


class TracerSum:
    """Sum of tracers of one field, e.g. the lensing shear and intrinsic alignments of a bin (G + I).

    Predictions are bilinear in the tracers: :meth:`Theory.angular_cl` of sums is the sum of the spectra of
    their terms (e.g. GG + GI + IG + II), so that bandpowers, correlation functions and Gaussian errors
    are those of the summed field.

    Parameters
    ----------
    name : str
        Name of the sum (e.g. 'shear_plus_ia_3'), ending with the bin index.
    tracers : sequence of Tracer
        Terms, of the same spin.
    """

    def __init__(self, name: str, tracers: Iterable[Tracer]) -> None:
        self.name = name
        self.tracers = tuple(tracers)
        if len({t.spin for t in self.tracers}) != 1:
            raise ValueError(f"tracers of a sum must have the same spin, got {[t.kind for t in self.tracers]}")
        # kind of the correlation types (see Theory.predict_2pt): that of the lensing shear if a term
        self.kind = 'shear' if any(t.kind == 'shear' for t in self.tracers) else self.tracers[0].kind

    @property
    def spin(self) -> int:
        """Spin of the terms."""
        return self.tracers[0].spin


# endregion

# ======================================================================================================================
# region Theory


class Theory:
    """Theory predictions between :class:`Tracer` objects, using CCL.

    Parameters
    ----------
    cosmo : pyccl.Cosmology
        Cosmology.
    p_of_k_a : pyccl.Pk2D or None, optional
        Matter power spectrum. Default is CCL's nonlinear one.
    """

    # CCL correlation type of each pair of tracer kinds (and component, for spin-2 pairs); intrinsic
    # alignments are spin-2 fields as shear
    _CORRELATION_TYPES = {
        ('density', 'density'): 'NN',
        ('density', 'shear'): 'NG',
        ('density', 'ia'): 'NG',
        ('shear', 'shear'): ('GG+', 'GG-'),
        ('shear', 'ia'): ('GG+', 'GG-'),
        ('ia', 'shear'): ('GG+', 'GG-'),
        ('ia', 'ia'): ('GG+', 'GG-'),
        ('convergence', 'convergence'): 'NN',
    }

    # smoothing scales (Mpc) considered by lognormal_smoothing: round values, for file names
    SMOOTHING_SCALES = (0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1., 1.2, 1.5, 2., 2.5, 3., 4., 5., 6., 8., 10., 12., 15., 20., 25., 30., 40., 50.)

    # non-Limber correction (see :meth:`_non_limber_correction`): computed up to this multipole, falling as 1 / ell^2 beyond
    NON_LIMBER_LMAX = 1000

    def __init__(self, cosmo: pyccl.Cosmology, p_of_k_a: pyccl.Pk2D | None = None) -> None:
        self.cosmo = cosmo
        self.p_of_k_a = p_of_k_a
        self._non_limber = {}  # non-Limber corrections of pairs of density tracers, keyed by their names
        self._smoothed_pk = {}  # power spectra with a Gaussian smoothing, keyed by the spectrum and the tracers' smoothing
        self._pt_calculator = None
        self._pt_correction_pk = {}  # corrections of perturbation theory to NLA, keyed by the tracers' amplitudes

    # ==================================================================================================================
    # region Power spectra

    def angular_cl(self, t1: Tracer | TracerSum, t2: Tracer | TracerSum, ell: np.ndarray, *, bb: bool = False, non_limber: bool = False) -> np.ndarray:
        """Return the angular power spectrum of two tracers.

        Parameters
        ----------
        t1, t2 : Tracer or TracerSum
            Tracers, or sums of tracers (the sum of the spectra of their terms).
        ell : array_like
            Multipoles.
        bb : bool, optional
            Return the B-mode spectrum instead: only intrinsic alignments beyond NLA (two ``'ia'``
            tracers, one with ``a2`` or ``a1delta``) have B modes, zero otherwise. Default is False.
        non_limber : bool, optional
            Add the non-Limber correction of two unsmoothed ``'density'`` tracers
            (:meth:`_non_limber_correction`), significant for thin shells below ell ~ 1000 but slow
            (minutes per pair). Other pairs stay in the Limber approximation. Default is False.

        Returns
        -------
        ndarray
            C_ell (E mode for spin-2 fields, or B with ``bb``; no pixel window), with the Gaussian
            smoothing of the tracers (see :class:`Tracer`), at their effective scale at each redshift.

        Notes
        -----
        Intrinsic alignments (``'ia'`` tracers) follow the NLA model with the matter power
        spectrum of the predictions; for TATT amplitudes (``a2`` or ``a1delta``), the terms
        beyond NLA of CCL's perturbation theory are added (:meth:`_pk_pt_correction`, with the
        same smoothing), which also give the B modes. The density weighting of the tidal
        torquing term is not modelled.
        """
        # sums of tracers: sum of the spectra of their terms
        if isinstance(t1, TracerSum) or isinstance(t2, TracerSum):
            terms1 = t1.tracers if isinstance(t1, TracerSum) else (t1,)
            terms2 = t2.tracers if isinstance(t2, TracerSum) else (t2,)
            return sum(self.angular_cl(a, b, ell, bb=bb, non_limber=non_limber) for a in terms1 for b in terms2)

        # smoothing of the tracers, independent of their order (key of the smoothed spectra)
        smoothing = tuple(sorted(((t.smooth_Mpc, t.smooth_nside or 0) for t in (t1, t2) if t.smooth_Mpc), key=repr))

        # intrinsic alignments beyond NLA (not with the convergence), with the IA leg first
        if t1.kind != 'ia' and t2.kind == 'ia':
            t1, t2 = t2, t1
        needs_pt = t1.kind == 'ia' and (t1._needs_pt or t2._needs_pt) and t2.kind != 'convergence'

        # B modes: only from the PT terms of two IA tracers
        if bb:
            if not (needs_pt and t2.kind == 'ia'):
                return np.zeros(len(np.atleast_1d(ell)))
            pk = self._pk_pt_correction(t1, t2, smoothing, bb=True)
            return self._limber(t1._ccl_pt_kernel(self.cosmo), t2._ccl_pt_kernel(self.cosmo), ell, pk)

        # E modes (or scalars): NLA (or matter) spectrum, plus the PT terms
        cl = self._limber(t1.ccl(self.cosmo), t2.ccl(self.cosmo), ell, self._apply_pk_smoothing(self.p_of_k_a, smoothing))
        if needs_pt:
            pk = self._pk_pt_correction(t1, t2, smoothing)
            kernel2 = t2._ccl_pt_kernel(self.cosmo) if t2.kind == 'ia' else t2.ccl(self.cosmo)
            cl = cl + self._limber(t1._ccl_pt_kernel(self.cosmo), kernel2, ell, pk)
        if non_limber and t1.kind == t2.kind == 'density' and not smoothing:
            cl = cl + self._non_limber_correction(t1, t2, ell)
        return cl

    def _pk_pt_correction(self, t1: Tracer, t2: Tracer, smoothing: tuple, *, bb: bool = False) -> pyccl.Pk2D:
        """Return the correction of perturbation theory to the NLA power spectrum, between intrinsic alignments and IA or matter.

        The difference between the TATT and NLA spectra of CCL's perturbation theory
        (:class:`pyccl.nl_pt.EulerianPTCalculator`, FAST-PT): the NLA term itself is computed
        with the matter power spectrum of the predictions (nonlinear, possibly with baryons).
        With ``bb``, the B-mode spectrum of two ``'ia'`` tracers instead (NLA has none, so the
        whole PT spectrum). Unsmoothed spectra are cached by the amplitudes of the tracers.

        Parameters
        ----------
        t1 : Tracer
            ``'ia'`` tracer.
        t2 : Tracer
            ``'ia'`` tracer, or another tracer for the matter (its bias is in its CCL kernel).
        smoothing : tuple
            Smoothing of the tracers (see :meth:`_apply_pk_smoothing`).
        bb : bool, optional
            Return the B-mode spectrum (two ``'ia'`` tracers only). Default is False (E mode).

        Returns
        -------
        pyccl.Pk2D
            Smoothed spectrum, to be integrated with the unit-amplitude kernels of the tracers
            (:meth:`Tracer._ccl_pt_kernel` for ``'ia'``, the CCL tracer otherwise).
        """
        if self._pt_calculator is None:
            self._pt_calculator = pyccl.nl_pt.EulerianPTCalculator(with_IA=True, with_NC=False, with_matter_1loop=False, cosmo=self.cosmo)
        # cache key: amplitudes and redshift range of the IA tracers (their PT tracers), and the mode
        ias = tuple(tuple(sorted(t.ia.items())) if t.kind == 'ia' else None for t in (t1, t2))
        key = (ias, tuple(float(np.max(t.z)) if t.kind == 'ia' else None for t in (t1, t2)), bb)
        if key not in self._pt_correction_pk:
            # second leg: intrinsic alignments, or the matter
            if t2.kind == 'ia':
                full2, nla2 = t2._pt_tracer(self.cosmo), t2._pt_tracer(self.cosmo, nla_only=True)
            else:
                full2 = nla2 = pyccl.nl_pt.PTMatterTracer()
            full = self._pt_calculator.get_biased_pk2d(t1._pt_tracer(self.cosmo), tracer2=full2, return_ia_bb=bb)
            if bb:
                # no B modes in NLA: the whole PT spectrum
                self._pt_correction_pk[key] = full
            else:
                # NLA is computed with the predictions' own matter spectrum: only the terms beyond it
                nla = self._pt_calculator.get_biased_pk2d(t1._pt_tracer(self.cosmo, nla_only=True), tracer2=nla2)
                self._pt_correction_pk[key] = full - nla
        return self._apply_pk_smoothing(self._pt_correction_pk[key], smoothing, name=('ia', key))

    def _apply_pk_smoothing(self, pk: pyccl.Pk2D | None, smoothing: tuple, name: str | tuple = 'matter') -> pyccl.Pk2D | None:
        r"""Return a power spectrum multiplied by the Gaussian smoothing of the tracers ``smoothing`` (cached by ``name``).

        Same grid as ``pk`` (CCL's nonlinear matter power spectrum if None), with the smoothing of
        each tracer at the comoving distance of each scale factor. Positive spectra are
        filtered in log (no underflow); others (e.g. perturbation-theory corrections, which
        change sign) linearly, extrapolated as constants beyond the grid's largest k (CCL's
        default quadratic extrapolation in ``log k`` would grow there instead of being smoothed away).

        Parameters
        ----------
        pk : pyccl.Pk2D or None
            Power spectrum to smooth (CCL's nonlinear matter power spectrum if None).
        smoothing : tuple
            Smoothing of the tracers, ``(smooth_Mpc, smooth_nside)`` of each smoothed one (see
            :meth:`angular_cl`): ``pk`` itself if empty.
        name : str or tuple, optional
            Name of ``pk`` for the cache (the spectrum must be the same for a given name).

        Returns
        -------
        pyccl.Pk2D or None
            Smoothed power spectrum, multiplied by ``exp(-k^2 sum(sigma^2) / 2)`` (the product of
            the Gaussian filters of the two tracers, of variances ``sigma^2``, one per smoothed tracer).

        Notes
        -----
        .. math::

            \tilde P(k, z) = P(k, z) \exp\left(-\frac{k^2}{2} \sum_t s_{{\rm eff}, t}^2(\chi(z))\right),

        summed over the smoothed tracers :math:`t` of the pair (:meth:`Tracer.effective_smooth_Mpc`): the product of
        their Gaussian filters, exact in the Limber approximation for a transverse smoothing.
        """
        if not smoothing:
            return pk
        if (name, smoothing) not in self._smoothed_pk:
            a_arr, lk_arr, pk_arr = (pk if pk is not None else self.cosmo.get_nonlin_power()).get_spline_arrays()
            # comoving variances of the tracers' Gaussians at each scale factor, and -log of their product at each k
            chi = pyccl.comoving_radial_distance(self.cosmo, a_arr)
            sigma2 = sum(Tracer.effective_smooth_Mpc(s, chi, nside or None)**2 for s, nside in smoothing)
            filt = np.exp(2. * lk_arr)[None, :] * sigma2[:, None] / 2.
            if np.all(pk_arr > 0.):
                smoothed = pyccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=np.log(pk_arr) - filt, is_logp=True)
            else:
                # spectra changing sign: linear, and constant beyond the grid
                smoothed = pyccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr * np.exp(-filt), is_logp=False, extrap_order_hik=0)
            self._smoothed_pk[name, smoothing] = smoothed
        return self._smoothed_pk[name, smoothing]

    def _limber(self, kernel1: pyccl.Tracer, kernel2: pyccl.Tracer, ell: np.ndarray, pk: pyccl.Pk2D | None) -> np.ndarray:
        """Return CCL's angular power spectrum, with the spline integration if the default one fails.

        Parameters
        ----------
        kernel1, kernel2 : pyccl.Tracer
            CCL tracers (radial kernels) of the two fields.
        ell : array_like
            Multipoles.
        pk : pyccl.Pk2D or None
            Power spectrum (CCL's nonlinear matter power spectrum if None).

        Returns
        -------
        ndarray
            C_ell (Limber approximation).
        """
        try:
            return pyccl.angular_cl(self.cosmo, kernel1, kernel2, ell, p_of_k_a=pk)
        except pyccl.CCLError:
            # e.g. kernels or spectra too sharp for the default (QAG) integration
            return pyccl.angular_cl(self.cosmo, kernel1, kernel2, ell, p_of_k_a=pk, limber_integration_method='spline')

    def _non_limber_correction(self, t1: Tracer, t2: Tracer, ell: np.ndarray) -> np.ndarray:
        r"""Return the non-Limber correction of the angular power spectrum of two ``'density'`` tracers (cached by their names).

        The difference between the exact spectrum and its Limber approximation, both with the matter power
        spectrum of the predictions at one effective redshift (the mean of the kernels), which suits narrow kernels
        (e.g. shells of matter):

        .. math::

            \Delta C_\ell = \frac{2}{\pi} \int dk\, k^2 P(k) I_1(k) I_2(k)
            - \int d\chi \frac{W_1 W_2}{\chi^2} P\left(\frac{\ell + 1/2}{\chi}\right),
            \qquad I_i(k) = \int d\chi\, W_i(\chi)\, j_\ell(k \chi),

        with the radial kernels :math:`W_i` of the CCL tracers. It is computed at about 70 multipoles up to
        :attr:`NON_LIMBER_LMAX`, in parallel threads, and interpolated; beyond, the relative correction falls as
        :math:`1 / \ell^2`. Zero below ell = 2.

        Parameters
        ----------
        t1, t2 : Tracer
            ``'density'`` tracers.
        ell : array_like
            Multipoles.

        Returns
        -------
        ndarray
            Correction, to add to the Limber C_ell of :meth:`angular_cl`.
        """
        key = tuple(sorted((t1.name, t2.name)))
        if key not in self._non_limber:
            self._non_limber[key] = self._non_limber_table(t1, t2)
        ells, delta, limber_lmax = self._non_limber[key]
        ell = np.asarray(ell, dtype=float)
        out = np.interp(ell, ells, delta, left=0., right=0.)
        hi = ell > ells[-1]
        if np.any(hi):
            # relative correction falling as 1 / ell^2, on the Limber spectrum at the effective redshift (computed at
            # 50 multipoles and interpolated in log)
            grid = np.geomspace(ells[-1], ell[hi].max(), 50)
            shape = np.exp(np.interp(np.log(ell[hi]), np.log(grid), np.log(np.abs(limber_lmax(grid)) + 1e-300)))
            out[hi] = delta[-1] * shape / abs(limber_lmax(ells[-1:])[0]) * (ells[-1] / ell[hi])**2
        return out

    def _non_limber_table(self, t1: Tracer, t2: Tracer) -> tuple:
        """Return the non-Limber correction of two ``'density'`` tracers at the multipoles where it is computed.

        The radial kernels are those of the CCL tracers, on 4001 comoving distances over the redshift range of each
        tracer; the power spectrum is that of the predictions at the scale factor of their mean comoving distance
        (weighted by the kernels). The exact spectrum integrates over k up to ``3 (ell + 1/2) / chi_min`` (plus the
        inverse width of the kernels), with 12 points per oscillation of the Bessel functions in k and in comoving
        distance. The multipoles (2 to 29, then 40 up to :attr:`NON_LIMBER_LMAX`) are computed in parallel threads.

        Parameters
        ----------
        t1, t2 : Tracer
            ``'density'`` tracers.

        Returns
        -------
        ells : ndarray
            Multipoles where the correction is computed.
        delta : ndarray
            Exact minus Limber C_ell at ``ells``.
        limber : callable
            Limber C_ell at the effective redshift, as a function of the multipoles, to extrapolate the correction
            beyond ``ells`` (see :meth:`_non_limber_correction`).
        """
        import os
        from concurrent.futures import ThreadPoolExecutor
        from scipy.special import spherical_jn
        cosmo = self.cosmo
        kernels = []
        for t in (t1, t2):
            chi_e = pyccl.comoving_radial_distance(cosmo, 1. / (1. + np.array([t.z.min(), t.z.max()])))
            chi = np.linspace(chi_e[0], chi_e[1], 4001)
            kernels.append((chi, np.asarray(t.ccl(cosmo).get_kernel(chi)).sum(axis=0)))
        w = np.concatenate([k[1] for k in kernels])
        chi_eff = np.sum(np.concatenate([k[0] for k in kernels]) * w) / np.sum(w)
        a_eff = float(pyccl.scale_factor_of_chi(cosmo, chi_eff))
        pk = self.p_of_k_a if self.p_of_k_a is not None else cosmo.get_nonlin_power()

        def power(k):
            return pk(np.atleast_1d(k), a_eff, cosmo)

        def limber(ell):
            # Limber spectrum at the effective redshift, on the overlap of the kernels
            chi = kernels[0][0]
            w2 = np.interp(chi, kernels[1][0], kernels[1][1], left=0., right=0.)
            return np.array([trapezoid(kernels[0][1] * w2 / chi**2 * power((l + 0.5) / chi), chi) for l in np.atleast_1d(ell)])

        chi_min = min(k[0][0] for k in kernels)
        chi_max = max(k[0][-1] for k in kernels)
        width = min(k[0][-1] - k[0][0] for k in kernels)

        def exact(ell):
            # k up to 3 (ell + 1/2) / chi_min (and the width of the kernels), 12 points per oscillation in k and in chi
            kmax = 3. * (ell + 0.5) / chi_min + 50. / width
            k = np.arange(0.2 * (ell + 0.5) / chi_max, kmax, 2. * np.pi / chi_max / 12.)
            integrals = []
            for chi, kern in kernels:
                c = np.linspace(chi[0], chi[-1], int(12 * (chi[-1] - chi[0]) * kmax / (2. * np.pi)) + 64)
                wc = np.interp(c, chi, kern)
                integrals.append(np.concatenate([trapezoid(wc * spherical_jn(ell, np.outer(k[s:s + 2000], c)), c, axis=1) for s in range(0, k.size, 2000)]))
            return 2. / np.pi * trapezoid(k**2 * power(k) * integrals[0] * integrals[1], k) - limber(ell)[0]

        ells = np.unique(np.concatenate([np.arange(2, 30), np.round(np.geomspace(30, self.NON_LIMBER_LMAX, 40))]).astype(int))
        with ThreadPoolExecutor(os.cpu_count()) as pool:  # scipy's Bessel functions release the GIL
            delta = np.array(list(pool.map(exact, ells[::-1])))[::-1]  # longest first
        return ells.astype(float), delta, limber

    # endregion

    # ==================================================================================================================
    # region Bandpowers

    def binned_cl(
            self,
            t1: Tracer,
            t2: Tracer,
            b: pymaster.NmtBin,
            bpws: np.ndarray | None = None,
            f_ell: np.ndarray | None = None,
            non_limber: bool = False) -> np.ndarray:
        """Return the angular power spectrum of two tracers, binned into bandpowers.

        Parameters
        ----------
        t1, t2 : Tracer
            Tracers.
        b : pymaster.NmtBin
            Binning (its ``lmax`` sets the multipoles used).
        bpws : ndarray or None, optional
            Bandpower windows of a workspace (``wsp.get_bandpower_windows()``, shape
            ``(n_cl, n_bands, n_cl, lmax + 1)``). Input power is put in the first
            (E or density) component, and, for two spin-2 fields, in the BB one for intrinsic
            alignments beyond NLA (see :meth:`angular_cl`), and all output components are returned. If
            None, C_ell (E mode) are averaged over each bandpower.
        f_ell : ndarray or None, optional
            Window applied to the fields (e.g. the pixel window), multiplying C_ell by ``f_ell**2``.
        non_limber : bool, optional
            Non-Limber correction of density pairs (see :meth:`angular_cl`). Default is False.

        Returns
        -------
        ndarray
            Shape ``(n_cl, n_bands)`` with ``bpws`` (e.g. EE, EB, BE, BB for two spin-2
            fields), else ``(1, n_bands)``.
        """
        ell = np.arange(b.lmax + 1)
        window = 1. if f_ell is None else f_ell**2
        cl = self.angular_cl(t1, t2, ell, non_limber=non_limber) * window
        if bpws is None:
            return b.bin_cell(cl[None, :])
        # E (or scalar) input in the first component of the windows
        cl_binned = bpws[:, :, 0, :] @ cl
        # B in the BB one, for two spin-2 fields only (zero but for intrinsic alignments)
        if bpws.shape[2] == 4:
            cl_binned = cl_binned + bpws[:, :, 3, :] @ (self.angular_cl(t1, t2, ell, bb=True) * window)
        return cl_binned

    def predict_cls(
            self,
            pairs: dict,
            *,
            b: pymaster.NmtBin | None = None,
            ell: np.ndarray | None = None,
            binning: str = 'wsp',
            bpws: np.ndarray | dict | None = None,
            f_ell: np.ndarray | None = None,
            noise: dict | None = None,
            error_bars: bool = False,
            stacked: bool = False,
            fsky: float = 1.,
            cross_covariance: bool = True,
            all_components: bool = False,
            non_limber: bool = False,
            progress: Callable | None = None) -> tuple:
        """Predict angular power spectra of tracer pairs, with optional Gaussian errors and covariance.

        Parameters
        ----------
        pairs : dict
            Tracer pairs to predict, ``{key: (t1, t2)}`` (e.g. ``{(i, j): (tracers[i], tracers[j])}``);
            outputs are keyed the same way.
        b : pymaster.NmtBin or None, optional
            Binning. Required unless ``binning='unbinned'`` with ``ell`` given.
        ell : ndarray or None, optional
            Multipoles for ``binning='unbinned'``. Default is the effective bandpower
            multipoles of ``b`` (needed for error bars, which use the bandpower widths).
        binning : {'wsp', 'average', 'unbinned'}, optional
            ``'wsp'`` (default): bandpowers binned with the bandpower windows ``bpws``.
            ``'average'``: bandpowers averaged over each bandpower of ``b``.
            ``'unbinned'``: C_ell at ``ell``. See :meth:`binned_cl`.
        bpws : ndarray or dict or None, optional
            Bandpower windows of a workspace (required with ``binning='wsp'``), for
            all pairs, or a dict of them keyed like ``pairs``. Spectra needed only for
            the covariance are binned with ``bpws`` if it is a single array, else averaged.
        f_ell : ndarray or None, optional
            Field window (e.g. pixel window), multiplying C_ell by ``f_ell**2``.
        noise : dict or None, optional
            White noise level N_ell of each tracer, keyed by tracer name, for the
            error bars (zero for missing tracers).
        error_bars : bool, optional
            If True, also return Gaussian error bars (diagonal in ell),
            ``cov(ab, cd) = (C_ac C_bd + C_ad C_bc) / ((2 ell + 1) fsky n_ell)``,
            with ``C_ell + N_ell`` for auto-spectra (computing the C_ell of other
            tracer pairs as needed).
        stacked : bool, optional
            If True, also return the spectra stacked into one vector (in the order
            of ``pairs``), and with ``error_bars`` their covariance.
        fsky : float, optional
            Sky fraction for the error bars.
        cross_covariance : bool, optional
            If False, the stacked covariance only has blocks between a pair and
            itself. Default is True.
        all_components : bool, optional
            With ``binning='wsp'``, return all components of the bandpower windows (shape
            ``(n_cl, n_bands)``, e.g. EE, EB, BE, BB) instead of the first one.
            Errors and stacking use the first component.
        non_limber : bool, optional
            Non-Limber correction of density pairs (see :meth:`angular_cl`): slow, for thin shells. Default is False.
        progress : callable or None, optional
            Wrapper for the loop over pairs (e.g. a tqdm function).

        Returns
        -------
        ell : ndarray
        cl : dict
            Keyed like ``pairs``.
        dcl : dict
            Only with ``error_bars``.
        cl_stacked, cov : ndarray
            Only with ``stacked`` (``cov`` only with ``error_bars`` too).
        """
        if binning not in ('wsp', 'average', 'unbinned'):
            raise ValueError(f"binning must be 'wsp', 'average' or 'unbinned', not {binning!r}")
        if binning == 'wsp' and bpws is None:
            raise ValueError("binning='wsp' needs the bandpower windows bpws")

        # multipoles: those given (unbinned), or the effective ones of the bandpowers, whose widths set the errors
        unbinned = binning == 'unbinned'
        nell = None
        if ell is None:
            ell = b.get_effective_ells()
            nell = b.get_nell_list()
        elif not unbinned:
            raise ValueError("ell can only be given with binning='unbinned'")
        if error_bars and nell is None:
            raise ValueError("error bars need the bandpower widths: use ell=None")

        noise = noise or {}
        self.cosmo.compute_growth()  # once, for all tracers

        def bpws_of(key: tuple | None) -> np.ndarray | None:
            # bandpower windows of one pair (None for bandpower averages)
            if binning != 'wsp':
                return None
            return bpws[key] if isinstance(bpws, dict) else bpws

        def predict(t1: Tracer, t2: Tracer, w: np.ndarray | None = None) -> np.ndarray:
            # C_ell of two tracers, unbinned or binned
            if unbinned:
                cl = self.angular_cl(t1, t2, ell, non_limber=non_limber)
                return cl if f_ell is None else cl * np.interp(ell, np.arange(len(f_ell)), f_ell)**2
            return self.binned_cl(t1, t2, b, bpws=w, f_ell=f_ell, non_limber=non_limber)

        # spectra of the pairs (all components with bandpower windows if all_components), keeping the first one per tracer pair
        cl = {}
        signal = {}  # first component of the C_ell of tracer pairs, keyed by tracer names
        for key, (t1, t2) in (progress or (lambda x, **k: x))(pairs.items(), desc="[predict_cls]"):
            w = bpws_of(key)
            res = predict(t1, t2, w)
            first = res if unbinned else res[0]
            signal[t1.name, t2.name] = signal[t2.name, t1.name] = first
            cl[key] = res if (all_components and not unbinned) else np.copy(first)

        outputs = [ell, cl]
        if error_bars or stacked:
            tracers = {t.name: t for t1, t2 in pairs.values() for t in (t1, t2)}

            def total(ta: str, tc: str) -> np.ndarray:
                # signal (computed if needed) plus noise for an auto-spectrum, of two tracers by name
                if (ta, tc) not in signal:
                    res = predict(tracers[ta], tracers[tc], None if isinstance(bpws, dict) else bpws_of(None))
                    signal[ta, tc] = signal[tc, ta] = res if unbinned else res[0]
                return signal[ta, tc] + (noise.get(ta, 0.) if ta == tc else 0.)

            # Gaussian errors cov(ab, cd) = (C_ac C_bd + C_ad C_bc) / ((2 ell + 1) fsky n_ell), from signal plus noise
            names = {key: (t1.name, t2.name) for key, (t1, t2) in pairs.items()}
            if error_bars:
                norm = (2 * ell + 1.) * fsky * nell
                dcl = {}
                for key, (ta, tb) in names.items():
                    dcl[key] = np.sqrt((total(ta, ta) * total(tb, tb) + total(ta, tb)**2) / norm)
                outputs.append(dcl)
            if stacked:
                # spectra stacked in the order of pairs, with their block-diagonal-in-ell covariance
                keys = list(pairs)
                n = len(ell)
                cl_stacked = np.concatenate([np.ravel(cl[key] if cl[key].ndim == 1 else cl[key][0]) for key in keys])
                outputs.append(cl_stacked)
                if error_bars:
                    cov = np.zeros([len(keys) * n, len(keys) * n])
                    for r, key1 in enumerate(keys):
                        ta, tb = names[key1]
                        for c, key2 in enumerate(keys):
                            if not cross_covariance and c != r:
                                continue
                            tc, td = names[key2]
                            block = (total(ta, td) * total(tb, tc) + total(ta, tc) * total(tb, td)) / norm
                            cov[r * n:(r + 1) * n, c * n:(c + 1) * n] = np.diag(block)
                    outputs.append(cov)
        return tuple(outputs)

    # endregion

    # ==================================================================================================================
    # region Correlations

    def predict_2pt(self, pairs: dict, theta: np.ndarray | dict, lmax: int, nell: int = 500, f_ell: np.ndarray | None = None) -> dict:
        """Predict real-space correlation functions of tracer pairs.

        Parameters
        ----------
        pairs : dict
            Tracer pairs to predict, ``{key: (t1, t2)}``, as in :meth:`predict_cls`; for a
            density-shear pair, ``t1`` is the density (lens) field. Outputs are keyed the same way.
        theta : array_like or dict
            Angular separations in arcmin, shared by all pairs or keyed like ``pairs``.
        lmax : int
            Maximum multipole of the C_ell used for the Hankel transform.
        nell : int, optional
            Number of log-spaced multipoles in ``[1, lmax]``.
        f_ell : ndarray or None, optional
            Window applied to the fields (e.g. the pixel window), at the multipoles ``0, 1, ...``
            up to at least ``lmax``: the C_ell are multiplied by ``f_ell**2`` (as in :meth:`binned_cl`)
            before the Hankel transform.

        Returns
        -------
        dict
            Per pair, ``w(theta)`` (density-density), ``gamma_t(theta)`` (density-shear or
            density-IA), or ``(xi_plus, xi_minus)`` (pairs of shear and IA, from C_EE + C_BB and
            C_EE - C_BB, see :meth:`angular_cl`).
        """
        ell = np.geomspace(1, lmax, nell)
        window = 1. if f_ell is None else np.interp(ell, np.arange(len(f_ell)), f_ell)**2
        xi = {}
        for key, (t1, t2) in pairs.items():
            corr = self._CORRELATION_TYPES[t1.kind, t2.kind]
            theta_deg = np.asarray(theta[key] if isinstance(theta, dict) else theta) / 60.
            cl = self.angular_cl(t1, t2, ell) * window
            if isinstance(corr, tuple):
                # xi+- of spin-2 fields: from C_EE +- C_BB
                cl_bb = self.angular_cl(t1, t2, ell, bb=True) * window
                xi[key] = tuple(self._correlation(ell, cl + sign * cl_bb, theta_deg, c) for c, sign in zip(corr, (1., -1.), strict=True))
            else:
                xi[key] = self._correlation(ell, cl, theta_deg, corr)
        return xi

    def _correlation(self, ell: np.ndarray, cl: np.ndarray, theta_deg: np.ndarray, corr: str) -> np.ndarray:
        """Return a correlation function from its C_ell, with CCL (FFTLog).

        CCL's FFTLog fails on spectra smoothed to vanishing values at high ell, and its power-law
        extrapolation diverges on numerical tails (e.g. of extrapolated power spectra): C_ell are
        cut at their first value below ``1e-10`` times their peak, after the peak.

        Parameters
        ----------
        ell : ndarray
            Log-spaced multipoles.
        cl : ndarray
            C_ell at ``ell``.
        theta_deg : ndarray
            Angular separations in degrees.
        corr : str
            Type of :func:`pyccl.correlation` (``'NN'``, ``'NG'``, ``'GG+'`` or ``'GG-'``).

        Returns
        -------
        ndarray
            Correlation function at ``theta_deg``; zero if ``cl`` is zero everywhere.
        """
        abs_cl = np.abs(cl)
        if not abs_cl.max() > 0.:
            return np.zeros(len(theta_deg))
        # cut at the first negligible value after the peak
        peak = np.argmax(abs_cl)
        negligible = np.flatnonzero(abs_cl[peak:] < 1e-10 * abs_cl[peak])
        last = peak + negligible[0] if len(negligible) else len(cl)
        return pyccl.correlation(self.cosmo, ell=ell[:last], C_ell=cl[:last], theta=theta_deg, type=corr)

    # endregion

    # ==================================================================================================================
    # region Lognormal

    def smoothed_variance(self, tracer: Tracer, chi: float, nside: int, smooth_Mpc: float | np.ndarray) -> float | np.ndarray:
        r"""Return the variance of a map of a tracer, smoothed by a Gaussian at the effective scale.

        As the maps of :meth:`pollux.map_builder.MapBuilder._smooth_alms`: Gaussian at
        :meth:`Tracer.effective_smooth_Mpc` (at least the pixel size), band-limited at
        ``3 nside - 1``.

        Parameters
        ----------
        tracer : Tracer
            Tracer of the map (e.g. the density of a shell).
        chi : float
            Comoving distance (Mpc) converting the smoothing scale to an angle.
        nside : int
            Nside of the map.
        smooth_Mpc : float or ndarray
            Comoving smoothing scales in Mpc.

        Returns
        -------
        float or ndarray
            Variance per smoothing scale.

        Notes
        -----
        .. math::

            \sigma^2(s) = \sum_{\ell = 2}^{3 N_{\rm side} - 1} \frac{2 \ell + 1}{4 \pi} C_\ell\, e^{-\ell (\ell + 1)\, \theta_s^2},
            \qquad \theta_s = s_{\rm eff}(\chi) / \chi.
        """
        lmax = 3 * nside - 1
        ell = np.arange(2, lmax + 1)
        ell_cl = np.unique(np.geomspace(2, lmax, 400).astype(int)).astype(float)
        cl = np.interp(ell, ell_cl, self.angular_cl(tracer, tracer, ell_cl))
        sigma = np.atleast_1d(Tracer.effective_smooth_Mpc(np.asarray(smooth_Mpc, dtype=float), chi, nside)) / chi
        variance = np.sum((2 * ell + 1) / (4 * np.pi) * cl * np.exp(-np.outer(sigma**2, ell * (ell + 1))), axis=1)
        return variance if np.ndim(smooth_Mpc) else variance[0]

    @staticmethod
    def lognormal_clipping(variance: float | np.ndarray, b: float) -> float | np.ndarray:
        r"""Return the fraction of pixels with ``1 + b * delta < 0``, for a lognormal density contrast.

        ``1 + delta = exp(G - s2 / 2)``, with ``G`` Gaussian of variance ``s2 = ln(1 + variance)``.
        For the projected density of the simulation shells this overestimates the clipped
        fraction (real voids are less empty), which makes it a conservative estimate.

        Parameters
        ----------
        variance : float or ndarray
            Variance of the density contrast.
        b : float
            Linear bias.

        Returns
        -------
        float or ndarray

        Notes
        -----
        With :math:`1 + \delta = e^{G - \sigma_G^2 / 2}`, :math:`G` Gaussian of variance :math:`\sigma_G^2 = \ln(1 + \sigma^2)`,
        :math:`1 + b \delta < 0` where :math:`G < c`, so that

        .. math::

            f_{\rm clip} = \Phi\left(\frac{c}{\sigma_G}\right), \qquad c = \ln\left(1 - \frac{1}{b}\right) + \frac{\sigma_G^2}{2},

        with :math:`\Phi` the standard normal cumulative distribution.
        """
        if b <= 1.:
            return 0. * np.asarray(variance)
        s2 = np.log1p(variance)
        return ndtr((np.log(1. - 1. / b) + s2 / 2.) / np.sqrt(s2))

    @staticmethod
    def lognormal_bias_ratio(variance: float | np.ndarray, b: float) -> float | np.ndarray:
        r"""Return the large-scale bias of ``max(0, 1 + b * delta)`` over ``b``, for a lognormal density contrast.

        From Price's theorem, ``b_eff = <dw/dG> / <w>`` for ``w = max(0, 1 + b * delta)`` and
        ``1 + delta = exp(G - s2 / 2)``. For the simulation shells this overestimates the
        bias deficit from clipping, which makes it a conservative estimate.

        Parameters
        ----------
        variance : float or ndarray
            Variance of the density contrast.
        b : float
            Linear bias.

        Returns
        -------
        float or ndarray
            ``b_eff / b``, at most 1.

        Notes
        -----
        With :math:`c` and :math:`\sigma_G` as in :meth:`lognormal_clipping`, for :math:`w = \max(0, 1 + b \delta)`,

        .. math::

            \frac{b_{\rm eff}}{b} = \frac{\langle \partial w / \partial G \rangle}{b \langle w \rangle}
            = \frac{\Phi\left((\sigma_G^2 - c) / \sigma_G\right)}{b\, \Phi\left((\sigma_G^2 - c) / \sigma_G\right) + (1 - b)\, \Phi(-c / \sigma_G)}.
        """
        if b <= 1.:
            return 1. + 0. * np.asarray(variance)
        s2 = np.log1p(variance)
        s = np.sqrt(s2)
        c = np.log(1. - 1. / b) + s2 / 2.  # clipped where G < c
        unclipped = ndtr((s2 - c) / s)  # <(1 + delta), unclipped pixels>
        return unclipped / (b * unclipped + (1. - b) * ndtr(-c / s))

    def lognormal_smoothing(
            self,
            tracers: list[Tracer],
            chi: np.ndarray,
            nside: int,
            b: float,
            max_clip: float | None = None,
            max_bias_error: float | None = None) -> np.ndarray:
        """Return a conservative smoothing scale per map, for galaxies placed with ``max(0, 1 + b * delta)``.

        For each map (e.g. shell), the smallest of :attr:`SMOOTHING_SCALES` that is at least
        the pixel size and keeps the clipped fraction below ``max_clip`` and the large-scale
        bias deficit ``1 - b_eff / b`` below ``max_bias_error``, for a lognormal density
        contrast of variance :meth:`smoothed_variance`. The lognormal overestimates both the
        clipping and the bias deficit of the simulation shells, by factors of 1.3 to 5
        (about 1.5 to 3 times the smoothing needed), so the scales are conservative.

        Parameters
        ----------
        tracers : list of Tracer
            Density tracer of each map, e.g. ``shell_builder.tracers``.
        chi : array_like
            Comoving distance (Mpc) of each map converting smoothing scales to angles, e.g.
            the far edges of the shells, ``shell_builder.shell_edges_comoving_distance[1:]``.
        nside : int
            Nside of the maps.
        b : float
            Linear bias (the largest of the tomographic bins).
        max_clip : float or None, optional
            Largest fraction of clipped pixels.
        max_bias_error : float or None, optional
            Largest large-scale bias deficit ``1 - b_eff / b``.

        Returns
        -------
        ndarray
            Smoothing scale per map, in Mpc (the largest of :attr:`SMOOTHING_SCALES`,
            with a warning, if none meets the criteria).
        """
        if max_clip is None and max_bias_error is None:
            raise ValueError("Give max_clip, max_bias_error or both.")
        scales = np.array(self.SMOOTHING_SCALES)
        smooth = np.zeros(len(tracers))
        for i, (tracer, d) in enumerate(zip(tracers, chi)):
            variance = self.smoothed_variance(tracer, d, nside, scales)
            ok = scales >= hp.nside2resol(nside) * d
            if max_clip is not None:
                ok &= self.lognormal_clipping(variance, b) <= max_clip
            if max_bias_error is not None:
                ok &= 1. - self.lognormal_bias_ratio(variance, b) <= max_bias_error
            if not ok.any():
                warnings.warn(f"No smoothing up to {scales[-1]} Mpc meets the criteria for map {i}: using {scales[-1]} Mpc.")
            smooth[i] = scales[np.argmax(ok)] if ok.any() else scales[-1]
        return smooth

    # endregion


# endregion
