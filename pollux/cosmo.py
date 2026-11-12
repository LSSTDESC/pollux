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

from typing import TYPE_CHECKING, Callable

import healpy as hp
import numpy as np
import pyccl
import sacc
from scipy.special import ndtr
import warnings

if TYPE_CHECKING:
    import pymaster

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

    @property
    def spin(self) -> int:
        """Spin of the field (2 for shear and intrinsic alignments, 0 otherwise)."""
        return 2 if self.kind in ('shear', 'ia') else 0

    @property
    def _needs_pt(self) -> bool:
        """Whether the predictions of the intrinsic alignments need perturbation theory: terms beyond NLA (``a2`` or ``a1delta``)."""
        return self.kind == 'ia' and any(self.ia.get(k) for k in ('a2', 'a1delta'))

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

    def mean_z_nz(self) -> float:
        """Mean redshift of the n(z) of the field (the source redshift for ``'convergence'``).

        For lensing fields this is the redshift of the sources, not of the structures
        they probe: see :meth:`mean_z_kernel`.
        """
        if self.kind == 'convergence':
            return self.z_source
        return np.trapz(self.nz * self.z, self.z) / np.trapz(self.nz, self.z)

    def mean_z_kernel(self, cosmo: pyccl.Cosmology, nchi: int = 1000) -> float:
        """Mean redshift weighted by the radial kernel of the field, i.e. where its signal comes from.

        Equal to :meth:`mean_z_nz` for density fields (whose kernel is their n(z)); for
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
        return np.trapz(z * kernel, chi) / np.trapz(kernel, chi)

    @staticmethod
    def effective_smooth_Mpc(smooth_Mpc: float | tuple, chi: float | np.ndarray, nside: int | None) -> float | np.ndarray:
        """Return the comoving smoothing scale actually applied at a distance: at least the pixel size.

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

    def __getstate__(self) -> dict:
        """Return the state for pickling, without the cached CCL tracer (rebuilt on demand)."""
        state = self.__dict__.copy()
        state['_ccl'] = None
        return state


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

    def __init__(self, cosmo: pyccl.Cosmology, p_of_k_a: pyccl.Pk2D | None = None) -> None:
        self.cosmo = cosmo
        self.p_of_k_a = p_of_k_a
        self._smoothed_pk = {}  # power spectra with a Gaussian smoothing, keyed by the spectrum and the tracers' smoothing
        self._pt_calculator = None
        self._pt_correction_pk = {}  # corrections of perturbation theory to NLA, keyed by the tracers' amplitudes

    def angular_cl(self, t1: Tracer, t2: Tracer, ell: np.ndarray) -> np.ndarray:
        """Return the angular power spectrum of two tracers.

        Parameters
        ----------
        t1, t2 : Tracer
            Tracers.
        ell : array_like
            Multipoles.

        Returns
        -------
        ndarray
            C_ell (E-mode for spin-2 fields; no pixel window), with the Gaussian smoothing of the
            tracers (see :class:`Tracer`), at their effective scale at each redshift.

        Notes
        -----
        Intrinsic alignments (``'ia'`` tracers) follow the NLA model with the matter power
        spectrum of the predictions; for TATT amplitudes (``a2`` or ``a1delta``), the terms
        beyond NLA of CCL's perturbation theory are added (:meth:`_pk_pt_correction`, with the
        same smoothing). The density weighting of the tidal torquing term is not modelled.
        """
        key = tuple(sorted(((t.smooth_Mpc, t.smooth_nside or 0) for t in (t1, t2) if t.smooth_Mpc), key=repr))
        cl = self._limber(t1.ccl(self.cosmo), t2.ccl(self.cosmo), ell, self._apply_smoothing(self.p_of_k_a, key))

        # intrinsic alignments beyond NLA, with the IA leg first
        if t1.kind != 'ia' and t2.kind == 'ia':
            t1, t2 = t2, t1
        if t1.kind == 'ia' and (t1._needs_pt or t2._needs_pt) and t2.kind != 'convergence':
            pk = self._pk_pt_correction(t1, t2, key)
            ccl2 = t2._ccl_pt_kernel(self.cosmo) if t2.kind == 'ia' else t2.ccl(self.cosmo)
            cl = cl + self._limber(t1._ccl_pt_kernel(self.cosmo), ccl2, ell, pk)
        return cl

    def _limber(self, ccl1: pyccl.Tracer, ccl2: pyccl.Tracer, ell: np.ndarray, pk: pyccl.Pk2D | None) -> np.ndarray:
        """Return CCL's angular power spectrum, with the spline integration if the default one fails."""
        try:
            return pyccl.angular_cl(self.cosmo, ccl1, ccl2, ell, p_of_k_a=pk)
        except pyccl.CCLError:
            return pyccl.angular_cl(self.cosmo, ccl1, ccl2, ell, p_of_k_a=pk, limber_integration_method='spline')

    def _apply_smoothing(self, pk: pyccl.Pk2D | None, key: tuple, name: str = 'matter') -> pyccl.Pk2D | None:
        """Return a power spectrum multiplied by the Gaussian smoothing of the tracers ``key`` (cached by ``name``).

        Same grid as ``pk`` (CCL's nonlinear matter power spectrum if None), with the smoothing of
        each tracer at the comoving distance of each scale factor. Positive spectra are
        filtered in log (no underflow).
        """
        if not key:
            return pk
        if (name, key) not in self._smoothed_pk:
            a_arr, lk_arr, pk_arr = (pk if pk is not None else self.cosmo.get_nonlin_power()).get_spline_arrays()
            chi = pyccl.comoving_radial_distance(self.cosmo, a_arr)
            sigma2 = sum(Tracer.effective_smooth_Mpc(s, chi, nside or None)**2 for s, nside in key)
            filt = np.exp(2. * lk_arr)[None, :] * sigma2[:, None] / 2.
            if np.all(pk_arr > 0.):
                smoothed = pyccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=np.log(pk_arr) - filt, is_logp=True)
            else:
                smoothed = pyccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr * np.exp(-filt), is_logp=False)
            self._smoothed_pk[name, key] = smoothed
        return self._smoothed_pk[name, key]

    def _pk_pt_correction(self, t1: Tracer, t2: Tracer, smooth_key: tuple) -> pyccl.Pk2D:
        """Return the correction of perturbation theory to the NLA power spectrum, between intrinsic alignments and IA or matter.

        The difference between the TATT and NLA spectra of CCL's perturbation theory
        (:class:`pyccl.nl_pt.EulerianPTCalculator`, FAST-PT): the NLA term itself is computed
        with the matter power spectrum of the predictions (nonlinear, possibly with baryons).
        ``t2`` is an ``'ia'`` tracer, or another tracer for the matter (its bias is in its CCL kernel).
        The spectrum is smoothed as the tracers (``smooth_key``, see :meth:`_apply_smoothing`).
        """
        if self._pt_calculator is None:
            self._pt_calculator = pyccl.nl_pt.EulerianPTCalculator(with_IA=True, with_NC=False, with_matter_1loop=False, cosmo=self.cosmo)
        ias = tuple(tuple(sorted(t.ia.items())) if t.kind == 'ia' else None for t in (t1, t2))
        key = (ias, tuple(float(np.max(t.z)) if t.kind == 'ia' else None for t in (t1, t2)))
        if key not in self._pt_correction_pk:
            # second leg: intrinsic alignments, or the matter
            if t2.kind == 'ia':
                full2, nla2 = t2._pt_tracer(self.cosmo), t2._pt_tracer(self.cosmo, nla_only=True)
            else:
                full2 = nla2 = pyccl.nl_pt.PTMatterTracer()
            full = self._pt_calculator.get_biased_pk2d(t1._pt_tracer(self.cosmo), tracer2=full2)
            nla = self._pt_calculator.get_biased_pk2d(t1._pt_tracer(self.cosmo, nla_only=True), tracer2=nla2)
            self._pt_correction_pk[key] = full - nla
        return self._apply_smoothing(self._pt_correction_pk[key], smooth_key, name=('ia', key))

    def binned_cl(self, t1: Tracer, t2: Tracer, b: pymaster.NmtBin, bpws: np.ndarray | None = None, f_ell: np.ndarray | None = None) -> np.ndarray:
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
            (E or density) component, and all output components are returned. If
            None, C_ell are averaged over each bandpower.
        f_ell : ndarray or None, optional
            Window applied to the fields (e.g. the pixel window), multiplying C_ell by ``f_ell**2``.

        Returns
        -------
        ndarray
            Shape ``(n_cl, n_bands)`` with ``bpws`` (e.g. EE, EB, BE, BB for two spin-2
            fields), else ``(1, n_bands)``.
        """
        cl = self.angular_cl(t1, t2, np.arange(b.lmax + 1))
        if f_ell is not None:
            cl = cl * f_ell**2
        if bpws is not None:
            return bpws[:, :, 0, :] @ cl
        return b.bin_cell(cl[None, :])

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
                cl = self.angular_cl(t1, t2, ell)
                return cl if f_ell is None else cl * np.interp(ell, np.arange(len(f_ell)), f_ell)**2
            return self.binned_cl(t1, t2, b, bpws=w, f_ell=f_ell)

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

            def total(a: str, c: str) -> np.ndarray:
                # signal (computed if needed) plus noise for an auto-spectrum
                if (a, c) not in signal:
                    res = predict(tracers[a], tracers[c], None if isinstance(bpws, dict) else bpws_of(None))
                    signal[a, c] = signal[c, a] = res if unbinned else res[0]
                return signal[a, c] + (noise.get(a, 0.) if a == c else 0.)

            # Gaussian errors cov(ab, cd) = (C_ac C_bd + C_ad C_bc) / ((2 ell + 1) fsky n_ell), from signal plus noise
            names = {key: (t1.name, t2.name) for key, (t1, t2) in pairs.items()}
            if error_bars:
                norm = (2 * ell + 1.) * fsky * nell
                dcl = {}
                for key, (a, bb) in names.items():
                    dcl[key] = np.sqrt((total(a, a) * total(bb, bb) + total(a, bb)**2) / norm)
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
                        a, bb = names[key1]
                        for c, key2 in enumerate(keys):
                            if not cross_covariance and c != r:
                                continue
                            cc, d = names[key2]
                            block = (total(a, d) * total(bb, cc) + total(a, cc) * total(bb, d)) / norm
                            cov[r * n:(r + 1) * n, c * n:(c + 1) * n] = np.diag(block)
                    outputs.append(cov)
        return tuple(outputs)

    def predict_2pt(self, pairs: dict, theta: np.ndarray | dict, lmax: int, nell: int = 500) -> dict:
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

        Returns
        -------
        dict
            Per pair, ``w(theta)`` (density-density), ``gamma_t(theta)`` (density-shear or
            density-IA), or ``(xi_plus, xi_minus)`` (pairs of shear and IA).
        """
        ell = np.geomspace(1, lmax, nell)
        xi = {}
        for key, (t1, t2) in pairs.items():
            corr = self._CORRELATION_TYPES[t1.kind, t2.kind]
            cl = self.angular_cl(t1, t2, ell)
            # CCL's FFTLog fails on spectra smoothed to vanishing values at high ell, or zero everywhere
            significant = np.flatnonzero(np.abs(cl) > 1e-30 * np.abs(cl).max())
            theta_deg = np.asarray(theta[key] if isinstance(theta, dict) else theta) / 60.
            if len(significant):
                last = significant[-1] + 1
                res = [pyccl.correlation(self.cosmo, ell=ell[:last], C_ell=cl[:last], theta=theta_deg, type=c) for c in np.atleast_1d(corr)]
            else:
                res = [np.zeros(len(theta_deg)) for c in np.atleast_1d(corr)]
            xi[key] = tuple(res) if isinstance(corr, tuple) else res[0]
        return xi

    # smoothing scales (Mpc) considered by lognormal_smoothing: round values, for file names
    SMOOTHING_SCALES = (0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1., 1.2, 1.5, 2., 2.5, 3., 4., 5., 6., 8., 10., 12., 15., 20., 25., 30., 40., 50.)

    def smoothed_variance(self, tracer: Tracer, chi: float, nside: int, smooth_Mpc: float | np.ndarray) -> float | np.ndarray:
        """Return the variance of a map of a tracer, smoothed by a Gaussian at the effective scale.

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
        """Return the fraction of pixels with ``1 + b * delta < 0``, for a lognormal density contrast.

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
        """
        if b <= 1.:
            return 0. * np.asarray(variance)
        s2 = np.log1p(variance)
        return ndtr((np.log(1. - 1. / b) + s2 / 2.) / np.sqrt(s2))

    @staticmethod
    def lognormal_bias_ratio(variance: float | np.ndarray, b: float) -> float | np.ndarray:
        """Return the large-scale bias of ``max(0, 1 + b * delta)`` over ``b``, for a lognormal density contrast.

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
