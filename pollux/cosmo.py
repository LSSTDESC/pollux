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

import numpy as np
import pyccl
import sacc

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
    kind : {'density', 'shear', 'convergence'}
        - ``'density'``: number counts (or matter density with ``bias=1``) with
          redshift distribution ``(z, nz)``, without RSD.
        - ``'shear'``: weak lensing of sources with redshift distribution ``(z, nz)``
          (spin 2; also used for the convergence of the same sources).
        - ``'convergence'``: lensing convergence of a single source plane at ``z_source``.
    z, nz : ndarray or None, optional
        Redshift distribution (need not be normalized), for ``'density'`` and ``'shear'``.
    bias : float, optional
        Linear bias, constant in redshift, for ``'density'``. Default is 1.
    z_source : float or None, optional
        Source redshift, for ``'convergence'``.
    smooth_Mpc : float or None, optional
        Comoving scale of a Gaussian smoothing of the field (e.g. galaxies placed
        from a smoothed density), for ``'density'``. Predictions multiply the matter
        power spectrum by ``exp(-k**2 smooth_Mpc**2 / 2)`` per smoothed tracer, which
        is exact in Limber for a transverse smoothing. Default is None.
    """

    KINDS = ('density', 'shear', 'convergence')

    def __init__(
            self,
            name: str,
            kind: str,
            z: np.ndarray | None = None,
            nz: np.ndarray | None = None,
            bias: float = 1.,
            z_source: float | None = None,
            smooth_Mpc: float | None = None) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        if kind == 'convergence' and z_source is None:
            raise ValueError("z_source is required for kind='convergence'")
        if kind != 'convergence' and (z is None or nz is None):
            raise ValueError(f"z and nz are required for kind={kind!r}")
        if smooth_Mpc and kind != 'density':
            raise ValueError("smooth_Mpc is only supported for kind='density'")
        self.name = name
        self.kind = kind
        self.z = None if z is None else np.asarray(z)
        self.nz = None if nz is None else np.asarray(nz)
        self.bias = bias
        self.z_source = z_source
        self.smooth_Mpc = smooth_Mpc or None
        self._ccl = None

    @property
    def spin(self) -> int:
        """Spin of the field (2 for shear, 0 otherwise)."""
        return 2 if self.kind == 'shear' else 0

    def ccl(self, cosmo: pyccl.Cosmology) -> pyccl.Tracer:
        """Return the CCL tracer for this field (cached for a given cosmology).

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology.

        Returns
        -------
        pyccl.Tracer
            ``NumberCountsTracer``, ``WeakLensingTracer`` or ``CMBLensingTracer`` (the convergence
            of a single source plane, at any redshift).
        """
        if self._ccl is None or self._ccl[0] is not cosmo:
            if self.kind == 'density':
                tracer = pyccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(self.z, self.nz), bias=(self.z, np.full_like(self.z, self.bias)))
            elif self.kind == 'shear':
                tracer = pyccl.WeakLensingTracer(cosmo, dndz=(self.z, self.nz))
            else:
                tracer = pyccl.CMBLensingTracer(cosmo, z_source=self.z_source)
            self._ccl = (cosmo, tracer)
        return self._ccl[1]

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

    # CCL correlation type of each pair of tracer kinds (and component, for shear-shear)
    _CORRELATION_TYPES = {
        ('density', 'density'): 'NN',
        ('density', 'shear'): 'NG',
        ('shear', 'shear'): ('GG+', 'GG-'),
        ('convergence', 'convergence'): 'NN',
    }

    def __init__(self, cosmo: pyccl.Cosmology, p_of_k_a: pyccl.Pk2D | None = None) -> None:
        self.cosmo = cosmo
        self.p_of_k_a = p_of_k_a
        self._smoothed_pk = {}  # matter power spectra with a Gaussian smoothing, keyed by sigma**2

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
            C_ell (E-mode for shear; no pixel window), with the Gaussian smoothing of the
            tracers (see :class:`Tracer`).
        """
        ccl1, ccl2 = t1.ccl(self.cosmo), t2.ccl(self.cosmo)
        pk = self.p_of_k_a
        sigma2 = (t1.smooth_Mpc or 0.)**2 + (t2.smooth_Mpc or 0.)**2
        if sigma2 > 0.:
            if sigma2 not in self._smoothed_pk:
                # same grid as the unsmoothed power spectrum, Gaussian filter applied to log P (no underflow)
                a_arr, lk_arr, pk_arr = (pk if pk is not None else self.cosmo.get_nonlin_power()).get_spline_arrays()
                log_pk = np.log(pk_arr) - np.exp(2. * lk_arr)[None, :] * sigma2 / 2.
                self._smoothed_pk[sigma2] = pyccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=log_pk, is_logp=True)
            pk = self._smoothed_pk[sigma2]
        try:
            return pyccl.angular_cl(self.cosmo, ccl1, ccl2, ell, p_of_k_a=pk)
        except pyccl.CCLError:
            return pyccl.angular_cl(self.cosmo, ccl1, ccl2, ell, p_of_k_a=pk, limber_integration_method='spline')

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
                return self.angular_cl(t1, t2, ell)
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
            Per pair, ``w(theta)`` (density-density), ``gamma_t(theta)`` (density-shear), or
            ``(xi_plus, xi_minus)`` (shear-shear).
        """
        ell = np.geomspace(1, lmax, nell)
        xi = {}
        for key, (t1, t2) in pairs.items():
            corr = self._CORRELATION_TYPES[t1.kind, t2.kind]
            cl = self.angular_cl(t1, t2, ell)
            theta_deg = np.asarray(theta[key] if isinstance(theta, dict) else theta) / 60.
            res = [pyccl.correlation(self.cosmo, ell=ell, C_ell=cl, theta=theta_deg, type=c) for c in np.atleast_1d(corr)]
            xi[key] = tuple(res) if isinstance(corr, tuple) else res[0]
        return xi
