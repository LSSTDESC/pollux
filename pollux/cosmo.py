"""Cosmology helpers: tracers and theory predictions, and reference cosmologies.

:class:`Tracer` describes an observable (a density, shear or convergence field
with its redshift distribution), from which CCL and SACC tracers are derived.
:class:`Theory` predicts angular power spectra and correlation functions between
tracers, for a given cosmology and matter power spectrum. Both are shared by the
map builders and the catalogs.

The module-level cosmologies are convenience definitions for examples/tests and
are not used by the main pipeline, which reads cosmology from simulation metadata.
"""

import numpy as np
import pyccl
import sacc

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


class MyCMBLensingTracer(pyccl.Tracer):
    """CCL tracer for the lensing convergence of a single source plane.

    Parameters
    ----------
    cosmo : pyccl.Cosmology
        Cosmology object.
    z_source : float
        Source redshift defining the lensing kernel.
    n_samples : int, optional
        Number of samples used internally to tabulate the kernel.
    """

    def __init__(self, cosmo, z_source, n_samples=100):
        super().__init__()

        # we need the distance functions at the C layer
        cosmo.compute_distances()
        kernel = pyccl.get_kappa_kernel(cosmo, z_source=z_source, n_samples=n_samples)
        if (cosmo['sigma_0'] == 0):
            self.add_tracer(cosmo, kernel=kernel, der_bessel=-1, der_angles=1)
        else:
            self._MG_add_tracer(cosmo, kernel, z_source, der_bessel=-1, der_angles=1)

        self.z_source = z_source


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
    """

    KINDS = ('density', 'shear', 'convergence')

    def __init__(self, name, kind, z=None, nz=None, bias=1., z_source=None):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        if kind == 'convergence' and z_source is None:
            raise ValueError("z_source is required for kind='convergence'")
        if kind != 'convergence' and (z is None or nz is None):
            raise ValueError(f"z and nz are required for kind={kind!r}")
        self.name = name
        self.kind = kind
        self.z = None if z is None else np.asarray(z)
        self.nz = None if nz is None else np.asarray(nz)
        self.bias = bias
        self.z_source = z_source
        self._ccl = None

    @property
    def spin(self):
        """Spin of the field (2 for shear, 0 otherwise)."""
        return 2 if self.kind == 'shear' else 0

    def ccl(self, cosmo):
        """Return the CCL tracer for this field (cached for a given cosmology).

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Cosmology.

        Returns
        -------
        pyccl.Tracer
            ``NumberCountsTracer``, ``WeakLensingTracer`` or :class:`MyCMBLensingTracer`.
        """
        if self._ccl is None or self._ccl[0] is not cosmo:
            if self.kind == 'density':
                tracer = pyccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(self.z, self.nz), bias=(self.z, np.full_like(self.z, self.bias)))
            elif self.kind == 'shear':
                tracer = pyccl.WeakLensingTracer(cosmo, dndz=(self.z, self.nz))
            else:
                tracer = MyCMBLensingTracer(cosmo, self.z_source)
            self._ccl = (cosmo, tracer)
        return self._ccl[1]

    def sacc(self):
        """Return the SACC tracer for this field.

        Returns
        -------
        sacc.tracers.BaseTracer
            ``NZTracer`` for fields with a redshift distribution, ``MiscTracer`` otherwise.
        """
        if self.kind == 'convergence':
            return sacc.tracers.MiscTracer(self.name)
        return sacc.tracers.NZTracer(self.name, self.z, self.nz)

    def mean_z(self):
        """Mean redshift of the field (the source redshift for ``'convergence'``)."""
        if self.kind == 'convergence':
            return self.z_source
        return np.trapz(self.nz * self.z, self.z) / np.trapz(self.nz, self.z)

    def __getstate__(self):
        # CCL tracers are rebuilt on demand (keeps tracers pickle-friendly)
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

    def __init__(self, cosmo, p_of_k_a=None):
        self.cosmo = cosmo
        self.p_of_k_a = p_of_k_a

    def angular_cl(self, t1, t2, ell):
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
            C_ell (E-mode for shear; no pixel window).
        """
        ccl1, ccl2 = t1.ccl(self.cosmo), t2.ccl(self.cosmo)
        try:
            return pyccl.angular_cl(self.cosmo, ccl1, ccl2, ell, p_of_k_a=self.p_of_k_a)
        except pyccl.CCLError:
            return pyccl.angular_cl(self.cosmo, ccl1, ccl2, ell, p_of_k_a=self.p_of_k_a, limber_integration_method='spline')

    def binned_cl(self, t1, t2, b, bpws=None, f_ell=None):
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

    def predict_cls(self, pairs, b=None, ell=None, binning='wsp', bpws=None, f_ell=None, noise=None, add_noise=False,
                    error_bars=False, stacked=False, fsky=1., cross_covariance=True, all_components=False, progress=None):
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
            White noise level N_ell of each tracer, keyed by tracer name; used for
            ``add_noise`` and error bars (zero for missing tracers).
        add_noise : bool, optional
            If True, add the noise bias to the predicted auto-spectra (error bars
            include the noise either way).
        error_bars : bool, optional
            If True, also return Gaussian error bars (diagonal in ell),
            ``cov(ab, cd) = (C_ac C_bd + C_ad C_bc) / ((2 ell + 1) fsky n_ell)``,
            with ``C_ell + N_ell`` for auto-spectra whatever ``add_noise`` (computing
            the C_ell of other tracer pairs as needed).
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
        self.cosmo.compute_growth()

        def bpws_of(key):
            if binning != 'wsp':
                return None
            return bpws[key] if isinstance(bpws, dict) else bpws

        def predict(t1, t2, w=None):
            if unbinned:
                return self.angular_cl(t1, t2, ell)
            return self.binned_cl(t1, t2, b, bpws=w, f_ell=f_ell)

        cl = {}
        signal = {}  # first component of the C_ell of tracer pairs, keyed by tracer names
        for key, (t1, t2) in (progress or (lambda x, **k: x))(pairs.items(), desc="[predict_cls]"):
            w = bpws_of(key)
            res = predict(t1, t2, w)
            first = res if unbinned else res[0]
            signal[t1.name, t2.name] = signal[t2.name, t1.name] = first
            out = res if (all_components and not unbinned) else np.copy(first)
            if add_noise and t1.name == t2.name and t1.name in noise:
                if unbinned:
                    out += noise[t1.name]
                else:
                    nl = noise[t1.name] * np.ones(b.lmax + 1)
                    if f_ell is not None:
                        nl *= f_ell**2
                    nl_binned = w[0, :, 0, :] @ nl if w is not None else b.bin_cell(nl[None, :])[0]
                    if out.ndim == 2:
                        out[0] += nl_binned
                    else:
                        out += nl_binned
            cl[key] = out

        outputs = [ell, cl]
        if error_bars or stacked:
            tracers = {t.name: t for t1, t2 in pairs.values() for t in (t1, t2)}

            def total(a, c):
                # signal (computed if needed) plus noise for an auto-spectrum
                if (a, c) not in signal:
                    res = predict(tracers[a], tracers[c], None if isinstance(bpws, dict) else bpws_of(None))
                    signal[a, c] = signal[c, a] = res if unbinned else res[0]
                return signal[a, c] + (noise.get(a, 0.) if a == c else 0.)

            names = {key: (t1.name, t2.name) for key, (t1, t2) in pairs.items()}
            if error_bars:
                norm = (2 * ell + 1.) * fsky * nell
                dcl = {}
                for key, (a, bb) in names.items():
                    dcl[key] = np.sqrt((total(a, a) * total(bb, bb) + total(a, bb)**2) / norm)
                outputs.append(dcl)
            if stacked:
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

    def correlation(self, t1, t2, theta, lmax, nell=500):
        """Return the real-space correlation function(s) of two tracers.

        Parameters
        ----------
        t1, t2 : Tracer
            Tracers; for a density-shear pair, ``t1`` is the density (lens) field.
        theta : array_like
            Angular separations in arcmin.
        lmax : int
            Maximum multipole of the C_ell used for the Hankel transform.
        nell : int, optional
            Number of log-spaced multipoles in ``[1, lmax]``.

        Returns
        -------
        ndarray or tuple of ndarray
            ``w(theta)`` (density-density), ``gamma_t(theta)`` (density-shear), or
            ``(xi_plus, xi_minus)`` (shear-shear).
        """
        corr = self._CORRELATION_TYPES[t1.kind, t2.kind]
        ell = np.geomspace(1, lmax, nell)
        cl = self.angular_cl(t1, t2, ell)
        theta_deg = np.asarray(theta) / 60.
        if isinstance(corr, tuple):
            return tuple(pyccl.correlation(self.cosmo, ell=ell, C_ell=cl, theta=theta_deg, type=c) for c in corr)
        return pyccl.correlation(self.cosmo, ell=ell, C_ell=cl, theta=theta_deg, type=corr)
