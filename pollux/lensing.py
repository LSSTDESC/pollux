import os

import healpy as hp
import healsparse
import numpy as np
import pyccl
import sacc
from tqdm.auto import trange

from .map_builder import MapBuilder
from .utils import quad


class MyCMBLensingTracer(pyccl.Tracer):
    """CCL tracer for CMB lensing convergence with configurable source redshift.

    Parameters
    ----------
    cosmo : pyccl.Cosmology
        Cosmology object.
    z_source : float
        Effective source redshift defining the CMB lensing kernel.
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


class LensingBuilder(MapBuilder):
    """Base class to build convergence and shear maps from matter shells.

    Parameters
    ----------
    shell_builder : pollux.shells.ShellBuilder
        Shell builder providing matter shells and geometry.
    nzs : {'zsnapshots'} or sequence
        If 'zsnapshots', produce one convergence map per shell edge (CMB-like).
        Otherwise, a list describing source n(z) per tomographic bin. Each entry
        can be a callable ``nz(z)`` or a tuple ``(z, n(z))``.
    kappa_dir : str or None, optional
        Output subdirectory under ``shell_builder.save_dir``.
    dtype : data-type, optional
        Output dtype for kappa and shear maps.
    sigma_e : float, optional
        Shape noise (ellipticity dispersion) used for noise spectra.
    nbar : float or array_like, optional
        Galaxy surface density in gal/arcmin^2 used for noise spectra. If a
        float is provided it applies uniformly to all maps; if an array_like
        is provided it must have length matching the number of maps and gives
        per-map values. Values are converted to per-steradian internally
        where needed.
    """

    def __init__(self, shell_builder, nzs, kappa_dir=None, dtype=np.float32, sigma_e=0.3, nbar=5., **kwargs):
        self.shell_builder = shell_builder
        if kappa_dir is None:
            save_dir = shell_builder.save_dir
        else:
            save_dir = os.path.join(shell_builder.save_dir, kappa_dir)
        self.reader = shell_builder.reader
        self.cosmo = shell_builder.cosmo
        self.nshells = shell_builder.nshells

        # initialize nzs and number of kappa maps
        self.nzs = nzs
        if nzs == 'zsnapshots':
            self.nkappas = self.shell_builder.nshells
        else:
            self.nkappas = len(nzs)

        # save n(z) info
        fname = os.path.join(save_dir, 'nzs.npy')
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(fname):
            np.save(fname, nzs)

        # normalize sigma_e: accept scalar or per-bin array
        sigma_arr = np.atleast_1d(sigma_e).astype(float)
        if sigma_arr.size == 1:
            sigma_arr = np.full(self.nkappas, float(sigma_arr[0]))
        elif sigma_arr.size != self.nkappas:
            raise ValueError(f"sigma_e must be a scalar or array of length {self.nkappas}, got {sigma_arr.size}")
        self.sigma_e = sigma_arr

        # normalize nbar: accept scalar or per-bin array (API units: gal/arcmin^2)
        nbar_arr = np.atleast_1d(nbar).astype(float)
        if nbar_arr.size == 1:
            nbar_arr = np.full(self.nkappas, float(nbar_arr[0]))
        elif nbar_arr.size != self.nkappas:
            raise ValueError(f"nbar must be a scalar or array of length {self.nkappas}, got {nbar_arr.size}")
        # store API-facing nbar in gal/arcmin^2 and an internal copy in gal/steradian
        self.nbar = nbar_arr
        self.nbar_sr = nbar_arr * (60.0 * 180.0 / np.pi)**2

        base_fname = f'kappa_{shell_builder.reader.kind}_seed{shell_builder.seed}'

        super().__init__(
            nside=shell_builder.nside, dtype=dtype, save_dir=save_dir, base_fname=base_fname, nmaps=self.nkappas, sacc_type="galaxy_convergence_cl", **kwargs)

    def _create_tracers_ccl(self, Nz=1000):
        """Create CCL lensing tracers.

        Parameters
        ----------
        Nz : int, optional
            Sampling used to tabulate n(z) when a callable is provided.

        Returns
        -------
        list
            List of CCL tracers (length ``self.nmaps``).
        """
        z_shell_bins = self.shell_builder.shell_edges_redshift()

        tracers = []
        if self.nzs == 'zsnapshots':
            for z_source in z_shell_bins[1:]:
                tracers.append(MyCMBLensingTracer(self.cosmo, z_source))
        else:
            for nz in self.nzs:
                if callable(nz):
                    zz = np.linspace(0, z_shell_bins[-1], Nz)
                    tracers.append(pyccl.WeakLensingTracer(self.cosmo, dndz=(zz, nz(zz))))
                else:
                    tracers.append(pyccl.WeakLensingTracer(self.cosmo, dndz=(nz[0], nz[1])))

        return tracers

    def _create_tracers_sacc(self, Nz=1000):
        """Create SACC tracers matching the lensing maps.

        Parameters
        ----------
        Nz : int, optional
            Sampling used to tabulate n(z) when a callable is provided.

        Returns
        -------
        list
            List of SACC tracers (length ``self.nmaps``).
        """
        z_shell_bins = self.shell_builder.shell_edges_redshift()

        tracers = []
        if self.nzs == 'zsnapshots':
            for i, _ in enumerate(z_shell_bins[1:]):
                tracers.append(sacc.tracers.MiscTracer(f"kappa_{i}"))
        else:
            for i, nz in enumerate(self.nzs):
                if callable(nz):
                    zz = np.linspace(0, z_shell_bins[-1], Nz)
                    tracers.append(sacc.tracers.NZTracer(f"kappa_{i}", zz, nz(zz)))
                else:
                    tracers.append(sacc.tracers.NZTracer(f"kappa_{i}", nz[0], nz[1]))

        return tracers

    def compute_shear_maps(self, ikappas=None, lmax=None, replicate_octant=False):
        """Compute shear (gamma1, gamma2) maps from convergence maps.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Convergence map indices to process. If None, process all.
        lmax : int or None, optional
            Maximum multipole used in harmonic transforms. Defaults to
            ``3*nside - 1``.
        replicate_octant : bool, optional
            If True, recompute a_lm from octant maps replicated to full sky
            (only relevant when ``octant_only=True``).

        Returns
        -------
        list or dict
            - If ``ikappas`` is a single int: returns a list ``[gamma1_map, gamma2_map]`` as HealSparse maps.
            - If ``ikappas`` is ``None`` or an iterable: returns a dict keyed by kappa index
              with list values ``[gamma1_map, gamma2_map]``. Also writes maps to disk.
        """
        if lmax is None:
            lmax = 3 * self.shell_builder.nside - 1

        ell = np.arange(3 * self.shell_builder.nside)
        fl = np.zeros(len(ell))
        fl[1:] = -np.sqrt((ell[1:] + 2.) * (ell[1:] - 1.) / (ell[1:] * (ell[1:] + 1.)))

        _, ipix = self.get_mask(nest=False, return_ipix=True)

        gammas = {}
        for ikappa in self.tqdm(ikappas, desc="[compute_shear_maps]"):
            g1 = self.check_maps(read=True, fname=self.get_fname(ikappa, what='gamma1'))
            g2 = self.check_maps(read=True, fname=self.get_fname(ikappa, what='gamma2'))
            if g1 is not None and g2 is not None:
                gammas[ikappa] = [g1, g2]
                continue

            # Get harmonic coefficients of convergence map
            klm = self.compute_alms(ikappa, save=not (replicate_octant), replicate_octant=replicate_octant, pol=False, lmax=lmax)[ikappa]

            # Convert to shear
            glm_E = hp.almxfl(klm, fl)

            # Generate shear map (ring-ordered)
            g1, g2 = hp.sphtfunc.alm2map_spin([glm_E, np.zeros_like(glm_E)], nside=self.nside, spin=2, lmax=lmax)

            gamma1 = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            gamma2 = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

            gamma1.update_values_pix(ipix, g1[ipix].astype(self.dtype), nest=False)
            gamma2.update_values_pix(ipix, g2[ipix].astype(self.dtype), nest=False)

            gamma1.write(self.get_fname(ikappa, what='gamma1'), clobber=True)
            gamma2.write(self.get_fname(ikappa, what='gamma2'), clobber=True)

            gammas[ikappa] = [gamma1, gamma2]

        return gammas[ikappas] if type(ikappas) is int else gammas

    def _noise_cls(self, imap):
        """Return the (white) shape-noise power spectrum.

        Parameters
        ----------
        imap : int
            Tomographic-bin index.

        Returns
        -------
        float
            Noise level N_ell (constant in ell).
        """
        return self.sigma_e[imap]**2 / self.nbar_sr[imap]


class BornLensingBuilder(LensingBuilder):
    """Convergence-map builder in the Born approximation."""

    def __init__(self, shell_builder, nzs, kappa_dir=None, dtype=np.float32, multiple_planes=False, nbar_from_maps=False, **kwargs):
        """Initialize a Born-approximation lensing builder.

        Parameters
        ----------
        multiple_planes : bool, optional
            If True, compute weights using a multiple-lens-planes approximation.
            If False, use Limber-style integrals of tracer kernels per shell.
        nbar_from_maps : bool, optional
            If True, estimate the mean density from the shell maps themselves.
            This can affect large-scale modes.
        """
        super().__init__(shell_builder, nzs, kappa_dir=kappa_dir, dtype=dtype, **kwargs)

        self.multiple_planes = multiple_planes
        self.nbar_from_maps = nbar_from_maps

        if multiple_planes:
            mpart = self.reader.mass_particles

            area = hp.nside2pixarea(self.nside)

            self.mpart_over_area = mpart / area

            zmax = self.shell_builder.shell_edges_redshift()[-1]
            self.dndz_norm = [quad(lambda z: t.get_dndz(z), 0., zmax) if isinstance(t, pyccl.tracers.NzTracer) else np.nan for t in self.get_tracers_ccl()]
            self.zlens = [self.shell_builder.mean_z(ishell) for ishell in range(self.nshells)]

    def compute_weight(self, ishell, ikappa):
        """Compute the shell weight contributing to a given convergence map.

        Parameters
        ----------
        ishell : int
            Shell index.
        ikappa : int
            Convergence map index.

        Returns
        -------
        float
            Scalar weight applied to the shell density/contrast when summing into
            the convergence map.
        """
        tracer = self.get_tracers_ccl()[ikappa]

        if self.multiple_planes:
            zmax = self.shell_builder.shell_edges_redshift()[-1]
            zlens = self.zlens[ishell]

            if zlens is None:
                zlens = self.shell_builder.mean_z(ishell)

            d_A_lens = pyccl.angular_diameter_distance(self.cosmo, 1. / (1. + zlens))  # needed to obtain surface mass density in 1/Mpc^2

            if isinstance(tracer, pyccl.tracers.NzTracer):

                def integrand(zsource, zlens, tracer):
                    sigma_critical = pyccl.background.sigma_critical(self.cosmo, a_lens=1. / (1. + zlens), a_source=1. / (1. + zsource))  # in Msun / Mpc^2
                    return tracer.get_dndz(zsource) / sigma_critical

                w = quad(integrand, zlens, zmax, args=(zlens, tracer)) / self.dndz_norm[ikappa]

            elif isinstance(tracer, pyccl.tracers.CMBLensingTracer):
                zsource = tracer.z_source
                if zsource > zlens:
                    w = 1. / pyccl.background.sigma_critical(self.cosmo, 1. / (1. + zlens), 1. / (1. + zsource))  # in Msun / Mpc^2
                else:
                    w = 0.
            else:
                raise ValueError

            return w * self.mpart_over_area / d_A_lens**2
        else:
            chi_min, chi_max = self.shell_builder.shell_edges_comoving_distance()[ishell:ishell + 2]
            return quad(lambda chi: tracer.get_kernel(chi), chi_min, chi_max)

    def construct_maps(self, ikappas=None):
        """Construct and save convergence maps by summing weighted shells.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Indices of convergence maps to build. If None, build all.

        Returns
        -------
        healsparse.HealSparseMap or dict
            - If ``ikappas`` is a single int: returns a single ``HealSparseMap``.
            - If ``ikappas`` is ``None`` or an iterable: returns a dict keyed
              by kappa index with ``HealSparseMap`` values.
        """

        # First check if requested kappas are already present, otherwise rebuild all
        kappas = self.check_maps(ikappas, read=True)
        if type(ikappas) is int:
            if kappas is not None:
                return kappas
        else:
            if None not in kappas.values():
                return kappas

        # Loop over shells and accumulate weighted contributions
        kappas = {ikappa: healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype) for ikappa in self.iter(ikappas)}
        for ishell in self.tqdm(range(self.nshells), desc="[construct_maps]"):
            shell = (self.shell_builder.load_maps(ishell)).astype(self.dtype)
            if self.shell_builder.save_density:
                density = shell
            else:
                if self.nbar_from_maps:
                    s = shell[:]
                    nbar = np.mean(s[s > 0.])
                else:
                    nbar = self.shell_builder.pixel_density(ishell)
                nbar = self.dtype(nbar)
                if self.multiple_planes:
                    density = shell - nbar
                else:
                    density = shell / nbar - 1.

            for ikappa, kappa in kappas.items():
                w = self.compute_weight(ishell, ikappa)
                kappas[ikappa] = healsparse.operations.sum_union([density * w, kappa])

        for ikappa, kappa in kappas.items():
            fname = self.get_fname(ikappa)
            kappa.write(fname, clobber=True)

        return kappas[ikappas] if type(ikappas) is int else kappas
