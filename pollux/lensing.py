"""Convergence and shear maps from the shells, in the Born approximation.

For tomographic redshift distributions (:class:`BornLensingBuilder`), or one source plane per shell
(:class:`SnapshotBornLensingBuilder`).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import TYPE_CHECKING, Any, Iterable

import healpy as hp
import healsparse as hsp
import numpy as np
import pyccl

from .io import layout
from .io.layout import PolluxIO
from .cosmo import Tracer
from .map_builder import MapBuilder, MapSum
from .utils.numerics import quad
from .utils.sky import spin2_factor

if TYPE_CHECKING:
    from .shells import ShellBuilder

logger = logging.getLogger(__name__)


class LensingBuilder(MapBuilder):
    """Base class to build convergence and shear maps from matter shells.

    Parameters
    ----------
    shell_builder : pollux.shells.ShellBuilder
        Shell builder providing matter shells and geometry.
    nzs : {'snapshots'} or sequence
        If 'snapshots', produce one convergence map per shell edge (CMB-like).
        Otherwise, a list describing source n(z) per tomographic bin. Each entry
        can be a callable ``nz(z)`` or a tuple ``(z, n(z))``.
    tag : str or None, optional
        Name of the n(z): maps go to the ``'kappa'`` product of the shell builder's files with
        ``nz=tag`` (``kappa/{tag}/``, see :class:`pollux.io.layout.PolluxIO`), ``kappa/default/`` when None.
    dtype : data-type, optional
        Output dtype for kappa and shear maps.
    **kwargs
        Forwarded to :class:`pollux.map_builder.MapBuilder`. The footprint, padding,
        octant replication, monopole removal and workspace directory default to
        those of ``shell_builder``.
    """

    def __init__(self, shell_builder: ShellBuilder, nzs: str | list, tag: str | None = None, dtype: np.dtype = np.float32, **kwargs: Any) -> None:
        self.shell_builder = shell_builder
        self.tag = tag
        self.reader = shell_builder.reader
        self.cosmo = shell_builder.cosmo
        self.nshells = shell_builder.nshells

        # initialize nzs and number of kappa maps
        self.nzs = nzs
        self.nkappas = self.shell_builder.nshells if nzs == 'snapshots' else len(nzs)

        kwargs.setdefault('mask', shell_builder.footprint)
        for k in ('padding_deg', 'apotype', 'padding_nside', 'replicate_octant', 'remove_monopole'):
            kwargs.setdefault(k, getattr(shell_builder, k))
        super().__init__(
            nside=shell_builder.nside,
            dtype=dtype,
            io=shell_builder.io.with_options(nz=tag or 'default'),
            product='kappa',
            nmaps=self.nkappas,
            sacc_type="galaxy_convergence_cl",
            **kwargs)

        # tomographic bins in file names (kappa_bin{b}), shells for snapshot lensing (kappa_{iii})
        if nzs != 'snapshots':
            self._index_format = PolluxIO.BIN_INDEX

        # configuration of the products (checked against an existing one), and the n(z)
        self.io.write_config(self.product, self.config())
        if nzs != 'snapshots':
            self._save_nzs()

    def config(self) -> dict:
        """Return the configuration of the products, for their ``config.yaml`` (:func:`pollux.io.config.write_config`).

        Returns
        -------
        dict
            ``product`` (``'kappa'``), ``parameters`` (builder, and n(z): ``'snapshots'``, or their number
            and a hash of their values) and ``inputs`` (the shells).
        """
        if self.nzs == 'snapshots':
            nz_parameters = 'snapshots'
        else:
            # callable n(z) by their values on the grid of the tracers
            zz = np.linspace(0, self.shell_builder.shell_edges_redshift[-1], 1000)
            nz_values = [np.asarray(nz(zz), dtype=float).tolist() if callable(nz) else [np.asarray(a, dtype=float).tolist() for a in nz] for nz in self.nzs]
            nz_parameters = dict(nbins=self.nkappas, sha1=hashlib.sha1(repr(nz_values).encode()).hexdigest())
        parameters = dict(builder=type(self).__name__, nzs=nz_parameters)
        return dict(product='kappa', parameters=parameters, inputs=[self.shell_builder.save_dir])

    def _save_nzs(self) -> None:
        """Save the redshift distributions, unless saved already.

        Distributions tabulated on a common redshift grid are saved as text (``nzs.txt``: columns z, then
        one per bin, as the usual n(z) files), others as ``nzs.npy``.
        """
        tables = [nz for nz in self.nzs if not callable(nz)]
        common = len(tables) == len(self.nzs) and all(np.array_equal(np.asarray(nz[0]), np.asarray(tables[0][0])) for nz in tables)
        fname = self.io.path(self.product, 'nzs', ext='txt' if common else 'npy')
        if os.path.exists(fname):
            return
        if not common:
            self.io.write_npy(fname, np.array(self.nzs, dtype=object))
            return
        with layout.atomic_write(fname) as tmp:
            np.savetxt(tmp, np.column_stack([tables[0][0]] + [nz[1] for nz in tables]), header='z ' + ' '.join(f'nz_{i}' for i in range(len(tables))))
        logger.debug("Saved to file %s.", fname)

    def _smooth_distance(self, ikappa: int) -> float:
        """Return the comoving distance of the source plane of a per-shell convergence map, converting smoothing scales to angles."""
        if self.nzs != 'snapshots':
            raise ValueError(
                "smoothing in Mpc is only meaningful for snapshot lensing (nzs='snapshots'); "
                "for n(z) bins there is no single comoving distance to convert Mpc to an angle.")
        return self.shell_builder.shell_edges_comoving_distance[ikappa + 1]

    def compute_shear_maps(
            self,
            ikappas: int | Iterable[int] | None = None,
            smooth_Mpc: float | None = None,
            lmax: int | None = None,
            save_field_map: bool = True) -> list[hsp.HealSparseMap] | dict[int, list[hsp.HealSparseMap]]:
        """Compute shear (gamma1, gamma2) maps from convergence maps.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Convergence map indices to process. If None, process all.
        smooth_Mpc : float or None, optional
            Comoving Gaussian smoothing scale in Mpc of the convergence a_lm before
            deriving shear (angular width at the source plane, see
            :meth:`_smooth_sigma`; only for per-shell maps, ``nzs='snapshots'``).
            Shear maps are saved with the suffix ``_{smooth_Mpc}Mpc``.
        lmax : int or None, optional
            Maximum multipole used in harmonic transforms. Defaults to
            ``3*nside - 1``.
        save_field_map : bool, optional
            With ``smooth_Mpc``, also save the smoothed convergence map, from the same
            a_lm (as :meth:`~pollux.map_builder.MapBuilder.compute_maps_from_alms`).
            Default is True.

        Returns
        -------
        list or dict
            - If ``ikappas`` is a single int: returns a list ``[gamma1_map, gamma2_map]`` as HealSparse maps.
            - If ``ikappas`` is ``None`` or an iterable: returns a dict keyed by kappa index
              with list values ``[gamma1_map, gamma2_map]``. Also writes maps to disk.
        """
        if lmax is None:
            lmax = 3 * self.shell_builder.nside - 1

        # E-mode of the shear from the convergence
        fl = spin2_factor(3 * self.shell_builder.nside - 1)

        _, ipix = self.get_mask(nest=False, return_ipix=True, padded=True)
        save_field_map = save_field_map and smooth_Mpc is not None

        gammas = {}
        for ikappa in self.tqdm(ikappas, desc="[compute_shear_maps]"):
            g1 = self.check_maps(read=True, fname=self.path(ikappa, what='gamma1', smooth_Mpc=smooth_Mpc))
            g2 = self.check_maps(read=True, fname=self.path(ikappa, what='gamma2', smooth_Mpc=smooth_Mpc))
            field_done = not save_field_map or os.path.exists(self.path(ikappa, what=self.smoothed_field_what, smooth_Mpc=smooth_Mpc))
            if g1 is not None and g2 is not None and field_done:
                gammas[ikappa] = [g1, g2]
                continue

            # Get harmonic coefficients of convergence map, smoothed in place if requested (and the smoothed convergence saved)
            klm = self.compute_alms(ikappa, pol=False, lmax=lmax)
            if smooth_Mpc is not None:
                self._smooth_alms(ikappa, klm, smooth_Mpc, lmax)
                if save_field_map:
                    self._save_map(
                        self._alm2hspmap(klm, ipix, lmax, mean=self._alm_field_mean(ikappa)),
                        self.path(ikappa, what=self.smoothed_field_what, smooth_Mpc=smooth_Mpc))

            # Convert to shear
            glm_E = hp.almxfl(klm, fl)

            # Generate shear map (ring-ordered)
            g1, g2 = hp.sphtfunc.alm2map_spin([glm_E, np.zeros_like(glm_E)], nside=self.nside, spin=2, lmax=lmax)

            gamma1 = self._make_hspmap(ipix, g1[ipix])
            gamma2 = self._make_hspmap(ipix, g2[ipix])

            gamma1_fname = self.path(ikappa, what='gamma1', smooth_Mpc=smooth_Mpc)
            gamma2_fname = self.path(ikappa, what='gamma2', smooth_Mpc=smooth_Mpc)
            self._save_map(gamma1, gamma1_fname)
            self._save_map(gamma2, gamma2_fname)

            gammas[ikappa] = [gamma1, gamma2]

        return gammas[ikappas] if np.isscalar(ikappas) else gammas

    def _create_tracers(self) -> list[Tracer]:
        """Create one lensing tracer per source redshift distribution (callables tabulated on 1000 redshifts).

        Returns
        -------
        list of pollux.cosmo.Tracer
            ``'shear'`` tracers named ``kappa_{i}`` (length ``self.nmaps``).
        """
        z_edges = self.shell_builder.shell_edges_redshift
        tracers = []
        for i, nz in enumerate(self.nzs):
            if callable(nz):
                zz = np.linspace(0, z_edges[-1], 1000)
                tracers.append(Tracer(f"kappa_{i}", 'shear', z=zz, nz=nz(zz)))
            else:
                tracers.append(Tracer(f"kappa_{i}", 'shear', z=nz[0], nz=nz[1]))
        return tracers

    def matter_power_spectrum(self) -> pyccl.Pk2D | None:
        """Return the matter power spectrum of the shell builder (see :meth:`MapBuilder.matter_power_spectrum`)."""
        return self.shell_builder.matter_power_spectrum()


class BornLensingBuilder(LensingBuilder):
    """Convergence-map builder in the Born approximation.

    Parameters
    ----------
    shell_builder, nzs, tag, dtype
        See :class:`LensingBuilder`.
    multiple_planes : bool, optional
        If True, compute weights using a multiple-lens-planes approximation.
        If False, use Limber-style integrals of tracer kernels per shell.
    nbar_from_maps : bool, optional
        If True, estimate the mean density from the shell maps themselves.
        This can affect large-scale modes.
    **kwargs
        Forwarded to :class:`LensingBuilder`.
    """

    def __init__(
            self,
            shell_builder: ShellBuilder,
            nzs: str | list,
            tag: str | None = None,
            dtype: np.dtype = np.float32,
            multiple_planes: bool = False,
            nbar_from_maps: bool = False,
            **kwargs: Any) -> None:
        # options set first, for the configuration written by LensingBuilder
        self.multiple_planes = multiple_planes
        self.nbar_from_maps = nbar_from_maps
        super().__init__(shell_builder, nzs, tag=tag, dtype=dtype, **kwargs)

        if multiple_planes:
            mpart = self.reader.mass_particles

            area = hp.nside2pixarea(self.nside)

            self.mpart_over_area = mpart / area

            zmax = self.shell_builder.shell_edges_redshift[-1]
            self.dndz_norm = [quad(lambda z: t.ccl(self.cosmo).get_dndz(z), 0., zmax) if t.kind == 'shear' else np.nan for t in self.tracers]
            self.zlens = [self.shell_builder.mean_z_volume(ishell) for ishell in range(self.nshells)]

    def config(self) -> dict:
        """Return the configuration of the products (:meth:`LensingBuilder.config`), with the options of the builder.

        Options at their defaults are left out, so that the configurations of products made before
        they were recorded stay valid.
        """
        config = super().config()
        config['parameters'].update({k: True for k in ('multiple_planes', 'nbar_from_maps') if getattr(self, k)})
        return config

    def construct_maps(self, ikappas: int | Iterable[int] | None = None) -> hsp.HealSparseMap | dict[int, hsp.HealSparseMap]:
        """Construct and save convergence maps by summing weighted shells.

        Parameters
        ----------
        ikappas : int or array_like of int or None, optional
            Indices of convergence maps to build. If None, build all.

        Returns
        -------
        hsp.HealSparseMap or dict
            - If ``ikappas`` is a single int: returns a single ``HealSparseMap``.
            - If ``ikappas`` is ``None`` or an iterable: returns a dict keyed
              by kappa index with ``HealSparseMap`` values.
        """

        # First check if requested kappas are already present, otherwise rebuild all
        kappas = self.check_maps(ikappas, read=True)
        if np.isscalar(ikappas):
            if kappas is not None:
                return kappas
        else:
            if None not in kappas.values():
                return kappas

        # Loop over shells (read once) and accumulate weighted contributions to all kappas:
        # kappa += w * density, with density = shell / nbar - 1 (or shell - nbar for multiple planes)
        mask = self.get_mask(nest=True, padded=True)
        kappa_sums = {ikappa: MapSum(mask, self.nside_c, self.dtype) for ikappa in self.iter(ikappas)}

        # only read shells contributing to at least one requested kappa
        weights = {ishell: {ikappa: self.compute_weight(ishell, ikappa) for ikappa in kappa_sums} for ishell in range(self.nshells)}
        ishells = [ishell for ishell, ws in weights.items() if any(w != 0. for w in ws.values())]

        for ishell, shell in self.shell_builder.iter_maps(ishells, desc="[construct_maps]"):
            if self.nbar_from_maps:
                s = shell[:]
                nbar = np.mean(s[s > 0.])
            else:
                nbar = self.shell_builder.pixel_density(ishell)
            nbar = self.dtype(nbar)

            for ikappa, kappa_sum in kappa_sums.items():
                w = weights[ishell][ikappa]
                if w == 0.:
                    continue
                if self.multiple_planes:
                    kappa_sum.add(shell, scale=w, offset=-w * nbar)
                else:
                    kappa_sum.add(shell, scale=w / nbar, offset=-w)

        kappas = {ikappa: kappa_sum.to_map() for ikappa, kappa_sum in kappa_sums.items()}
        for ikappa, kappa in kappas.items():
            fname = self.path(ikappa)
            self._save_map(kappa, fname)

        return kappas[ikappas] if np.isscalar(ikappas) else kappas

    def _noise_cls(self, imap: int) -> float:
        r"""Return the white noise of a convergence map: the particle shot noise of its shells, propagated.

        The map sums the density contrast of the shells with weights ``w_i``, so its noise is
        ``sum_i w_i^2 N_i``, with ``N_i`` the shot noise of shell ``i`` (:meth:`ShellBuilder._noise_cls`,
        with its subsampling); with ``multiple_planes`` the weights apply to the particle counts, i.e. to the
        density contrast times the number of particles per pixel.

        Parameters
        ----------
        imap : int
            Convergence map index.

        Returns
        -------
        float
            Noise level N_ell (constant in ell).

        Notes
        -----
        .. math::

            N_\ell^{\kappa_j} = \sum_i w_{ij}^2\, N_\ell^{\delta_i},

        with :math:`w_{ij}` the weights of :meth:`compute_weight`, multiplied by :math:`\bar N_i` with ``multiple_planes``.
        """
        noise = 0.
        for ishell in range(self.nshells):
            w = self.compute_weight(ishell, imap)
            if w != 0.:
                if self.multiple_planes:
                    w = w * self.shell_builder.pixel_density(ishell)
                noise += w**2 * self.shell_builder._noise_cls(ishell)
        return noise

    def compute_weight(self, ishell: int, ikappa: int) -> float:
        r"""Compute the shell weight contributing to a given convergence map.

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

        Notes
        -----
        The convergence of map :math:`j` sums the shells :math:`i` with weights :math:`w_{ij}`. By default (Born
        approximation, integrating the lensing kernel :math:`q_j` of CCL over each shell),

        .. math::

            \kappa_j = \sum_i w_{ij}\, \delta_i, \qquad
            w_{ij} = \int_{\chi_i}^{\chi_{i+1}} q_j(\chi)\, d\chi, \qquad
            q_j(\chi) = \frac{3}{2} \Omega_m H_0^2 \frac{\chi}{a(\chi)} \int_\chi^\infty d\chi'\, n_j(\chi') \frac{\chi' - \chi}{\chi'},

        with :math:`\delta_i = N_i / \bar N_i - 1` the density contrast of the particle counts :math:`N_i` per pixel
        (:math:`\bar N_i` expected from the simulation, or the mean of the map with ``nbar_from_maps``). With
        ``multiple_planes``, each shell is a lens plane at its volume-weighted mean redshift :math:`z_i`, of surface
        density :math:`m_p (N_i - \bar N_i) / (\Omega_{\rm pix} D_A(z_i)^2)`, with :math:`m_p` the mass of a particle
        and :math:`\Omega_{\rm pix}` the pixel area:

        .. math::

            \kappa_j = \sum_i w_{ij} (N_i - \bar N_i), \qquad
            w_{ij} = \frac{m_p}{\Omega_{\rm pix} D_A(z_i)^2}
            \frac{\int_{z_i}^{z_{\max}} n_j(z_s)\, \Sigma_{\rm crit}^{-1}(z_i, z_s)\, dz_s}{\int_0^{z_{\max}} n_j(z_s)\, dz_s},

        or :math:`w_{ij} = m_p / (\Omega_{\rm pix} D_A(z_i)^2 \Sigma_{\rm crit}(z_i, z_s))` for a single source plane
        at :math:`z_s > z_i` (zero otherwise).
        """
        tracer = self.tracers[ikappa]

        if self.multiple_planes:
            zmax = self.shell_builder.shell_edges_redshift[-1]
            zlens = self.zlens[ishell]

            if zlens is None:
                zlens = self.shell_builder.mean_z_volume(ishell)

            d_A_lens = pyccl.angular_diameter_distance(self.cosmo, 1. / (1. + zlens))  # needed to obtain surface mass density in 1/Mpc^2

            if tracer.kind == 'shear':

                def integrand(zsource: float, zlens: float, tracer: pyccl.Tracer) -> float:
                    # source n(z) over the critical surface density
                    sigma_critical = pyccl.background.sigma_critical(self.cosmo, a_lens=1. / (1. + zlens), a_source=1. / (1. + zsource))  # in Msun / Mpc^2
                    return tracer.get_dndz(zsource) / sigma_critical

                w = quad(integrand, zlens, zmax, args=(zlens, tracer.ccl(self.cosmo))) / self.dndz_norm[ikappa]

            elif tracer.kind == 'convergence':
                zsource = tracer.z_source
                if zsource > zlens:
                    w = 1. / pyccl.background.sigma_critical(self.cosmo, 1. / (1. + zlens), 1. / (1. + zsource))  # in Msun / Mpc^2
                else:
                    w = 0.
            else:
                raise ValueError

            return w * self.mpart_over_area / d_A_lens**2
        else:
            chi_min, chi_max = self.shell_builder.shell_edges_comoving_distance[ishell:ishell + 2]
            tracer_ccl = tracer.ccl(self.cosmo)
            return quad(lambda chi: tracer_ccl.get_kernel(chi), chi_min, chi_max)


class SnapshotBornLensingBuilder(BornLensingBuilder):
    """Born-approximation lensing builder with one convergence map per shell snapshot.

    Each map uses a CMB-like lensing kernel sourced at the redshift edge of its
    shell, equivalent to ``BornLensingBuilder(..., nzs='snapshots')``. Prefer
    constructing via :meth:`pollux.shells.ShellBuilder.snapshot_lensing_builder`.

    Parameters
    ----------
    shell_builder : pollux.shells.ShellBuilder
        Shell builder providing matter shells and geometry.
    tag : str or None, optional
        Name of the output subdirectory, ``kappa/{tag}`` (default ``kappa/snapshots``).
    **kwargs
        Forwarded to :class:`BornLensingBuilder` (e.g. ``multiple_planes``).
    """

    def __init__(self, shell_builder: ShellBuilder, tag: str | None = 'snapshots', **kwargs: Any) -> None:
        super().__init__(shell_builder, nzs='snapshots', tag=tag, **kwargs)

    def _create_tracers(self) -> list[Tracer]:
        """Create one convergence tracer per shell, with its source plane at the shell's far edge.

        Returns
        -------
        list of pollux.cosmo.Tracer
            ``'convergence'`` tracers named ``kappa_{i}``.
        """
        return [Tracer(f"kappa_{i}", 'convergence', z_source=z_source) for i, z_source in enumerate(self.shell_builder.shell_edges_redshift[1:])]
