import itertools
import warnings
import os

import healpy as hp
import healsparse
import numba
import numpy as np
import pyccl
import sacc
import yaml
from astropy.io import fits
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm, trange
from parallel_sort import argsort as pargsort
from collections.abc import Iterable

from .map_builder import MapBuilder
from .utils import quad, split_bins, arange_endpoint, inter_diff
from .reader import DummyReader
from .lensing import BornLensingBuilder


class ShellBuilder(MapBuilder):
    """Construct HealSparse shell maps from simulation particle data.

    A *shell* is a radial bin in comoving distance (or equivalently redshift).
    For each shell, this builder produces a HealSparse map of particle counts
    (or optionally density / density contrast), computed by reading simulation
    sub-volumes, applying optional isometries, and replicating the periodic box
    to cover the shell volume.

    This class also provides utilities to:
    - define shell edges from snapshot redshifts,
    - compute expected pixel/sky densities,
    - combine per-subvolume shell maps into final maps,
    - compute tidal-tensor maps from density maps.

    Parameters
    ----------
    nshells : int
        Number of shells to build.
    reader : pollux.reader.Reader
        Simulation reader returning particle positions for each snapshot/sub-volume.
    nside : int
        Healpix nside of the shell maps.
    save_dir : str or None
        Output directory where maps and metadata are written.
    kind : {'snapshot', 'lightcone'}, optional
        Shell construction mode. 'snapshot' builds shells out of snapshot boxes;
        'lightcone' uses reader-provided shell edges (typically already binned).
    seed : int, optional
        Random seed used to draw isometries/rotflips and for subsampling.
    dtype : data-type, optional
        dtype used to store shell map values.
    save_density : bool, optional
        If True, save density maps (HealSparse dense output via
        ``generate_healpix_map``). If False, save count maps and expose density
        contrast via :meth:`_field_map`.
    apply_grid_isometry : bool or sequence, optional
        If True, apply one random affine isometry per shell to the underlying
        particle distribution: ``xyz' = A(xyz + b)`` with random orthogonal-ish
        rotation/permutation/flip (A) and translation (b).
        If an iterable of length ``nshells`` is provided, it is interpreted as
        user-provided ``[(A0, b0), (A1, b1), ...]``.
    group_isometries : bool or float or None, optional
        If set, reuse the same isometry across multiple consecutive shells until
        a maximum radial width is reached. If True, uses ``dchi_max = boxsize``.
        If float, uses ``dchi_max = group_isometries * boxsize``.
    apply_replicate_rotflip : bool or dict, optional
        If True, apply HACC-style discrete rotations (6) and sign flips (8) to
        each replicate (48 configs). If a dict is provided, it maps replicate
        indices ``(ix, iy, iz)`` to a rotation code in ``[0, 48)``.
        This option is mutually exclusive with ``apply_grid_isometry``.
    subsample_box : float or dict or None, optional
        Particle subsampling configuration.
        - If float in (0, 1], keep that fraction of particles per sub read.
        - If dict with key ``'densitymax_z'``: choose ratios per shell to keep
          per-pixel S/N comparable to the shell at that redshift.
        - If dict with key ``'clmax_r'``: choose ratios per shell so that shot
          noise is below theory C_ell at lmax by the requested factor.

    Other Parameters
    ----------------
    **kwargs
        Forwarded to :class:`pollux.map_builder.MapBuilder`.

    Notes
    -----
    - ``apply_grid_isometry`` and ``apply_replicate_rotflip`` do not commute;
      only one should be used at a time.
    - For octant-only footprints, particles outside x>0, y>0, z>0 are discarded.
    """
    
    def __init__(
            self,
            nshells,
            reader,
            nside,
            save_dir,
            kind='snapshot',
            seed=12345,
            dtype=np.uint64,
            save_density=False,
            apply_grid_isometry=False,
            group_isometries=None,
            apply_replicate_rotflip=False,
            subsample_box=None,
            **kwargs):
        self.nshells = nshells
        self.reader = reader
        self.cosmo = reader.cosmo
        self.kind = kind
        self.seed = seed
        self.save_density = save_density

        # Options for isometries and replicate rotation/flips
        assert (not apply_grid_isometry) or (
            not apply_replicate_rotflip
        )  # they do not commute and grid isometry is slow and can be applied once, whereas replicate_rotflip is fast but must be applied to each replicate
        self.apply_grid_isometry = apply_grid_isometry
        self.apply_replicate_rotflip = apply_replicate_rotflip

        base_fname = f'shell_{self.reader.kind}_seed{self.seed}'

        if dtype == np.int32:
            warnings.warn(f"The max value for {dtype} might not be enough : {np.iinfo(dtype)}")

        super().__init__(nside=nside, dtype=dtype, save_dir=save_dir, base_fname=base_fname, nmaps=self.nshells, sacc_type="galaxy_density_cl", **kwargs)

        # Particle subsampling
        assert type(subsample_box) is dict or type(subsample_box) is float or subsample_box is None
        if type(subsample_box) is dict:
            keys = subsample_box.keys()
            assert 'densitymax_z' in keys or 'clmax_r' in keys
        self.subsample_box = subsample_box

        # Draw random grid isometries
        if isinstance(self.apply_grid_isometry, Iterable):
            # user-provided isometries
            assert len(self.apply_grid_isometry) == self.nshells
            for i in range(self.nshells):
                A, b = self.apply_grid_isometry[i]
                assert A.shape == (3, 3)
                assert b.shape == (3,)
            self.Abs = self.apply_grid_isometry
            self.apply_grid_isometry = True
        elif self.apply_grid_isometry:
            # draw random isometries
            self.Abs = self.random_isometries(group_isometries)
            self.apply_grid_isometry = True
        else:
            # no isometries
            self.Abs = [(np.zeros((3, 3), dtype=self.dtype), np.zeros(3, dtype=self.dtype)) for _ in range(self.nshells)]
            self.apply_grid_isometry = False

        # Draw random replicate rotation/flips
        if type(self.apply_replicate_rotflip) is dict:
            # user-provided replicate rotation/flips
            warnings.warn("Shell builder using user-provided replicate rot/flip: make sure not to miss any!")
            self.replicate_rotflip = self.apply_replicate_rotflip
            self.apply_replicate_rotflip = True
        elif self.apply_replicate_rotflip:
            # draw random replicate rotation/flips
            ## first get number of replicates per direction
            rmax = self.shell_edges_comoving_distance()[-1]
            boxsize = self.reader.boxsize()
            nreplicate = int(np.ceil(rmax / boxsize)) + 2  # not too sure about that 2, but it's fast to check...

            ## then draw random rotation/flips (6 rotations, 8 flips, so max 48)
            rng = np.random.default_rng(self.seed)
            self.replicate_rotflip = {}
            for ix in range(-nreplicate, nreplicate + 1):
                for iy in range(-nreplicate, nreplicate + 1):
                    for iz in range(-nreplicate, nreplicate + 1):
                        # self.replicate_rotflip[ix, iy, iz] = rng.randint(48)
                        self.replicate_rotflip[ix, iy, iz] = rng.integers(low=0, high=48)
            self.apply_replicate_rotflip = True
        else:
            # no replicate rotation/flips
            self.replicate_rotflip = {}
            self.apply_replicate_rotflip = False

        # Convert replicate_rotflip to array for numba compatibility
        if len(self.replicate_rotflip) > 0:
            replicate_rotflip = np.empty((len(self.replicate_rotflip), 4), dtype=np.int64)
            for i, (k, v) in enumerate(self.replicate_rotflip.items()):
                replicate_rotflip[i, :3] = k
                replicate_rotflip[i, 3] = v
            self.replicate_rotflip = replicate_rotflip
        else:
            self.replicate_rotflip = np.zeros((1, 4), dtype=np.int64)

        # Try getting shell information from reader, if not dummy.
        try:
            # make sure snapshots are larger than the shells
            z_shell_bins = self.shell_edges_redshift()
            chi_shell_bins = self.shell_edges_comoving_distance()
            boxsize = self.reader.boxsize()
            diff = np.diff(chi_shell_bins)
            print(f"  - Box size = {boxsize:.2f} Mpc")
            if not np.all(diff < boxsize):
                message = "Some shells are deeper than the simulation box!\n"
                for ishell in range(self.nshells):
                    if diff[ishell] > boxsize:
                        message += f"Shell {ishell} ({z_shell_bins[ishell]:.2f}<z<{z_shell_bins[ishell+1]:.2f}) has width {diff[ishell]:.2f} Mpc\n"
                warnings.warn(message)
        except NotImplementedError:
            warnings.warn("Reader does not implement `shell_edges_redshift` or `shell_edges_comoving_distance` or `boxsize.")
            assert isinstance(reader, DummyReader)

        # Dump useful metadata
        if self.save_dir is not None:
            try:
                self.save_info()
            except:
                print("Could not save the metadata info.")

    def shell_edges_comoving_distance(self):
        """Return comoving distances to shell edges.

        Returns
        -------
        ndarray
            Comoving radial distances (Mpc) of all shell edges.
            Length is ``nshells + 1``.
        """
        return pyccl.comoving_radial_distance(self.cosmo, 1. / (1 + self.shell_edges_redshift()))

    def shell_edges_redshift(self):
        """Return redshifts of shell edges.

        Returns
        -------
        ndarray
            Redshifts of all shell edges. Length is ``nshells + 1`` for
            ``kind='snapshot'`` and depends on the reader for ``kind='lightcone'``.

        Raises
        ------
        ValueError
            If ``self.kind`` is not recognized, or if ``shell_edges_match`` is invalid.
        """
        if self.kind == 'snapshot':
            if not hasattr(self, 'z_shell_bins'):
                if self.shell_edges_match == 'center_redshift':
                    func = lambda z: z
                    ifunc = func
                elif self.shell_edges_match == 'center_scale_factor':
                    func = lambda z: 1. / (1. + z)
                    ifunc = lambda a: 1. / a - 1.
                elif self.shell_edges_match == 'center_comoving_distance':
                    func = lambda z: 1e-3 * pyccl.comoving_radial_distance(self.cosmo, 1. / (1. + z))
                    ifunc = None
                elif self.shell_edges_match == 'center_comoving_volume':
                    func = lambda z: (1e-3 * pyccl.comoving_radial_distance(self.cosmo, 1. / (1. + z)))**3
                    ifunc = None
                else:
                    raise ValueError

                self.z_shell_bins = split_bins(self.reader.zsnapshots, func=func, ifunc=ifunc, zmax=self.reader.zmax)

            return self.z_shell_bins

        elif self.kind == 'lightcone':
            return self.reader.zsnapshots

        else:
            raise ValueError

    def shell_info(self, fname=None):
        """Compute and optionally save a structured summary of shell metadata.

        Parameters
        ----------
        fname : str or None, optional
            If provided, write the table to this filename. If it contains
            ``'.txt'``, the output is plain text via :func:`numpy.savetxt`,
            otherwise it is saved as ``.npy`` via :func:`numpy.save`.

        Returns
        -------
        ndarray
            Structured numpy array with fields including shell index, edge
            redshifts, comoving distances, and snapshot identifiers.
        """
        z_shell_bins = self.shell_edges_redshift()
        names = ['ishell', 'zsnapshot', 'zmin', 'zmax', 'chisnapshot', 'chimin', 'chimax', 'snapshot_id']
        dtypes = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'int']
        out = np.array(
            list(
                zip(
                    np.arange(self.nshells), self.reader.zsnapshots, z_shell_bins[:-1], z_shell_bins[1:],
                    self.cosmo.comoving_radial_distance(1 / (1 + self.reader.zsnapshots)), self.cosmo.comoving_radial_distance(1 / (1 + z_shell_bins[:-1])),
                    self.cosmo.comoving_radial_distance(1 / (1 + z_shell_bins[1:])), self.reader.snapshots)),
            dtype=list(zip(names, dtypes)))
        if fname is not None:
            if '.txt' in fname:
                np.savetxt(fname, out, header=' '.join(names), fmt='%8i %16.8f %16.8f %16.8f %16.8f %16.8f %16.8f %8i')
            else:
                np.save(fname, out)
        return out

    def save_info(self, ext='npy'):
        """Write shell metadata, cosmology, and simulation info to disk.

        This method writes:
        - ``{base_fname}_shell_info.{ext}`` : shell edge and density summaries
        - ``cosmo_ccl.yaml`` : CCL cosmology (when serializable)
        - ``sim_info.yaml`` : key simulation and map parameters

        Parameters
        ----------
        ext : {'npy', 'txt'}, optional
            Output format for shell summary.

        Returns
        -------
        None
        """
        # Shell info for later reference and compare to existing as a safe check
        assert ext in ['npy', 'txt']
        fname = os.path.join(self.save_dir, self.base_fname + '_shell_info.' + ext)

        z_shell_bins = self.shell_edges_redshift()
        pixel_density = [self.pixel_density(ishell) for ishell in range(self.nshells)]
        sky_density = [self.sky_density(ishell) for ishell in range(self.nshells)]
        if self.kind == 'snapshot':
            names = ['ishell', 'zsnapshot', 'zmin', 'zmax', 'chisnapshot', 'chimin', 'chimax', 'snapshot_id', 'pixel_density', 'sky_density']
            dtypes = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'int', 'float', 'float']
            shell_info = np.array(
                list(
                    zip(
                        np.arange(self.nshells), self.reader.zsnapshots, z_shell_bins[:-1], z_shell_bins[1:],
                        self.cosmo.comoving_radial_distance(1 / (1 + self.reader.zsnapshots)), self.cosmo.comoving_radial_distance(1 / (1 + z_shell_bins[:-1])),
                        self.cosmo.comoving_radial_distance(1 / (1 + z_shell_bins[1:])), self.reader.snapshots, pixel_density, sky_density)),
                dtype=list(zip(names, dtypes)))
            fmt = '%8i %16.8f %16.8f %16.8f %16.8f %16.8f %16.8f %8i %16.8f %16.8f'
        elif self.kind == 'lightcone':
            names = ['ishell', 'zmin', 'zmax', 'pixel_density', 'sky_density']
            dtypes = ['int', 'float', 'float', 'float', 'float']
            shell_info = np.array(
                list(zip(np.arange(self.nshells), z_shell_bins[:-1], z_shell_bins[1:], pixel_density, sky_density)), dtype=list(zip(names, dtypes)))
            fmt = '%8i %16.8f %16.8f %16.8f %16.8f'
        else:
            raise ValueError

        if ext == 'npy':
            if os.path.exists(fname):
                shell_info_existing = np.load(fname)
                try:
                    for name, dtype in zip(names, dtypes):
                        if dtype == 'int':
                            np.testing.assert_array_equal(
                                shell_info[name], shell_info_existing[name], err_msg=f"Shell info {name} does not match existing file {fname}.")
                        elif dtype == 'float':
                            np.testing.assert_allclose(
                                shell_info[name], shell_info_existing[name], rtol=1e-5, err_msg=f"Shell info {name} does not match existing file {fname}.")
                except TypeError:
                    warnings.warn('Shell info file exists, different type!')
            else:
                np.save(fname, shell_info)
        else:
            np.savetxt(fname, shell_info, header=' '.join(names), fmt=fmt)

        # Save cosmology
        fname = os.path.join(self.save_dir, "cosmo_ccl.yaml")
        if not os.path.exists(fname):
            try:
                self.reader.cosmo.write_yaml(os.path.join(self.save_dir, "cosmo_ccl.yaml"))
            except ValueError:
                warnings.warn("Cannot write cosmo yaml file, likely due to using non-serializable options (eg P(k)).")

        # Save simulation and maps info
        data = {
            "nparticles": self.reader.nparticles,
            "boxsize": self.reader.boxsize(),
            "mass_particles": self.reader.mass_particles,
            "box_comoving_density": self.reader.box_comoving_density(),
            "kind": self.reader.kind,
            "snapshots": self.reader.snapshots.tolist(),
            "nsnapshots": self.reader.nsnapshots,
            "zsnapshots": self.reader.zsnapshots.tolist(),
            "nshells": self.nshells,
            "nside": self.nside,
            "nside_c": self.nside_c,
            "npix": self.npix,
        }
        fname = os.path.join(self.save_dir, 'sim_info.yaml')
        if not os.path.exists(fname):
            yaml.dump(data, open(fname, "w"))

    def snapshot_lensing_builder(self, kappa_dir='kappa_zsnapshots'):
        """Return a snapshot-lensing builder (CMB-like) using shell snapshots.

        Parameters
        ----------
        kappa_dir : str, optional
            Output directory name (relative to save_dir) for kappa maps.

        Returns
        -------
        pollux.lensing.BornLensingBuilder
            Lensing builder configured with ``nzs='zsnapshots'``.
        """
        return BornLensingBuilder(self, nzs='zsnapshots', kappa_dir=kappa_dir, wsp_dir=self.wsp_dir)

    def _dndz(self, zz):
        """Return a normalized redshift distribution proportional to dV/dz.

        Parameters
        ----------
        zz : ndarray
            Redshift sampling points.

        Returns
        -------
        ndarray
            Values proportional to comoving volume element per unit redshift,
            normalized to unit integral over the provided ``zz`` grid.

        Notes
        -----
        This is used to build shell NumberCounts tracers with volume weighting.
        """
        aa = 1. / (1. + zz)
        dVdz = pyccl.comoving_angular_distance(self.cosmo, aa)**2 / pyccl.background.h_over_h0(self.cosmo, aa)
        return dVdz / np.trapz(dVdz, zz)  # rough normalization

    def _create_tracers_ccl(self, Nz=1000):
        """Create one CCL NumberCounts tracer per shell.

        Parameters
        ----------
        Nz : int, optional
            Number of redshift samples used within each shell.

        Returns
        -------
        list
            List of :class:`pyccl.NumberCountsTracer` objects of length ``nshells``.
        """
        z_shell_bins = self.shell_edges_redshift()

        tracers = []
        for ishell in range(self.nshells):
            zz = np.linspace(z_shell_bins[ishell], z_shell_bins[ishell + 1], Nz)
            dndz = self._dndz(zz)
            tracers.append(pyccl.NumberCountsTracer(self.cosmo, has_rsd=False, dndz=(zz, dndz), bias=(zz, np.ones_like(zz))))

        return tracers

    def _create_tracers_sacc(self, Nz=1000):
        """Create one SACC NZ tracer per shell.

        Parameters
        ----------
        Nz : int, optional
            Number of redshift samples used within each shell.

        Returns
        -------
        list
            List of :class:`sacc.tracers.NZTracer` objects of length ``nshells``.
        """
        z_shell_bins = self.shell_edges_redshift()

        tracers = []
        for ishell in range(self.nshells):
            zz = np.linspace(z_shell_bins[ishell], z_shell_bins[ishell + 1], Nz)
            dndz = self._dndz(zz)
            tracers.append(sacc.tracers.NZTracer(f"density_{ishell}", zz, dndz))

        return tracers

    def mean_z(self, ishell):
        """Compute the volume-weighted mean redshift of a shell.

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        float
            Mean redshift in the shell, weighted by the comoving volume element.
        """
        z_shell_bins = self.shell_edges_redshift()
        zmin = z_shell_bins[ishell]
        zmax = z_shell_bins[ishell + 1]

        dVdz = lambda z: pyccl.comoving_angular_distance(self.cosmo, 1. / (1. + z))**2 / pyccl.background.h_over_h0(self.cosmo, 1. / (1. + z))

        return quad(lambda z: z * dVdz(z), zmin, zmax) / quad(dVdz, zmin, zmax)

    def random_isometries(self, group_isometries=None, get_shell_identifier=False):
        """Generate random affine isometries for shells.

        Parameters
        ----------
        group_isometries : bool or float or None, optional
            If provided, reuse a given isometry across multiple shells until the
            cumulative shell width exceeds ``dchi_iso_max`` (see class docstring).
        get_shell_identifier : bool, optional
            If True, also return a mapping from shell index to isometry id.

        Returns
        -------
        list
            List of length ``nshells`` of ``(A, b)`` pairs where:
            - ``A`` is a (3,3) matrix (rotation/permutation/flip),
            - ``b`` is a (3,) translation vector.
        dict, optional
            Only if ``get_shell_identifier=True``. Maps ``ishell -> iso_id``.
        """
        rng = np.random.default_rng(self.seed)
        boxsize = self.reader.boxsize()
        chi_shell_bins = self.shell_edges_comoving_distance()

        # maximum width a given isometry can be applied to
        if group_isometries is not None:
            # if grouped, fix as function of box size
            if type(group_isometries) is bool:
                assert group_isometries
                dchi_iso_max = boxsize
            if type(group_isometries) is float:
                dchi_iso_max = boxsize * group_isometries
        else:
            # if not grouped, set to 0 to force drawing one isometry per shell
            dchi_iso_max = 0.

        Abs = []
        n_isometries = 0
        dchi_iso = np.inf
        shell_identifier = {}

        for ishell in range(self.nshells):
            dchi_shell = chi_shell_bins[ishell + 1] - chi_shell_bins[ishell]
            if dchi_iso + dchi_shell > dchi_iso_max:
                # if grouped shell width is larger than maximum width for isometry, draw a new one
                A, b = random_isometry(rng, boxsize)
                n_isometries += 1
                dchi_iso = 0.
            dchi_iso += dchi_shell
            Abs.append([A.astype(self.reader.dtype), b.astype(self.reader.dtype)])
            shell_identifier[ishell] = n_isometries

        if group_isometries is not None:
            print(f"  - Using {n_isometries} isometries grouped within {dchi_iso_max:.2f} Mpc.")
        else:
            print(f"  - Using {n_isometries} isometries.")

        if get_shell_identifier:
            return Abs, shell_identifier
        else:
            return Abs

    def estimate_replicates(self, ishell):
        """Estimate how many periodic-box replicates are needed (volume ratio).

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        float
            Approximate number of box volumes required to fill the shell volume,
            accounting for octant-only mode if enabled.
        """
        chi_shell_bins = self.shell_edges_comoving_distance()
        box_volume = self.reader.boxsize()**3

        shell_volume = 4. / 3. * np.pi * (chi_shell_bins[ishell + 1]**3 - chi_shell_bins[ishell]**3)
        if self.octant_only:
            shell_volume /= 8.

        return shell_volume / box_volume

    def get_replicates(self, **kwargs):
        """Compute replicate indices intersecting each (snapshot, sub-volume) job.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`construct_maps_sub` (e.g. ``fast=...``), with
            ``get_replicates_only=True`` enforced.

        Returns
        -------
        list
            List of tuples ``(ijob, replicates)`` where ``replicates`` is a list
            of (ix, iy, iz) replicate indices intersecting the shell.
        """
        replicates = []
        for ijob in trange(self.reader.njobs, desc='[get_replicates]'):
            replicates.append((ijob, self.construct_maps_sub(None, None, ijob=ijob, get_replicates_only=True, **kwargs)))
        return replicates

    def subsample_ratio(self, ishell):
        """Return the particle subsampling ratio applied for a given shell.

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        float
            Fraction of particles retained when reading sub-volumes for this shell.

        Raises
        ------
        ValueError
            If ``subsample_box`` has an unsupported type or key.
        """
        if type(self.subsample_box) is dict:
            if not hasattr(self, 'subsample_ratios'):
                if 'densitymax_z' in self.subsample_box.keys():
                    # estimated number of particles per pixel
                    ngal_pix = np.array([self.estimate_replicates(ishell) for ishell in range(self.nshells)]) * self.reader.nparticles / self.npix
                    if self.octant_only:
                        ngal_pix *= 8.
                    snr_pix = 1. / np.sqrt(ngal_pix)  # S/N per pixel
                    # S/N at redshift subsample_box_auto_z
                    snr_pix_cut = np.interp(self.subsample_box['densitymax_z'], [self.mean_z(ishell) for ishell in range(self.nshells)], snr_pix)
                    # fraction of particles to maintain S/N of z>subsample_box_auto_z
                    self.subsample_ratios = np.minimum(1., (snr_pix / snr_pix_cut)**2)
                elif 'clmax_r' in self.subsample_box.keys():
                    # estimated number of particles per pixel
                    ngal_pix = np.array([self.estimate_replicates(ishell) for ishell in range(self.nshells)]) * self.reader.nparticles / self.npix
                    if self.octant_only:
                        ngal_pix *= 8.
                    # compute theory Cl's
                    cl = self.predict_cls(auto_only=True, approximate=True)[1]
                    # fix particle density such that shot-noise is 100 times lower than Cl at lmax
                    nbar_th = self.subsample_box['clmax_r'] / np.array([cl[i, i][-1] for i in range(self.nshells)])
                    # convert to desired number of particles per pixel
                    ngal_th = nbar_th * hp.nside2pixarea(self.nside, degrees=False)
                    self.subsample_ratios = np.minimum(1., ngal_th / ngal_pix)
                else:
                    raise ValueError
            return self.subsample_ratios[ishell]
        elif type(self.subsample_box) is float:
            return self.subsample_box
        elif self.subsample_box is None:
            return 1.
        else:
            raise ValueError

    def box_comoving_density(self, ishell):
        """Return the comoving particle number density after subsampling.

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        float
            Effective comoving particle density (Mpc^-3).
        """
        return self.reader.box_comoving_density() * self.subsample_ratio(ishell)

    def pixel_density(self, ishell):
        """Return the expected mean number of particles per pixel in a shell.

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        float
            Mean particles per pixel expected from comoving density and shell volume.
        """
        chi_shell_bins = self.shell_edges_comoving_distance()

        chi_max = chi_shell_bins[ishell + 1]
        chi_min = chi_shell_bins[ishell]
        pixel_volume = 4. * np.pi / 3. * (chi_max**3 - chi_min**3) / self.npix

        return pixel_volume * self.box_comoving_density(ishell)

    def sky_density(self, ishell, degrees=False):
        """Return the expected mean number of particles per unit solid angle.

        Parameters
        ----------
        ishell : int
            Shell index.
        degrees : bool, optional
            If True, return per square degree. If False, return per steradian.

        Returns
        -------
        float
            Mean surface density (particles / sr or particles / deg^2).
        """
        area = hp.nside2pixarea(self.nside, degrees=degrees)
        return self.pixel_density(ishell) / area

    def combine_maps_subs(self, ishells=None, add_orphans=False):
        """Sum per-subvolume shell maps into final per-shell maps.

        Parameters
        ----------
        ishells : int or array_like of int or None, optional
            Shell indices to combine. If None, combines all shells.
        add_orphans : bool, optional
            If True, also add an ``orphan`` map stored under ``sub='orphan'``.
            This is used by some interpolation/lightcone workflows.

        Returns
        -------
        healsparse.HealSparseMap or dict
            - If ``ishells`` is a single int: returns a single combined
              ``HealSparseMap``.
            - If ``ishells`` is ``None`` or an iterable: returns a dict keyed
              by shell index with combined ``HealSparseMap`` values.
        """
        shells = {}

        _, ipix = self.get_mask(nest=True, return_ipix=True)
        for ishell in self.tqdm(ishells, desc="[combine_maps_subs]"):
            fname = self.get_fname(ishell)
            shell = self.check_maps(read=True, fname=fname)
            if shell is not None:
                shells[ishell] = shell
                continue

            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            shell.update_values_pix(ipix, self.dtype(0), nest=True)  # initialize to zero on valid pixels
            isubs = self.reader.get_isubs_for_snapshot(ishell)
            for isub in tqdm(isubs, desc=f"[combine_maps_subs: shell {ishell}]", leave=False):
                tmp_shell = self.load_maps(ishell, sub=isub)
                shell = healsparse.operations.sum_union([shell, tmp_shell])

            if add_orphans:
                print('Adding the orphan as well!')
                tmp_shell = healsparse.HealSparseMap.read(self.get_fname(ishell, sub='orphan'))
                shell = healsparse.operations.sum_union([shell, tmp_shell])

            shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)

            shells[ishell] = shell

        return shells[ishells] if type(ishells) is int else shells

    def construct_maps(self, imaps=None, **kwargs):
        """Construct all requested shell maps.

        This constructs sub-maps for each (shell, subvolume) and then calls
        :meth:`combine_maps_subs` to produce final shell maps.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Shell indices to construct. If None, constructs all shells.
        **kwargs
            Forwarded to :meth:`construct_maps_sub`.

        Returns
        -------
        healsparse.HealSparseMap or dict
            - If ``imaps`` is a single int: returns a single combined
              ``HealSparseMap``.
            - If ``imaps`` is ``None`` or an iterable: returns a dict keyed
              by shell index with combined ``HealSparseMap`` values.
        """
        # first construct shells on simulation subs
        for ishell in self.tqdm(imaps, desc="[construct_maps]"):
            isubs = self.reader.get_isubs_for_snapshot(ishell)
            for isub in tqdm(isubs, desc=f"[construct_maps: shell {ishell}]", leave=False):
                self.construct_maps_sub(ishell, isub, **kwargs)

        # then combine into final shells
        return self.combine_maps_subs(imaps)

    def _field_map(self, ishell, nest=True):
        """Return a dense Healpix map for one shell.

        Parameters
        ----------
        ishell : int
            Shell index.
        nest : bool, optional
            Output ordering.

        Returns
        -------
        ndarray
            If ``save_density=True``, returns a dense healpix density map derived
            from the HealSparse map.
            Otherwise returns a dense healpix *density contrast* map delta, with:
            - delta = -1 on masked (valid-but-empty) pixels by convention,
            - delta = shell_count/pixel_density - 1 on pixels containing data.
        """
        shell = self.load_maps(ishell)

        if self.save_density:
            return shell.generate_healpix_map(nest=nest)
        else:
            spix = shell.valid_pixels
            s = shell[spix]

            if not nest:
                spix = hp.nest2ring(self.nside, spix)

            _, ipix = self.get_mask(nest=nest, return_ipix=True)

            density = np.zeros(self.npix)
            density[ipix] = -1.
            density[spix] = s / self.pixel_density(ishell) - 1.

            return density

    def _noise_cls(self, ishell):
        """Return the Poisson shot-noise level for shell density maps.

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        float
            White noise level N_ell = 1 / nbar, where nbar is the expected sky
            density for this shell (per steradian).
        """
        return 1. / self.sky_density(ishell)

    def compute_tidal_tensor_maps(self, ishells=None, smooth_Mpc=None, lmax=None, replicate_octant=False):
        """Compute tidal tensor component maps from shell density maps.

        For each shell, computes a spin-2 transform of the E-mode a_lm of the
        density map, and combines it with the density (trace) to obtain
        approximate 2D tidal components (t11, t22, t12) on the sky.

        Parameters
        ----------
        ishells : int or array_like of int or None, optional
            Shell indices to process. If None, process all.
        smooth_Mpc : float or None, optional
            If provided, smooths a_lm with a Gaussian of angular width
            ``sigma = smooth_Mpc / chi_shell``.
        lmax : int or None, optional
            Maximum multipole used for harmonic transforms. Defaults to
            ``3*nside - 1``.
        replicate_octant : bool, optional
            If True, recompute a_lm from octant maps replicated to full sky
            (irrelevant if ``octant_only=False``).

        Returns
        -------
        list or dict
            - If ``ishells`` is a single int: returns a list ``[t11, t22, t12]`` as HealSparse maps.
            - If ``ishells`` is ``None`` or an iterable: returns a dict keyed by shell index
              with list values ``[t11, t22, t12]``. Also writes maps to disk as
              ``what='t11'/'t22'/'t12'`` or with ``_{smooth_Mpc}Mpc`` suffix if smoothing is applied.
        """
        if lmax is None:
            lmax = 3 * self.nside - 1

        _, ipix = self.get_mask(nest=False, return_ipix=True)

        if smooth_Mpc is not None:
            assert smooth_Mpc > 0.
            whats = [f't11_{smooth_Mpc}Mpc', f't22_{smooth_Mpc}Mpc', f't12_{smooth_Mpc}Mpc']
        else:
            whats = ['t11', 't22', 't12']

        ts = {}
        for ishell in self.tqdm(ishells, desc="[compute_tidal_tensor]"):
            alm_E = self.compute_alms(ishell, save=not (replicate_octant), replicate_octant=replicate_octant, pol=False, lmax=lmax)

            if smooth_Mpc is not None:
                sigma = smooth_Mpc / self.shell_edges_comoving_distance()[ishell]
                hp.sphtfunc.smoothalm(
                    alm_E,
                    sigma=sigma,
                    pol=False,
                    inplace=True,
                )

            Q, U = hp.alm2map_spin((alm_E, np.zeros_like(alm_E)), self.nside, spin=2, lmax=lmax)

            Q = Q[ipix]
            U = U[ipix]

            density = self.get_field_map(ishell, nest=False)[ipix]

            # In 2D cartesian, s11(k)+s22(k)=delta(k)
            # In the spherical polarization framework, we have:
            # Q is 'g1' = (t11 - t22)
            # delta is t11 + t22
            # U is 'g2' = t12
            # t11 is (Q +delta)/2
            # t22 is (delta-Q)/2
            # Need to subtract off the trace, affecting s_11 and s_22... (-delta/3)
            # t11 = (delta+Q)/2 - density/3
            # t22 = (delta-Q)/2 - density/3
            # t12 = U
            vals = [density / 6. + Q / 2., density / 6. - Q / 2., U] # t11, t22, t12
            
            ts[ishell] = []
            for what, val in zip(whats, vals):
                hsp = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
                hsp.update_values_pix(ipix, val.astype(self.dtype), nest=False)
                hsp.write(self.get_fname(ishell, what=what), clobber=True, format=self.save_format)
                ts[ishell].append(hsp)

        return ts[ishells] if type(ishells) is int else ts


class SnapshotShellBuilder(ShellBuilder):
    """ShellBuilder specialized for snapshot-based shells.

    This builder defines shell edges from snapshot redshifts using the selected
    ``shell_edges_match`` strategy and constructs each shell by replicating the
    periodic snapshot box in comoving coordinates.
    """

    def __init__(self, reader, nside, save_dir, shell_edges_match='center_scale_factor', **kwargs):
        """Initialize a snapshot-based shell builder.

        Parameters
        ----------
        reader : pollux.reader.Reader
            Simulation snapshot reader.
        nside : int
            Healpix nside of the shell maps.
        save_dir : str or None
            Output directory.
        shell_edges_match : {'center_redshift','center_scale_factor','center_comoving_distance','center_comoving_volume'}, optional
            Defines how shell edges are placed relative to snapshot centers.
        **kwargs
            Forwarded to :class:`ShellBuilder`.
        """

        self.shell_edges_match = shell_edges_match

        super().__init__(reader.nsnapshots, reader, nside, save_dir, kind='snapshot', **kwargs)

    def construct_maps_sub(self, ishell, isub, ijob=None, verbose=False, get_replicates_only=False, fast=True):
        """Construct a single shell map for one snapshot/sub-volume.

        Parameters
        ----------
        ishell : int
            Shell index.
        isub : int
            Sub-volume index.
        ijob : int or None, optional
            If provided, overrides (ishell, isub) using ``reader.get_job_args(ijob)``.
        verbose : bool, optional
            If True, prints additional diagnostic output.
        get_replicates_only : bool, optional
            If True, does not build the map and instead returns replicate indices
            (ix, iy, iz) intersecting the shell.
        fast : bool, optional
            Reserved for alternate implementations; currently unused in the shown code.

        Returns
        -------
        healsparse.HealSparseMap or list
            If ``get_replicates_only=False``, returns the constructed HealSparse shell map.
            If ``get_replicates_only=True``, returns replicate indices (implementation-dependent).

        Notes
        -----
        This method:
        - reads a sub-volume from the reader,
        - optionally subsamples particles,
        - applies optional isometries / rotflips,
        - replicates the box as needed to cover the shell,
        - bins particles into healpix and writes a HealSparse map.
        """
        if ijob is not None:
            ishell, isub = self.reader.get_job_args(ijob)
        fname = self.get_fname(ishell, sub=isub)

        # check if shell has already been created
        shell = self.check_maps(read=True, fname=fname)
        if shell is not None:
            if verbose:
                print(f"Found and loaded map at {fname}.")
            return shell

        # get shell radii
        boxsize = self.dtype(self.reader.boxsize())
        chi_shell_bins = self.shell_edges_comoving_distance()
        rmin2, rmax2 = chi_shell_bins[ishell]**2, chi_shell_bins[ishell + 1]**2

        # read box
        xyz, box_edges_original = self.reader.read_sub(ishell, isub, read_box_edges_only=get_replicates_only)
        if verbose:
            print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

        # particle sub-sampling
        if self.subsample_box is not None:
            ratio = self.subsample_ratio(ishell)
            nparticles_kept = int(len(xyz) * ratio)
            if verbose:
                print(f"Keeping {nparticles_kept} particles, ie a fraction {ratio}.")
            if nparticles_kept < len(xyz):
                rng = np.random.default_rng(self.seed + ishell)
                xyz = xyz[rng.choice(len(xyz), size=nparticles_kept, replace=False), :]

        # initialize shell
        shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

        nside = self.nside
        order = hp.nside2order(self.nside)
        ncap = (nside * nside - nside) << 1
        npix = self.npix

        uipix, c = _construct_maps_sub_helper(
            True, xyz, box_edges_original, boxsize, rmin2, rmax2, self.apply_grid_isometry, self.Abs[ishell][0], self.Abs[ishell][1],
            self.apply_replicate_rotflip, self.replicate_rotflip, self.octant_only, nside, order, ncap, npix)
        if len(uipix) > 0:
            if verbose:
                print(f"Found {np.sum(c)} particles in the shell.")
            shell.update_values_pix(uipix, c.astype(self.dtype), nest=False)
        else:
            if verbose:
                print("Found no particle in the shell.")

        if self.save_dir is not None:
            shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
            if verbose:
                print(f"Saved to file {fname}")

        return shell

    def construct_catalog_sub(self, ishell, isub, verbose=False, **kwargs):
        """Construct a particle catalog for a given shell/sub-volume.

        Parameters
        ----------
        ishell : int
            Shell index.
        isub : int
            Sub-volume index.
        verbose : bool, optional
            If True, prints additional diagnostic output.
        **kwargs
            Forwarded to ``reader.read_sub``.

        Returns
        -------
        idx : ndarray
            Indices of selected particles in the input array (per replicate, concatenated).
        ra : ndarray
            Right ascension in degrees.
        dec : ndarray
            Declination in degrees.
        r : ndarray
            Comoving radial distance (Mpc).
        """
        # get shell radii
        chi_shell_bins = self.shell_edges_comoving_distance()
        rmin2, rmax2 = chi_shell_bins[ishell]**2, chi_shell_bins[ishell + 1]**2
        boxsize = self.reader.boxsize()

        # read box
        xyz, box_edges_original = self.reader.read_sub(ishell, isub, **kwargs)
        if verbose:
            print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

        idx, ra, dec, r = _construct_catalog_sub_helper(
            True, xyz, box_edges_original, boxsize, rmin2, rmax2, self.apply_grid_isometry, self.Abs[ishell][0], self.Abs[ishell][1],
            self.apply_replicate_rotflip, self.replicate_rotflip, self.octant_only)
        if len(idx) > 0:
            return np.concatenate(idx), np.concatenate(ra), np.concatenate(dec), np.concatenate(r)
        else:
            return [np.array([])] * 4

    def construct_catalog(self, ishell, extra_columns=[], **kwargs):
        """Construct and save a per-shell (halo/particle) FITS catalog.

        Parameters
        ----------
        ishell : int
            Shell index.
        extra_columns : list of str, optional
            Extra properties to load using ``reader.read_props`` and store as
            additional FITS columns.
        **kwargs
            Forwarded to :meth:`construct_catalog_sub` / ``reader.read_sub``.

        Returns
        -------
        astropy.io.fits.HDUList
            FITS HDU list (primary HDU + one table HDU per sub-volume).

        Notes
        -----
        Output filename is ``what='haloes'`` under :meth:`get_fname`.
        """
        isubs = self.reader.get_isubs_for_snapshot(ishell)

        fname = self.get_fname(ishell, what='haloes')

        if os.path.exists(fname):
            try:
                return fits.open(fname)
            except:
                pass

        hdul = [fits.PrimaryHDU()]

        for isub in self.tqdm(isubs, desc=f"[construct_halo_catalog: shell {ishell}]"):
            # Construct catalog for simulation sub-volume/snapshot
            res = self.construct_catalog_sub(ishell, isub, **kwargs)

            # Build columns for output files
            idx = fits.Column(name='ihaloes', format='J', array=res[0])
            ra = fits.Column(name='ra', format='E', array=res[1])
            dec = fits.Column(name='dec', format='E', array=res[2])
            r = fits.Column(name='chi', format='E', array=res[3])
            a = pyccl.scale_factor_of_chi(self.cosmo, res[3])
            z = fits.Column(name='z', format='E', array=1. / a - 1.)
            cols = [idx, ra, dec, r, z]

            # Add extra columns
            if len(extra_columns) > 0:
                tmp = self.reader.read_props(ishell, isub, extra_columns, ihaloes=res[0])
                for name, array in tmp.items():
                    cols.append(fits.Column(name=name, format='E', array=array))

            header = fits.Header()
            header[f'ISUB'] = isub
            header[f'ZSNAP'] = self.reader.zsnapshots[ishell]

            hdul.append(fits.BinTableHDU.from_columns(cols, header))

        fits.HDUList(hdul).writeto(fname, overwrite=True)

        return hdul

    def read_catalogs_props(self, ishell, variables):
        """Read variables from a previously constructed catalog (and/or reader).

        Parameters
        ----------
        ishell : int
            Shell index.
        variables : list of str
            Variables to return. The special variables ``'ra'``, ``'dec'``,
            ``'chi'`` and ``'z'`` are read directly from the catalog; any other
            names are loaded from the reader via :meth:`reader.read_props`.

        Returns
        -------
        dict
            Dictionary mapping variable name to 1D concatenated numpy arrays.
        """
        fname = self.get_fname(ishell, what='haloes')
        hdul = fits.open(fname)

        out = {k: [] for k in variables}

        catalog_variables = [v for v in variables if v in ['ra', 'dec', 'chi', 'z']]
        props_variables = [v for v in variables if v not in ['ra', 'dec', 'chi', 'z']]

        for hdu in self.tqdm(hdul[1:], desc="[read_catalogs_props]"):
            if len(props_variables) > 0:
                isub = hdu.header['ISUB']
                tmp = self.reader.read_props(ishell, isub, props_variables, ihaloes=hdu.data['ihaloes'])
                for k in props_variables:
                    out[k].append(tmp[k])
            if len(catalog_variables) > 0:
                for k in catalog_variables:
                    out[k].append(hdu.data[k])

        for k in variables:
            out[k] = np.concatenate(out[k])

        return out


class LightconeSlicingBuilder(ShellBuilder):
    """ShellBuilder that slices a lightcone by stacking multiple snapshots.

    For each shell, this builder reads multiple lightcone snapshots whose
    redshift range intersects the shell, concatenates particles, then bins them
    into a shell map.
    """

    def __init__(self, reader, nside, save_dir, shell_edges_match='center_scale_factor', **kwargs):
        """Initialize a lightcone slicing builder.

        Parameters
        ----------
        reader : pollux.reader.Reader
            Lightcone-capable reader providing ``zsnapshots_fulllist`` and
            ``read_sub(isnapshot, isub)`` for lightcone snapshots.
        nside : int
            Healpix nside of the shell maps.
        save_dir : str or None
            Output directory.
        shell_edges_match : str, optional
            Shell-edge placement strategy (same meaning as for SnapshotShellBuilder).
        **kwargs
            Forwarded to :class:`ShellBuilder`.
        """

        self.shell_edges_match = shell_edges_match

        super().__init__(reader.nsnapshots, reader, nside, save_dir, kind='snapshot', **kwargs)

    def construct_maps_sub(self, ishell, isub, ijob=None, verbose=False):
        """Construct a single shell map for one lightcone sub-volume.

        Parameters
        ----------
        ishell : int
            Shell index.
        isub : int
            Sub-volume index.
        ijob : int or None, optional
            If provided, overrides (ishell, isub) using ``reader.get_job_args(ijob)``.
        verbose : bool, optional
            If True, prints additional diagnostic output.

        Returns
        -------
        healsparse.HealSparseMap
            Constructed HealSparse shell map for this sub-volume.
        """
        if ijob is not None:
            ishell, isub = self.reader.get_job_args(ijob)
        fname = self.get_fname(ishell, sub=isub)

        # check if shell has already been created
        shell = self.check_maps(read=True, fname=fname)
        if shell is not None:
            if verbose:
                print(f"Found and loaded map at {fname}.")
            return shell

        # get shell radii and redshifts
        boxsize = self.dtype(self.reader.boxsize())
        chi_shell_bins = self.shell_edges_comoving_distance()
        rmin2, rmax2 = chi_shell_bins[ishell]**2, chi_shell_bins[ishell + 1]**2
        z_shell_bins = self.shell_edges_redshift()
        zmin, zmax = z_shell_bins[ishell], z_shell_bins[ishell + 1]

        # get lightcone snapshot redshifts
        zsnapshot_fulllist = self.reader.zsnapshots_fulllist

        # read box over all snapshots that intersect the shell
        xyz = []
        for isnapshot in range(len(zsnapshot_fulllist) - 1):
            if zmin > zsnapshot_fulllist[isnapshot + 1]:
                continue
            if zmax < zsnapshot_fulllist[isnapshot]:
                continue
            xyz.append(self.reader.read_sub(isnapshot, isub))
        xyz = np.vstack(xyz)

        # define box edges
        if len(xyz) > 0:
            box_lims = np.stack([np.min(xyz, axis=0), np.max(xyz, axis=0)])
            cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
            box_edges_original = box_lims[0] + (box_lims[1] - box_lims[0]) * cube_nodes.astype(self.dtype)
        else:
            box_edges_original = np.zeros((8, 3)).astype(self.dtype)

        if verbose:
            r2_xyz = xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2
            rmin2_xyz = r2_xyz.min()
            rmax2_xyz = r2_xyz.max()
            print(f"Shell {ishell} (z=[{zmin:.3f},{zmax:.3f}]):")
            print(f"  - Particle radial distance range in box: [{np.sqrt(rmin2_xyz):.3f}, {np.sqrt(rmax2_xyz):.3f}] Mpc")
            print(f"  - Shell radial distance range: [{np.sqrt(rmin2):.3f}, {np.sqrt(rmax2):.3f}] Mpc")
            print(f"  - Number of particles in box: {len(xyz)}")

        # initialize shell
        shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

        nside = self.nside
        order = hp.nside2order(self.nside)
        ncap = (nside * nside - nside) << 1
        npix = self.npix

        uipix, c = _construct_maps_sub_helper(
            False, xyz, box_edges_original, boxsize, rmin2, rmax2, self.apply_grid_isometry, self.Abs[ishell][0], self.Abs[ishell][1],
            self.apply_replicate_rotflip, self.replicate_rotflip, self.octant_only, nside, order, ncap, npix)
        if len(uipix) > 0:
            if verbose:
                print(f"Found {np.sum(c)} particles in the shell.")
            shell.update_values_pix(uipix, c.astype(self.dtype), nest=False)
        else:
            if verbose:
                print("Found no particle in the shell.")

        if self.save_dir is not None:
            shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
            if verbose:
                print(f"Saved to file {fname}")

        return shell

    def combine_maps_subs(self, ishells=None):
        """Combine per-sub shell maps into final shells (lightcone slicing mode).

        Parameters
        ----------
        ishells : int or array_like of int or None, optional
            Shell indices to combine. If None, combines all shells.

        Returns
        -------
        healsparse.HealSparseMap or dict
            - If ``ishells`` is a single int: returns a single combined
              ``HealSparseMap``.
            - If ``ishells`` is ``None`` or an iterable: returns a dict keyed
              by shell index with combined ``HealSparseMap`` values.
        """
        shells = {}

        _, ipix = self.get_mask(nest=True, return_ipix=True)
        for ishell in self.tqdm(ishells, desc="[combine_maps_subs]"):
            fname = self.get_fname(ishell)
            shell = self.check_maps(read=True, fname=fname)
            if shell is not None:
                shells[ishell] = shell
                continue
            
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            shell.update_values_pix(ipix, self.dtype(0), nest=True) # initialize to zero on valid pixels
            isubs = range(1000)
            n_subs_added = 0
            for isub in tqdm(isubs, desc=f"[combine_maps_subs: shell {ishell}]", leave=False):
                try:
                    tmp_shell = self.load_maps(ishell, sub=isub)
                    shell = healsparse.operations.sum_union([shell, tmp_shell])
                    n_subs_added += 1
                except FileNotFoundError:
                    continue
            if n_subs_added != 0:
                shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
            else:
                print('Zero sub files found for shell number ' + str(ishell) + '!')

            shells[ishell] = shell
            
        return shells[ishells] if type(ishells) is int else shells


class SnapshotInterpBuilder(ShellBuilder):
    """ShellBuilder that interpolates particle worldlines between snapshots.

    This builder treats consecutive snapshot boxes as endpoints of particle
    worldlines and finds their intersections with a lightcone defined by a
    sequence of scale factors. It supports orphan handling when particles appear
    or disappear between snapshots.
    """

    def __init__(self, reader, nside, save_dir, **kwargs):
        """Initialize a snapshot-interpolation builder.

        Parameters
        ----------
        reader : pollux.reader.Reader
            Reader providing at least consecutive snapshots and particle ids.
        nside : int
            Healpix nside of the shell maps.
        save_dir : str or None
            Output directory.
        **kwargs
            Forwarded to :class:`ShellBuilder`.

        Notes
        -----
        This builder uses ``kind='lightcone'`` with ``nshells = nsnapshots - 1``.
        """

        super().__init__(reader.nsnapshots - 1, reader, nside, save_dir, kind='lightcone', **kwargs)

    def construct_maps_sub(self, isnap, isub, ijob=None, verbose=False, replicate_ixyz=None, get_replicates_only=False, fast=True, do_orphan_ishell=None):
        """Construct one interpolated lightcone shell map for a sub-volume.

        Parameters
        ----------
        isnap : int
            Snapshot index (defines the interval [isnap, isnap+1]).
        isub : int
            Sub-volume index.
        ijob : int or None, optional
            If provided, overrides (isnap, isub) using ``reader.get_job_args(ijob)``.
        verbose : bool, optional
            If True, prints diagnostic information about overlap/particles.
        replicate_ixyz : tuple(int,int,int) or None, optional
            If provided, only compute the specified replicate (ix, iy, iz).
        get_replicates_only : bool, optional
            If True, return only replicate indices intersecting the shell and
            do not build the map.
        fast : bool, optional
            Reserved for alternate implementations; currently unused in the shown code.
        do_orphan_ishell : int or None, optional
            If provided, build the orphan shell (stored under ``sub='orphan'``)
            by loading previously dumped orphan particle arrays.

        Returns
        -------
        healsparse.HealSparseMap or list
            HealSparse map, or replicate list when ``get_replicates_only=True``.

        Notes
        -----
        The lightcone is approximated by a tabulated chi(a) curve between the two
        snapshots, and particle worldlines (linear in a) are intersected with it
        to determine sky positions.
        """
        if ijob is not None:
            isnap, isub = self.reader.get_job_args(ijob)

        if do_orphan_ishell is not None:
            isnap = do_orphan_ishell

        # check if shell exists and/or can be saved
        if do_orphan_ishell is not None:
            fname = self.get_fname(isnap, sub='orphan')
        else:
            fname = self.get_fname(isnap, sub=isub)

        # check if shell has already been created
        shell = self.check_maps(read=True, fname=fname)
        if shell is not None:
            if verbose:
                print(f"Found and loaded map at {fname}.")
            return shell

        # get lightcone line
        z_shell_bins = self.shell_edges_redshift()
        a_i = 1. / (1. + z_shell_bins[isnap + 1])
        a_f = 1. / (1. + z_shell_bins[isnap])
        a_lc = arange_endpoint(a_i, a_f, 0.005)
        chi2_lc = self.cosmo.comoving_radial_distance(a_lc)**2
        rmin2, rmax2 = chi2_lc[-1], chi2_lc[0]

        # get number of replicates per direction
        boxsize = self.reader.boxsize()
        # not too sure about that 2, but it's fast to check...
        nreplicate = int(np.ceil(np.sqrt(rmax2) / boxsize)) + 2

        # read box
        if do_orphan_ishell is not None:
            keys = ['x', 'y', 'z']

            # load orphan particles and sort to match them
            xyzi = np.concatenate(
                [np.load(self.get_fname(isnap, sub=isub, what='orphan_i', format='npy')) for isub in self.reader.get_isubs_for_snapshot(isnap)])
            xyzf = np.concatenate(
                [np.load(self.get_fname(isnap, sub=isub, what='orphan_f', format='npy')) for isub in self.reader.get_isubs_for_snapshot(isnap)])
            if verbose:
                print(f"Loaded {len(xyzi)} orphan particles.")

            # sort id's and match particles
            sidxi = np.argsort(xyzi['id'])
            sidxf = np.argsort(xyzf['id'])
            xyz_i = np.vstack([xyzi[sidxi][k] for k in keys]).T
            xyz_f = np.vstack([xyzf[sidxf][k] for k in keys]).T
            if verbose:
                np.testing.assert_array_equal(xyzi['id'][sidxi], xyzf['id'][sidxf])
                print("Matched particles.")

            # move particles that went through box faces
            part_boundary_fix(xyz_i, xyz_f, boxsize, fixed='i')

            # recompute box edges accordingly
            box_lims = np.stack(
                [np.min([np.min(xyz_i, axis=0), np.min(xyz_f, axis=0)], axis=0),
                 np.max([np.max(xyz_i, axis=0), np.max(xyz_f, axis=0)], axis=0)])
            cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
            box_edges_original = (box_lims[0] + (box_lims[1] - box_lims[0]) * cube_nodes).astype(self.dtype)

        else:
            xyz_i, box_edges_original, idx_i = self.reader.read_sub(isnap, isub, extra_variables=['id'])
            xyz_f, box_edges_original, idx_f = self.reader.read_sub(isnap + 1, isub, extra_variables=['id'])

            if verbose:
                print(f"Loaded box {isub} (snapshot {isnap}) with {len(xyz_i)} particles.")
                print(f"Loaded box {isub} (snapshot {isnap + 1}) with {len(xyz_f)} particles.")

            # match the indices of particles
            idx_i_in, idx_f_in, idx_i_out, idx_f_out = inter_diff(idx_i['id'], idx_f['id'])
            assert len(idx_i_in) == len(idx_f_in)

            # dump orphan particles
            np.save(
                self.get_fname(isnap, sub=isub, what='orphan_i', format='npy'),
                np.rec.array((*xyz_i[idx_i_out].T, idx_i['id'][idx_i_out]), names=['x', 'y', 'z', 'id']))
            np.save(
                self.get_fname(isnap, sub=isub, what='orphan_f', format='npy'),
                np.rec.array((*xyz_f[idx_f_out].T, idx_f['id'][idx_f_out]), names=['x', 'y', 'z', 'id']))

            # keep matched particles
            xyz_i = xyz_i[idx_i_in]
            xyz_f = xyz_f[idx_f_in]

            if verbose:
                print(f"Found {len(idx_i_in)} particles in both boxes, {len(idx_i_out)}/{len(idx_f_out)} orphans in initial/final boxes saved.")

        # particle sub-sampling
        if self.subsample_box is not None:
            ratio = self.subsample_ratio(isnap)
            nparticles_kept = int(len(xyz_i) * ratio)
            if verbose:
                print(f"Keeping {nparticles_kept} particles, ie a fraction {ratio}.")
            if nparticles_kept < len(xyz_i):
                rng = np.random.default_rng(self.seed + isnap)
                s = rng.choice(len(xyz_i), size=nparticles_kept, replace=False)
                xyz_i = xyz_i[s, :]
                xyz_f = xyz_f[s, :]

        # initialize shell
        shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

        nside = self.nside
        order = hp.nside2order(self.nside)
        ncap = (nside * nside - nside) << 1
        npix = self.npix

        if False:
            uipix, c = _construct_maps_sub_helper(
                xyz, box_edges_original, self.Abs[isnap][0], self.Abs[isnap][1], nreplicate, boxsize, rmin2, rmax2, self.apply_grid_isometry, self.octant_only,
                nside, order, ncap, npix)
            if len(uipix) > 0:
                shell.update_values_pix(uipix, c.astype(self.dtype), nest=False)

        else:
            iso_applied = False  # to xyz

            # precompute isometry on particles (since replicates are then only a shift away)
            if self.apply_grid_isometry:
                box_edges = Axpb(*self.Abs[isnap], box_edges_original)
            else:
                box_edges = box_edges_original

            replicates = []
            # replicate the box
            for ix in range(-nreplicate, nreplicate + 1):
                for iy in range(-nreplicate, nreplicate + 1):
                    for iz in range(-nreplicate, nreplicate + 1):
                        # check if we want to do only one replicate
                        if replicate_ixyz is not None:
                            if not np.array_equal(replicate_ixyz, [ix, iy, iz]):
                                continue

                        # compute shift
                        shift_original = (np.array([ix, iy, iz], dtype=self.reader.dtype) * self.reader.boxsize())[None, :]
                        if self.apply_grid_isometry:
                            shift = Ax(self.Abs[isnap][0], shift_original)
                        else:
                            shift = shift_original

                        # apply isometry to box edges
                        box_edges_shifted = shift + box_edges

                        # check if box intersects octant, otherwise skip
                        # (to do so, we check if the box is fully contained in the x<0 or y<0 or z<0 half spaces)
                        if self.octant_only and np.any(np.all(box_edges_shifted <= 0., axis=0)):
                            continue

                        # check if box intersects with shell, otherwise skip
                        # first, check if box fully inside inner sphere
                        if np.all(xyz2r2(box_edges_shifted) < rmin2):
                            continue

                        # second, check if box intersects with outer sphere
                        sphere_center = -self.Abs[isnap][1] if self.apply_grid_isometry else np.zeros(3)
                        if not AABB_sphere_intersect(box_edges_original + shift_original, sphere_center, rmax2):
                            continue

                        # if we got this far, then this replicate intersects with the shell
                        replicates.append([ix, iy, iz])
                        if verbose:
                            print(f"Found overlap for box {isub} (snapshot {isnap}) for replicate {(ix,iy,iz)}.")
                        if get_replicates_only:
                            continue

                        # now, add particles to shell

                        # apply random isometries (once) if necessary
                        if self.apply_grid_isometry and not iso_applied:
                            xyz_i = Axpb(*self.Abs[isnap], xyz_i)
                            xyz_f = Axpb(*self.Abs[isnap], xyz_f)
                            iso_applied = True

                        # compute unique pixels into which particles fall and their counts, all done in vec2pix_unique (using numba) as follows:
                        # 1 - remove particles out of shell based on radius
                        # 2 - remove particles out of octant if octant_only
                        # 3 - compute corresponding healpy pixels of all particles (removed particles fast set to -1) with vec2pix and loc2pix (translated into python from Healpix code)
                        # 4 - get unique pixels and counts (ignoring -1's)
                        ipix = _worldline2pix(nside, order, ncap, npix, a_lc, chi2_lc, *(xyz_i.T), *(xyz_f.T), *(shift[0].T), rmin2, rmax2, self.octant_only)
                        uipix, c = _unique_pix_pos(ipix, npix)

                        if len(uipix) > 0:
                            if verbose:
                                print(f"Found {np.sum(c)} particles in the shell.")

                            # add particles to shell
                            tmp_shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
                            tmp_shell.update_values_pix(uipix, c.astype(self.dtype), nest=False)

                            shell = healsparse.operations.sum_union([shell, tmp_shell])
                        else:
                            if verbose:
                                print("Found no particle in the shell.")

            if verbose:
                print(f"Found overlap for box {isub} (shell {isnap}) for {len(replicates)} replicates.")

            if get_replicates_only:
                return replicates

        if self.save_dir is not None:
            shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
            if verbose:
                print(f"Saved to file {fname}")

        return shell


# @numba.njit(parallel=False)
# def _add_at_one(a, index):
#     n = len(index)
#     for i in numba.prange(n):
#         a[index[i]] += 1


@numba.njit
def xyz2r2(xyz):
    """Return squared radius r^2 for an array of Cartesian vectors.

    Parameters
    ----------
    xyz : ndarray
        Array of shape (N, 3).

    Returns
    -------
    ndarray
        Array of shape (N,) giving ``x^2 + y^2 + z^2``.
    """
    return xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2


# @numba.njit(parallel=True)
# def xyz2r(xyz):
#     return np.sqrt(xyz2r2(xyz))

# @numba.njit(parallel=True)
# def xyzr2tp(xyz, r):
#     assert len(xyz) == len(r)
#     n = len(xyz)
#     x = xyz[:, 0]
#     y = xyz[:, 1]
#     z = xyz[:, 2]
#     theta = np.arccos(z / r)
#     phi = np.arctan2(x, y)
#     return theta, phi


@numba.njit(parallel=True)
def Axpb(A, b, xyz):
    """Apply an affine transform ``A(x + b)`` to a set of vectors.

    Parameters
    ----------
    A : ndarray
        Matrix of shape (3, 3).
    b : ndarray
        Translation vector of shape (3,).
    xyz : ndarray
        Array of shape (N, 3).

    Returns
    -------
    ndarray
        Transformed vectors of shape (N, 3).
    """
    return np.dot(xyz + b, A)


@numba.njit
def Ax(A, xyz):
    """Apply a linear transform ``A x`` to vectors.

    Parameters
    ----------
    A : ndarray
        Matrix of shape (3, 3).
    xyz : ndarray
        Array of shape (N, 3) or (1, 3).

    Returns
    -------
    ndarray
        Transformed vectors (same leading shape as input).
    """
    return np.dot(xyz, A)


# @numba.njit(parallel=True)
# def Axpb_inv(A, b, xyz):
#     return np.dot(xyz, A.T) - b


@numba.njit
def rotflip(rot, xyz, box_size):
    """Apply HACC-style discrete rotations (axis permutations) and sign flips.

    The 48 possible transforms are indexed by ``rot`` in [0, 48):
    - ``rot >> 3`` selects one of 6 coordinate permutations,
    - ``rot & 7`` selects one of 8 independent sign flips.

    Parameters
    ----------
    rot : int
        Rotation/flip code in [0, 48).
    xyz : ndarray
        Positions of shape (N, 3) in box coordinates [0, box_size].
    box_size : float
        Side length of the simulation box.

    Returns
    -------
    ndarray
        Transformed positions of shape (N, 3) in [0, box_size].

    Notes
    -----
    The transform is applied about the box center. A replicate translation must
    still be applied after this operation.
    """
    x = xyz[:, 0] - box_size / 2.
    y = xyz[:, 1] - box_size / 2.
    z = xyz[:, 2] - box_size / 2.

    rot_type = rot >> 3
    # rotations by 90 deg are swapping coordinates, so 3!=6 possibilities
    if rot_type == 0:
        xtmp = x
        ytmp = y
        ztmp = z
    elif rot_type == 1:
        xtmp = x
        ytmp = z
        ztmp = y
    elif rot_type == 2:
        xtmp = y
        ytmp = x
        ztmp = z
    elif rot_type == 3:
        xtmp = y
        ytmp = z
        ztmp = x
    elif rot_type == 4:
        xtmp = z
        ytmp = x
        ztmp = y
    elif rot_type == 5:
        xtmp = z
        ytmp = y
        ztmp = x
    else:
        raise ValueError

    # sign flips are after (2^3=8 possibilities)
    sign_type = rot & ((1 << 3) - 1)
    if ((sign_type & (1 << 0)) != 0):
        xtmp = -xtmp
    if ((sign_type & (1 << 1)) != 0):
        ytmp = -ytmp
    if ((sign_type & (1 << 2)) != 0):
        ztmp = -ztmp

    xtmp += box_size / 2.
    ytmp += box_size / 2.
    ztmp += box_size / 2.

    # note depending on replication value you will still need to translate this box to the appropriate place in the lightcone after performing this rotation, and if velocities are required you'll need to apply a similar shift to those
    return np.column_stack((xtmp, ytmp, ztmp))


def random_isometry(rng, boxsize):
    """Draw a random affine isometry (rotation+flip and translation) for a box.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    boxsize : float
        Box side length.

    Returns
    -------
    A : ndarray
        Random 3x3 matrix (rotation times diagonal sign flip).
    b : ndarray
        Random translation vector of length 3 in [0, boxsize).
    """
    pcomb = list(itertools.product([1, -1], repeat=3))  # all permuted orders
    r = Rotation.random(random_state=rng).as_matrix()
    p = pcomb[rng.choice(len(pcomb))]
    A = np.dot(r, np.diag(np.array(p)))
    b = rng.random(3) * boxsize
    return A, b


def AABB_sphere_intersect(box_edges, s, r2):
    """Test whether an axis-aligned box intersects a sphere.

    Parameters
    ----------
    box_edges : ndarray
        Box vertices of shape (8, 3) (or any set of points whose min/max define
        the AABB).
    s : ndarray
        Sphere center (3,).
    r2 : float
        Sphere radius squared.

    Returns
    -------
    bool
        True if the box intersects the sphere.

    Notes
    -----
    Implements the classic AABB-sphere overlap test:
    https://stackoverflow.com/a/4579069
    """
    c1, c2 = box_edges.min(axis=0), box_edges.max(axis=0)
    d2 = r2
    for i in range(3):
        if s[i] < c1[i]:
            d2 -= (s[i] - c1[i])**2
        elif s[i] > c2[i]:
            d2 -= (s[i] - c2[i])**2
    return d2 > 0.


@numba.njit
def _AABB_sphere_intersect(box_edges, s, r2):
    """Numba version of :func:`AABB_sphere_intersect`."""
    d2 = r2
    for i in range(3):
        c1 = np.min(box_edges[:, i])
        c2 = np.max(box_edges[:, i])
        if s[i] < c1:
            d2 -= (s[i] - c1)**2
        elif s[i] > c2:
            d2 -= (s[i] - c2)**2
    return d2 > 0.


inv_halfpi = 2. / np.pi


@numba.njit
def _loc2pix(nside, order, ncap, npix, z, phi, sth):
    """Convert (z, phi) on the unit sphere to a Healpix pixel index (RING-like).

    Parameters
    ----------
    nside : int
        Healpix nside.
    order : int
        Healpix order = log2(nside).
    ncap : int
        Number of pixels in the polar caps (Healpix internal constant).
    npix : int
        Total number of pixels (= 12*nside^2).
    z : float
        Cosine of colatitude (= cos(theta)).
    phi : float
        Longitude angle in radians.
    sth : float
        sin(theta) (distance to z-axis on the unit sphere).

    Returns
    -------
    int
        Pixel index.

    Notes
    -----
    This is a Python/Numba translation of the Healpix ``loc2pix`` logic used by
    ``vec2pix``. It is used internally to avoid calling healpy in tight loops.
    """
    za = np.abs(z)
    tt = phi * inv_halfpi % 4.
    if za <= 2. / 3.:
        nl4 = 4 * nside
        temp1 = nside * (0.5 + tt)
        temp2 = nside * z * 0.75
        jp = int(temp1 - temp2)  # index of  ascending edge line
        jm = int(temp1 + temp2)  # index of descending edge line

        # ring number counted from z=2/3
        ir = nside + 1 + jp - jm  # in {1,2n+1}
        kshift = 1 - (ir % 2)  # kshift=1 if ir even, 0 otherwise

        t1 = jp + jm - nside + kshift + 1 + nl4 + nl4
        if order > 0:
            ip = (t1 >> 1) & (nl4 - 1)
        else:
            ip = ((t1 >> 1) % nl4)  # in {0,4n-1}

        return ncap + (ir - 1) * nl4 + ip

    else:  # Equatorial region
        tp = tt - np.floor(tt)
        if za < 0.99:
            tmp = nside * np.sqrt(3 * (1 - za))
        else:
            tmp = nside * sth / np.sqrt((1. + za) / 3.)

        jp = int(tp * tmp)  # increasing edge line index
        jm = int((1. - tp) * tmp)  # decreasing edge line index

        ir = jp + jm + 1  # ring number counted from the closest pole
        ip = int(tt * ir)  # in {0,4*ir-1}

        if z > 0:
            return 2 * ir * (ir - 1) + ip
        else:
            return npix - 2 * ir * (ir + 1) + ip


@numba.njit
def _vec2pix_sub(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """Convert one shifted Cartesian position to a Healpix pixel index with cuts.

    Parameters
    ----------
    nside, order, ncap, npix : int
        Healpix geometry parameters.
    x, y, z : float
        Particle position in box coordinates.
    shiftx, shifty, shiftz : float
        Replicate translation applied before binning.
    rmin2, rmax2 : float
        Shell radial cuts (squared).
    octant_only : bool
        If True, require x>=0, y>=0, z>=0 after shifting.

    Returns
    -------
    int
        Pixel index, or -1 if the point is rejected by cuts.
    """
    # apply shift
    tmpx = x + shiftx
    tmpy = y + shifty
    tmpz = z + shiftz

    # get radius and remove particles out of shell
    r2 = tmpx**2 + tmpy**2 + tmpz**2
    if r2 < rmin2 or r2 >= rmax2:
        return -1

    # remove particles out of octant if octant_only
    if octant_only:  # check if particle out of octant
        if (tmpx < 0.) or (tmpy < 0.) or (tmpz < 0.):
            return -1

    # get longitude (phi), altitude (nz) and distance to z-axis (sth) of particle projected on the unit sphere
    phi = np.arctan2(tmpy, tmpx)
    xl = 1. / np.sqrt(r2)
    nz = tmpz * xl
    sth = np.sqrt(tmpx**2 + tmpy**2) * xl

    return _loc2pix(nside, order, ncap, npix, nz, phi, sth)


@numba.njit(parallel=True)
def _vec2pix(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """Vectorized/parallel version of :func:`_vec2pix_sub`.

    Parameters
    ----------
    x, y, z : ndarray
        1D arrays of equal length.
    shiftx, shifty, shiftz : float
        Translation applied to all points.
    rmin2, rmax2 : float
        Shell cuts.
    octant_only : bool
        Octant cut flag.

    Returns
    -------
    ndarray
        Pixel indices (int), with -1 for rejected particles.
    """
    n = len(x)
    ipix = np.zeros(n, dtype=np.int_)
    for i in numba.prange(n):
        ipix[i] = _vec2pix_sub(nside, order, ncap, npix, x[i], y[i], z[i], shiftx, shifty, shiftz, rmin2, rmax2, octant_only)
    return ipix


@numba.njit
def _vec2radec_sub(x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """Convert one shifted Cartesian position to (ra, dec, r) with cuts.

    Returns
    -------
    (bool, float, float, float)
        (accepted, ra_deg, dec_deg, r_Mpc). If not accepted, returns (False, 0, 0, 0).
    """
    # apply shift
    tmpx = x + shiftx
    tmpy = y + shifty
    tmpz = z + shiftz

    # get radius and remove particles out of shel
    r2 = tmpx**2 + tmpy**2 + tmpz**2
    if r2 < rmin2 or r2 >= rmax2:
        return (False, 0., 0., 0.)

    # remove particles out of octant if octant_only
    if octant_only:  # check if particle out of octant
        if (tmpx < 0.) or (tmpy < 0.) or (tmpz < 0.):
            return (False, 0., 0, 0.)

    r = np.sqrt(r2)
    theta = np.arccos(tmpz / r)
    phi = np.arctan2(tmpy, tmpx)

    ra = np.rad2deg(phi)
    dec = 90. - np.rad2deg(theta)

    return (True, ra, dec, r)


@numba.njit(parallel=True)
def _vec2radec(x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """Vectorized/parallel conversion to (ra, dec, r) with shell/octant cuts.

    Returns
    -------
    idx : ndarray of int
        Indices of accepted particles (same length as input, -1 if rejected).
    ra : ndarray
        Right ascension in degrees (undefined where idx=-1).
    dec : ndarray
        Declination in degrees (undefined where idx=-1).
    r : ndarray
        Radius in same units as input coordinates (undefined where idx=-1).
    """
    n = len(x)
    ra = np.zeros(n, dtype=x.dtype)
    dec = np.zeros(n, dtype=x.dtype)
    r = np.zeros(n, dtype=x.dtype)
    idx = np.zeros(n, dtype=np.int_)
    for i in numba.prange(n):
        good, tmpra, tmpdec, tmpr = _vec2radec_sub(x[i], y[i], z[i], shiftx, shifty, shiftz, rmin2, rmax2, octant_only)
        if good:
            idx[i] = i
            ra[i] = tmpra
            dec[i] = tmpdec
            r[i] = tmpr
        else:
            idx[i] = -1

    return idx, ra, dec, r


@numba.njit
def _unique_pix_pos(ipix, npix):
    """Count occurrences of each pixel index and return the non-zero ones.

    Parameters
    ----------
    ipix : ndarray of int
        Pixel indices with -1 marking rejected particles.
    npix : int
        Total number of pixels.

    Returns
    -------
    u : ndarray of int
        Pixel indices with non-zero counts.
    c : ndarray of int
        Counts aligned with ``u``.
    """
    n = len(ipix)
    c = np.zeros(npix, dtype=np.int_)
    for i in range(n):
        pos = ipix[i]
        if pos >= 0:
            c[pos] += 1
    u = np.argwhere(c)[:, 0]
    return u, c[u]


@numba.njit
def _construct_maps_sub_helper(
        do_replicate, xyz_original, box_edges_original, boxsize, rmin2, rmax2, apply_grid_isometry, A, b, apply_replicate_rotflip, rep2rot, octant_only, nside,
        order, ncap, npix):
    """Numba helper to bin particles from one subvolume into pixels (with replication).

    Parameters
    ----------
    do_replicate : bool
        If True, replicate the periodic box across (ix,iy,iz) shifts.
    xyz_original : ndarray
        Particle positions (N, 3).
    box_edges_original : ndarray
        Box vertices (8, 3).
    boxsize : float
        Box side length.
    rmin2, rmax2 : float
        Shell radial cuts (squared).
    apply_grid_isometry : bool
        Whether to apply affine isometry (A, b).
    A, b : ndarray
        Isometry parameters.
    apply_replicate_rotflip : bool
        Whether to apply discrete rotflip per replicate.
    rep2rot : ndarray
        Array of shape (M, 4) encoding (ix,iy,iz, rotcode).
    octant_only : bool
        If True, apply octant cut.
    nside, order, ncap, npix : int
        Healpix geometry parameters.

    Returns
    -------
    u : ndarray of int
        Pixel indices with non-zero counts.
    count_u : ndarray
        Counts for those pixels.

    Notes
    -----
    This helper performs the core counting loop used by snapshot/lightcone
    shell construction.
    """
    rot_dict = numba.typed.Dict()
    for i in range(len(rep2rot)):
        ix, iy, iz = rep2rot[i, :3]
        rot_dict[ix, iy, iz] = rep2rot[i, 3]

    # precompute isometry on particles (since replicates are then only a shift away)
    if apply_grid_isometry:
        xyz = Axpb(A, b, xyz_original)
        box_edges = Axpb(A, b, box_edges_original)
    else:
        if not apply_replicate_rotflip:
            xyz = xyz_original
            box_edges = box_edges_original

    n = len(xyz_original)
    dtype = xyz_original.dtype
    count = np.zeros(npix, dtype=np.uint)
    if do_replicate:
        nreplicate = int(np.ceil(np.sqrt(rmax2) / boxsize)) + 2
    else:
        nreplicate = 0

    # replicate the box
    for ix in range(-nreplicate, nreplicate + 1):
        for iy in range(-nreplicate, nreplicate + 1):
            for iz in range(-nreplicate, nreplicate + 1):
                # compute shift
                shift_original = np.expand_dims(np.array([ix, iy, iz]) * boxsize, 0).astype(dtype)
                if apply_grid_isometry:
                    shift = Ax(A, shift_original)
                else:
                    shift = shift_original

                if apply_replicate_rotflip:
                    if (ix, iy, iz) not in rot_dict:
                        continue
                    else:
                        box_edges = rotflip(rot_dict[ix, iy, iz], box_edges_original, boxsize).astype(dtype)

                # apply shift to box edges
                box_edges_shifted = shift + box_edges

                ## check if box intersects octant, otherwise skip
                # (to do so, we check if the box is fully contained in the x<0 or y<0 or z<0 half spaces)
                if octant_only:
                    if np.any(np.array([np.all(box_edges_shifted[:, i] <= 0.) for i in range(3)])):
                        continue

                ## check if box intersects with shell, otherwise skip
                # first, check if box fully inside inner sphere
                if np.all(xyz2r2(box_edges_shifted) < rmin2):
                    continue

                # second, check if box intersects with outer sphere
                if apply_grid_isometry:
                    if not _AABB_sphere_intersect(box_edges_original + shift_original, -b, rmax2):
                        continue
                else:
                    if not _AABB_sphere_intersect(box_edges_shifted, np.zeros(3, dtype=dtype), rmax2):
                        continue

                ## add particles to shell
                # rotate/flip replicate
                if apply_replicate_rotflip:
                    xyz = rotflip(rot_dict[ix, iy, iz], xyz_original, boxsize).astype(dtype)

                # compute corresponding healpy pixels of all particles (particles out of shell/octant quickly set to -1) with vec2pix and loc2pix (translated into python from Healpix code)
                ipix = _vec2pix(nside, order, ncap, npix, xyz[:, 0], xyz[:, 1], xyz[:, 2], shift[0, 0], shift[0, 1], shift[0, 2], rmin2, rmax2, octant_only)

                for i in range(n):
                    pos = ipix[i]
                    if pos >= 0:
                        count[pos] += 1

    u = np.argwhere(count)[:, 0]
    return u, count[u]


@numba.njit
def _construct_catalog_sub_helper(
        do_replicate, xyz_original, box_edges_original, boxsize, rmin2, rmax2, apply_grid_isometry, A, b, apply_replicate_rotflip, rep2rot, octant_only):
    """Numba helper to build (idx, ra, dec, r) catalogs for one subvolume.

    Parameters
    ----------
    do_replicate : bool
        If True, replicate the periodic box across shifts.
    xyz_original : ndarray
        Particle positions (N, 3).
    box_edges_original : ndarray
        Box vertices (8, 3).
    boxsize : float
        Box side length.
    rmin2, rmax2 : float
        Shell radial cuts (squared).
    apply_grid_isometry, A, b, apply_replicate_rotflip, rep2rot, octant_only
        Same meaning as in :func:`_construct_maps_sub_helper`.

    Returns
    -------
    idx, ra, dec, r : list
        Lists of arrays (one per kept replicate) that can be concatenated by the caller.
    """
    rot_dict = numba.typed.Dict()
    for i in range(len(rep2rot)):
        ix, iy, iz = rep2rot[i, :3]
        rot_dict[ix, iy, iz] = rep2rot[i, 3]

    # precompute isometry on particles (since replicates are then only a shift away)
    if apply_grid_isometry:
        xyz = Axpb(A, b, xyz_original)
        box_edges = Axpb(A, b, box_edges_original)
    else:
        if not apply_replicate_rotflip:
            xyz = xyz_original
            box_edges = box_edges_original

    idx = []
    ra = []
    dec = []
    r = []

    dtype = xyz.dtype
    if do_replicate:
        nreplicate = int(np.ceil(np.sqrt(rmax2) / boxsize)) + 2
    else:
        nreplicate = 0

    # replicate the box
    for ix in range(-nreplicate, nreplicate + 1):
        for iy in range(-nreplicate, nreplicate + 1):
            for iz in range(-nreplicate, nreplicate + 1):
                # compute shift
                shift_original = np.expand_dims(np.array([ix, iy, iz]) * boxsize, 0).astype(dtype)
                if apply_grid_isometry:
                    shift = Ax(A, shift_original)
                else:
                    shift = shift_original

                # apply shift to box edges
                box_edges_shifted = shift + box_edges

                ## check if box intersects octant, otherwise skip
                # (to do so, we check if the box is fully contained in the x<0 or y<0 or z<0 half spaces)
                if octant_only:
                    if np.any(np.array([np.all(box_edges_shifted[:, i] <= 0.) for i in range(3)])):
                        continue

                ## check if box intersects with shell, otherwise skip
                # first, check if box fully inside inner sphere
                if np.all(xyz2r2(box_edges_shifted) < rmin2):
                    continue

                # second, check if box intersects with outer sphere
                if apply_grid_isometry:
                    if not _AABB_sphere_intersect(box_edges_original + shift_original, -b, rmax2):
                        continue
                else:
                    if not _AABB_sphere_intersect(box_edges_shifted, np.zeros(3, dtype=dtype), rmax2):
                        continue

                ## add particles to catalog
                # rotate/flip replicate
                if apply_replicate_rotflip:
                    xyz = rotflip(rot_dict[ix, iy, iz], xyz_original, boxsize).astype(dtype)

                # compute ra, dec and radial comoving distance (idx is -1 for particles to reject)
                tmpidx, tmpra, tmpdec, tmpr = _vec2radec(xyz[:, 0], xyz[:, 1], xyz[:, 2], shift[0, 0], shift[0, 1], shift[0, 2], rmin2, rmax2, octant_only)

                # add to output
                w = tmpidx >= 0
                if np.any(w):
                    idx.append(tmpidx[w])
                    ra.append(tmpra[w])
                    dec.append(tmpdec[w])
                    r.append(tmpr[w])

    return idx, ra, dec, r


@numba.njit
def intersect_lightcone(chi2_i, chi2_f, a, chi2):
    """Find the scale factor where a particle worldline intersects a lightcone.

    Parameters
    ----------
    chi2_i, chi2_f : float
        Squared comoving radii of the particle at initial/final snapshot.
    a : ndarray
        Scale factor grid defining the lightcone segment.
    chi2 : ndarray
        Squared comoving radial distance of the lightcone at each ``a`` sample.

    Returns
    -------
    float
        Intersecting scale factor a_int (linear interpolation between samples).

    Notes
    -----
    Assumes the particle radius evolves linearly with a between endpoints in
    squared-radius space, which is the approximation used in the caller.
    """
    a_i = a[0]
    a_f = a[-1]
    # y = chi2 - (chi2_i + (chi2_f - chi2_i) * (a - a_i) / (a_f - a_i))
    y_last = 0.
    dchida = (chi2_f - chi2_i) / (a_f - a_i)
    m = len(a)
    for j in range(1, m):
        #     if y[j] < 0.:
        #         return a[j - 1] - y[j - 1] * (a[j - 1] - a[j]) / (y[j - 1] - y[j])
        y_new = chi2[j] - (chi2_i + dchida * (a[j] - a_i))

        if y_new < 0.:
            return a[j - 1] - y_last * (a[j - 1] - a[j]) / (y_last - y_new)
        else:
            y_last = y_new


@numba.njit
def _worldline2pix_sub(nside, order, ncap, npix, a_lc, chi2_lc, x_i, y_i, z_i, x_f, y_f, z_f, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """Compute the Healpix pixel of a particle/lightcone intersection (single).

    Parameters
    ----------
    a_lc : ndarray
        Scale factor samples defining the lightcone along the snapshot interval.
    chi2_lc : ndarray
        Squared comoving distances chi(a)^2 along the same samples.
    (x_i,y_i,z_i), (x_f,y_f,z_f) : float
        Initial/final particle positions.
    shiftx, shifty, shiftz : float
        Replicate translation.
    rmin2, rmax2 : float
        Shell radial cuts.
    octant_only : bool
        Octant cut flag.

    Returns
    -------
    int
        Pixel index, or -1 if rejected.
    """
    # apply shift
    tmpx_i = x_i + shiftx
    tmpy_i = y_i + shifty
    tmpz_i = z_i + shiftz

    tmpx_f = x_f + shiftx
    tmpy_f = y_f + shifty
    tmpz_f = z_f + shiftz

    # get radius and remove particles out of shell
    r2_i = tmpx_i**2 + tmpy_i**2 + tmpz_i**2
    r2_f = tmpx_f**2 + tmpy_f**2 + tmpz_f**2
    if r2_f < rmin2 or r2_i >= rmax2:  # it is enough to test initial/final radii as it cannot evolve faster than light
        return -1

    # remove particles out of octant if octant_only
    if octant_only:  # check if particle out of octant
        if (tmpx_i < 0. and tmpx_f < 0.) or (tmpy_i < 0. and tmpy_f < 0.) or (tmpz_i < 0. and tmpz_f < 0.):
            return -1

    # find intersection of particle wordline and lightcone (ie find scale factor when particle comoving radius intersects the comoving radial distance function)
    a_int = intersect_lightcone(r2_i, r2_f, a_lc, chi2_lc)

    a_i = a_lc[0]
    a_f = a_lc[-1]

    assert a_i <= a_int and a_int < a_f

    a_ratio = (a_int - a_i) / (a_f - a_i)
    tmpx = tmpx_i + (tmpx_f - tmpx_i) * a_ratio
    tmpy = tmpy_i + (tmpy_f - tmpy_i) * a_ratio
    tmpz = tmpz_i + (tmpz_f - tmpz_i) * a_ratio

    # remove particles that intersect lightcone out of shell
    r2 = tmpx**2 + tmpy**2 + tmpz**2
    if r2 < rmin2 or r2 >= rmax2:
        return -1

    # remove particles out of octant if octant_only
    if octant_only:  # check if particle out of octant
        if (tmpx < 0.) or (tmpy < 0.) or (tmpz < 0.):
            return -1

    # get longitude (phi), altitude (nz) and distance to z-axis (sth) of particle projected on the unit sphere
    phi = np.arctan2(tmpy, tmpx)
    xl = 1. / np.sqrt(r2)
    nz = tmpz * xl
    sth = np.sqrt(tmpx**2 + tmpy**2) * xl

    return _loc2pix(nside, order, ncap, npix, nz, phi, sth)


@numba.njit(parallel=True)
def _worldline2pix(nside, order, ncap, npix, a_lc, chi_lc, x_i, y_i, z_i, x_f, y_f, z_f, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """Parallel worldline-to-pixel conversion for snapshot interpolation.

    Returns
    -------
    ndarray
        Pixel indices (int) of length N, with -1 for rejected particles.
    """
    n = len(x_i)
    ipix = np.zeros(n, dtype=np.int_)
    for i in numba.prange(n):
        ipix[i] = _worldline2pix_sub(
            nside, order, ncap, npix, a_lc, chi_lc, x_i[i], y_i[i], z_i[i], x_f[i], y_f[i], z_f[i], shiftx, shifty, shiftz, rmin2, rmax2, octant_only)
    return ipix


@numba.njit
def _inter_diff_helper(ar, sar_idx, ar1_size):
    """Helper to compute intersection/differences of two arrays after sorting.

    Parameters
    ----------
    ar : ndarray
        Concatenated array ``[ar1, ar2]``.
    sar_idx : ndarray
        Indices that sort ``ar``.
    ar1_size : int
        Length of ar1.

    Returns
    -------
    ar1_indices_in, ar2_indices_in, ar1_indices_out, ar2_indices_out : ndarray
        Indices of elements present in both arrays, and elements present only in
        ar1 or only in ar2 (with indices referred to each original array).
    """
    sar = ar[sar_idx]

    mask = sar[1:] == sar[:-1]

    ar1_indices_in = sar_idx[:-1][mask]
    ar2_indices_in = sar_idx[1:][mask] - ar1_size

    flag = np.concatenate((np.array([True]), ~mask, np.array([True])))
    z = sar_idx[flag[1:] & flag[:-1]]
    w = z >= ar1_size

    ar1_indices_out = z[~w]
    ar2_indices_out = z[w] - ar1_size

    return ar1_indices_in, ar2_indices_in, ar1_indices_out, ar2_indices_out


def inter_diff(ar1, ar2):
    """Return indices of intersecting and non-intersecting elements of two arrays.

    Parameters
    ----------
    ar1, ar2 : ndarray
        1D arrays to compare (typically particle ids). Duplicates are treated
        via adjacent matches in the sorted concatenation.

    Returns
    -------
    ar1_indices_in : ndarray
        Indices into ar1 that are present in ar2.
    ar2_indices_in : ndarray
        Indices into ar2 that are present in ar1.
    ar1_indices_out : ndarray
        Indices into ar1 that are not present in ar2.
    ar2_indices_out : ndarray
        Indices into ar2 that are not present in ar1.
    """
    ar = np.concatenate((ar1, ar2))
    sar_idx = pargsort(ar)

    return _inter_diff_helper(ar, sar_idx, ar1.size)


@numba.njit
def part_boundary_fix_helper(xi, xf, boxsize, fixed='i'):
    """Resolve periodic boundary crossings between two positions along one axis.

    Parameters
    ----------
    xi, xf : float
        Initial and final coordinate along one axis in [0, boxsize).
    boxsize : float
        Periodic box size.
    fixed : {'i', 'f'}, optional
        Which endpoint should remain fixed while shifting the other by +/- boxsize.

    Returns
    -------
    (float, float)
        Adjusted (xi, xf) after applying periodic correction.

    Notes
    -----
    If the displacement exceeds boxsize/2, the function assumes a wrap-around
    occurred and shifts one endpoint accordingly.
    """
    if np.abs(xf - xi) < boxsize / 2.:
        return (xi, xf)
    else:
        if fixed == 'i':
            if xi < xf:
                return (xi, xf - boxsize)
            else:
                return (xi, xf + boxsize)
        elif fixed == 'f':
            if xi < xf:
                return (xi + boxsize, xf)
            else:
                return (xi - boxsize, xf)
        else:
            raise ValueError


@numba.njit(parallel=True)
def part_boundary_fix(xyzi, xyzf, boxsize, fixed='i'):
    """Apply :func:`part_boundary_fix_helper` to all particles and axes.

    Parameters
    ----------
    xyzi, xyzf : ndarray
        Arrays of shape (N, 3) containing initial and final positions.
    boxsize : float
        Periodic box size.
    fixed : {'i', 'f'}, optional
        Which endpoint should remain fixed.

    Returns
    -------
    None
        Arrays are modified in-place.
    """
    n = len(xyzi)
    for i in numba.prange(n):
        for k in range(3):
            xyzi[i, k], xyzf[i, k] = part_boundary_fix_helper(xyzi[i, k], xyzf[i, k], boxsize=boxsize, fixed=fixed)
