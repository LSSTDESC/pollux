from __future__ import annotations

import itertools
import logging
import numbers
import warnings
import os
from functools import cached_property
from typing import Any

import healpy as hp
import healsparse as hsp
import numba
import numpy as np
import pyccl
import yaml
from astropy.io import fits
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm, trange
from parallel_sort import argsort as pargsort
from collections.abc import Iterable

from .cosmo import Tracer
from .map_builder import MapBuilder, MapSum
from .utils import quad, split_bins, arange_endpoint
from .reader import DummyReader, Reader
from .lensing import SnapshotBornLensingBuilder

logger = logging.getLogger(__name__)


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
    save_dir : str or None, optional
        Output directory where maps and metadata are written. Mutually
        exclusive with ``root_dir``.
    root_dir : str or None, optional
        Root directory from which ``save_dir`` is derived as
        ``{root_dir}/nside{nside}/seed{seed}/shells``. Mutually exclusive
        with ``save_dir``.
    base_fname : str or None, optional
        Prefix of the output file names. Default is ``shell_{reader.kind}``.
    seed : int, optional
        Random seed used to draw isometries/rotflips and for subsampling.
    dtype : data-type, optional
        dtype used to store shell map values.
    mask : ndarray or 'octant' or None, optional
        Footprint specification forwarded to :class:`~pollux.map_builder.MapBuilder`.
        Default is ``'octant'``. For ``'octant'``, particles outside
        x>0, y>0, z>0 are discarded during map construction.
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
        Particle subsampling configuration:

        - If float in (0, 1], keep that fraction of particles per sub read.
        - If dict with key ``'densitymax_z'``: choose ratios per shell to keep
          per-pixel S/N comparable to the shell at that redshift.
        - If dict with key ``'clmax_r'``: choose ratios per shell so that shot
          noise is below theory C_ell at lmax by the requested factor.
    **kwargs
        Forwarded to :class:`pollux.map_builder.MapBuilder`.

    Notes
    -----
    - ``apply_grid_isometry`` and ``apply_replicate_rotflip`` do not commute;
      only one should be used at a time.

    Attributes
    ----------
    kind : {'snapshot', 'lightcone'}
        Shell construction mode. 'snapshot' builds shells out of snapshot boxes;
        'lightcone' uses reader-provided shell edges (typically already binned).
    add_orphans : bool
        Whether orphan particles should be added in :meth:`combine_maps_subs`.
        ``False`` here; overridden to ``True`` in :class:`SnapshotInterpBuilder`.
    smoothed_field_what : str
        ``what`` of the smoothed density maps (``'density'``), see
        :meth:`~pollux.map_builder.MapBuilder.compute_maps_from_alms`.
    """

    add_orphans = False
    kind = 'snapshot'
    smoothed_field_what = 'density'

    def __init__(
            self,
            nshells: int,
            reader: Reader,
            nside: int,
            save_dir: str | None = None,
            root_dir: str | None = None,
            seed: int = 12345,
            dtype: np.dtype = np.uint64,
            mask: np.ndarray | str | None = 'octant',
            apply_grid_isometry: bool | list = False,
            group_isometries: bool | float | None = None,
            apply_replicate_rotflip: bool | dict = False,
            subsample_box: dict | None = None,
            base_fname: str | None = None,
            **kwargs: Any) -> None:
        self.nshells = nshells
        self.reader = reader
        self.cosmo = reader.cosmo
        self.seed = seed

        # Options for isometries and replicate rotation/flips
        # they do not commute, and the grid isometry is slow and applied once whereas replicate_rotflip is fast but applied to each replicate
        if apply_grid_isometry and apply_replicate_rotflip:
            raise ValueError("apply_grid_isometry and apply_replicate_rotflip cannot be combined")
        self.apply_grid_isometry = apply_grid_isometry
        self.apply_replicate_rotflip = apply_replicate_rotflip

        if root_dir is not None:
            if save_dir is not None:
                raise ValueError('Provide either root_dir or save_dir, not both.')
            # maps built on a padded footprint are kept apart from unpadded ones
            padding = kwargs.get('padding_deg') if mask is not None else None
            shells_dir = f'shells_pad{padding:g}' if padding else 'shells'
            save_dir = os.path.join(root_dir, f'nside{nside}', f'seed{seed}', shells_dir)
            kwargs.setdefault('wsp_dir', os.path.join(root_dir, f'nside{nside}'))

        if base_fname is None:
            base_fname = f'shell_{reader.kind}'

        if dtype == np.int32:
            warnings.warn(f"The max value for {dtype} might not be enough : {np.iinfo(dtype)}")

        super().__init__(
            nside=nside, dtype=dtype, save_dir=save_dir, base_fname=base_fname, nmaps=self.nshells, sacc_type="galaxy_density_cl", mask=mask, **kwargs)

        # Healpix geometry passed to the pixelization kernels: nside, order, ncap, npix (see _loc2pix)
        self._healpix_geometry = (nside, hp.nside2order(nside), (nside * nside - nside) << 1, self.npix)

        # Particle subsampling
        if not (subsample_box is None or isinstance(subsample_box, (dict, numbers.Real))):
            raise TypeError(f"subsample_box must be None, a float or a dict, not {subsample_box!r}")
        if isinstance(subsample_box, dict) and not {'densitymax_z', 'clmax_r'} & subsample_box.keys():
            raise ValueError(f"subsample_box must have a 'densitymax_z' or 'clmax_r' key, not {list(subsample_box)}")
        self.subsample_box = subsample_box

        # Draw random grid isometries
        if isinstance(self.apply_grid_isometry, Iterable):
            # user-provided isometries
            if len(self.apply_grid_isometry) != self.nshells:
                raise ValueError(f"apply_grid_isometry must have one (A, b) isometry per shell ({self.nshells})")
            for i in range(self.nshells):
                A, b = self.apply_grid_isometry[i]
                if A.shape != (3, 3):
                    raise ValueError(f"Isometry {i}: A must have shape (3, 3), not {A.shape}")
                if b.shape != (3,):
                    raise ValueError(f"Isometry {i}: b must have shape (3,), not {b.shape}")
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
        if isinstance(self.apply_replicate_rotflip, dict):
            # user-provided replicate rotation/flips
            warnings.warn("Shell builder using user-provided replicate rot/flip: make sure not to miss any!")
            self.replicate_rotflip = self.apply_replicate_rotflip
            self.apply_replicate_rotflip = True
        elif self.apply_replicate_rotflip:
            # draw random replicate rotation/flips
            # first get number of replicates per direction
            rmax = self.shell_edges_comoving_distance[-1]
            boxsize = self.reader.boxsize
            nreplicate = int(np.ceil(rmax / boxsize)) + 2  # not too sure about that 2, but it's fast to check...

            # then draw random rotation/flips (6 rotations, 8 flips, so max 48)
            rng = np.random.default_rng(self.seed)
            self.replicate_rotflip = {}
            for ix in range(-nreplicate, nreplicate + 1):
                for iy in range(-nreplicate, nreplicate + 1):
                    for iz in range(-nreplicate, nreplicate + 1):
                        self.replicate_rotflip[ix, iy, iz] = rng.integers(low=0, high=48)
            self.apply_replicate_rotflip = True
        else:
            # no replicate rotation/flips
            self.replicate_rotflip = {}
            self.apply_replicate_rotflip = False

        # Try getting shell information from reader, if not dummy.
        try:
            # make sure snapshots are larger than the shells
            z_edges = self.shell_edges_redshift
            boxsize = self.reader.boxsize
            diff = np.diff(self.shell_edges_comoving_distance)
            logger.info("  - Box size = %.2f Mpc.", boxsize)
            if not np.all(diff < boxsize):
                message = "Some shells are deeper than the simulation box!\n"
                for ishell in range(self.nshells):
                    if diff[ishell] > boxsize:
                        message += f"Shell {ishell} ({z_edges[ishell]:.2f}<z<{z_edges[ishell+1]:.2f}) has width {diff[ishell]:.2f} Mpc\n"
                warnings.warn(message)
        except NotImplementedError:
            warnings.warn("Reader does not implement `shell_edges_redshift` or `shell_edges_comoving_distance` or `boxsize.")
            assert isinstance(reader, DummyReader)

        # Dump useful metadata
        try:
            self.save_info()
        except Exception:
            warnings.warn("Could not save the metadata info.")

    @cached_property
    def shell_edges_comoving_distance(self) -> np.ndarray:
        """Comoving distances to shell edges.

        ndarray
            Comoving radial distances (Mpc) of all shell edges.
            Length is ``nshells + 1``.
        """
        return pyccl.comoving_radial_distance(self.cosmo, 1. / (1 + self.shell_edges_redshift))

    @cached_property
    def shell_edges_redshift(self) -> np.ndarray:
        """Redshifts of shell edges.

        ndarray
            Redshifts of all shell edges. Length is ``nshells + 1`` for
            ``kind='snapshot'`` and depends on the reader for ``kind='lightcone'``.

        Raises
        ------
        ValueError
            If ``self.kind`` is not recognized, or if ``shell_edges_match`` is invalid.
        """
        if self.kind == 'snapshot':
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

            return split_bins(self.reader.zsnapshots, func=func, ifunc=ifunc, zmax=self.reader.zmax)

        elif self.kind == 'lightcone':
            return self.reader.zsnapshots

        else:
            raise ValueError

    def mean_z_volume(self, ishell: int) -> float:
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
        z_edges = self.shell_edges_redshift
        zmin = z_edges[ishell]
        zmax = z_edges[ishell + 1]

        dVdz = lambda z: pyccl.comoving_angular_distance(self.cosmo, 1. / (1. + z))**2 / pyccl.background.h_over_h0(self.cosmo, 1. / (1. + z))

        return quad(lambda z: z * dVdz(z), zmin, zmax) / quad(dVdz, zmin, zmax)

    def shell_info(self, fname: str | None = None) -> np.ndarray:
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
        z_edges = self.shell_edges_redshift
        names = ['ishell', 'zsnapshot', 'zmin', 'zmax', 'chisnapshot', 'chimin', 'chimax', 'snapshot_id']
        dtypes = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'int']
        out = np.array(
            list(
                zip(
                    np.arange(self.nshells), self.reader.zsnapshots, z_edges[:-1], z_edges[1:],
                    self.cosmo.comoving_radial_distance(1 / (1 + self.reader.zsnapshots)), self.cosmo.comoving_radial_distance(1 / (1 + z_edges[:-1])),
                    self.cosmo.comoving_radial_distance(1 / (1 + z_edges[1:])), self.reader.snapshots)),
            dtype=list(zip(names, dtypes)))
        if fname is not None:
            if '.txt' in fname:
                np.savetxt(fname, out, header=' '.join(names), fmt='%8i %16.8f %16.8f %16.8f %16.8f %16.8f %16.8f %8i')
            else:
                np.save(fname, out)
            logger.debug("Saved to file %s.", fname)
        return out

    def save_info(self, ext: str = 'npy') -> None:
        """Write shell metadata, cosmology, and simulation info to disk.

        This method writes:
        - ``{base_fname}_info.{ext}`` : shell edge and density summaries
        - ``cosmo_ccl.yaml`` : CCL cosmology (when serializable)
        - ``sim_info.yaml`` : key simulation and map parameters

        Parameters
        ----------
        ext : {'npy', 'txt'}, optional
            Output format for shell summary.
        """
        # Shell info for later reference and compare to existing as a safe check
        if ext not in ['npy', 'txt']:
            raise ValueError(f"ext must be 'npy' or 'txt', not {ext!r}")
        fname = os.path.join(self.save_dir, self.base_fname + '_info.' + ext)

        # table of the shells: redshift and distance edges, densities
        z_edges = self.shell_edges_redshift
        pixel_density = [self.pixel_density(ishell) for ishell in range(self.nshells)]
        sky_density = [self.sky_density(ishell) for ishell in range(self.nshells)]
        if self.kind == 'snapshot':
            names = ['ishell', 'zsnapshot', 'zmin', 'zmax', 'chisnapshot', 'chimin', 'chimax', 'snapshot_id', 'pixel_density', 'sky_density']
            dtypes = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'int', 'float', 'float']
            shell_info = np.array(
                list(
                    zip(
                        np.arange(self.nshells), self.reader.zsnapshots, z_edges[:-1], z_edges[1:],
                        self.cosmo.comoving_radial_distance(1 / (1 + self.reader.zsnapshots)), self.cosmo.comoving_radial_distance(1 / (1 + z_edges[:-1])),
                        self.cosmo.comoving_radial_distance(1 / (1 + z_edges[1:])), self.reader.snapshots, pixel_density, sky_density)),
                dtype=list(zip(names, dtypes)))
            fmt = '%8i %16.8f %16.8f %16.8f %16.8f %16.8f %16.8f %8i %16.8f %16.8f'
        elif self.kind == 'lightcone':
            names = ['ishell', 'zmin', 'zmax', 'pixel_density', 'sky_density']
            dtypes = ['int', 'float', 'float', 'float', 'float']
            shell_info = np.array(list(zip(np.arange(self.nshells), z_edges[:-1], z_edges[1:], pixel_density, sky_density)), dtype=list(zip(names, dtypes)))
            fmt = '%8i %16.8f %16.8f %16.8f %16.8f'
        else:
            raise ValueError

        # compare with an existing table (the configuration must not have changed), or save it
        if ext == 'npy':
            if os.path.exists(fname):
                shell_info_existing = np.load(fname)
                logger.debug("Found and loaded existing shell info at %s.", fname)
                try:
                    for name, dtype in zip(names, dtypes):
                        if dtype == 'int':
                            np.testing.assert_array_equal(
                                shell_info[name], shell_info_existing[name], err_msg=f"Shell info {name} does not match existing file {fname}.")
                        elif dtype == 'float':
                            np.testing.assert_allclose(
                                shell_info[name], shell_info_existing[name], rtol=1e-5, err_msg=f"Shell info {name} does not match existing file {fname}.")
                except (TypeError, AssertionError) as e:
                    warnings.warn(f'Shell info file exists but does not match current config: {e}.')
            else:
                np.save(fname, shell_info)
                logger.debug("Saved to file %s.", fname)
        else:
            np.savetxt(fname, shell_info, header=' '.join(names), fmt=fmt)
            logger.debug("Saved to file %s.", fname)

        # Save cosmology
        fname = os.path.join(self.save_dir, "cosmo_ccl.yaml")
        if not os.path.exists(fname):
            try:
                self.reader.cosmo.write_yaml(os.path.join(self.save_dir, "cosmo_ccl.yaml"))
                logger.debug("Saved to file %s.", fname)
            except ValueError:
                warnings.warn("Cannot write cosmo yaml file, likely due to using non-serializable options (eg P(k)).")
        else:
            logger.debug("Found existing cosmology file at %s, skipping.", fname)

        # Save simulation and maps info
        data = {
            "nparticles": self.reader.nparticles,
            "boxsize": self.reader.boxsize,
            "mass_particles": self.reader.mass_particles,
            "box_comoving_density": self.reader.box_comoving_density,
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
            with open(fname, "w") as f:
                yaml.dump(data, f)
            logger.debug("Saved to file %s.", fname)
        else:
            logger.debug("Found existing simulation info file at %s, skipping.", fname)

    def random_isometries(self, group_isometries: bool | float | None = None, get_shell_identifier: bool = False) -> list:
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
        boxsize = self.reader.boxsize
        chi_edges = self.shell_edges_comoving_distance

        # maximum width a given isometry can be applied to
        if group_isometries is not None:
            # if grouped, fix as function of box size
            if group_isometries is True:
                dchi_iso_max = boxsize
            elif isinstance(group_isometries, numbers.Real) and not isinstance(group_isometries, bool):
                dchi_iso_max = boxsize * group_isometries
            else:
                raise ValueError(f"group_isometries must be None, True or a float, not {group_isometries!r}")
        else:
            # if not grouped, set to 0 to force drawing one isometry per shell
            dchi_iso_max = 0.

        Abs = []
        n_isometries = 0
        dchi_iso = np.inf
        shell_identifier = {}

        for ishell in range(self.nshells):
            dchi_shell = chi_edges[ishell + 1] - chi_edges[ishell]
            if dchi_iso + dchi_shell > dchi_iso_max:
                # if grouped shell width is larger than maximum width for isometry, draw a new one
                A, b = _random_isometry(rng, boxsize)
                n_isometries += 1
                dchi_iso = 0.
            dchi_iso += dchi_shell
            Abs.append([A.astype(self.reader.dtype), b.astype(self.reader.dtype)])
            shell_identifier[ishell] = n_isometries

        if group_isometries is not None:
            logger.info("  - Using %d isometries grouped within %.2f Mpc.", n_isometries, dchi_iso_max)
        else:
            logger.info("  - Using %d isometries.", n_isometries)

        if get_shell_identifier:
            return Abs, shell_identifier
        else:
            return Abs

    def estimate_replicates(self, ishell: int) -> float:
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
        chi_edges = self.shell_edges_comoving_distance
        box_volume = self.reader.boxsize**3

        shell_volume = 4. / 3. * np.pi * (chi_edges[ishell + 1]**3 - chi_edges[ishell]**3) * self.fsky

        return shell_volume / box_volume

    def get_valid_replicas(self) -> list:
        """Return valid replica indices for each (snapshot, sub-volume) job.

        Reads only box edges (no particle data) and calls
        :meth:`_get_valid_replicas` to determine which periodic-box replicas
        overlap each shell.

        Returns
        -------
        list
            List of tuples ``(ijob, replicates)`` where ``replicates`` is a list
            of (ix, iy, iz) replica indices that may overlap the shell.
        """
        replicates = []
        for ijob in trange(self.reader.njobs, desc='[get_valid_replicas]'):
            ishell, isub = self.reader.job_args(ijob)
            replicates.append((ijob, self._valid_replicas_sub(ishell, isub)))
        return replicates

    def _valid_replicas_sub(self, ishell: int, isub: int) -> list[tuple]:
        """Return replica indices (ix, iy, iz) that may overlap a given shell.

        Thin wrapper around :meth:`_valid_replicas_box` that reads
        ``box_edges_original`` from the reader before delegating.

        Parameters
        ----------
        ishell : int
            Shell index.
        isub : int
            Sub-volume index.

        Returns
        -------
        list of tuple
            Valid replica indices as (ix, iy, iz) integer triples.
        """
        _, box_edges_original = self.reader.read_sub(ishell, isub, read_box_edges_only=True)
        return self._valid_replicas_box(ishell, box_edges_original)

    def _valid_replicas_box(self, ishell: int, box_edges_original: np.ndarray) -> list[tuple]:
        """Geometry-level replica filter for a given box.

        Applies octant, sphere-intersection, and mask-footprint tests to the
        box defined by ``box_edges_original``, and returns the replica indices
        that could contribute particles to shell ``ishell``.  This is the single
        source of truth for replica geometry filtering; the numba map-building
        helpers iterate only over the list returned here.

        Parameters
        ----------
        ishell : int
            Shell index.
        box_edges_original : ndarray, shape (8, 3)
            Box corner positions in the original (un-shifted) frame.

        Returns
        -------
        list of tuple
            Valid replica indices as (ix, iy, iz) integer triples.
        """
        chi_edges = self.shell_edges_comoving_distance
        rmin2 = chi_edges[ishell]**2
        rmax2 = chi_edges[ishell + 1]**2
        boxsize = self.reader.boxsize
        dtype = box_edges_original.dtype
        nreplicate = int(np.ceil(np.sqrt(rmax2) / boxsize)) + 2
        replicas = list(itertools.product(range(-nreplicate, nreplicate + 1), repeat=3))
        logger.debug("Shell %d: checking %d^3 = %d replicates.", ishell, 2 * nreplicate + 1, (2 * nreplicate + 1)**3)

        A, b = self.Abs[ishell]
        # precompute isometry on box edges (replicates are then only a shift away)
        if self.apply_grid_isometry:
            box_edges = _Axpb(A, b, box_edges_original).astype(dtype)
        else:
            if not self.apply_replicate_rotflip:
                box_edges = box_edges_original

        if self.apply_replicate_rotflip:
            rot_dict = self.replicate_rotflip

        # dense footprint mask for the overlap test (cached on self; vectorized
        # boolean indexing below is both faster and far lighter than building a
        # Python set of tens of millions of pixel indices)
        mask_dense = self.get_mask(nest=True, padded=True) if self.footprint is not None else None

        valid = []
        # replicate the box
        for (ix, iy, iz) in self.tqdm(replicas, desc=f"[_valid_replicas_box] shell {ishell}"):
            # compute shift
            shift_original = np.expand_dims(np.array([ix, iy, iz]) * boxsize, 0).astype(dtype)
            shift = _Ax(A, shift_original) if self.apply_grid_isometry else shift_original

            if self.apply_replicate_rotflip:
                if (ix, iy, iz) not in rot_dict:
                    continue
                else:
                    box_edges = _rotflip(rot_dict[(ix, iy, iz)], box_edges_original, boxsize).astype(dtype)

            # apply shift to box edges
            box_edges_shifted = box_edges + shift

            # check if box intersects octant, otherwise skip
            # (to do so, we check if the box is fully contained in the x<0 or y<0 or z<0 half spaces)
            if self.octant_only and np.any(np.all(box_edges_shifted <= 0., axis=0)):
                continue

            # check if box intersects with shell, otherwise skip
            # first, check if box fully inside inner sphere
            if np.all(np.sum(box_edges_shifted**2, axis=1) < rmin2):
                continue

            # second, check if box intersects with outer sphere
            if self.apply_grid_isometry:
                if not _AABB_sphere_intersect(box_edges_original + shift_original, -b, rmax2):
                    continue
            else:
                if not _AABB_sphere_intersect(box_edges_shifted, np.zeros(3, dtype=dtype), rmax2):
                    continue

            # check if replica sky footprint overlaps the mask
            if mask_dense is not None:
                rep_ipix = _query_replica_ipix(box_edges_shifted, self.nside)
                # rep_ipix is None means the replica is guaranteed to overlap
                # every pixel (full-sky fallback), so skip the check entirely
                if rep_ipix is not None and not mask_dense[rep_ipix].any():
                    continue

            valid.append((ix, iy, iz))

        logger.debug("Shell %d: %d valid replicates out of %d.", ishell, len(valid), len(replicas))

        return valid

    def subsample_ratio(self, ishell: int) -> float:
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
        if isinstance(self.subsample_box, dict):
            if not hasattr(self, 'subsample_ratios'):
                # particles per full-sky pixel: fsky in estimate_replicates cancels fsky in
                # the per-footprint-pixel denominator, leaving pure geometry with no mask needed
                chi_edges = self.shell_edges_comoving_distance
                shell_volumes = 4. / 3. * np.pi * (chi_edges[1:]**3 - chi_edges[:-1]**3)
                ngal_pix = shell_volumes * self.reader.nparticles / (self.reader.boxsize**3 * self.npix)
                if 'densitymax_z' in self.subsample_box.keys():
                    snr_pix = 1. / np.sqrt(ngal_pix)  # S/N per pixel
                    # S/N at redshift subsample_box_auto_z
                    snr_pix_cut = np.interp(self.subsample_box['densitymax_z'], [self.mean_z_volume(ishell) for ishell in range(self.nshells)], snr_pix)
                    # fraction of particles to maintain S/N of z>subsample_box_auto_z
                    self.subsample_ratios = np.minimum(1., (snr_pix / snr_pix_cut)**2)
                elif 'clmax_r' in self.subsample_box.keys():
                    # compute theory Cl's
                    cl = self.predict_cls(auto_only=True, binning='unbinned')[1]
                    # fix particle density such that shot-noise is 100 times lower than Cl at lmax
                    nbar_th = self.subsample_box['clmax_r'] / np.array([cl[i, i][-1] for i in range(self.nshells)])
                    # convert to desired number of particles per pixel
                    ngal_th = nbar_th * hp.nside2pixarea(self.nside, degrees=False)
                    self.subsample_ratios = np.minimum(1., ngal_th / ngal_pix)
            return self.subsample_ratios[ishell]
        if self.subsample_box is None:
            return 1.
        return float(self.subsample_box)

    def _subsample(self, ishell: int, *arrays: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Subsample particles for a given shell, applying the same draw to all arrays.

        Parameters
        ----------
        ishell : int
            Shell index (determines the subsampling ratio and the RNG seed).
        *arrays : ndarray
            One or more particle arrays sharing the same leading length (e.g.
            ``xyz``, or ``xyz_i`` and ``xyz_f`` for worldline interpolation).
            The same index selection is applied to every array.

        Returns
        -------
        ndarray or tuple of ndarray
            Subsampled array(s). A single array is returned directly; multiple
            arrays are returned as a tuple.
        """
        ratio = self.subsample_ratio(ishell)
        n = len(arrays[0])
        nkept = int(n * ratio)
        logger.debug("Keeping %d particles, ie a fraction %s.", nkept, ratio)
        if nkept >= n:
            return arrays[0] if len(arrays) == 1 else arrays
        rng = np.random.default_rng(self.seed + ishell)
        idx = rng.choice(n, size=nkept, replace=False)
        out = tuple(a[idx] for a in arrays)
        return out[0] if len(out) == 1 else out

    def box_comoving_density(self, ishell: int) -> float:
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
        return self.reader.box_comoving_density * self.subsample_ratio(ishell)

    def pixel_density(self, ishell: int) -> float:
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
        chi_edges = self.shell_edges_comoving_distance

        chi_max = chi_edges[ishell + 1]
        chi_min = chi_edges[ishell]
        pixel_volume = 4. * np.pi / 3. * (chi_max**3 - chi_min**3) / self.npix

        return pixel_volume * self.box_comoving_density(ishell)

    def sky_density(self, ishell: int, degrees: bool = False) -> float:
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

    def construct_maps(self, imaps: int | Iterable[int] | None = None, **kwargs: Any) -> hsp.HealSparseMap | dict[int, hsp.HealSparseMap]:
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
        hsp.HealSparseMap or dict
            - If ``imaps`` is a single int: returns a single combined
              ``HealSparseMap``.
            - If ``imaps`` is ``None`` or an iterable: returns a dict keyed
              by shell index with combined ``HealSparseMap`` values.
        """
        # first construct shells on simulation subs
        for ishell in self.tqdm(imaps, desc="[construct_maps]"):
            isubs = self.reader.isubs_for_snapshot(ishell)
            for isub in tqdm(isubs, desc=f"[construct_maps: shell {ishell}]", leave=False):
                self.construct_maps_sub(ishell, isub, **kwargs)

        # then combine into final shells
        return self.combine_maps_subs(imaps)

    def combine_maps_subs(self,
                          ishells: int | Iterable[int] | None = None,
                          allow_missing_subs: bool = False) -> hsp.HealSparseMap | dict[int, hsp.HealSparseMap]:
        """Sum per-subvolume shell maps into final per-shell maps.

        Parameters
        ----------
        ishells : int or array_like of int or None, optional
            Shell indices to combine. If None, combines all shells.
        allow_missing_subs : bool, optional
            If True, iterate sub indices and skip missing files silently.
            Deprecated — only kept as a workaround for incomplete runs where
            not all sub-files were produced. Prefer completing all subs first.

        Returns
        -------
        hsp.HealSparseMap or dict
            - If ``ishells`` is a single int: returns a single combined
              ``HealSparseMap``.
            - If ``ishells`` is ``None`` or an iterable: returns a dict keyed
              by shell index with combined ``HealSparseMap`` values.

        Notes
        -----
        Orphan particles are added when ``self.add_orphans`` is True
        (set on :class:`SnapshotInterpBuilder`; False everywhere else).
        """
        shells = {}

        mask = self.get_mask(nest=True, padded=True)
        for ishell in self.tqdm(ishells, desc="[combine_maps_subs]"):
            fname = self.get_fname(ishell)
            shell = self.check_maps(read=True, fname=fname)
            if shell is not None:
                shells[ishell] = shell
                continue

            isubs = list(self.reader.isubs_for_snapshot(ishell))
            if self.add_orphans:
                isubs.append('orphan')
            shell_sum = MapSum(mask, self.nside_c, self.dtype)
            for _, sub_map in self.iter_maps(ishell, isubs=isubs, allow_missing=allow_missing_subs, desc=f"[combine_maps_subs: shell {ishell}]"):
                shell_sum.add(sub_map)
            shell = shell_sum.to_map()

            self._save_map(shell, fname)

            shells[ishell] = shell

        return shells[ishells] if np.isscalar(ishells) else shells

    def _count_to_shell(self, count: np.ndarray) -> hsp.HealSparseMap:
        """Convert dense per-pixel particle counts into a HealSparse shell map.

        Parameters
        ----------
        count : ndarray of int
            Dense RING-ordered array of length ``self.npix`` holding the number of
            particles per pixel (e.g. accumulated with :func:`_count_pix`).

        Returns
        -------
        hsp.HealSparseMap
            Shell map of dtype ``self.dtype`` whose valid pixels are the non-empty
            pixels of ``count`` inside the padded footprint.

        Notes
        -----
        The footprint mask is applied here, on non-empty pixels only, which is
        equivalent to (and cheaper than) filtering every particle hit.
        """
        uipix = np.flatnonzero(count)
        # drop out-of-footprint pixels (equivalent to filtering every particle)
        if self.footprint is not None:
            uipix = uipix[self.get_mask(nest=False, padded=True)[uipix]]
        c = count[uipix]
        logger.debug("Found %d particles in the shell.", int(c.sum(dtype=np.int64)))
        return self._make_hspmap(hp.ring2nest(self.nside, uipix), c, nest=True)

    def field_hspmap(self, ishell: int) -> hsp.HealSparseMap:
        """Return the density contrast of one shell as a HealSparse map.

        Parameters
        ----------
        ishell : int
            Shell index.

        Returns
        -------
        hsp.HealSparseMap
            Density contrast relative to the cosmic mean, on the padded footprint:
            ``shell_count / pixel_density - 1`` on pixels containing particles, and
            -1 on empty pixels.
        """
        shell = self.load_maps(ishell)
        _, ipix = self.get_mask(nest=True, return_ipix=True, padded=True)
        # -1 (no particle) on the padded footprint, then the density contrast where there are particles
        density = self._make_hspmap(ipix, np.full(len(ipix), -1.), nest=True)
        spix = shell.valid_pixels
        density.update_values_pix(spix, (shell[spix] / self.pixel_density(ishell) - 1.).astype(self.dtype), nest=True)
        return density

    def _smooth_sigma(self, ishell: int, smooth_Mpc: float) -> float:
        """Return the angular width (radians) of a comoving smoothing scale at the far edge of a shell."""
        return smooth_Mpc / self.shell_edges_comoving_distance[ishell + 1]

    def compute_tidal_tensor_maps(
            self,
            ishells: int | Iterable[int] | None = None,
            smooth_Mpc: float | None = None,
            lmax: int | None = None,
            save_field_map: bool = True) -> list[hsp.HealSparseMap] | dict[int, list[hsp.HealSparseMap]]:
        """Compute tidal tensor component maps from shell density maps.

        For each shell, computes a spin-2 transform of the E-mode a_lm of the
        density map, and combines it with the density (trace) to obtain
        approximate 2D tidal components (t11, t22, t12) on the sky.

        Parameters
        ----------
        ishells : int or array_like of int or None, optional
            Shell indices to process. If None, process all.
        smooth_Mpc : float or None, optional
            Comoving Gaussian smoothing scale in Mpc of the density a_lm (angular width
            at the shell's far edge, see :meth:`_smooth_sigma`).
        lmax : int or None, optional
            Maximum multipole used for harmonic transforms. Defaults to
            ``3*nside - 1``.
        save_field_map : bool, optional
            With ``smooth_Mpc``, also save the smoothed density map, from the same
            a_lm (as :meth:`~pollux.map_builder.MapBuilder.compute_maps_from_alms`).
            Default is True.

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

        _, ipix = self.get_mask(nest=False, return_ipix=True, padded=True)
        whats = ['t11', 't22', 't12']
        save_field_map = save_field_map and smooth_Mpc is not None

        ts = {}
        for ishell in self.tqdm(ishells, desc="[compute_tidal_tensor]"):
            t11 = self.check_maps(read=True, fname=self.get_fname(ishell, what='t11', smooth_Mpc=smooth_Mpc))
            t22 = self.check_maps(read=True, fname=self.get_fname(ishell, what='t22', smooth_Mpc=smooth_Mpc))
            t12 = self.check_maps(read=True, fname=self.get_fname(ishell, what='t12', smooth_Mpc=smooth_Mpc))
            field_done = not save_field_map or os.path.exists(self.get_fname(ishell, what=self.smoothed_field_what, smooth_Mpc=smooth_Mpc))
            if t11 is not None and t22 is not None and t12 is not None and field_done:
                ts[ishell] = [t11, t22, t12]
                continue

            alm = self.compute_alms(ishell, pol=False, lmax=lmax)

            if smooth_Mpc is not None:
                # smoothed density, from the a_lm smoothed in place (reused for the tidal field below)
                self._smooth_alms(ishell, alm, smooth_Mpc, lmax)
                density_hspmap = self._alm2hspmap(alm, ipix, lmax, mean=self._alm_field_mean(ishell))
                if save_field_map:
                    self._save_map(density_hspmap, self.get_fname(ishell, what=self.smoothed_field_what, smooth_Mpc=smooth_Mpc))
                density = density_hspmap.get_values_pix(ipix, nest=False)
            else:
                density = self.field_map(ishell, nest=False)[ipix]

            Q, U = hp.alm2map_spin((alm, np.zeros_like(alm)), self.nside, spin=2, lmax=lmax)
            Q = Q[ipix]
            U = U[ipix]

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
            vals = [density / 6. + Q / 2., density / 6. - Q / 2., U]  # t11, t22, t12

            ts[ishell] = []
            for what, val in zip(whats, vals, strict=True):
                hspmap = self._make_hspmap(ipix, val)
                tensor_fname = self.get_fname(ishell, what=what, smooth_Mpc=smooth_Mpc)
                self._save_map(hspmap, tensor_fname)
                ts[ishell].append(hspmap)

        return ts[ishells] if np.isscalar(ishells) else ts

    def _create_tracers(self) -> list[Tracer]:
        """Create one density tracer per shell, with a volume-weighted top-hat redshift distribution (1000 redshifts per shell).

        Returns
        -------
        list of pollux.cosmo.Tracer
            ``'density'`` tracers named ``density_{ishell}``, of length ``nshells``.
        """
        z_edges = self.shell_edges_redshift

        tracers = []
        for ishell in range(self.nshells):
            zz = np.linspace(z_edges[ishell], z_edges[ishell + 1], 1000)
            aa = 1. / (1. + zz)
            dVdz = pyccl.comoving_angular_distance(self.cosmo, aa)**2 / pyccl.background.h_over_h0(self.cosmo, aa)
            tracers.append(Tracer(f"density_{ishell}", 'density', z=zz, nz=dVdz / np.trapz(dVdz, zz)))

        return tracers

    def _noise_cls(self, ishell: int) -> float:
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

    def snapshot_lensing_builder(self, tag: str | None = 'zsnapshots') -> SnapshotBornLensingBuilder:
        """Return a lensing builder with one convergence map per shell snapshot.

        Parameters
        ----------
        tag : str or None, optional
            Tag used to name the output subdirectory: ``kappa_{tag}``.
            Default is ``'zsnapshots'`` gives ``kappa_zsnapshots/``.

        Returns
        -------
        pollux.lensing.SnapshotBornLensingBuilder
        """
        return SnapshotBornLensingBuilder(self, tag=tag)


class SnapshotShellBuilder(ShellBuilder):
    """ShellBuilder specialized for snapshot-based shells.

    This builder defines shell edges from snapshot redshifts using the selected
    ``shell_edges_match`` strategy and constructs each shell by replicating the
    periodic snapshot box in comoving coordinates.

    Parameters
    ----------
    reader : pollux.reader.Reader
        Simulation snapshot reader.
    nside : int
        Healpix nside of the shell maps.
    save_dir : str or None, optional
        Explicit output directory.  Mutually exclusive with ``root_dir``.
    root_dir : str or None, optional
        Root directory; the builder computes
        ``{root_dir}/nside{nside}/seed{seed}/shells/`` automatically.
    shell_edges_match : {'center_redshift','center_scale_factor','center_comoving_distance','center_comoving_volume'}, optional
        Defines how shell edges are placed relative to snapshot centers.
    **kwargs
        Forwarded to :class:`ShellBuilder`.
    """

    def __init__(
            self,
            reader: Reader,
            nside: int,
            save_dir: str | None = None,
            root_dir: str | None = None,
            shell_edges_match: str = 'center_scale_factor',
            **kwargs: Any) -> None:
        self.shell_edges_match = shell_edges_match

        super().__init__(reader.nsnapshots, reader, nside, save_dir=save_dir, root_dir=root_dir, **kwargs)

    def construct_maps_sub(self, ishell: int, isub: int, ijob: int | None = None) -> hsp.HealSparseMap:
        """Construct a single shell map for one snapshot/sub-volume.

        Parameters
        ----------
        ishell : int
            Shell index.
        isub : int
            Sub-volume index.
        ijob : int or None, optional
            If provided, overrides (ishell, isub) using ``reader.job_args(ijob)``.

        Returns
        -------
        hsp.HealSparseMap
            Constructed HealSparse shell map for this sub-volume.

        Notes
        -----
        This method reads a sub-volume, optionally subsamples particles, applies
        optional isometries / rotflips, replicates the box to cover the shell,
        bins particles into healpix, and writes a HealSparse map.
        """
        if ijob is not None:
            ishell, isub = self.reader.job_args(ijob)
        fname = self.get_fname(ishell, sub=isub)

        shell = self.check_maps(read=True, fname=fname)
        if shell is not None:
            return shell

        # get shell radii
        chi_edges = self.shell_edges_comoving_distance
        rmin2, rmax2 = chi_edges[ishell]**2, chi_edges[ishell + 1]**2
        boxsize = self.reader.boxsize

        # precompute valid replicas (reads only box edges) before loading particles
        valid_replicas = np.array(self._valid_replicas_sub(ishell, isub), dtype=np.int64).reshape(-1, 3)

        if len(valid_replicas) > 0:
            # read box
            xyz, _ = self.reader.read_sub(ishell, isub)
            logger.debug("Loaded box %d (shell %d) with %d particles.", isub, ishell, len(xyz))

            # particle sub-sampling
            xyz = self._subsample(ishell, xyz)

            A, b = self.Abs[ishell]

            # apply grid isometry once; _rotflip (if any) is applied per replica below
            dtype = xyz.dtype
            if self.apply_grid_isometry:
                xyz_work = _Axpb(A, b, xyz)
            elif not self.apply_replicate_rotflip:
                xyz_work = xyz

            count = np.zeros(self.npix, dtype=np.int32)  # per-pixel counts: int32 is enough
            ipix = np.empty(len(xyz), dtype=np.int32 if self.npix < 2**31 else np.int64)  # reused for all replicas
            buf = np.empty_like(ipix)  # buffer for scatter-add operation
            for ix, iy, iz in valid_replicas:
                shift_orig = np.expand_dims(np.array([ix, iy, iz]) * boxsize, 0).astype(dtype)
                shift = _Ax(A, shift_orig) if self.apply_grid_isometry else shift_orig
                if self.apply_replicate_rotflip:
                    xyz_work = _rotflip(self.replicate_rotflip[ix, iy, iz], xyz, boxsize).astype(dtype)

                # bin particles into healpix pixels
                _vec2pix(
                    ipix, *self._healpix_geometry, xyz_work[:, 0], xyz_work[:, 1], xyz_work[:, 2], shift[0, 0], shift[0, 1], shift[0, 2], rmin2, rmax2,
                    self.octant_only)

                # scatter-add (out-of-shell particles have ipix = -1 and are skipped)
                _count_pix(count, ipix, buf)

            shell = self._count_to_shell(count)
            del count, ipix, buf
        else:
            logger.debug("No valid replica — skipping particle load.")
            shell = self._make_hspmap()

        self._save_map(shell, fname)

        return shell

    def construct_catalogs_sub(self, ishell: int, isub: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct a particle catalog for a given shell/sub-volume.

        Parameters
        ----------
        ishell : int
            Shell index.
        isub : int
            Sub-volume index.
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
        chi_edges = self.shell_edges_comoving_distance
        rmin2, rmax2 = chi_edges[ishell]**2, chi_edges[ishell + 1]**2
        boxsize = self.reader.boxsize

        # precompute valid replicas (reads only box edges) before loading particles
        valid_replicas = np.array(self._valid_replicas_sub(ishell, isub), dtype=np.int64).reshape(-1, 3)
        if len(valid_replicas) == 0:
            return np.array([], dtype=np.intp), np.array([]), np.array([]), np.array([])

        # read box
        xyz, _ = self.reader.read_sub(ishell, isub, **kwargs)
        logger.debug("Loaded box %d (shell %d) with %d particles.", isub, ishell, len(xyz))

        # particle sub-sampling
        xyz = self._subsample(ishell, xyz)

        A, b = self.Abs[ishell]

        # apply grid isometry once; _rotflip (if any) is applied per replica below
        dtype = xyz.dtype
        if self.apply_grid_isometry:
            xyz_work = _Axpb(A, b, xyz)
        elif not self.apply_replicate_rotflip:
            xyz_work = xyz

        idx_list, ra_list, dec_list, r_list = [], [], [], []
        for ix, iy, iz in valid_replicas:
            shift_orig = np.expand_dims(np.array([ix, iy, iz]) * boxsize, 0).astype(dtype)
            shift = _Ax(A, shift_orig) if self.apply_grid_isometry else shift_orig
            if self.apply_replicate_rotflip:
                xyz_work = _rotflip(self.replicate_rotflip[ix, iy, iz], xyz, boxsize).astype(dtype)

            tmpidx, tmpra, tmpdec, tmpr = _vec2radec(
                xyz_work[:, 0], xyz_work[:, 1], xyz_work[:, 2], shift[0, 0], shift[0, 1], shift[0, 2], rmin2, rmax2, self.octant_only)
            w = tmpidx >= 0
            if np.any(w):
                idx_list.append(tmpidx[w])
                ra_list.append(tmpra[w])
                dec_list.append(tmpdec[w])
                r_list.append(tmpr[w])

        if idx_list:
            idx = np.concatenate(idx_list)
            ra = np.concatenate(ra_list)
            dec = np.concatenate(dec_list)
            r = np.concatenate(r_list)
        else:
            idx = np.array([], dtype=np.intp)
            ra = dec = r = np.array([])

        # post-filter: drop particles whose sky position falls outside the mask
        if self.footprint is not None and len(idx) > 0:
            ipix = hp.ang2pix(self.nside, ra, dec, nest=False, lonlat=True)
            _, mask_ipix_ring = self.get_mask(nest=False, return_ipix=True, padded=True)
            keep = np.isin(ipix, mask_ipix_ring)
            idx, ra, dec, r = idx[keep], ra[keep], dec[keep], r[keep]

        return idx, ra, dec, r

    def construct_catalogs(self,
                           ishells: int | Iterable[int] | None = None,
                           extra_columns: list[str] | None = None,
                           **kwargs: Any) -> fits.HDUList | dict[int, fits.HDUList]:
        """Construct and save a per-shell (halo/particle) FITS catalog.

        Parameters
        ----------
        ishells : int or array_like of int or None, optional
            Shell indices to construct. If None, constructs all shells.
        extra_columns : list of str, optional
            Extra properties to load using ``reader.read_props`` and store as
            additional FITS columns.
        **kwargs
            Forwarded to :meth:`construct_catalog_sub` / ``reader.read_sub``.

        Returns
        -------
        astropy.io.fits.HDUList or dict
            - If ``ishells`` is a single int: returns a FITS HDU list
              (primary HDU + one table HDU per sub-volume).
            - If ``ishells`` is ``None`` or an iterable: returns a dict keyed
              by shell index with FITS HDU list values.

        Notes
        -----
        Output filename is ``what='catalog'`` under :meth:`get_fname`.
        """
        extra_columns = extra_columns or []
        out = {}

        for ishell in self.tqdm(ishells, desc="[construct_catalogs]"):
            isubs = self.reader.isubs_for_snapshot(ishell)

            fname = self.get_fname(ishell, what='catalog')

            if os.path.exists(fname):
                try:
                    out[ishell] = fits.open(fname)
                    logger.debug("Found and loaded catalog at %s.", fname)
                    continue
                except Exception as e:
                    warnings.warn(f"Cannot read catalog at {fname}: {e}.")

            hdul = [fits.PrimaryHDU()]

            for isub in self.tqdm(isubs, desc=f"[construct_catalogs: shell {ishell}]"):
                # Construct catalog for simulation sub-volume/snapshot
                res = self.construct_catalogs_sub(ishell, isub, **kwargs)

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
                header['ISUB'] = isub
                header['ZSNAP'] = self.reader.zsnapshots[ishell]

                hdul.append(fits.BinTableHDU.from_columns(cols, header))

            fits.HDUList(hdul).writeto(fname, overwrite=True)
            logger.debug("Saved to file %s.", fname)

            out[ishell] = hdul

        return out[ishells] if np.isscalar(ishells) else out

    def load_catalogs(self, ishells: int | Iterable[int] | None = None, variables: list[str] | None = None) -> dict:
        """Load previously constructed per-shell catalogs as concatenated arrays.

        Parameters
        ----------
        ishells : int or array_like of int or None, optional
            Shell indices to load. If None, loads all shells.
        variables : list of str or None, optional
            If None, return every column stored in the catalog (as saved by
            :meth:`construct_catalogs`, e.g. ``'ihaloes'``, ``'ra'``, ``'dec'``,
            ``'chi'``, ``'z'``, and any ``extra_columns``). If given, any
            requested variable missing from the catalog is fetched from the
            reader via :meth:`_read_props`, and the returned dict is restricted
            to exactly the requested variables.

        Returns
        -------
        dict
            - If ``ishells`` is a single int: mapping from column/variable
              name to 1D concatenated numpy arrays.
            - If ``ishells`` is ``None`` or an iterable: mapping from shell
              index to the per-shell dict described above.
        """
        out = {}

        for ishell in self.tqdm(ishells, desc="[load_catalogs]"):
            fname = self.get_fname(ishell, what='catalog')
            with fits.open(fname) as hdul:
                columns = hdul[1].columns.names
                cat = {k: np.concatenate([hdu.data[k] for hdu in hdul[1:]]) for k in columns}
            logger.debug("Found and loaded catalog at %s.", fname)

            if variables is None:
                out[ishell] = cat
            else:
                missing = [v for v in variables if v not in cat]
                if missing:
                    logger.info("Shell %d: %s not in saved catalog, fetching from reader.", ishell, missing)
                    cat.update(self._read_props(ishell, missing))
                out[ishell] = {k: cat[k] for k in variables}

        return out[ishells] if np.isscalar(ishells) else out

    def _read_props(self, ishells: int | Iterable[int] | None, variables: list[str], save: bool = False) -> dict:
        """Read variables not stored in the catalog directly from the reader.

        Helper for callers (e.g. :class:`~pollux.baryons.BaryonifiedShellBuilder`)
        that already have the catalog-native columns from :meth:`load_catalogs`
        and only need to fetch the remaining ``variables`` (e.g. halo
        properties not saved via ``extra_columns``) from the original
        simulation files.

        Parameters
        ----------
        ishells : int or array_like of int or None
            Shell indices to read. If None, reads all shells.
        variables : list of str
            Variable names to load via :meth:`reader.read_props`.
        save : bool, optional
            If True, checks for and reuses a cached ``what='catalog_props'``
            FITS file (per shell) instead of re-reading the original
            simulation files, and updates that cache with any variables
            fetched from the reader on this call. If False, always reads
            fresh from the reader and never touches the cache file.

        Returns
        -------
        dict
            - If ``ishells`` is a single int: mapping from variable name to 1D
              concatenated numpy arrays.
            - If ``ishells`` is ``None`` or an iterable: mapping from shell
              index to the per-shell variable dicts described above.
        """
        out = {}

        for ishell in self.tqdm(ishells, desc="[_read_props]"):
            props_fname = self.get_fname(ishell, what='catalog_props')
            # properties saved by a previous call
            cached = {}
            if save and os.path.exists(props_fname):
                with fits.open(props_fname) as props_hdul:
                    cached = {k: np.array(props_hdul[1].data[k]) for k in props_hdul[1].columns.names}
                logger.debug("Found and loaded catalog_props at %s.", props_fname)

            # fetch the others from the reader, one sub-volume (catalog HDU) at a time
            missing = [v for v in variables if v not in cached]
            if missing:
                fname = self.get_fname(ishell, what='catalog')
                fetched = {k: [] for k in missing}
                with fits.open(fname) as hdul:
                    logger.debug("Found and loaded catalog at %s.", fname)
                    for hdu in self.tqdm(hdul[1:], desc=f"[_read_props: shell {ishell}]"):
                        tmp = self.reader.read_props(ishell, hdu.header['ISUB'], missing, ihaloes=hdu.data['ihaloes'])
                        for k in missing:
                            fetched[k].append(tmp[k])
                for k in missing:
                    fetched[k] = np.concatenate(fetched[k])

                cached.update(fetched)

                # save all properties (as float32) for later calls
                if save:
                    cols = [fits.Column(name=k, format='E', array=v) for k, v in cached.items()]
                    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(props_fname, overwrite=True)
                    logger.debug("Saved to file %s.", props_fname)

            out[ishell] = {k: cached[k] for k in variables}

        return out[ishells] if np.isscalar(ishells) else out


class LightconeSlicingBuilder(ShellBuilder):
    """ShellBuilder that slices a lightcone by stacking multiple snapshots.

    For each shell, this builder reads multiple lightcone snapshots whose
    redshift range intersects the shell, concatenates particles, then bins them
    into a shell map.

    Parameters
    ----------
    reader : pollux.reader.Reader
        Lightcone-capable reader providing ``zsnapshots_fulllist`` and
        ``read_sub(isnapshot, isub)`` for lightcone snapshots.
    nside : int
        Healpix nside of the shell maps.
    save_dir : str or None, optional
        Explicit output directory.  Mutually exclusive with ``root_dir``.
    root_dir : str or None, optional
        Root directory; the builder computes
        ``{root_dir}/nside{nside}/seed{seed}/shells/`` automatically.
    shell_edges_match : str, optional
        Shell-edge placement strategy (same meaning as for SnapshotShellBuilder).
    **kwargs
        Forwarded to :class:`ShellBuilder`.
    """

    def __init__(
            self,
            reader: Reader,
            nside: int,
            save_dir: str | None = None,
            root_dir: str | None = None,
            shell_edges_match: str = 'center_scale_factor',
            **kwargs: Any) -> None:
        self.shell_edges_match = shell_edges_match

        super().__init__(reader.nsnapshots, reader, nside, save_dir=save_dir, root_dir=root_dir, **kwargs)

    def construct_maps_sub(self, ishell: int, isub: int, ijob: int | None = None) -> hsp.HealSparseMap:
        """Construct a single shell map for one lightcone sub-volume.

        Parameters
        ----------
        ishell : int
            Shell index.
        isub : int
            Sub-volume index.
        ijob : int or None, optional
            If provided, overrides (ishell, isub) using ``reader.job_args(ijob)``.

        Returns
        -------
        hsp.HealSparseMap
            Constructed HealSparse shell map for this sub-volume.
        """
        if ijob is not None:
            ishell, isub = self.reader.job_args(ijob)
        fname = self.get_fname(ishell, sub=isub)

        # check if shell has already been created
        shell = self.check_maps(read=True, fname=fname)
        if shell is not None:
            return shell

        # get shell radii and redshifts
        chi_edges = self.shell_edges_comoving_distance
        rmin2, rmax2 = chi_edges[ishell]**2, chi_edges[ishell + 1]**2
        z_edges = self.shell_edges_redshift
        zmin, zmax = z_edges[ishell], z_edges[ishell + 1]

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

        if logger.isEnabledFor(logging.DEBUG):
            r2_xyz = xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2
            rmin2_xyz = r2_xyz.min()
            rmax2_xyz = r2_xyz.max()
            logger.debug(
                "Shell %d (z=[%.3f,%.3f]): particle radial distance range in box [%.3f, %.3f] Mpc, "
                "shell radial distance range [%.3f, %.3f] Mpc, %d particles in box.", ishell, zmin, zmax, np.sqrt(rmin2_xyz), np.sqrt(rmax2_xyz),
                np.sqrt(rmin2), np.sqrt(rmax2), len(xyz))

        # lightcone-slicing builder stacks particles from multiple snapshots without replication
        A, b = self.Abs[ishell]
        dtype = xyz.dtype
        xyz_work = _Axpb(A, b, xyz) if self.apply_grid_isometry else xyz
        shift_orig = np.zeros((1, 3), dtype=dtype)
        shift = _Ax(A, shift_orig) if self.apply_grid_isometry else shift_orig

        # bin particles into healpix pixels
        ipix = np.empty(len(xyz), dtype=np.int32 if self.npix < 2**31 else np.int64)
        buf = np.empty_like(ipix)
        _vec2pix(
            ipix, *self._healpix_geometry, xyz_work[:, 0], xyz_work[:, 1], xyz_work[:, 2], shift[0, 0], shift[0, 1], shift[0, 2], rmin2, rmax2,
            self.octant_only)

        count = np.zeros(self.npix, dtype=np.int32)  # per-pixel counts: int32 is enough
        # scatter-add (out-of-shell particles have ipix = -1 and are skipped)
        _count_pix(count, ipix, buf)
        shell = self._count_to_shell(count)
        del count, ipix, buf

        self._save_map(shell, fname)

        return shell


class SnapshotInterpBuilder(ShellBuilder):
    """ShellBuilder that interpolates particle worldlines between snapshots.

    This builder treats consecutive snapshot boxes as endpoints of particle
    worldlines and finds their intersections with a lightcone defined by a
    sequence of scale factors. It supports orphan handling when particles appear
    or disappear between snapshot subvolumes.

    Parameters
    ----------
    reader : pollux.reader.Reader
        Reader providing at least consecutive snapshots and particle ids.
    nside : int
        Healpix nside of the shell maps.
    save_dir : str or None, optional
        Explicit output directory.  Mutually exclusive with ``root_dir``.
    root_dir : str or None, optional
        Root directory; the builder computes
        ``{root_dir}/nside{nside}/seed{seed}/shells/`` automatically.
    **kwargs
        Forwarded to :class:`ShellBuilder`.

    Notes
    -----
    This builder uses ``kind='lightcone'`` with ``nshells = nsnapshots - 1``.
    """

    add_orphans = True
    kind = 'lightcone'

    def __init__(self, reader: Reader, nside: int, save_dir: str | None = None, root_dir: str | None = None, **kwargs: Any) -> None:
        super().__init__(reader.nsnapshots - 1, reader, nside, save_dir=save_dir, root_dir=root_dir, **kwargs)

    def construct_maps_sub(self, ishell: int, isub: int, ijob: int | None = None, do_orphan_ishell: int | None = None) -> hsp.HealSparseMap:
        """Construct one interpolated lightcone shell map for a sub-volume.

        Parameters
        ----------
        ishell : int
            Shell index: the interval between snapshots ``ishell`` and ``ishell + 1``.
        isub : int
            Sub-volume index.
        ijob : int or None, optional
            If provided, overrides (ishell, isub) using ``reader.job_args(ijob)``.
        do_orphan_ishell : int or None, optional
            If provided, build the orphan shell (stored under ``sub='orphan'``)
            by loading previously dumped orphan particle arrays.

        Returns
        -------
        hsp.HealSparseMap
            Constructed HealSparse shell map.

        Notes
        -----
        The lightcone is approximated by a tabulated chi(a) curve between the two
        snapshots, and particle worldlines (linear in a) are intersected with it
        to determine sky positions via :func:`_worldline2pix`.
        """
        if ijob is not None:
            ishell, isub = self.reader.job_args(ijob)
        if do_orphan_ishell is not None:
            ishell = do_orphan_ishell

        fname = self.get_fname(ishell, sub='orphan' if do_orphan_ishell is not None else isub)

        shell = self.check_maps(read=True, fname=fname)
        if shell is not None:
            return shell

        # get lightcone line
        z_edges = self.shell_edges_redshift
        a_i = 1. / (1. + z_edges[ishell + 1])
        a_f = 1. / (1. + z_edges[ishell])
        a_lc = arange_endpoint(a_i, a_f, 0.005)
        chi2_lc = self.cosmo.comoving_radial_distance(a_lc)**2
        rmin2, rmax2 = chi2_lc[-1], chi2_lc[0]
        boxsize = self.reader.boxsize

        # precompute valid replicas for the non-orphan case before loading particles
        if do_orphan_ishell is None:
            valid_replicas = np.array(self._valid_replicas_sub(ishell, isub), dtype=np.int64).reshape(-1, 3)
            if len(valid_replicas) == 0:
                shell = self._make_hspmap()
                self._save_map(shell, fname)
                return shell

        # read particles
        if do_orphan_ishell is not None:
            keys = ['x', 'y', 'z']
            xyzi = np.concatenate([np.load(self.get_fname(ishell, sub=isub, what='orphan_i', format='npy')) for isub in self.reader.isubs_for_snapshot(ishell)])
            xyzf = np.concatenate([np.load(self.get_fname(ishell, sub=isub, what='orphan_f', format='npy')) for isub in self.reader.isubs_for_snapshot(ishell)])
            logger.debug("Loaded %d orphan particles.", len(xyzi))

            # sort id's and match particles
            sidxi = np.argsort(xyzi['id'])
            sidxf = np.argsort(xyzf['id'])
            xyz_i = np.vstack([xyzi[sidxi][k] for k in keys]).T
            xyz_f = np.vstack([xyzf[sidxf][k] for k in keys]).T
            if logger.isEnabledFor(logging.DEBUG):
                np.testing.assert_array_equal(xyzi['id'][sidxi], xyzf['id'][sidxf])
                logger.debug("Matched particles.")

            # move particles that went through box faces
            _part_boundary_fix(xyz_i, xyz_f, boxsize, fixed='i')

            # recompute box edges accordingly
            box_lims = np.stack(
                [np.min([np.min(xyz_i, axis=0), np.min(xyz_f, axis=0)], axis=0),
                 np.max([np.max(xyz_i, axis=0), np.max(xyz_f, axis=0)], axis=0)])
            cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)
            box_edges_original = (box_lims[0] + (box_lims[1] - box_lims[0]) * cube_nodes).astype(self.dtype)

        else:
            # read box
            xyz_i, box_edges_original, idx_i = self.reader.read_sub(ishell, isub, extra_variables=['id'])
            xyz_f, box_edges_original, idx_f = self.reader.read_sub(ishell + 1, isub, extra_variables=['id'])

            logger.debug("Loaded box %d (snapshot %d) with %d particles.", isub, ishell, len(xyz_i))
            logger.debug("Loaded box %d (snapshot %d) with %d particles.", isub, ishell + 1, len(xyz_f))

            # match the indices of particles
            idx_i_in, idx_f_in, idx_i_out, idx_f_out = _inter_diff(idx_i['id'], idx_f['id'])
            assert len(idx_i_in) == len(idx_f_in)

            # dump orphan particles
            orphan_i_fname = self.get_fname(ishell, sub=isub, what='orphan_i', format='npy')
            np.save(orphan_i_fname, np.rec.array((*xyz_i[idx_i_out].T, idx_i['id'][idx_i_out]), names=['x', 'y', 'z', 'id']))
            logger.debug("Saved to file %s.", orphan_i_fname)

            orphan_f_fname = self.get_fname(ishell, sub=isub, what='orphan_f', format='npy')
            np.save(orphan_f_fname, np.rec.array((*xyz_f[idx_f_out].T, idx_f['id'][idx_f_out]), names=['x', 'y', 'z', 'id']))
            logger.debug("Saved to file %s.", orphan_f_fname)

            # keep matched particles
            xyz_i = xyz_i[idx_i_in]
            xyz_f = xyz_f[idx_f_in]

            logger.debug("Found %d particles in both boxes, %d/%d orphans in initial/final boxes saved.", len(idx_i_in), len(idx_i_out), len(idx_f_out))

        # particle sub-sampling
        xyz_i, xyz_f = self._subsample(ishell, xyz_i, xyz_f)

        # for the orphan case box_edges_original spans all subs and is only known after loading
        if do_orphan_ishell is not None:
            valid_replicas = np.array(self._valid_replicas_box(ishell, box_edges_original), dtype=np.int64).reshape(-1, 3)
        A, b = self.Abs[ishell]

        # apply grid isometry once; _rotflip (if any) is applied per replica below
        dtype = xyz_i.dtype
        if self.apply_grid_isometry:
            xyz_i_work = _Axpb(A, b, xyz_i)
            xyz_f_work = _Axpb(A, b, xyz_f)
        elif not self.apply_replicate_rotflip:
            xyz_i_work = xyz_i
            xyz_f_work = xyz_f

        count = np.zeros(self.npix, dtype=np.int32)  # per-pixel counts: int32 is enough
        ipix = np.empty(len(xyz_i), dtype=np.int32 if self.npix < 2**31 else np.int64)  # reused for all replicas
        buf = np.empty_like(ipix)
        for ix, iy, iz in valid_replicas:
            shift_orig = np.expand_dims(np.array([ix, iy, iz]) * boxsize, 0).astype(dtype)
            shift = _Ax(A, shift_orig) if self.apply_grid_isometry else shift_orig
            if self.apply_replicate_rotflip:
                xyz_i_work = _rotflip(self.replicate_rotflip[ix, iy, iz], xyz_i, boxsize).astype(dtype)
                xyz_f_work = _rotflip(self.replicate_rotflip[ix, iy, iz], xyz_f, boxsize).astype(dtype)

            # bin particles into healpix pixels
            _worldline2pix(
                ipix, *self._healpix_geometry, a_lc, chi2_lc, xyz_i_work[:, 0], xyz_i_work[:, 1], xyz_i_work[:, 2], xyz_f_work[:, 0], xyz_f_work[:, 1],
                xyz_f_work[:, 2], shift[0, 0], shift[0, 1], shift[0, 2], rmin2, rmax2, self.octant_only)

            # scatter-add (out-of-shell particles have ipix = -1 and are skipped)
            _count_pix(count, ipix, buf)

        shell = self._count_to_shell(count)
        del count, ipix, buf

        self._save_map(shell, fname)

        return shell


@numba.njit(parallel=True)
def _Axpb(A: np.ndarray, b: np.ndarray, xyz: np.ndarray) -> np.ndarray:
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
def _Ax(A: np.ndarray, xyz: np.ndarray) -> np.ndarray:
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


@numba.njit
def _rotflip(rot: int, xyz: np.ndarray, boxsize: float) -> np.ndarray:
    """Apply HACC-style discrete rotations (axis permutations) and sign flips.

    The 48 possible transforms are indexed by ``rot`` in [0, 48):
    - ``rot >> 3`` selects one of 6 coordinate permutations,
    - ``rot & 7`` selects one of 8 independent sign flips.

    Parameters
    ----------
    rot : int
        Rotation/flip code in [0, 48).
    xyz : ndarray
        Positions of shape (N, 3) in box coordinates [0, boxsize].
    boxsize : float
        Side length of the simulation box.

    Returns
    -------
    ndarray
        Transformed positions of shape (N, 3) in [0, boxsize].

    Notes
    -----
    The transform is applied about the box center. A replicate translation must
    still be applied after this operation.
    """
    x = xyz[:, 0] - boxsize / 2.
    y = xyz[:, 1] - boxsize / 2.
    z = xyz[:, 2] - boxsize / 2.

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

    xtmp += boxsize / 2.
    ytmp += boxsize / 2.
    ztmp += boxsize / 2.

    # note: depending on the replication, this box still needs to be translated to its place in the lightcone after this rotation,
    # and velocities, if required, need a similar shift
    return np.column_stack((xtmp, ytmp, ztmp))


def _random_isometry(rng: np.random.Generator, boxsize: float) -> tuple[np.ndarray, np.ndarray]:
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


def _AABB_sphere_intersect(box_edges: np.ndarray, s: np.ndarray, r2: float) -> bool:
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


def _query_replica_ipix(corners: np.ndarray, nside: int, nest: bool = True) -> np.ndarray | None:
    """Return HEALPix pixels overlapping a box replica.

    Projects the 8 box corners onto the unit sphere, builds a CCW-ordered
    convex hull in a local tangent plane, and calls :func:`healpy.query_polygon`
    with ``inclusive=True``.  If the polygon spans more than a hemisphere or
    the centroid is near the origin (box contains the observer), ``None`` is
    returned to signal that the replica overlaps every pixel unconditionally,
    without materializing the full pixel array.

    Parameters
    ----------
    corners : ndarray, shape (8, 3)
        Cartesian coordinates of the 8 box corners in the observer frame.
    nside : int
        HEALPix NSIDE.
    nest : bool, optional
        If True (default), return pixel indices in NESTED ordering.

    Returns
    -------
    ndarray or None
        Pixel indices overlapping the replica footprint, or ``None`` if the
        replica is guaranteed to overlap every pixel (full-sky fallback).
    """
    # Project corners onto the unit sphere
    unit = corners / np.linalg.norm(corners, axis=1, keepdims=True)

    # Build a tangent-plane frame at the centroid of the projected corners
    c = unit.mean(axis=0)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-10:
        # Centroid near origin: box straddles the observer, covers the full sky
        return None
    c /= norm_c
    ref = np.array([0., 0., 1.]) if abs(c[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(c, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)

    # A corner more than a quarter-turn from the centroid means the box spans
    # more than a hemisphere: an orthographic (dot-product) tangent-plane
    # projection would fold such points onto the wrong side, so its 2D convex
    # hull can come back in an order that traces a non-convex polygon on the
    # sphere and crashes query_polygon's C++ backend instead of raising a
    # catchable Python exception. Use a gnomonic (central) projection instead,
    # which preserves great-circle straightness within the hemisphere, and
    # fall back to full-sky whenever a corner falls outside it.
    dot_c = unit @ c
    if np.any(dot_c <= 1e-6):
        return None

    proj2d = np.column_stack([(unit @ e1) / dot_c, (unit @ e2) / dot_c])
    hull_2d = ConvexHull(proj2d)
    verts = unit[hull_2d.vertices]

    try:
        return hp.query_polygon(nside, verts, inclusive=True, nest=nest)
    except Exception:
        # Polygon spans more than a hemisphere: keep as full-sky candidate
        return None


inv_halfpi = 2. / np.pi


@numba.njit
def _loc2pix(nside: int, order: int, ncap: int, npix: int, z: float, phi: float, sth: float) -> int:
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
    if za <= 2. / 3.:  # equatorial region
        nl4 = 4 * nside
        temp1 = nside * (0.5 + tt)
        temp2 = nside * z * 0.75
        jp = int(temp1 - temp2)  # index of ascending edge line
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

    else:  # polar caps
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
def _vec2pix_kernel(
        nside: int, order: int, ncap: int, npix: int, x: float, y: float, z: float, shiftx: float, shifty: float, shiftz: float, rmin2: float, rmax2: float,
        octant_only: bool) -> int:
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
def _vec2pix(
        ipix: np.ndarray, nside: int, order: int, ncap: int, npix: int, x: np.ndarray, y: np.ndarray, z: np.ndarray, shiftx: float, shifty: float,
        shiftz: float, rmin2: float, rmax2: float, octant_only: bool) -> None:
    """Vectorized/parallel version of :func:`_vec2pix_kernel`, filling ``ipix`` in place.

    Parameters
    ----------
    ipix : ndarray of int
        Output pixel indices, same length as ``x``, with -1 for rejected particles.
    nside, order, ncap, npix : int
        Healpix geometry parameters (see :func:`_loc2pix`).
    x, y, z : ndarray
        Particle positions in box coordinates (1D arrays of equal length).
    shiftx, shifty, shiftz : float
        Replicate translation applied before binning.
    rmin2, rmax2 : float
        Shell radial cuts (squared).
    octant_only : bool
        If True, require x>=0, y>=0, z>=0 after shifting.
    """
    for i in numba.prange(len(x)):
        ipix[i] = _vec2pix_kernel(nside, order, ncap, npix, x[i], y[i], z[i], shiftx, shifty, shiftz, rmin2, rmax2, octant_only)


@numba.njit
def _vec2radec_kernel(x: float, y: float, z: float, shiftx: float, shifty: float, shiftz: float, rmin2: float, rmax2: float,
                      octant_only: bool) -> tuple[bool, float, float, float]:
    """Convert one shifted Cartesian position to (ra, dec, r) with cuts.

    Parameters
    ----------
    x, y, z : float
        Particle position in box coordinates.
    shiftx, shifty, shiftz : float
        Replicate translation applied first.
    rmin2, rmax2 : float
        Shell radial cuts (squared).
    octant_only : bool
        If True, require x>=0, y>=0, z>=0 after shifting.

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
def _vec2radec(x: np.ndarray, y: np.ndarray, z: np.ndarray, shiftx: float, shifty: float, shiftz: float, rmin2: float, rmax2: float,
               octant_only: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized/parallel version of :func:`_vec2radec_kernel`.

    Parameters
    ----------
    x, y, z : ndarray
        Particle positions in box coordinates (1D arrays of equal length).
    shiftx, shifty, shiftz : float
        Replicate translation applied first.
    rmin2, rmax2 : float
        Shell radial cuts (squared).
    octant_only : bool
        If True, require x>=0, y>=0, z>=0 after shifting.

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
        good, tmpra, tmpdec, tmpr = _vec2radec_kernel(x[i], y[i], z[i], shiftx, shifty, shiftz, rmin2, rmax2, octant_only)
        if good:
            idx[i] = i
            ra[i] = tmpra
            dec[i] = tmpdec
            r[i] = tmpr
        else:
            idx[i] = -1

    return idx, ra, dec, r


@numba.njit
def _intersect_lightcone(chi2_i: float, chi2_f: float, a: np.ndarray, chi2: np.ndarray) -> float:
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
        y_new = chi2[j] - (chi2_i + dchida * (a[j] - a_i))

        if y_new < 0.:
            return a[j - 1] - y_last * (a[j - 1] - a[j]) / (y_last - y_new)
        else:
            y_last = y_new


@numba.njit
def _worldline2pix_kernel(
        nside: int, order: int, ncap: int, npix: int, a_lc: np.ndarray, chi2_lc: np.ndarray, x_i: float, y_i: float, z_i: float, x_f: float, y_f: float,
        z_f: float, shiftx: float, shifty: float, shiftz: float, rmin2: float, rmax2: float, octant_only: bool) -> int:
    """Compute the Healpix pixel of a particle/lightcone intersection (single).

    Parameters
    ----------
    nside, order, ncap, npix : int
        Healpix geometry parameters (see :func:`_loc2pix`).
    a_lc : ndarray
        Scale factor samples defining the lightcone along the snapshot interval.
    chi2_lc : ndarray
        Squared comoving distances chi(a)^2 along the same samples.
    x_i, y_i, z_i, x_f, y_f, z_f : float
        Initial and final particle positions.
    shiftx, shifty, shiftz : float
        Replicate translation.
    rmin2, rmax2 : float
        Shell radial cuts (squared).
    octant_only : bool
        If True, require x>=0, y>=0, z>=0 after shifting.

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
    a_int = _intersect_lightcone(r2_i, r2_f, a_lc, chi2_lc)

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
def _worldline2pix(
        ipix: np.ndarray, nside: int, order: int, ncap: int, npix: int, a_lc: np.ndarray, chi2_lc: np.ndarray, x_i: np.ndarray, y_i: np.ndarray,
        z_i: np.ndarray, x_f: np.ndarray, y_f: np.ndarray, z_f: np.ndarray, shiftx: float, shifty: float, shiftz: float, rmin2: float, rmax2: float,
        octant_only: bool) -> None:
    """Parallel version of :func:`_worldline2pix_kernel`, filling ``ipix`` in place.

    Parameters
    ----------
    ipix : ndarray of int
        Output pixel indices, same length as ``x_i``, with -1 for rejected particles.
    nside, order, ncap, npix : int
        Healpix geometry parameters (see :func:`_loc2pix`).
    a_lc, chi2_lc : ndarray
        Lightcone samples, see :func:`_worldline2pix_kernel`.
    x_i, y_i, z_i, x_f, y_f, z_f : ndarray
        Initial and final particle positions (1D arrays of equal length).
    shiftx, shifty, shiftz : float
        Replicate translation.
    rmin2, rmax2 : float
        Shell radial cuts (squared).
    octant_only : bool
        If True, require x>=0, y>=0, z>=0 after shifting.
    """
    for i in numba.prange(len(x_i)):
        ipix[i] = _worldline2pix_kernel(
            nside, order, ncap, npix, a_lc, chi2_lc, x_i[i], y_i[i], z_i[i], x_f[i], y_f[i], z_f[i], shiftx, shifty, shiftz, rmin2, rmax2, octant_only)


@numba.njit(parallel=True)
def _count_pix(count: np.ndarray, ipix: np.ndarray, buf: np.ndarray, nranges: int = 4096) -> None:
    """Add one count per particle to a dense per-pixel count array, in place.

    Parallel histogram: valid pixel indices are first grouped by pixel range into
    ``buf`` (counting then scattering them per chunk of particles), then each range
    is counted by a single thread.

    Parameters
    ----------
    count : ndarray of int
        Dense per-pixel count array, updated in place.
    ipix : ndarray of int
        Pixel index of each particle (e.g. from :func:`_vec2pix`), with -1 marking
        rejected particles, which are skipped.
    buf : ndarray of int
        Work array, at least as long as ``ipix``, with a dtype able to hold pixel indices.
    nranges : int, optional
        Number of pixel ranges.
    """
    n = len(ipix)
    nchunks = numba.get_num_threads()
    shift = 0
    while (len(count) - 1) >> shift >= nranges:
        shift += 1

    # number of valid particles of each chunk in each pixel range
    hist = np.zeros((nchunks, nranges), np.int64)
    for c in numba.prange(nchunks):
        for i in range(c * n // nchunks, (c + 1) * n // nchunks):
            p = ipix[i]
            if p >= 0:
                hist[c, p >> shift] += 1

    # position in buf of each (chunk, range) block, grouped by range
    offset = np.empty((nchunks, nranges), np.int64)
    start = np.empty(nranges + 1, np.int64)
    total = 0
    for k in range(nranges):
        start[k] = total
        for c in range(nchunks):
            offset[c, k] = total
            total += hist[c, k]
    start[nranges] = total

    # scatter valid pixel indices into buf, grouped by range
    for c in numba.prange(nchunks):
        o = offset[c].copy()
        for i in range(c * n // nchunks, (c + 1) * n // nchunks):
            p = ipix[i]
            if p >= 0:
                k = p >> shift
                buf[o[k]] = p
                o[k] += 1

    # count each range
    for k in numba.prange(nranges):
        for j in range(start[k], start[k + 1]):
            count[buf[j]] += 1


@numba.njit
def _inter_diff_kernel(ar: np.ndarray, sar_idx: np.ndarray, ar1_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _inter_diff(ar1: np.ndarray, ar2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    return _inter_diff_kernel(ar, sar_idx, ar1.size)


@numba.njit
def _part_boundary_fix_kernel(xi: float, xf: float, boxsize: float, fixed: str = 'i') -> tuple[float, float]:
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
def _part_boundary_fix(xyzi: np.ndarray, xyzf: np.ndarray, boxsize: float, fixed: str = 'i') -> None:
    """Apply :func:`_part_boundary_fix_kernel` to all particles and axes.

    Parameters
    ----------
    xyzi, xyzf : ndarray
        Arrays of shape (N, 3) containing initial and final positions.
    boxsize : float
        Periodic box size.
    fixed : {'i', 'f'}, optional
        Which endpoint should remain fixed.

    Notes
    -----
    Arrays are modified in-place.
    """
    n = len(xyzi)
    for i in numba.prange(n):
        for k in range(3):
            xyzi[i, k], xyzf[i, k] = _part_boundary_fix_kernel(xyzi[i, k], xyzf[i, k], boxsize=boxsize, fixed=fixed)
