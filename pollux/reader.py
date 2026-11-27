"""Readers of N-body simulations (HACC, SLICS, ...): particles and halos of the snapshots or lightcones, cosmology and shell geometry."""

from __future__ import annotations

import itertools
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pyccl
from pynverse import inversefunc

from .utils.numerics import subsample_ppf, trim, trunc_norm

logger = logging.getLogger(__name__)

os.environ["GENERICIO_NO_MPI"] = "y"
try:
    import pygio
except Exception:
    warnings.warn("Unable to import pygio.")


class Reader(ABC):
    """Abstract base class for particle readers.

    A reader provides a uniform interface to access particle positions and
    optional per-particle properties from N-body simulations, possibly split
    into snapshots and sub-volumes.

    Parameters
    ----------
    cosmo : pyccl.Cosmology
        Cosmology object.
    kind : str
        Reader identifier used for naming outputs.
    unit : {'Mpc', 'Mpc_over_h'}
        Unit of particle coordinates returned by :meth:`read_sub`.
    nparticles : int
        Total number of particles in the simulation.
    zsnapshots : array_like
        Snapshot redshifts (ordered by increasing redshift).
    zmax : float or None
        Maximum redshift considered by downstream products. If not None, all
        values in ``zsnapshots`` must be strictly smaller than ``zmax``.
    nsubs : int
        Number of sub-volumes per snapshot.
    dtype : data-type, optional
        Floating dtype used for returned particle coordinates (e.g. ``np.float32``).
        This controls the dtype of arrays returned by :meth:`read_sub` (and the
        ``read_props`` of halo readers). Do not confuse this with builder/map dtypes which
        control stored map precision; the name is intentionally generic but
        means coordinates for readers.

    Attributes
    ----------
    asnapshots : ndarray
        Snapshot scale factors, defined as ``1/(1+zsnapshots)``.
    nsnapshots : int
        Number of snapshots.
    njobs : int
        Total number of independent jobs (snapshot, sub-volume).
    mass_particles : float
        Particle mass in Msun (computed from mean matter density and box volume).

    Notes
    -----
    Subclasses must implement :attr:`boxsize` and :meth:`read_sub` (abstract). Halo readers also read
    properties per object (``read_props``, e.g. :meth:`HACCHaloReader.read_props`), for the
    ``extra_columns`` of the halo catalogs (:meth:`pollux.shells.SnapshotShellBuilder.construct_catalogs`).
    """

    def __init__(
        self,
        cosmo: pyccl.Cosmology,
        kind: str,
        unit: str,
        nparticles: int,
        zsnapshots: NDArray,
        zmax: float | None,
        nsubs: int,
        dtype: Any = np.float64,
    ) -> None:
        self.cosmo = cosmo
        self.kind = kind

        if unit not in ["Mpc", "Mpc_over_h"]:
            raise ValueError("`unit` must be one of {'Mpc', 'Mpc_over_h'}.")
        self.unit = unit

        self.nparticles = int(nparticles)

        # Derived particle mass (assumes boxsize is in Mpc, not Mpc/h)
        rho_m = pyccl.background.rho_x(self.cosmo, 1.0, "matter", is_comoving=True)  # Msun / Mpc^3
        vol = self.boxsize**3  # Mpc^3
        self.mass_particles = rho_m * vol / self.nparticles if self.nparticles > 0 else np.nan

        self.zsnapshots = np.asarray(zsnapshots)
        self.zmax = zmax
        if zmax is not None:
            assert np.all(self.zsnapshots < zmax)
        self.asnapshots = 1.0 / (1.0 + self.zsnapshots)
        self.nsnapshots = len(self.zsnapshots)
        self.nsubs = int(nsubs)

        # Independent jobs (snapshot, sub-volume)
        self._job_args = list(itertools.product(range(self.nsnapshots), range(self.nsubs)))
        self.njobs = len(self._job_args)

        self.dtype = dtype

    @property
    @abstractmethod
    def boxsize(self) -> float:
        """Comoving box side length in Mpc."""

    @property
    def box_comoving_density(self) -> float:
        """Comoving particle number density in Mpc^-3, consistent with :attr:`boxsize`."""
        return self.nparticles / self.boxsize**3

    # ==================================================================================================================
    # region Sub-volumes

    def job_args(self, ijob: int) -> tuple[int, int]:
        """Return ``(isnapshot, isub)`` for a global job index.

        Parameters
        ----------
        ijob : int
            Index in the Cartesian product of snapshots and sub-volumes.

        Returns
        -------
        isnapshot : int
            Snapshot index.
        isub : int
            Sub-volume index.
        """
        return self._job_args[ijob]

    def isubs_for_snapshot(self, isnapshot: int) -> NDArray:
        """Return the sub-volume indices of the jobs of one snapshot.

        Parameters
        ----------
        isnapshot : int
            Snapshot index.

        Returns
        -------
        ndarray of int
            Sub-volume indices ``isub`` of the jobs whose snapshot is ``isnapshot``.
        """
        return np.array([x[1] for x in self._job_args if x[0] == isnapshot])  # in case self._job_args is overwritten in subclasses

    def ijobs_for_snapshot(self, isnapshot: int) -> NDArray:
        """Return global job indices for all sub-volumes of one snapshot.

        Parameters
        ----------
        isnapshot : int
            Snapshot index.

        Returns
        -------
        ndarray of int
            Job indices whose snapshot component equals ``isnapshot``.
        """
        return np.array([i for i, x in enumerate(self._job_args) if x[0] == isnapshot])  # in case self._job_args is overwritten in subclasses

    # endregion

    # ==================================================================================================================
    # region Reading

    @abstractmethod
    def read_sub(
        self,
        isnapshot: int,
        isub: int,
        assert_xyz_in_box: bool = False,
        read_box_edges_only: bool = False,
        extra_variables: list[str] | None = None,
    ) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, dict[str, NDArray]]:
        """Read a snapshot sub-volume and return particle positions.

        Parameters
        ----------
        isnapshot : int
            Snapshot index.
        isub : int
            Sub-volume index.
        assert_xyz_in_box : bool, optional
            If True, assert that returned coordinates fall within the returned
            sub-volume bounding box.
        read_box_edges_only : bool, optional
            If True, return only the sub-volume box edges and skip particle I/O.
        extra_variables : list of str, optional
            Additional fields to return (reader-specific). If None, no extra
            fields are returned.

        Returns
        -------
        xyz : ndarray
            Particle positions of shape ``(N, 3)``.
        box_edges : ndarray
            Cube vertices of the sub-volume, shape ``(8, 3)`` (or compatible).
        extra : dict, optional
            Returned only if ``extra_variables`` is not empty. maps variable name
            to a 1D array aligned with the returned particles.
        """

    # endregion


class SLICSReader(Reader):
    """Reader for SLICS particle snapshots.

    This reader supports the SLICS binary particle files produced by Joachim
    Harnois-Deraps.

    Parameters
    ----------
    path : str
        Directory containing SLICS particle files named like ``{z:.3f}xv{isub}.dat``.
    cosmo : pyccl.Cosmology
        Cosmology object.
    boxsize : float, optional
        Box side length in Mpc. Default is ``505/0.6898``.
    nparticles : int, optional
        Total number of particles. Default is ``1536**3``.
    nodes_dim : int, optional
        Number of MPI tasks per dimension (total sub-volumes = ``nodes_dim**3``).
        Default is 4.
    ncc : int, optional
        Integer used by the SLICS file format. Default is 768.
    rnc : float, optional
        Scale used by the SLICS file format. Default is 3072.0.
    zmax : float or None, optional
        Maximum redshift considered by downstream products.
    """

    def __init__(
        self,
        path: str,
        cosmo: pyccl.Cosmology,
        boxsize: float = 505.0 / 0.6898,
        nparticles: int = 1536**3,
        nodes_dim: int = 4,
        ncc: int = 768,
        rnc: float = 3072.0,
        zmax: float | None = None,
    ) -> None:
        self.path = path
        self._boxsize = float(boxsize)
        self.nodes_dim = int(nodes_dim)
        self.ncc = int(ncc)
        self.rnc = float(rnc)

        # Snapshot redshifts inferred from the available files
        zsnapshots = np.sort([float(x[:5]) for x in filter(lambda _x: "xv0.dat" in _x, os.listdir(self.path))])

        # Number of sub boxes
        nsubs = self.nodes_dim**3

        super().__init__(cosmo, "slics", "Mpc", nparticles, zsnapshots, zmax, nsubs, dtype=np.float32)

    @property
    def boxsize(self) -> float:
        """Simulation box side length in Mpc."""
        return self._boxsize

    def read_sub(
        self,
        isnapshot: int,
        isub: int,
        assert_xyz_in_box: bool = False,
        read_box_edges_only: bool = False,
        extra_variables: list[str] | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Read a SLICS snapshot sub-volume.

        Parameters
        ----------
        isnapshot : int
            Snapshot index.
        isub : int
            Sub-volume index in ``[0, nodes_dim**3)``.
        assert_xyz_in_box : bool, optional
            If True, validate that particle coordinates lie within the returned
            box edges.
        read_box_edges_only : bool, optional
            If True, return only box edges.
        extra_variables : list of str, optional
            Not supported by this reader. Must be None or empty.

        Returns
        -------
        xyz : ndarray
            Particle positions of shape ``(N, 3)`` in Mpc.
        box_edges : ndarray
            Cube vertices of the sub-volume, shape ``(8, 3)``.

        Raises
        ------
        ValueError
            If ``extra_variables`` is provided and non-empty.
        """
        if extra_variables not in (None, []):
            raise ValueError("SLICSReader does not support `extra_variables`.")

        z = self.zsnapshots[isnapshot]
        fname = os.path.join(self.path, f"{z:.3f}xv{isub}.dat")

        # node coordinates of the sub-volume: isub = i + j * nodes_dim + k * nodes_dim**2
        n = self.nodes_dim
        if not 0 <= isub < n**3:
            raise ValueError("Invalid sub-volume index.")
        node_coords = np.array([[isub % n, (isub // n) % n, isub // n**2]])

        # vertices of the sub-volume
        cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)
        box_edges = (node_coords + cube_nodes) * self.ncc * self._boxsize / self.rnc

        if read_box_edges_only:
            return np.zeros((17, 3), dtype=np.float32), box_edges

        # particles (x, y, z, vx, vy, vz) after a 12-value header, in node-local grid units
        with open(fname, "rb") as file:
            logger.debug("Reading %s.", fname)
            data = np.fromfile(file, dtype=np.float32)[12:]
            data = data.reshape((data.size // 6, 6))

        # global positions in Mpc, wrapped in the periodic box
        xyz = ((data[:, :3] + node_coords * self.ncc) % self.rnc) * self._boxsize / self.rnc

        if assert_xyz_in_box:
            for i in range(3):
                if not (np.all(np.min(box_edges[:, i]) <= xyz[:, i]) and np.all(xyz[:, i] <= np.max(box_edges[:, i]))):
                    raise RuntimeError(f"Particles outside the sub-volume along axis {i}")

        return xyz.astype(np.float32), box_edges.astype(np.float32)


class _HACCBaseReader(Reader):
    """Base of the HACC readers: the parameters of the run and the scale factors of its time steps."""

    @property
    def boxsize(self) -> float:
        """Simulation box side length in Mpc."""
        return self.boxsize_h / self.cosmo["h"]

    @staticmethod
    def _read_run(
        path: str,
        run_params: str,
        matter_power_spectrum: Any | None = None,
    ) -> tuple[pyccl.Cosmology, dict[str, Any], NDArray, float, float, float]:
        """Read the parameter file of a HACC run and build its cosmology.

        Parameters
        ----------
        path : str
            Root path to HACC simulation outputs.
        run_params : str
            Relative path to parameter file.
        matter_power_spectrum : object, optional
            Matter power spectrum to use when building cosmology from parameters.

        Returns
        -------
        cosmo : pyccl.Cosmology
            Cosmology object.
        params : dict
            Raw parameter dictionary.
        snapshots : ndarray
            Snapshot step indices.
        boxsize_h : float
            Box size in Mpc/h.
        sim_zmax : float
            Initial redshift.
        sim_zmin : float
            Final redshift.
        """
        fname = os.path.join(path, run_params)
        logger.info("  - Reading parameter file at %s.", fname)
        params = _HACCBaseReader._read_param_file(fname)

        # cosmology (HACC's DEUT is Omega_b h^2)
        cosmo_dict = dict(
            Omega_c=float(params["OMEGA_CDM"]),
            Omega_b=float(params["DEUT"]) / float(params["HUBBLE"])**2,
            h=float(params["HUBBLE"]),
            n_s=float(params["NS"]),
            sigma8=float(params["SS8"]),
            Neff=3.046,
            w0=float(params["W_DE"]),
            wa=float(params["WA_DE"]),
            T_CMB=2.726,
            Omega_k=0.0,
        )
        if matter_power_spectrum is not None:
            cosmo_dict["matter_power_spectrum"] = matter_power_spectrum
        cosmo = pyccl.Cosmology(**cosmo_dict)
        logger.info("  - Cosmology = %s.", cosmo)

        # time steps, redshift range and box size (in Mpc/h)
        nsteps = int(params["N_STEPS"])
        sim_zmax = float(params["Z_IN"])
        sim_zmin = float(params["Z_FIN"])
        boxsize_h = float(params["RL"])

        # snapshot steps of the parameter file, or those found in the output folder if they differ
        snapshots = np.array(list(map(int, params["ALIVE_DUMP"].split())))
        snapshot_dir = os.path.join(path, "output")
        if os.path.exists(snapshot_dir):
            try:
                snapshot_dirs = os.listdir(snapshot_dir)
                snapshots_output = np.array([i for i in range(nsteps) if f"STEP{i}" in snapshot_dirs])
                np.testing.assert_array_equal(snapshots, snapshots_output)
            except AssertionError:
                snapshots = snapshots_output
                warnings.warn("Using snapshots found in the `output` folder, which do not match those in the parameter file.")
        logger.info("  - Found %d snapshots: %s.", len(snapshots), snapshots)

        return cosmo, params, snapshots, boxsize_h, sim_zmax, sim_zmin

    @staticmethod
    def _read_param_file(fname: str) -> dict[str, str]:
        """Read a HACC parameter file into a dict (see :meth:`_read_run`).

        The expected format is one entry per line::

            KEY value...

        Comment lines starting with ``#`` and blank lines are ignored.

        Parameters
        ----------
        fname : str
            File path to the parameter file.

        Returns
        -------
        dict
            Mapping from parameter names to raw string values.

        Notes
        -----
        Values are not parsed to numeric types here; callers are expected to cast
        them to float/int as needed.
        """
        with open(fname, 'r') as f:
            params = {}
            for x in f:
                if not (x.startswith('#') or x.startswith('\n')):
                    try:
                        k, v = x.rstrip().split(' ', 1)
                        params[k] = v
                    except ValueError:
                        pass
        return params

    @staticmethod
    def _step_scale_factors(nsteps: int, sim_zmax: float, sim_zmin: float) -> NDArray:
        """Return the scale factors of the time steps of a HACC run, linear in scale factor from ``sim_zmax`` to ``sim_zmin``.

        Parameters
        ----------
        nsteps : int
            Number of time steps of the run.
        sim_zmax, sim_zmin : float
            Initial and final redshifts of the run.

        Returns
        -------
        ndarray
            Scale factor of each step, indexed by step (length ``nsteps``).
        """
        return np.linspace(1.0 / (1.0 + sim_zmax), 1.0 / (1.0 + sim_zmin), nsteps + 1)[1:]

    @staticmethod
    def _subsample_snapshots(
        snapshots: NDArray,
        a_steps: NDArray,
        snap_sub_aminstep: float = 0.0,
        snap_sub_zmax: float | None = None,
        snap_sub_match_dist_n: bool | int = False,
        snap_sub_every_n: int = 1,
    ) -> NDArray:
        """Subsample HACC snapshots based on various criteria.

        Parameters
        ----------
        snapshots : ndarray
            Initial snapshot steps.
        a_steps : ndarray
            Scale factor of each time step (:meth:`_step_scale_factors`).
        snap_sub_aminstep : float, optional
            Minimum scale factor step.
        snap_sub_zmax : float or None, optional
            Maximum redshift cutoff.
        snap_sub_match_dist_n : bool or int, optional
            Target for distribution matching.
        snap_sub_every_n : int, optional
            Keep every n-th snapshot.

        Returns
        -------
        ndarray
            Subsampled snapshot steps.
        """
        # Subsample: minimum step in scale factor
        if snap_sub_aminstep > 0.0:
            _, idx = trim(a_steps[snapshots], snap_sub_aminstep, return_indices=True)
            logger.info("  - Subsampling: minimum a step %s, returns %d snapshots.", snap_sub_aminstep, len(idx))
            snapshots = snapshots[idx]

        # Subsample: maximum redshift
        if snap_sub_zmax is not None:
            w = np.argwhere(a_steps[snapshots] > 1.0 / (1.0 + snap_sub_zmax))
            if len(w) > 0:
                snapshots = snapshots[w[0, 0]:]
                logger.info("  - Subsampling: max redshift %s, returns %d snapshots.", snap_sub_zmax, len(snapshots))

        # Subsample: match inferred bimodal distribution
        if snap_sub_match_dist_n:
            aa = a_steps[snapshots]

            # multimodal distribution in scale factor, need ppf (computed numerically)
            s1, s2, lam = 0.12, 0.27, 0.4
            amin = np.min(aa)
            t1 = trunc_norm(amin, 1.0, 0.5, s1)
            t2 = trunc_norm(amin, 1.0, 1.0, s2)
            cdf = lambda a: lam * t1.cdf(a) + (1 - lam) * t2.cdf(a)
            ppf = inversefunc(cdf, domain=[amin, 1.0], image=[0.0, 1.0])

            idx, _ = subsample_ppf(aa, snap_sub_match_dist_n, ppf)
            if len(idx) < len(snapshots):
                logger.info("  - Subsampling: match distribution, returns %d snapshots.", len(idx))
            snapshots = snapshots[idx]

        # Subsample: keep every n-th snapshot
        snapshots = snapshots[::-snap_sub_every_n]
        if snap_sub_every_n > 1:
            logger.info("  - Subsampling: keeping every %d snapshots, returns %d snapshots.", snap_sub_every_n, len(snapshots))

        return snapshots


class HACCReader(_HACCBaseReader):
    """Reader for HACC snapshot particle data via GenericIO.

    The reader parses the HACC run parameter file to build a CCL cosmology and to
    infer the set of snapshots available on disk. The data are organized in
    sub-volumes indexed by ``isub``.

    Parameters
    ----------
    path : str
        Root path to the HACC simulation outputs.
    run_params : str
        Relative path to the HACC parameter file.
    snap_sub_zmax : float or None, optional
        If provided, keep only snapshots with ``z <= snap_sub_zmax``.
    snap_sub_aminstep : float, optional
        Minimum step in scale factor used to downsample snapshots.
    snap_sub_match_dist_n : bool or int, optional
        If truthy, subsample snapshots to match a target (bimodal) distribution
        in scale factor, keeping roughly this many snapshots.
    snap_sub_every_n : int, optional
        Keep every n-th snapshot after other selections. Default is 1.
    snapshots : array_like of int or None, optional
        If provided, use this list of snapshot steps. Otherwise, use the steps
        from the parameter file (and optionally check the output directory).
    group_along_x : bool, optional
        If True, group sub-boxes along x to form larger contiguous volumes
        (reduces the number of sub-volumes).
    matter_power_spectrum : object, optional
        Forwarded to :class:`pyccl.Cosmology` as ``matter_power_spectrum``.
    **kwargs
        Ignored, so that the same options can configure :class:`HACCHaloReader`
        (e.g. ``halo_path``).
    """

    def __init__(
        self,
        path: str,
        run_params: str,
        snap_sub_zmax: float | None = None,
        snap_sub_aminstep: float = 0.0,
        snap_sub_match_dist_n: bool | int = False,
        snap_sub_every_n: int = 1,
        snapshots: NDArray | None = None,
        group_along_x: bool = False,
        matter_power_spectrum: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.path = path
        self.group_along_x = group_along_x
        self.variables: list[str] = ["x", "y", "z"]

        logger.info("*** HACC reader ***")

        # Read run parameters and build cosmology
        cosmo, params, snapshots_all, boxsize_h, sim_zmax, sim_zmin = self._read_run(path, run_params, matter_power_spectrum=matter_power_spectrum)

        nsteps = int(params["N_STEPS"])
        a_steps = self._step_scale_factors(nsteps, sim_zmax, sim_zmin)

        # Handle snapshot subsampling
        if snapshots is not None:
            self.snapshots = snapshots
        else:
            self.snapshots = self._subsample_snapshots(
                snapshots_all,
                a_steps,
                snap_sub_aminstep=snap_sub_aminstep,
                snap_sub_zmax=snap_sub_zmax,
                snap_sub_match_dist_n=snap_sub_match_dist_n,
                snap_sub_every_n=snap_sub_every_n,
            )

        zsnapshots = 1.0 / a_steps[self.snapshots] - 1.0

        # Simulation properties
        self.boxsize_h = boxsize_h
        nparticles = int(params["NP"])**3
        dtype = np.float32

        # Simulation box limits (hard-coded for 64 sub-volumes)
        self.lims = np.array([[
            [np.floor(isub / 16) * 0.5, np.floor(isub / 16) * 0.5 + 2.5],
            [np.floor(isub / 4) % 4, (np.floor(isub / 4) % 4) + 1],
            [isub % 4, (isub % 4) + 1],
        ] for isub in range(64)])

        nsub = 16 if self.group_along_x else 64
        super().__init__(cosmo, "hacc", "Mpc_over_h", nparticles, zsnapshots, snap_sub_zmax, nsub, dtype=dtype)

    def read_sub(
        self,
        isnapshot: int,
        isub: int,
        assert_xyz_in_box: bool = False,
        read_box_edges_only: bool = False,
        extra_variables: list[str] | None = None,
    ) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, dict[str, NDArray]]:
        """Read a HACC snapshot sub-volume.

        Parameters
        ----------
        isnapshot : int
            Snapshot index.
        isub : int
            Sub-volume index.
        assert_xyz_in_box : bool, optional
            If True, check the returned positions lie within ``box_edges``.
        read_box_edges_only : bool, optional
            If True, return only ``box_edges`` and skip particle I/O.
        extra_variables : list of str or None, optional
            Additional GenericIO variables to load and return.

        Returns
        -------
        xyz : ndarray
            Particle positions of shape ``(N, 3)`` in Mpc.
        box_edges : ndarray
            Cube vertices of the sub-volume, shape ``(8, 3)``.
        extra : dict, optional
            Only if ``extra_variables`` is provided and non-empty.
        """
        if extra_variables is None:
            extra_variables = []
        if assert_xyz_in_box and read_box_edges_only:
            raise ValueError("assert_xyz_in_box and read_box_edges_only cannot be combined")

        # vertices of the sub-volume (or of its column along x, if grouped)
        snapshot = self.snapshots[isnapshot]
        lims = np.array([[0, 4], self.lims[isub][1], self.lims[isub][2]] if self.group_along_x else [self.lims[isub][i] for i in range(3)])
        box_lims = lims * self.boxsize / 4.0
        cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)
        box_edges = np.stack([box_lims[i, cube_nodes[:, i]] for i in range(3)], axis=1).astype(self.dtype)

        if read_box_edges_only:
            return np.zeros((17, 3), dtype=self.dtype), box_edges

        if pygio is None:
            raise RuntimeError("pygio is required to read HACC data but could not be imported.")

        # read the GenericIO file of the sub-volume (the 4 files along x, if grouped)
        out = {k: [] for k in (self.variables + extra_variables)}
        nx = 4 if self.group_along_x else 1
        for ix in range(nx):
            ifile = ix * 16 + isub
            name = self._get_name(snapshot, ifile)
            logger.debug("Reading %s.", name)
            data = pygio.read_genericio(name, variables=self.variables + extra_variables)
            for k in (self.variables + extra_variables):
                out[k].append(data[k])

        # positions in Mpc
        xyz = np.array([np.concatenate(out[k]) for k in self.variables]).T
        xyz /= self.cosmo["h"]

        if assert_xyz_in_box:
            for i in range(3):
                if not (np.all(np.min(box_edges[:, i]) <= xyz[:, i]) and np.all(xyz[:, i] <= np.max(box_edges[:, i]))):
                    raise RuntimeError(f"Particles outside the sub-volume along axis {i}")
        else:
            # otherwise, edges of the box actually spanned by the particles (which may slightly exceed the sub-volume)
            box_lims = np.stack([np.min(xyz, axis=0), np.max(xyz, axis=0)])
            cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)
            box_edges = box_lims[0] + (box_lims[1] - box_lims[0]) * cube_nodes

        if len(extra_variables) > 0:
            return xyz.astype(self.dtype), box_edges.astype(self.dtype), {k: np.concatenate(out[k]) for k in extra_variables}
        return xyz.astype(self.dtype), box_edges.astype(self.dtype)

    def _get_name(self, step: int, isub: int) -> str:
        """Return the GenericIO particle file of one step and sub-volume.

        Parameters
        ----------
        step : int
            Time step of the snapshot.
        isub : int
            Sub-volume.

        Returns
        -------
        str
        """
        return os.path.join(self.path, "output", f"STEP{step}/m000.full.mpicosmo.{step}#{isub}")


class HACCLightconeReader(_HACCBaseReader):
    """Reader for HACC lightcone particle data.

    This reader loads GenericIO lightcone files stored under a ``Lightcones/``
    directory, for :class:`~pollux.shells.LightconeShellBuilder`.

    Unlike the other readers, :meth:`read_sub` returns only the particle positions
    (lightcone files have no sub-volume edges) and takes none of the other readers'
    options (``assert_xyz_in_box``, ``read_box_edges_only``, ``extra_variables``), so
    it cannot serve the snapshot-based shell builders.

    Parameters
    ----------
    path : str
        Root path to the HACC simulation outputs.
    run_params : str
        Relative path to the HACC parameter file.
    lightcone_path : str
        Path (relative to ``path``) containing lightcone files.
    snapshots : array_like of int or None, optional
        If provided, use this list of snapshot steps. Otherwise, use the steps
        from the parameter file (and optionally check the output directory).
    snap_sub_zmax : float or None, optional
        Maximum redshift considered by downstream products.
    snap_sub_every_n : int, optional
        Keep every n-th snapshot. Default is 1.
    matter_power_spectrum : object, optional
        Forwarded to :class:`pyccl.Cosmology`.
    nsub : int, optional
        Number of lightcone sub-volumes (files) per step. Default is 32.
    """

    def __init__(
        self,
        path: str,
        run_params: str,
        lightcone_path: str,
        snapshots: NDArray | None = None,
        snap_sub_zmax: float | None = None,
        snap_sub_every_n: int = 1,
        matter_power_spectrum: Any | None = None,
        nsub: int = 32,
    ) -> None:
        self.path = path
        self.lightcone_path = lightcone_path
        self.variables: list[str] = ["x", "y", "z"]

        logger.info("*** HACC lightcone reader ***")

        # Read run parameters and build cosmology
        cosmo, params, snapshots_param, boxsize_h, sim_zmax, sim_zmin = self._read_run(path, run_params, matter_power_spectrum=matter_power_spectrum)

        nsteps = int(params["N_STEPS"])

        if snapshots is None:
            snapshots_param_list = snapshots_param
        else:
            snapshots_param_list = snapshots

        a_steps = self._step_scale_factors(nsteps, sim_zmax, sim_zmin)

        # Store full list for lightcone indexing
        self.snapshots_fulllist = snapshots_param_list[::-1]
        self.zsnapshots_fulllist = 1.0 / a_steps[self.snapshots_fulllist] - 1.0

        # Subsample by every n-th snapshot
        self.snapshots = snapshots_param_list[::-snap_sub_every_n]
        if snap_sub_every_n > 1:
            logger.info("  - Subsampling: keeping every %d snapshots, returns %d snapshots.", snap_sub_every_n, len(self.snapshots))

        zsnapshots = 1.0 / a_steps[self.snapshots] - 1.0

        # Simulation properties
        self.boxsize_h = boxsize_h
        nparticles = int(params["NP"])**3
        dtype = np.float32

        # Sub-volume limits (for consistency with HACCReader)
        self.lims = np.array([[
            [np.floor(ibox / 16) * 0.5, np.floor(ibox / 16) * 0.5 + 2.5],
            [np.floor(ibox / 4) % 4, (np.floor(ibox / 4) % 4) + 1],
            [ibox % 4, (ibox % 4) + 1],
        ] for ibox in range(64)])

        super().__init__(cosmo, "hacc", "Mpc_over_h", nparticles, zsnapshots, snap_sub_zmax, int(nsub), dtype=dtype)

    def read_sub(self, isnapshot: int, isub: int) -> NDArray:
        """Read one lightcone file and return particle positions.

        Unlike :meth:`Reader.read_sub`, returns positions only (no ``box_edges``) and
        takes no options.

        Parameters
        ----------
        isnapshot : int
            Snapshot index in the *full* snapshot list used for lightcones.
        isub : int
            Sub-volume index.

        Returns
        -------
        xyz : ndarray
            Particle positions of shape ``(N, 3)`` in Mpc.
        """
        if pygio is None:
            raise RuntimeError("pygio is required to read HACC lightcone data but could not be imported.")

        snapshot = self.snapshots_fulllist[isnapshot]
        file = self._get_name(snapshot, isub)
        out = pygio.read_genericio(file, variables=["x", "y", "z"], print_stats=False)
        xyz = {k: data / self.cosmo["h"] for k, data in out.items()}
        return np.column_stack([xyz["x"], xyz["y"], xyz["z"]])

    def _get_name(self, step: int, isub: int) -> str:
        """Return the GenericIO lightcone file of one step and sub-volume.

        Parameters
        ----------
        step : int
            Time step of the snapshot.
        isub : int
            Sub-volume.

        Returns
        -------
        str
        """
        return os.path.join(self.path, self.lightcone_path, f"STEP{step}", f"m000.lc.mpicosmo.{step}#{isub}")


class HACCHaloReader(HACCReader):
    """Reader for HACC halo catalogs (GenericIO halo properties files).

    Parameters
    ----------
    halo_path : str
        Path (relative to ``path``) containing halo property files.
    *args, **kwargs
        Forwarded to :class:`HACCReader`.
    """

    def __init__(self, halo_path: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kind = "hacc_halo"
        self.variables: list[str] = ["fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"]
        self.halo_path = halo_path
        assert not self.group_along_x

    def read_props(
            self,
            isnapshot: int,
            isub: int,
            props_variables: list[str],
            ihaloes: slice | NDArray = slice(-1),
    ) -> dict[str, NDArray]:
        """Read halo properties for one snapshot sub-volume.

        Parameters
        ----------
        isnapshot : int
            Snapshot index.
        isub : int
            Sub-volume index.
        props_variables : list of str
            Variable names to load from the halo properties files.
        ihaloes : slice or array_like, optional
            Row selection applied on the loaded arrays.

        Returns
        -------
        dict
            Mapping from property name to selected array.
        """
        if pygio is None:
            raise RuntimeError("pygio is required to read HACC halo data but could not be imported.")

        snapshot = self.snapshots[isnapshot]
        name = self._get_name(snapshot, isub)
        logger.debug("Reading %s.", name)
        data = pygio.read_genericio(name, variables=props_variables)
        return {k: data[k][ihaloes] for k in props_variables}

    def _get_name(self, step: int, isub: int) -> str:
        """Return the GenericIO halo file of one step and sub-volume.

        Parameters
        ----------
        step : int
            Time step of the snapshot.
        isub : int
            Sub-volume.

        Returns
        -------
        str
        """
        return os.path.join(self.path, self.halo_path, f"STEP{step}/m000-{step}.haloproperties#{isub}")


class SkySimReader(Reader):
    """Reader for SkySim-like density map outputs (file-based snapshot selection).

    This reader infers snapshot steps from a directory containing files named
    ``density_map_{step}_dens.bin`` and converts the step list to redshifts using
    a linear scale-factor schedule.

    Parameters
    ----------
    path : str
        Directory containing density map files.
    cosmo : pyccl.Cosmology
        Cosmology object.
    nsteps : int, optional
        Number of time steps in the simulation. Default is 500.
    sim_zmin, sim_zmax : float, optional
        Simulation redshift bounds used to build the linear scale-factor grid.
    aminstep : float, optional
        Minimum step in scale factor used to downsample snapshots.
    zmax : float or None, optional
        Maximum redshift considered by downstream products.
    """

    def __init__(
        self,
        path: str,
        cosmo: pyccl.Cosmology,
        nsteps: int = 500,
        sim_zmin: float = 0.0,
        sim_zmax: float = 200.0,
        aminstep: float = 0.0,
        zmax: float | None = None,
    ) -> None:
        self.path = path
        self.aminstep = aminstep

        # snapshots: the steps with a mass sheet (but step 259)
        mass_sheets_fname = list(sorted(filter(lambda s: s.endswith(".bin"), os.listdir(path))))
        if "density_map_259_dens.bin" in mass_sheets_fname:
            mass_sheets_fname.remove("density_map_259_dens.bin")
        snapshots = np.array([i for i in range(nsteps) if f"density_map_{i}_dens.bin" in mass_sheets_fname])

        # their scale factors, with steps linear in a between the initial and final redshifts
        amax = 1.0 / (1.0 + sim_zmin)
        amin = 1.0 / (1.0 + sim_zmax)
        asnapshots_all = np.linspace(amin, amax, nsteps + 1)[1:]
        asnapshots_all = asnapshots_all[snapshots]

        # keep snapshots spaced by at least aminstep in a
        asnapshots, idx = trim(asnapshots_all, self.aminstep, return_indices=True)
        if len(idx) < len(snapshots):
            logger.info("Subsampling snapshots with aminstep=%s returns %d snapshots.", aminstep, len(idx))
        zsnapshots = 1.0 / asnapshots[::-1] - 1.0
        self.snapshots = snapshots[list(reversed(idx))]

        # redshifts up to the first snapshot beyond zmax
        if zmax is not None:
            try:
                i = np.argwhere(zsnapshots > zmax)[0, 0]
                zsnapshots = zsnapshots[:i + 1]
            except IndexError:  # no snapshot beyond zmax
                pass

        # fixed properties of the simulation
        nparticles = 10240**3
        self.boxsize_h = 3e3
        dtype = np.float32
        nsub = 1

        super().__init__(cosmo, "skysim", "Mpc_over_h", nparticles, zsnapshots, zmax, nsub, dtype=dtype)

    @property
    def boxsize(self) -> float:
        """Simulation box side length in Mpc."""
        return self.boxsize_h / self.cosmo["h"]

    def read_sub(
        self,
        isnapshot: int,
        isub: int,
        assert_xyz_in_box: bool = False,
        read_box_edges_only: bool = False,
        extra_variables: list[str] | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Not available: the outputs of SkySim are mass sheets (Healpix density maps), not particles.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("SkySim outputs are mass sheets (Healpix density maps), not particles.")


class SyntheticReader(Reader):
    """Synthetic simulation, for tests and examples: particles drawn uniformly at random in a periodic box.

    The box is split into ``nsplit**3`` cubic sub-volumes, each holding ``nparticles_per_sub`` particles per snapshot,
    drawn on the fly from a generator seeded by ``(seed, isnapshot, isub)``: the same particles at every read, without
    any file. With ``nparticles_per_sub=0``, a reader without particles (e.g. for a shell builder of predictions only).

    Parameters
    ----------
    cosmo : pyccl.Cosmology or None, optional
        Cosmology. Default is ``pyccl.CosmologyVanillaLCDM()``.
    boxsize : float, optional
        Box side length in Mpc. Default is 500.
    zsnapshots : array_like, optional
        Snapshot redshifts, increasing. Default is ``(0.1, 0.3, 0.5)``.
    zmax : float or None, optional
        Maximum redshift of the products (above the last snapshot). Default is None.
    nsplit : int, optional
        Sub-volumes per dimension. Default is 1.
    nparticles_per_sub : int, optional
        Particles per sub-volume and snapshot. Default is 10000.
    seed : int, optional
        Seed of the particle positions. Default is 0.
    dtype : data-type, optional
        Type of the coordinates. Default is float32.
    """

    def __init__(
        self,
        cosmo: pyccl.Cosmology | None = None,
        boxsize: float = 500.,
        zsnapshots: NDArray | tuple = (0.1, 0.3, 0.5),
        zmax: float | None = None,
        nsplit: int = 1,
        nparticles_per_sub: int = 10000,
        seed: int = 0,
        dtype: Any = np.float32,
    ) -> None:
        self._boxsize = float(boxsize)
        self.nsplit = int(nsplit)
        self.nparticles_per_sub = int(nparticles_per_sub)
        self.seed = int(seed)
        cosmo = pyccl.CosmologyVanillaLCDM() if cosmo is None else cosmo
        super().__init__(cosmo, 'synthetic', 'Mpc', self.nsplit**3 * self.nparticles_per_sub, zsnapshots, zmax, self.nsplit**3, dtype=dtype)
        self.snapshots = list(range(self.nsnapshots))

    @property
    def boxsize(self) -> float:
        """Box side length in Mpc."""
        return self._boxsize

    def read_sub(
        self,
        isnapshot: int,
        isub: int,
        assert_xyz_in_box: bool = False,
        read_box_edges_only: bool = False,
        extra_variables: list[str] | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Return the particles of one sub-volume of one snapshot, and the vertices of the sub-volume.

        Parameters
        ----------
        isnapshot : int
            Snapshot index.
        isub : int
            Sub-volume index, in ``[0, nsplit**3)``.
        assert_xyz_in_box : bool, optional
            If True, check that the particles are in the box.
        read_box_edges_only : bool, optional
            If True, return no particles, only the vertices.
        extra_variables : list of str or None, optional
            Not supported (no particle identifiers or properties).

        Returns
        -------
        xyz : ndarray
            Positions, shape ``(nparticles_per_sub, 3)``, in Mpc.
        box_edges : ndarray
            Vertices of the sub-volume, shape ``(8, 3)``.

        Raises
        ------
        ValueError
            If ``extra_variables`` are requested.
        """
        if extra_variables not in (None, []):
            raise ValueError("A SyntheticReader has no extra variables.")
        side = self._boxsize / self.nsplit
        corner = np.array(np.unravel_index(isub, (self.nsplit,) * 3)) * side
        cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)
        box_edges = (corner + cube_nodes * side).astype(self.dtype)
        if read_box_edges_only:
            return np.zeros((0, 3), dtype=self.dtype), box_edges
        rng = np.random.default_rng([self.seed, isnapshot, isub])
        xyz = (corner + rng.uniform(high=side, size=(self.nparticles_per_sub, 3))).astype(self.dtype)
        if assert_xyz_in_box and not np.all((0. <= xyz) & (xyz <= self._boxsize)):
            raise RuntimeError("Particles outside the box")
        return xyz, box_edges
