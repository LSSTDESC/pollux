import itertools
import os

import numpy as np
import pyccl
from pynverse import inversefunc

from .utils import (get_trunc_norm, get_z_shell_fz, read_params, subsample_ppf, trim_array_min_diff)

os.environ["GENERICIO_NO_MPI"] = "y"
try:
    import pygio
except:
    print("Unable to import pygio")


class Reader:
    """A generic class to read particles from N-body simulations.
    """

    def __init__(self, kind, unit, nparticles, zsnapshots, nsubs, binning_match, dtype=np.float64):
        self.kind = kind

        assert unit in ["Mpc", "Mpc_over_h"]
        self.unit = unit

        self.nparticles = nparticles

        self.zsnapshots = zsnapshots
        self.asnapshots = 1. / (1. + zsnapshots)
        self.nshells = len(zsnapshots)
        self.nsubs = nsubs
        self.binning_match = binning_match

        # independent jobs
        self._sub_args = list(itertools.product(range(self.nshells), range(nsubs)))
        self.njobs = len(self._sub_args)

        self.dtype = dtype

    def get_ishell(self, isub):
        return self._sub_args[isub][0]

    def get_isubs_for_shell(self, ishell):
        return np.argwhere([x[0] == ishell for x in self._sub_args])[:, 0]

    def get_com_shell_bins(self, cosmo):
        return pyccl.comoving_radial_distance(cosmo, 1. / (1 + self.get_z_shell_bins(cosmo)))

    def get_z_shell_bins(self, cosmo, return_fzbins=False):
        if hasattr(self, 'z_shell_bins'):
            return self.z_shell_bins
        else:
            if self.binning_match == 'center_redshift':
                func = lambda z: z
                ifunc = func
            elif self.binning_match == 'center_scale_factor':
                func = lambda z: 1. / (1. + z)
                ifunc = lambda a: 1. / a - 1.
            elif self.binning_match == 'center_comoving_distance':
                func = lambda z: 1e-3 * pyccl.comoving_radial_distance(cosmo, 1. / (1. + z))
                ifunc = None
            elif self.binning_match == 'center_comoving_volume':
                func = lambda z: (1e-3 * pyccl.comoving_radial_distance(cosmo, 1. / (1. + z)))**3
                ifunc = None
            else:
                raise ValueError
            self.z_shell_bins = get_z_shell_fz(self.zsnapshots, func=func, ifunc=ifunc, return_fzbins=return_fzbins)

            return self.z_shell_bins

    def get_shell_info(self, cosmo, fname=None):
        z_shell_bins = self.get_z_shell_bins(cosmo)
        names = ['ishell', 'zsnapshot', 'zmin', 'zmax', 'snapshot_id']
        dtypes = ['int', 'float', 'float', 'float', 'int']
        out = np.array(
            list(zip(np.arange(self.nshells), self.zsnapshots, z_shell_bins[:-1], z_shell_bins[1:], self.snapshots)),
            dtype=list(zip(names, dtypes)))
        if fname is not None:
            np.savetxt(fname, out, header=' '.join(names), fmt='%8i %16.8f %16.8f %16.8f %8i')
        return out

    def get_boxsize(self, cosmo):
        raise NotImplementedError

    def read_sub(self, isub, verbose=False, assert_xyz_in_box=False, read_box_edges_only=False, h_to_divide_by=None):
        raise NotImplementedError

    def get_box_comoving_density(self, cosmo):
        """Computes the comoving particle density of the simulation.

        Returns
        -------
            Comoving particle density of the simulation.
        """
        return self.nparticles / self.get_boxsize(cosmo)**3


class SLICSReader(Reader):

    def __init__(
            self,
            path,
            boxsize=505. / 0.6898,
            nparticles=1536**3,
            nodes_dim=4,
            ncc=768,
            rnc=3072.0,
            binning_match='center_comoving_distance'):
        """Reader for SLICS simulations by Joachim Harnois-Deraps.

        Parameters
        ----------
        path
            Path of simulation snapshots.
        boxsize, optional
            Box size in Mpc, by default 505./0.6898
        nparticles, optional
            Total number of particles, by default 1536**3
        nodes_dim, optional
            Number of MPI task per dimension (4^3 = 64 cubes), by default 4
        ncc, optional
            Integer, by default 768
        rnc, optional
            Scale, by default 3072.0
        """
        self.path = path
        self.boxsize = boxsize
        self.nodes_dim = nodes_dim  # that’s the
        self.ncc = ncc
        self.rnc = rnc

        # simulation snapshot redshifts
        zsnapshots = np.sort([float(x[:5]) for x in filter(lambda _x: "xv0.dat" in _x, os.listdir(self.path))])

        # number of sub boxes (constant here)
        ncube = self.nodes_dim**3

        super().__init__('slics', 'Mpc', nparticles, zsnapshots, ncube, binning_match, dtype=np.float32)

    def get_boxsize(self, cosmo):
        return self.boxsize

    def read_sub(self, isub, verbose=False, assert_xyz_in_box=False, read_box_edges_only=False, h_to_divide_by=None):
        """Reads a sub volume of a simulation snapshot.

        Parameters
        ----------
        isub
            Index of the sub-volume element and the snapshot. This should be computed automatically.
        verbose, optional
            Print feedback, by default False
        assert_xyz_in_box, optional
            Check whether particle , by default False
        read_box_edges_only, optional
            Read only the sub-volume cube nodes, by default False

        Returns
        -------
            Particle positions and sub-volume cube nodes, to quickly determine if volume/shell intersect.
        """
        ishell, i = self._sub_args[isub]
        z = self.zsnapshots[ishell]
        filename = os.path.join(self.path, f'{z:.3f}xv{i}.dat')

        for k1 in range(self.nodes_dim):
            for j1 in range(self.nodes_dim):
                for i1 in range(self.nodes_dim):
                    if (i == i1 + j1 * self.nodes_dim + k1 * self.nodes_dim**2):
                        node_coords = np.array([[i1, j1, k1]])
                        break

        cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
        box_edges = (node_coords + cube_nodes) * self.ncc * self.boxsize / self.rnc

        if read_box_edges_only:
            return np.zeros((17, 3)), box_edges

        with open(filename, 'rb') as file:
            if verbose:
                print("Reading", filename)
            data = np.fromfile(file, dtype=np.float32)[12:]
            n = len(data)
            data = data.reshape((n // 6, 6))

        xyz = ((data[:, :3] + node_coords * self.ncc) % self.rnc) * self.boxsize / self.rnc

        if assert_xyz_in_box:
            for i in range(3):
                assert np.all(np.min(box_edges[:, i]) <= xyz[:, i]) and np.all(xyz[:, i] <= np.max(box_edges[:, i]))

        assert h_to_divide_by is None

        return xyz, box_edges


class HACCReader(Reader):

    def __init__(
            self,
            path,
            read_cosmo_params=False,
            nsteps=500,
            sim_zmin=0.,
            sim_zmax=200.,
            zmax=None,
            aminstep=0.,
            sampling_match_n=False,
            binning_match='center_scale_factor',
            group_along_x=False):
        """Reader for SLICS simulations by Joachim Harnois-Deraps.

        Parameters
        ----------
        path
            Path of simulation snapshots.
        boxsize, optional
            Box size in Mpc, by default 505./0.6898
        nparticles, optional
            Total number of particles, by default 1536**3
        nodes_dim, optional
            Number of MPI task per dimension (4^3 = 64 cubes), by default 4
        ncc, optional
            Integer, by default 768
        rnc, optional
            Scale, by default 3072.0
        """
        self.path = path
        self.aminstep = aminstep
        self.group_along_x = group_along_x

        if read_cosmo_params:
            fname = os.path.join(path, 'run/run000/indat.params')
            params = read_params(fname)
            # create cosmo object that can be used by builder
            self.cosmo = pyccl.Cosmology(
                Omega_c=float(params['OMEGA_CDM']),
                Omega_b=float(params['DEUT']) / float(params['HUBBLE'])**2,
                h=float(params['HUBBLE']),
                n_s=float(params['NS']),
                sigma8=float(params['SS8']),
                Omega_k=0.)
            # read and override parameters to determine redshifts of snapshots
            nsteps = int(params['N_STEPS'])
            sim_zmax = float(params['Z_IN'])
            sim_zmin = float(params['Z_FIN'])

        # simulation snapshot redshifts : step are constant in scale factor, but only a fraction of steps are saved
        snapshot_dirs = os.listdir(os.path.join(path, 'output'))
        snapshots = np.array([i for i in range(nsteps) if f'STEP{i}' in snapshot_dirs])
        print(f"Found {len(snapshots)} snapshots")

        # helper function : we only work on the list of snapshots to avoid mistakes in book-keeping
        def get_asnapshots(idx):
            asnapshots_all = np.linspace(1. / (1. + sim_zmax), 1. / (1. + sim_zmin), nsteps + 1)[1:]
            return asnapshots_all[idx]

        # subsample snapshots : minimum step in scale factor
        if aminstep > 0.:
            _, idx = trim_array_min_diff(get_asnapshots(snapshots), aminstep, return_indices=True)
            print(f"Subsampling snapshots : minimum a step {aminstep}, returns {len(idx)} snapshots")
            snapshots = snapshots[idx]

        # subsample snapshots : maximum redshift
        if zmax is not None:
            w = np.argwhere(get_asnapshots(snapshots) > 1. / (1. + zmax))
            if len(w) > 0:
                snapshots = snapshots[w[0, 0]:]
                print(f"Subsampling snapshots : max redshift {zmax}, returns {len(snapshots)} snapshots")

        # subsample snapshots : match inferred bimodal distribution
        if sampling_match_n:
            aa = get_asnapshots(snapshots)

            # multimodal distribution in scale factor, need ppf (computed numerically)
            s1, s2, lam = 0.12, 0.27, 0.4
            amin = np.min(aa)
            t1 = get_trunc_norm(amin, 1., 0.5, s1)
            t2 = get_trunc_norm(amin, 1., 1., s2)
            cdf = lambda a: lam * t1.cdf(a) + (1 - lam) * t2.cdf(a)
            ppf = inversefunc(cdf, domain=[amin, 1.], image=[0., 1.])

            idx, _ = subsample_ppf(aa, sampling_match_n, ppf)
            if len(idx) < len(snapshots):
                print(f"Subsampling snapshots : match distribution, returns {len(idx)} snapshots")
            snapshots = snapshots[idx]

        # done with subsampling !
        self.snapshots = snapshots[::-1]  # snapshots ordered by redshift
        zsnapshots = 1. / get_asnapshots(self.snapshots) - 1.

        # simulation properties
        step = self.snapshots[0]
        name = os.path.join(self.path, 'output', f'STEP{step}/m000.full.mpicosmo.{step}#0')
        self.boxsize_h = pygio.read_phys_scale(name)[0]
        nparticles = pygio.read_num_elems(name[:-2])
        dtype = pygio.read_variable_dtypes(name)['x']

        # simulation box limits
        self.lims = np.array([[[np.floor(ibox / (16)) * 0.5,
                                np.floor(ibox / (16)) * 0.5 + 2.5],
                               [np.floor(ibox / 4) % 4, (np.floor(ibox / 4) % 4) + 1], [ibox % 4, (ibox % 4) + 1]]
                              for ibox in range(64)])

        nsub = 16 if self.group_along_x else 64
        super().__init__('hacc', 'Mpc_over_h', nparticles, zsnapshots, nsub, binning_match, dtype=dtype)

    def get_boxsize(self, cosmo):
        return self.boxsize_h / cosmo["h"]

    def read_sub(self, isub, verbose=False, assert_xyz_in_box=False, read_box_edges_only=False, h_to_divide_by=None):
        """Reads a sub volume of a simulation snapshot.

        Parameters
        ----------
        isub
            Index of the sub-volume element and the snapshot. This should be computed automatically.
        verbose, optional
            Print feedback, by default False
        assert_xyz_in_box, optional
            Check whether particle , by default False
        read_box_edges_only, optional
            Read only the sub-volume cube nodes, by default False

        Returns
        -------
            Particle positions and sub-volume cube nodes, to quickly determine if volume/shell intersect.
        """
        istep, ibox = self._sub_args[isub]
        step = self.snapshots[istep]

        if not hasattr(self, 'cosmo'):
            assert h_to_divide_by is not None, "HACC uses Mpc/h units, so we need h here!"
        else:
            h_to_divide_by = self.cosmo['h']
        boxsize = self.boxsize_h / h_to_divide_by

        if self.group_along_x:
            box_lims = np.array([[0, 4], self.lims[ibox][1], self.lims[ibox][2]]) * boxsize / 4.
        else:
            box_lims = np.array([self.lims[ibox][i] for i in range(3)]) * boxsize / 4.
        cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
        box_edges = np.zeros(cube_nodes.shape)
        for i in range(3):
            box_edges[:, i] = box_lims[i, cube_nodes[:, i]]  #there's propbably a fancy numpy way to do this...

        if read_box_edges_only:
            return np.zeros((17, 3)), box_edges

        variables = ['x', 'y', 'z']
        out = {k: [] for k in variables}

        nx = 4 if self.group_along_x else 1
        for ix in range(nx):
            ifile = ix * 16 + ibox
            name = os.path.join(self.path, 'output', f'STEP{step}/m000.full.mpicosmo.{step}#{ifile}')
            print(name)
            if verbose:
                print("Reading", name)
            data = pygio.read_genericio(name, variables=variables)
            for k in variables:
                out[k].append(data[k])
        xyz = np.array([np.concatenate(out[k]) for k in variables]).T

        xyz /= h_to_divide_by

        if assert_xyz_in_box:
            for i in range(3):
                assert np.all(np.min(box_edges[:, i]) <= xyz[:, i]) and np.all(xyz[:, i] <= np.max(box_edges[:, i]))

        return xyz, box_edges.astype(self.dtype)


class DummyReader(Reader):
    """A generic class to read particles from N-body simulations.
    """

    def __init__(self, boxsize=200., nshells=3, nsplit=2, nparticule_per_sub=10, zmax=2.):
        self.boxsize = boxsize
        self.xyzs = np.random.uniform(high=self.boxsize, size=(nsplit**3 * nshells, nparticule_per_sub, 3))
        self.zmax = zmax

        nparticles = nsplit**3 * nparticule_per_sub
        zsnapshots = np.zeros(nshells)
        nsubs = nsplit**3

        super().__init__('dummy', 'Mpc', nparticles, zsnapshots, nsubs, dtype=np.float64)

    def get_com_shell_bins(self, cosmo):
        chi_max = pyccl.comoving_radial_distance(cosmo, 1. / (1. + self.zmax))
        return np.linspace(0, chi_max, self.nshells + 1)

    def get_boxsize(self, cosmo):
        return self.boxsize

    def read_sub(self, isub, verbose=False, assert_xyz_in_box=False, read_box_edges_only=False, h_to_divide_by=None):
        cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
        box_edges = cube_nodes * self.boxsize
        xyz = self.xyzs[isub]
        return xyz, box_edges

    def get_box_comoving_density(self, cosmo):
        return self.nparticles / self.get_boxsize(cosmo)**3


class SkySimReader(HACCReader):

    def __init__(
            self,
            path,
            nsteps=500,
            sim_zmin=0.,
            sim_zmax=200.,
            aminstep=0.,
            zmax=None,
            binning_match='center_scale_factor'):
        self.path = path
        self.aminstep = aminstep

        mass_sheets_fname = list(sorted(filter(lambda s: s.endswith('.bin'), os.listdir(path))))
        mass_sheets_fname.remove('density_map_259_dens.bin')
        snapshots = np.array([i for i in range(nsteps) if f'density_map_{i}_dens.bin' in mass_sheets_fname])

        amax = 1. / (1. + sim_zmin)
        amin = 1. / (1. + sim_zmax)
        asnapshots_all = np.linspace(amin, amax, nsteps + 1)[1:]
        asnapshots_all = asnapshots_all[snapshots]

        asnapshots, idx = trim_array_min_diff(asnapshots_all, self.aminstep, return_indices=True)
        if len(idx) < len(snapshots):
            print(f"Subsampling snapshots with aminstep={aminstep} returns {len(idx)} snapshots")
        zsnapshots = 1. / asnapshots[::-1] - 1.
        self.snapshots = snapshots[list(reversed(idx))]

        if zmax is not None:
            try:
                i = np.argwhere(zsnapshots > zmax)[0, 0]
                zsnapshots = zsnapshots[:i + 1]
            except:
                pass

        # some simulation properties
        nparticles = 10240**3
        self.boxsize_h = 3e3
        dtype = np.float32
        nsub = 1

        Reader.__init__(self, 'skysim', 'Mpc_over_h', nparticles, zsnapshots, nsub, binning_match, dtype=dtype)
