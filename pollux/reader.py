import itertools
import os
import warnings

import numpy as np
import pyccl
from pynverse import inversefunc

from .utils import (compute_shell_edges, read_params, subsample_ppf, trim_array, trunc_norm)

os.environ["GENERICIO_NO_MPI"] = "y"
try:
    import pygio
except:
    print("Unable to import pygio")


class Reader:

    def __init__(self, cosmo, kind, unit, nparticles, zsnapshots, zmax, nsubs, shell_edges_match, dtype=np.float64):
        """`Reader` is a generic class to read particles from N-body simulations.

        Parameters
        ----------
        cosmo
            CCL cosmology object
        kind
            Kind of reader subclass (for saving files)
        unit
            Particle coordinate unit, 'Mpc' or 'Mpc_over_h'
        nparticles
            Total number of particles
        zsnapshots
            Redshifts of snapshots
        zmax
            Maximum redshift considered
        nsubs
            Number of sub-boxes
        shell_edges_match
            Method to define shells for snapshots redshifts, must be 'center_redshift', 'center_scale_factor', 'center_comoving_distance' or 'center_comoving_volume'
        dtype, optional
            dtype of particle data, by default np.float64
        """
        self.cosmo = cosmo
        self.kind = kind

        assert unit in ["Mpc", "Mpc_over_h"]
        self.unit = unit

        self.nparticles = nparticles
        rho_m = pyccl.background.rho_x(self.cosmo, 1., 'matter', is_comoving=True)  #  in units of Msun / Mpc^3
        vol = self.boxsize()**3  # in Mpc^3
        self.mass_particles = rho_m * vol / self.nparticles  # in Msun

        self.zsnapshots = zsnapshots
        self.zmax = zmax
        if zmax is not None:
            assert np.all(self.zsnapshots < zmax)
        self.asnapshots = 1. / (1. + zsnapshots)
        self.nshells = len(zsnapshots)
        self.nsubs = nsubs
        self.shell_edges_match = shell_edges_match

        # independent jobs
        self._sub_args = list(itertools.product(range(self.nshells), range(nsubs)))
        self.njobs = len(self._sub_args)

        self.dtype = dtype

    def get_ishell(self, isub):
        """
        Get shell index from overall sub-volume/snapshot index
        """
        return self._sub_args[isub][0]

    def get_isubs_for_shell(self, ishell):
        """
        Get sub-volume/snapshot indices for a given shell index
        """
        return np.argwhere([x[0] == ishell for x in self._sub_args])[:, 0]

    def shell_edges_comoving_distance(self):
        """
        Compute comoving distance to shell edges
        """
        return pyccl.comoving_radial_distance(self.cosmo, 1. / (1 + self.shell_edges_redshift()))

    def shell_edges_redshift(self, return_fzbins=False):
        """Compute redshift to shell edges

        Parameters
        ----------
        return_fzbins, optional
            Whether to return the value of the function used to find center between snapshot redshifts, by default False

        Returns
        -------
            Redshifts of shell edges
        """
        if hasattr(self, 'z_shell_bins'):
            return self.z_shell_bins
        else:
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

            res = compute_shell_edges(self.zsnapshots, func=func, ifunc=ifunc, zmax=self.zmax, return_fzbins=return_fzbins)

            if return_fzbins:
                self.z_shell_bins = res[0]
            else:
                self.z_shell_bins = res

            return self.z_shell_bins

    def shell_info(self, fname=None):
        z_shell_bins = self.shell_edges_redshift()
        names = ['ishell', 'zsnapshot', 'zmin', 'zmax', 'snapshot_id']
        dtypes = ['int', 'float', 'float', 'float', 'int']
        out = np.array(list(zip(np.arange(self.nshells), self.zsnapshots, z_shell_bins[:-1], z_shell_bins[1:], self.snapshots)), dtype=list(zip(names, dtypes)))
        if fname is not None:
            if '.txt' in fname:
                np.savetxt(fname, out, header=' '.join(names), fmt='%8i %16.8f %16.8f %16.8f %8i')
            else:
                np.save(fname, out)
        return out

    def boxsize(self):
        raise NotImplementedError

    def read_sub(self, isub, verbose=False, assert_xyz_in_box=False, read_box_edges_only=False):
        """Reads a sub volume of a simulation snapshot.

        Parameters
        ----------
        isub
            Index of the sub-volume/snapshot element. This should be computed automatically.
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
        raise NotImplementedError

    def read_props(self, isub, props_variables, ihaloes=slice(-1), verbose=False):
        """Reads extra properties for a sub volume of a simulation snapshot.

        Parameters
        ----------
        isub
            Index of the sub-volume/snapshot element. This should be computed automatically.
        props_variables
            Name of the variables to read
        ihaloes, optional
            Index of the row to keep, by default slice(-1)
        verbose, optional
            Print feedback, by default False
        """
        raise NotImplementedError

    def box_comoving_density(self):
        """Computes the comoving particle density of the simulation.

        Returns
        -------
            Comoving particle density of the simulation.
        """
        return self.nparticles / self.boxsize()**3


class SLICSReader(Reader):

    def __init__(
            self,
            path,
            cosmo,
            boxsize=505. / 0.6898,
            nparticles=1536**3,
            nodes_dim=4,
            ncc=768,
            rnc=3072.0,
            zmax=None,
            shell_edges_match='center_comoving_distance'):
        """Reader for SLICS simulations by Joachim Harnois-Deraps.

        Parameters
        ----------
        path
            Path of simulation snapshots
        cosmo
            CCL cosmology object
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
        zmax, optional
            Maximum redshift considered, by default None
        shell_edges_match, optional
            Method to define shells, by default 'center_comoving_distance'
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

        super().__init__(cosmo, 'slics', 'Mpc', nparticles, zsnapshots, zmax, ncube, shell_edges_match, dtype=np.float32)

    def boxsize(self):
        return self.boxsize

    def read_sub(self, isub, verbose=False, assert_xyz_in_box=False, read_box_edges_only=False):
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

        # assert h_to_divide_by is None

        return xyz, box_edges


class HACCReader(Reader):

    def __init__(
            self,
            path,
            run_params='run/run000/indat.params',
            snap_sub_zmax=None,
            snap_sub_aminstep=0.,
            snap_sub_match_dist_n=False,
            snap_sub_every_n=1,
            shell_edges_match='center_scale_factor',
            group_along_x=False,
            matter_power_spectrum=None,
    ):
        """Reader for HACC simulations

        Parameters
        ----------
        path
            Root folder where particle data is located
        cosmo, optional
            CCL cosmology object, by default None
        read_cosmo_params, optional
            Whether to read cosmology from HACC run files, by default False
        nsteps, optional
            Number of steps in simulation, by default 500
        sim_zmin, optional
            Minimum redshift in simulation, by default 0.
        sim_zmax, optional
            Maximum redshift in simulation, by default 200.
        zmax, optional
            Maximum redshift considered, by default None
        aminstep, optional
            Minimum step in scale factor to downsample snapshots, by default 0.
        sampling_match_n, optional
            Number of snapshots to try to keep while matching roughly optimal distribution, by default False
        shell_edges_match, optional
            Method to define shells, by default 'center_comoving_distance'
        group_along_x, optional
            Whether to group sub-boxes along x to define contiguous box volumes, by default False
        """
        self.path = path
        self.group_along_x = group_along_x
        self.variables = ['x', 'y', 'z']

        print("*** HACC reader ***")

        # read run parameters
        fname = os.path.join(path, run_params)
        print(f"  - Reading parameter file at", fname)
        params = read_params(fname)

        # create cosmo object that can be used by builder  
        cosmo_dict = dict(
            Omega_c=float(params['OMEGA_CDM']),
            Omega_b=float(params['DEUT']) / float(params['HUBBLE'])**2,
            h=float(params['HUBBLE']),
            n_s=float(params['NS']),
            sigma8=float(params['SS8']),
            Neff=float(params['N_EFF_MASSLESS']),
            w0=float(params['W_DE']),
            wa=float(params['WA_DE']),
            # T_CMB=float(params['T_CMB']),
            T_CMB=2.726,
            Omega_k=0.)
        if matter_power_spectrum is not None:
            cosmo_dict['matter_power_spectrum']=matter_power_spectrum
        cosmo = pyccl.Cosmology(**cosmo_dict)
        print(f"  - Cosmology", cosmo)

        # read parameters to determine redshifts of snapshots
        nsteps = int(params['N_STEPS'])
        sim_zmax = float(params['Z_IN'])
        sim_zmin = float(params['Z_FIN'])

        # simulation snapshot redshifts : step are constant in scale factor, but only a fraction of steps are saved
        snapshots = np.array(list(map(int, params['ALIVE_DUMP'].split())))
        snapshot_dir = os.path.join(path, 'output')
        if os.path.exists(snapshot_dir):
            try:
                snapshot_dirs = os.listdir(snapshot_dir)
                snapshots_output = np.array([i for i in range(nsteps) if f'STEP{i}' in snapshot_dirs])
                np.testing.assert_array_equal(snapshots, snapshots_output)
            except AssertionError:
                snapshots = snapshots_output
                warnings.warn("We are using the snapshots found in the `output` folder, which do not match those in the parameter file.")
        print(f"  - Found {len(snapshots)} snapshots :", snapshots)

        # helper function : we only work on the list of snapshots to avoid mistakes in book-keeping
        def get_asnapshots(idx):
            asnapshots_all = np.linspace(1. / (1. + sim_zmax), 1. / (1. + sim_zmin), nsteps + 1)[1:]
            return asnapshots_all[idx]

        # subsample snapshots : minimum step in scale factor
        if snap_sub_aminstep > 0.:
            _, idx = trim_array(get_asnapshots(snapshots), snap_sub_aminstep, return_indices=True)
            print(f"  - Subsampling : minimum a step {snap_sub_aminstep}, returns {len(idx)} snapshots")
            snapshots = snapshots[idx]

        # subsample snapshots : maximum redshift
        if snap_sub_zmax is not None:
            w = np.argwhere(get_asnapshots(snapshots) > 1. / (1. + snap_sub_zmax))
            if len(w) > 0:
                snapshots = snapshots[w[0, 0]:]
                print(f"  - Subsampling : max redshift {snap_sub_zmax}, returns {len(snapshots)} snapshots")

        # subsample snapshots : match inferred bimodal distribution
        if snap_sub_match_dist_n:
            aa = get_asnapshots(snapshots)

            # multimodal distribution in scale factor, need ppf (computed numerically)
            s1, s2, lam = 0.12, 0.27, 0.4
            amin = np.min(aa)
            t1 = trunc_norm(amin, 1., 0.5, s1)
            t2 = trunc_norm(amin, 1., 1., s2)
            cdf = lambda a: lam * t1.cdf(a) + (1 - lam) * t2.cdf(a)
            ppf = inversefunc(cdf, domain=[amin, 1.], image=[0., 1.])

            idx, _ = subsample_ppf(aa, snap_sub_match_dist_n, ppf)
            if len(idx) < len(snapshots):
                print(f"  - Subsampling : match distribution, returns {len(idx)} snapshots")
            snapshots = snapshots[idx]

        # subsample snapshots : every n'ths snapshots
        self.snapshots = snapshots[::-snap_sub_every_n]  # snapshots ordered by redshift

        if snap_sub_every_n > 1:
            print(f"  - Subsampling : keeping every {snap_sub_every_n} snapshots, returns {len(self.snapshots)} snapshots")

        # done with subsampling !
        zsnapshots = 1. / get_asnapshots(self.snapshots) - 1.

        # simulation properties

        if False:
            step = self.snapshots[0]
            name = os.path.join(self.path, 'output', f'STEP{step}/m000.full.mpicosmo.{step}')
            self.boxsize_h = pygio.read_phys_scale(name)[0]
            nparticles = pygio.read_num_elems(name)
            dtype = pygio.read_variable_dtypes(name)['x']
        else:
            self.boxsize_h = float(params['RL'])
            nparticles = int(params['NP'])**3
            dtype = np.float32

        # simulation box limits
        self.lims = np.array([[[np.floor(ibox / (16)) * 0.5, np.floor(ibox / (16)) * 0.5 + 2.5], [np.floor(ibox / 4) % 4, (np.floor(ibox / 4) % 4) + 1],
                               [ibox % 4, (ibox % 4) + 1]] for ibox in range(64)])

        nsub = 16 if self.group_along_x else 64
        super().__init__(cosmo, 'hacc', 'Mpc_over_h', nparticles, zsnapshots, snap_sub_zmax, nsub, shell_edges_match, dtype=dtype)

    def boxsize(self):
        return self.boxsize_h / self.cosmo["h"]

    def _read_sub_helper(self, isub):
        istep, ibox = self._sub_args[isub]
        step = self.snapshots[istep]

        boxsize = self.boxsize_h / self.cosmo["h"]

        if self.group_along_x:
            box_lims = np.array([[0, 4], self.lims[ibox][1], self.lims[ibox][2]]) * boxsize / 4.
        else:
            box_lims = np.array([self.lims[ibox][i] for i in range(3)]) * boxsize / 4.

        cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
        box_edges = np.zeros(cube_nodes.shape)

        for i in range(3):
            box_edges[:, i] = box_lims[i, cube_nodes[:, i]]  #there's propbably a fancy numpy way to do this...

        return box_edges.astype(self.dtype), ibox, step

    def _get_name(self, step, ifile):
        return os.path.join(self.path, 'output', f'STEP{step}/m000.full.mpicosmo.{step}#{ifile}')

    def read_sub(self, isub, verbose=False, assert_xyz_in_box=False, read_box_edges_only=False):
        box_edges, ibox, step = self._read_sub_helper(isub)

        if read_box_edges_only:
            return np.zeros((17, 3), dtype=self.dtype), box_edges

        out = {k: [] for k in self.variables}

        nx = 4 if self.group_along_x else 1
        for ix in range(nx):
            ifile = ix * 16 + ibox
            name = self._get_name(step, ifile)
            if verbose:
                print("Reading", name)
            data = pygio.read_genericio(name, variables=self.variables)
            for k in self.variables:
                out[k].append(data[k])
        xyz = np.array([np.concatenate(out[k]) for k in self.variables]).T

        xyz /= self.cosmo["h"]

        if assert_xyz_in_box:
            for i in range(3):
                assert np.all(np.min(box_edges[:, i]) <= xyz[:, i]) and np.all(xyz[:, i] <= np.max(box_edges[:, i]))
        else:
            box_lims = np.stack([np.min(xyz, axis=0), np.max(xyz, axis=0)])
            cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
            box_edges = box_lims[0] + (box_lims[1] - box_lims[0]) * cube_nodes

        return xyz, box_edges.astype(self.dtype)


class HACCHaloReader(HACCReader):

    def __init__(self, halo_path='analysis/Halos/b0168/Props', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variables = ['fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z']
        self.halo_path = halo_path
        assert not self.group_along_x

    def _get_name(self, step, ifile):
        return os.path.join(self.path, self.halo_path, f'STEP{step}/m000-{step}.haloproperties#{ifile}')

    def read_props(self, isub, props_variables, ihaloes=slice(-1), verbose=False):

        istep, ibox = self._sub_args[isub]
        step = self.snapshots[istep]

        # nx = 4 if self.group_along_x else 1
        # for ix in range(nx):
        ix = 0
        ifile = ix * 16 + ibox
        name = self._get_name(step, ifile)
        if verbose:
            print("Reading", name)
        data = pygio.read_genericio(name, variables=props_variables)

        return {k: data[k][ihaloes] for k in props_variables}


class SkySimReader(HACCReader):

    def __init__(self, path, cosmo, nsteps=500, sim_zmin=0., sim_zmax=200., aminstep=0., zmax=None, shell_edges_match='center_scale_factor'):
        self.path = path
        self.aminstep = aminstep

        mass_sheets_fname = list(sorted(filter(lambda s: s.endswith('.bin'), os.listdir(path))))
        mass_sheets_fname.remove('density_map_259_dens.bin')
        snapshots = np.array([i for i in range(nsteps) if f'density_map_{i}_dens.bin' in mass_sheets_fname])

        amax = 1. / (1. + sim_zmin)
        amin = 1. / (1. + sim_zmax)
        asnapshots_all = np.linspace(amin, amax, nsteps + 1)[1:]
        asnapshots_all = asnapshots_all[snapshots]

        asnapshots, idx = trim_array(asnapshots_all, self.aminstep, return_indices=True)
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

        Reader.__init__(self, cosmo, 'skysim', 'Mpc_over_h', nparticles, zsnapshots, zmax, nsub, shell_edges_match, dtype=dtype)


class MockReader(Reader):

    def __init__(self, cosmo, boxsize=200., nshells=3, nsplit=2, nparticule_per_sub=10, zmax=2.):
        self.boxsize = boxsize
        self.xyzs = np.random.uniform(high=self.boxsize, size=(nsplit**3 * nshells, nparticule_per_sub, 3))

        nparticles = nsplit**3 * nparticule_per_sub
        zsnapshots = np.zeros(nshells)
        nsubs = nsplit**3

        super().__init__(cosmo, 'dummy', 'Mpc', nparticles, zsnapshots, zmax, nsubs, dtype=np.float64)

    def shell_edges_comoving_distance(self, cosmo):
        chi_max = pyccl.comoving_radial_distance(cosmo, 1. / (1. + self.zmax))
        return np.linspace(0, chi_max, self.nshells + 1)

    def boxsize(self, cosmo):
        return self.boxsize

    def read_sub(self, isub, verbose=False, assert_xyz_in_box=False, read_box_edges_only=False):
        cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
        box_edges = cube_nodes * self.boxsize
        xyz = self.xyzs[isub]
        return xyz, box_edges

    def box_comoving_density(self, cosmo):
        return self.nparticles / self.boxsize(cosmo)**3


class DummyReader(Reader):

    def __init__(self, cosmo, kind, unit, nsnapshots):
        super().__init__(
            cosmo, kind, unit, nparticles=0., zsnapshots=np.zeros(nsnapshots), zmax=None, nsubs=0, shell_edges_match='center_scale_factor', dtype=np.float64)
