import itertools
import os

import healpy as hp
import numpy as np
import pyccl

from .utils import trim_array_min_diff

os.environ["GENERICIO_NO_MPI"] = "y"
try:
    import pygio
except:
    print("Unable to import pygio")


class Reader:
    """A generic class to read particles from N-body simulations.
    """

    def __init__(self, kind, unit="Mpc"):
        self.kind = kind
        self.unit = unit
        assert unit in ["Mpc", "Mpc_over_h"]

    def get_ishell(self, isub):
        return self._sub_args[isub][0]

    def get_isubs_for_shell(self, ishell):
        return np.argwhere([x[0] == ishell for x in self._sub_args])[:, 0]

    def get_fname_sub(self, save_dir, isub, nside, seed, save_format):
        return os.path.join(
            save_dir, f'shell_{self.kind}_z{self.get_ishell(isub)}_sub{isub}_nside{nside}_seed{seed}.' + save_format)

    def get_fname_shell(self, save_dir, isub, nside, seed, save_format):
        return os.path.join(save_dir,
                            f'shell_{self.kind}_z{self.get_ishell(isub)}_nside{nside}_seed{seed}.' + save_format)

    def get_com_shell_bins(self, cosmo):
        raise NotImplementedError

    def get_boxsize(self, cosmo):
        raise NotImplementedError

    def read_sub(self, isub, verbose=False, assert_xyz_in_box=False, read_box_edges_only=False, h_to_divide_by=None):
        raise NotImplementedError


class SLICSReader(Reader):

    def __init__(self, path, boxsize=505. / 0.6898, nparticles=1536**3, nodes_dim=4, ncc=768, rnc=3072.0):
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
        self.nparticles = nparticles
        self.nodes_dim = nodes_dim  # that’s the
        self.ncc = ncc
        self.rnc = rnc

        # simulation snapshot redshifts
        self.zsnapshots = np.sort([float(x[:5]) for x in filter(lambda _x: "xv0.dat" in _x, os.listdir(self.path))])
        self.nshells = len(self.zsnapshots)

        # number of sub boxes (constant here)
        self.ncube = self.nodes_dim**3

        # independent jobs
        self._sub_args = list(itertools.product(range(self.nshells), range(self.ncube)))
        self.nsubs = len(self._sub_args)

        # define approximately even split of significant jobs (here, one job per cube accross shells)
        self.jobs = [range(c, self.nsubs, self.ncube) for c in range(self.ncube)]
        self.njobs = len(self.jobs)

        super().__init__('slics', 'Mpc')

    def get_com_shell_bins(self, cosmo):
        """Computes the comoving distances for the particle shells. Here, snapshots are at fixed comoving distance, therefore we use this to determine the furthest shell.

        Parameters
        ----------
        cosmo
            Cosmology object from CCL.

        Returns
        -------
            Comving distances.
        """
        # TODO : make it a property

        # compute comoving distances of the shell centers and widths (assuming)
        com_snapshots = pyccl.comoving_radial_distance(cosmo, 1. / (1. + self.zsnapshots))

        # compute bins in comoving distance
        com_shell_bins = np.zeros(self.nshells + 1)
        com_shell_bins[1:-1] = 0.5 * (com_snapshots[1:] + com_snapshots[:-1])
        com_shell_bins[-1] = 2 * com_shell_bins[-2] - com_shell_bins[-3]  # fixed comoving distance width

        return com_shell_bins

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

    def get_box_comoving_density(self):
        """Computes the comoving particle density of the simulation.

        Returns
        -------
            Comoving particle density of the simulation.
        """
        return self.nparticles / self.boxsize**3 * np.ones(self.nshells)

    def get_pixel_density(self, ishell, cosmo, nside):
        """Computes the expected mean number of particle per pixels from the simulation density.

        Parameters
        ----------
        ishell
            Index of the shell.
        cosmo
            Cosmology object from CCL.
        nside
            Resolution parameter nside of the shell.

        Returns
        -------
            Mean number of particles per pixel.
        """
        npix = hp.nside2npix(nside)
        com_shell_bins = self.get_com_shell_bins(cosmo)

        #     z_shell_bins = 1./pyccl.scale_factor_of_chi(cosmo, com_shell_bins) - 1.

        #     z_low = z_shell_bins[ishell]
        #     z_up = z_shell_bins[ishell+1]
        # pixel_volume = (cosmo.comoving_volume(z_up).value - cosmo.comoving_volume(z_low).value)/n_pix

        chi_max = com_shell_bins[ishell + 1]
        chi_min = com_shell_bins[ishell]
        pixel_volume = 4. * np.pi / 3. * (chi_max**3 - chi_min**3) / npix

        return pixel_volume * self.get_box_comoving_density()


class HACCReader(Reader):

    def __init__(self, path, nsteps=500, zmax=4., zmin=0., aminstep=0., group_along_x=True):
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
        self.nsteps = nsteps
        self.zmax = zmax
        self.zmin = zmin
        self.aminstep = aminstep
        self.group_along_x = group_along_x

        # simulation snapshot redshifts : step are constant in scale factor, but only a fraction of steps are saved
        snapshot_dirs = os.listdir(os.path.join(path, 'output'))
        snapshots = np.array([i for i in range(nsteps) if f'STEP{i}' in snapshot_dirs])

        amax = 1. / (1. + zmin)
        amin = 1. / (1. + zmax)
        asnapshots = amin + (amax - amin) * (snapshots - snapshots[0]) / (snapshots[-1] - snapshots[0])
        self.asnapshots, _idx = trim_array_min_diff(asnapshots, self.aminstep, return_indices=True)
        if len(_idx) < len(snapshots):
            print(f"Subsampling snapshots with aminstep={aminstep} returns {len(_idx)} snapshots")
        self.snapshots = snapshots[_idx]

        self.asnapshots = self.asnapshots[::-1]
        self.snapshots = self.snapshots[::-1]

        self.zsnapshots = 1. / self.asnapshots - 1.
        self.nshells = len(self.zsnapshots)

        # number of sub boxes (constant here)
        self.ncube = 64

        # independent jobs
        # self._sub_args = list(itertools.product(range(self.nshells), range(self.ncube)))
        if self.group_along_x:
            self._sub_args = list(
                itertools.product(range(self.nshells), range(16))
            )  # there are 64 subfiles, but on the x-axis, they are not continuous, so we might as well load them all together and make one continious volume
        else:
            self._sub_args = list(itertools.product(range(self.nshells), range(64)))
        self.nsubs = len(self._sub_args)

        self.lims = np.array([[[np.floor(ibox / (16)) * 0.5,
                                np.floor(ibox / (16)) * 0.5 + 2.5],
                               [np.floor(ibox / 4) % 4, (np.floor(ibox / 4) % 4) + 1], [ibox % 4, (ibox % 4) + 1]]
                              for ibox in range(64)])

        # some simulation properties
        step = self.snapshots[0]
        name = os.path.join(self.path, 'output', f'STEP{step}/m000.full.mpicosmo.{step}#0')
        self.boxsize_h = pygio.read_phys_scale(name)[0]
        self.nparticles = pygio.read_num_elems(name[:-2])

        # define approximately even split of significant jobs (here, one job per cube accross shells)
        # self.jobs = [range(c, self.nsubs, self.ncube) for c in range(self.ncube)]
        # self.njobs = len(self.jobs)

        super().__init__('hacc', 'Mpc_over_h')

    def get_com_shell_bins(self, cosmo):
        """Computes the comoving distances for the particle shells. Here, snapshots are at fixed scale factor, therefore we use this to determine the furthest shell.

        Parameters
        ----------
        cosmo
            Cosmology object from CCL.

        Returns
        -------
            Comving distances.
        """
        # TODO : make it a property

        amids = 0.5 * (self.asnapshots[1:] + self.asnapshots[:-1])

        # compute bins in comoving distance
        com_shell_bins = np.zeros(self.nshells + 1)
        com_shell_bins[1:-1] = pyccl.comoving_radial_distance(cosmo, amids)
        com_shell_bins[-1] = pyccl.comoving_radial_distance(cosmo,
                                                            2 * amids[-1] - amids[-2])  # fixed scake factor width

        return com_shell_bins

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

        assert h_to_divide_by is not None, "HACC uses Mpc/h units, so we need h here!"
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

        return xyz, box_edges
