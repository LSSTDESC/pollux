import os

import healsparse
import numpy as np
import pyccl
import yaml
from astropy.io import fits


class PolluxIO:

    def __init__(self, sim_path, seed, tomo="SRDv1_nz", catalog="poisson", interp_pixels=True, bias=1):
        """`PolluxIO` creates an interface to Pollux/HACC simulations data products.

        Parameters
        ----------
        sim_path
            Path to simulation files.
        seed
            Seed used for pseudo-independent realizations.
        tomo, optional
            Name of the tomographic redshift distributions used for lensing products, by default "SRDv1_nz" (the other option is "kappa_zsnapshots" for individual shells).
        catalog, optional
            Source catalog sampling method, by default "poisson" (other options are "uniform" and "multinomial").
        interp_pixels, optional
            Whether to interpolate between adjacent pixels when creating the mock catalogs, by default True.
        bias, optional
            Value for the galaxy bias used for the Poisson mock catalogs.
        """
        self.sim_path = sim_path
        self.seed = seed
        if tomo is not None:
            self.tomo = tomo
        self.catalog = catalog
        self.interp_pixels = interp_pixels
        self.bias = bias
        print("*** PolluxIO ***")
        print(" - path =", sim_path)

        # Basic simulations info
        print(" - simulations info")
        info = yaml.safe_load(open(os.path.join(self.sim_path, "sim_info.yaml"), "r"))
        for k, v in info.items():
            setattr(self, k, v)
            print(f"    * {k} = {v}")

        # Shell info
        self.shell_info = np.load(os.path.join(self.sim_path, f"shell_{self.kind}_seed{self.seed}_shell_info.npy"))

        # Cosmology
        self.cosmo = pyccl.Cosmology.read_yaml(os.path.join(self.sim_path, "cosmo_ccl.yaml"))
        print(" - cosmo", self.cosmo)

        # Redshift distributions
        if tomo is not None:
            self.nzs = np.load(os.path.join(self.sim_path, tomo, f"nzs.npy"))

    def load_shell(self, i, density=False):
        fname = os.path.join(
            self.sim_path,
            f"shell_{self.kind}_seed{self.seed}_nside{self.nside}_imap{i}.fits",
        )
        shell = healsparse.HealSparseMap.read(fname)

        if density:
            return shell / self.shell_info["pixel_density"][i] - 1.0
        else:
            return shell

    def load_haloes(self, i):
        fname = os.path.join(
            self.sim_path,
            f"shell_{self.kind}_seed{self.seed}_nside{self.nside}_haloes_props_imap{i}.fits",
        )
        return fits.open(fname)
    
    def load_IA(self, i):
        fname = os.path.join(
            self.sim_path,
            f"shell_{self.kind}_seed{self.seed}_nside{self.nside}_t11_imap{i}.fits",
        )
        t11 = healsparse.HealSparseMap.read(fname)

        fname = os.path.join(
            self.sim_path,
            f"shell_{self.kind}_seed{self.seed}_nside{self.nside}_t12_imap{i}.fits",
        )
        t12 = healsparse.HealSparseMap.read(fname)
        
        fname = os.path.join(
            self.sim_path,
            f"shell_{self.kind}_seed{self.seed}_nside{self.nside}_t22_imap{i}.fits",
        )
        t22 = healsparse.HealSparseMap.read(fname)

        return t11, t12, t22

    def load_kappa(self, i):
        fname = os.path.join(
            self.sim_path,
            self.tomo,
            f"kappa_{self.kind}_seed{self.seed}_nside{self.nside}_imap{i}.fits",
        )
        return healsparse.HealSparseMap.read(fname)

    def load_shear(self, i):
        fname = os.path.join(
            self.sim_path,
            self.tomo,
            f"kappa_{self.kind}_seed{self.seed}_nside{self.nside}_gamma1_imap{i}.fits",
        )
        g1 = healsparse.HealSparseMap.read(fname)

        fname = os.path.join(
            self.sim_path,
            self.tomo,
            f"kappa_{self.kind}_seed{self.seed}_nside{self.nside}_gamma2_imap{i}.fits",
        )
        g2 = healsparse.HealSparseMap.read(fname)

        return g1, g2

    def load_catalog(self, i):
        interp_pixels_dict = {True: 'interppixels',False: 'nointerppixels'}
        if self.catalog == 'poisson':
            fname = os.path.join(
                self.sim_path,
                self.tomo,
                f"kappa_{self.kind}_seed{self.seed}_nside{self.nside}_cat{self.catalog}_{interp_pixels_dict[self.interp_pixels]}_b{self.bias}_imap{i}.fits",
            )
        elif self.catalog == 'uniform':
            fname = os.path.join(
                self.sim_path,
                self.tomo,
                f"kappa_{self.kind}_seed{self.seed}_nside{self.nside}_cat{self.catalog}_{interp_pixels_dict[self.interp_pixels]}_imap{i}.fits",
            )
        return fits.open(fname)
