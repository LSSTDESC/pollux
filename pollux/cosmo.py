import pyccl

cosmo_slics = pyccl.Cosmology(Omega_c=0.2905 - 0.0447, Omega_b=0.0447, h=0.6898, n_s=0.969, sigma8=0.826, Omega_k=0.)

cosmo_dc2 = pyccl.Cosmology(Omega_c=0.220, Omega_b=0.02258 / (0.71**2), h=0.71, n_s=0.963, sigma8=0.8, Omega_k=0.)
