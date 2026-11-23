"""Baryonification of a shell (:meth:`pollux.baryons.BaryonifiedShellBuilder._baryonify`) against BaryonForge's runner,
on a small map with a few halos, in one process and in several."""
from types import SimpleNamespace

import numpy as np
import pyccl
import pytest

bfg = pytest.importorskip('BaryonForge')
from pollux.baryons import BaryonifiedShellBuilder, s19_profiles  # noqa: E402

NSIDE = 128
MASS_DEF = ('200', 'critical')
# the S19 parameters of the validation runs (with a shorter projection)
S19_PARAMS = dict(
    epsilon_max=20,
    cutoff=1e30,
    proj_cutoff=100,
    epsilon=4.0,
    cdelta=None,
    a=0.3,
    n=2.0,
    theta_ej=4,
    theta_co=0.1,
    M_c=1.42857e14,
    mu_beta=0.4,
    gamma=2,
    delta=7,
    mu_theta_ej=0.0,
    mu_theta_co=0.0,
    A=0.045,
    M1=3.57143e11,
    eta=0.3,
    eta_delta=0.3,
    tau=-1.5,
    tau_delta=0,
    epsilon_h=0.015,
    p=0.3,
    q=0.707)


@pytest.fixture(scope='module')
def shell():
    """A count map, halos on it, and the state of a builder that _baryonify uses (its model tabulated at their z)."""
    import healpy as hp
    cosmo = pyccl.Cosmology(Omega_c=0.22, Omega_b=0.045, h=0.71, n_s=0.963, sigma8=0.8, matter_power_spectrum='linear')
    cosmo_dict = {**cosmo.to_dict(), 'Omega_m': cosmo['Omega_m']}
    dmo, dmb = s19_profiles(S19_PARAMS, MASS_DEF)
    pix = bfg.HealPixel(NSIDE)
    model = bfg.Profiles.Baryonification2D(
        bfg.ConvolvedProfile(dmo, pix), bfg.ConvolvedProfile(dmb, pix), cosmo, mass_def=pyccl.halos.massdef.MassDef(*MASS_DEF))
    model.setup_interpolator(
        z_min=0.19, z_max=0.26, N_samples_z=2, z_linear_sampling=True, M_min=1e12, M_max=1e15, N_samples_Mass=10, N_samples_R=50, verbose=False)
    rng = np.random.default_rng(1)
    count = rng.poisson(50., hp.nside2npix(NSIDE)).astype(np.float64)
    count[:hp.nside2npix(NSIDE) // 50] = 0.  # some empty pixels
    n = 40
    halos = dict(
        ra=rng.uniform(0., 60., n),
        dec=rng.uniform(-30., 30., n),
        M=(10**rng.uniform(13., 14.5, n)).astype(np.float32),
        z=rng.uniform(0.2, 0.25, n).astype(np.float32))  # float32, as the halo catalogs
    builder = SimpleNamespace(cosmo_linear=cosmo, model=model, mass_def=MASS_DEF, epsilon_max=20., nside=NSIDE, nworkers=1)
    return count, halos, builder, cosmo_dict


def baryonify(shell, nworkers):
    count, halos, builder, _ = shell
    builder.nworkers = nworkers
    return BaryonifiedShellBuilder._baryonify(builder, count, halos['ra'], halos['dec'], halos['M'], halos['z'])


def test_baryonify(shell):
    count, halos, builder, cosmo_dict = shell
    catalog = bfg.utils.HaloLightConeCatalog(halos['ra'], halos['dec'], halos['M'], halos['z'], cosmo_dict)
    runner = bfg.Runners.BaryonifyShell(catalog, bfg.utils.LightconeShell(map=count, cosmo=cosmo_dict), epsilon_max=20., model=builder.model, verbose=False)
    expected = runner.process()
    one = baryonify(shell, 1)
    assert np.max(np.abs(expected - count)) > 0.1  # the halos move mass
    assert np.allclose(one, expected, rtol=0., atol=1e-9 * count.max())
    assert np.isclose(one.sum(), count.sum(), rtol=1e-12)
    # in parallel: the same, up to the order of the sums of the displacements
    assert np.allclose(baryonify(shell, 4), one, rtol=0., atol=1e-12 * count.max())
