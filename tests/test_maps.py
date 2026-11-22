"""Formats of the maps: the pixelization kernels in both orderings, the conversions between dense RING and HealSparse
maps, and the products of the map builders in each format (``map_format``, ``sub_format``), which must be the same
maps."""
import os
import shutil

import healpy as hp
import healsparse as hsp
import numpy as np
import pytest

import pollux
from pollux.shells import _vec2pix_nest, _vec2pix_ring
from pollux.utils import healpix as H

NSIDE = 32
FORMATS = [('healsparse', 'healsparse'), ('healpix', 'healpix'), ('healsparse', 'healpix'), ('healpix', 'healsparse')]


def directions(rng, n=200000):
    """Random directions, and directions near the poles and the boundary between the polar caps and the equator."""
    v = rng.normal(size=(n, 3))
    zs = np.concatenate(
        [np.cos(np.arccos(2. / 3.) + rng.normal(scale=1e-9, size=n // 10)), 1. - 10.**rng.uniform(-16, -1, n // 10), -1. + 10.**rng.uniform(-16, -1, n // 10)])
    ph = rng.uniform(0, 2 * np.pi, len(zs))
    ph[:4000] = np.repeat(np.arange(4) * np.pi / 2, 1000)  # on the boundaries of the base pixels
    s = np.sqrt(1 - zs**2)
    return np.concatenate([v, np.stack([s * np.cos(ph), s * np.sin(ph), zs], 1)])


@pytest.mark.parametrize('nside', [1, 32, 8192])
def test_vec2pix(nside):
    v = directions(np.random.default_rng(nside))
    geometry = (nside, hp.nside2order(nside), (nside * nside - nside) << 1, hp.nside2npix(nside))
    for dtype in (np.float64, np.float32):
        x, y, z = (np.ascontiguousarray(v[:, k].astype(dtype)) for k in range(3))
        ipix = np.empty(len(x), dtype=np.int64)
        for vec2pix, nest in ((_vec2pix_ring, False), (_vec2pix_nest, True)):
            vec2pix(ipix, *geometry, x, y, z, 0., 0., 0., 0., np.inf, False)
            assert np.array_equal(ipix, hp.vec2pix(nside, x.astype(np.float64), y.astype(np.float64), z.astype(np.float64), nest=nest))


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32])
def test_conversions(dtype):
    rng = np.random.default_rng(0)
    npix = hp.nside2npix(NSIDE)
    values = (rng.normal(size=npix) * 100).astype(dtype)
    theta, phi = hp.pix2ang(NSIDE, np.arange(npix))
    for valid in (None, (theta < 1.) & (phi < 2.)):
        ref = hsp.HealSparseMap.make_empty(8, NSIDE, dtype=dtype)
        ipix = np.arange(npix) if valid is None else np.flatnonzero(valid)
        ref.update_values_pix(ipix, values[ipix], nest=False)
        m = H.dense_to_hsp(values, 8, valid=valid)
        assert np.array_equal(m.valid_pixels, ref.valid_pixels)
        assert np.array_equal(m[m.valid_pixels], ref[ref.valid_pixels])
        for nest in (False, True):
            dense = H.hsp_to_dense(ref, nest=nest)
            assert np.array_equal(dense, ref.get_values_pix(np.arange(npix), nest=nest))
        if dtype != np.int32:
            dense = ref.generate_healpix_map(nest=False)
            assert np.array_equal(H.dense_to_hsp(dense, 8).valid_pixels, ref.valid_pixels)
            ra, dec = rng.uniform(0, 360, 2000), rng.uniform(-90, 90, 2000)
            for allow_partial in (True, False):
                assert np.array_equal(H.interpolate_pos(dense, ra, dec, allow_partial), ref.interpolate_pos(ra, dec, allow_partial=allow_partial))
            assert np.array_equal(H.values_pos(dense, ra, dec), H.values_pos(ref, ra, dec))
    assert np.array_equal(H.reorder(values, True), hp.reorder(values, r2n=True))
    assert np.array_equal(H.reorder(values, False), hp.reorder(values, n2r=True))


def run(root, map_format='healsparse', sub_format='healsparse', octant=False):
    """Shells, their a_lm, density and tidal maps, and convergence and shear maps per shell."""
    reader = pollux.SyntheticReader(boxsize=200., zsnapshots=(0.02, 0.04, 0.06), nparticles_per_sub=20000)
    footprint = pollux.Footprint(NSIDE, pollux.PolluxIO(str(root)), **(dict(mask='octant', padding_deg=10.) if octant else {}))
    # random isometry of the box (as in production): with a box corner on the observer, the octant fails (_query_replica_ipix)
    builder = pollux.shells.SnapshotShellBuilder(reader, footprint, seed=0, apply_grid_isometry=True, map_format=map_format, sub_format=sub_format)
    written = [builder.construct_maps()]
    builder.compute_alms()
    written.append(builder.compute_tidal_tensor_maps(smooth_Mpc=10.))
    lensing = builder.snapshot_lensing_builder()
    written.append(lensing.construct_maps())
    lensing.compute_alms()
    written.append(lensing.compute_shear_maps())
    # the methods writing maps return them as written, in the format of the builder
    for maps in written:
        for m in maps.values():
            assert all(H.is_dense(mi) == (map_format == 'healpix') for mi in (m if isinstance(m, list) else [m]))
    return builder, lensing


def products(builder, lensing):
    """The maps of the products, as dense RING maps, and their a_lm."""
    out = {}
    for ishell in range(builder.nmaps):
        out['shell', ishell] = builder.read_maps(ishell, dense=True)
        out['alm', ishell] = builder.read_alms(ishell)
        out['density', ishell] = builder.field_map(ishell, smooth_Mpc=10.)
        for what in ('t11', 't22', 't12'):
            out[what, ishell] = builder.read_maps(ishell, what=what, smooth_Mpc=10., dense=True)
        out['kappa', ishell] = lensing.read_maps(ishell, dense=True)
        out['kappa_alm', ishell] = lensing.read_alms(ishell)
        for what in ('gamma1', 'gamma2'):
            out[what, ishell] = lensing.read_maps(ishell, what=what, dense=True)
    return out


@pytest.mark.parametrize('octant', [False, True])
def test_formats(tmp_path, octant):
    ref = products(*run(tmp_path / 'ref', octant=octant))
    assert np.all(ref['shell', 1][ref['shell', 1] != hp.UNSEEN] >= 0.) and np.any(ref['shell', 1] > 0.)
    for map_format, sub_format in FORMATS[1:]:
        builder, lensing = run(tmp_path / f'{map_format}_{sub_format}', map_format, sub_format, octant=octant)
        ext = '.npy' if map_format == 'healpix' else '.fits'
        assert builder.path(0).endswith(ext) and builder.path(0, sub=0).endswith('.npz' if sub_format == 'healpix' else '.fits')
        assert os.path.exists(lensing.path(0, what='gamma1'))
        out = products(builder, lensing)
        for key, value in ref.items():
            assert np.array_equal(out[key], value), (map_format, sub_format, key)


def test_legacy_alms(tmp_path):
    # a_lm of earlier runs, in healpy's FITS files without the mean of the field: the same smoothed maps
    builder, _ = run(tmp_path)
    density = builder.field_map(1, smooth_Mpc=10.)
    fname = builder._path_alms(1)
    alm = builder.read_alms(1)
    os.remove(fname)
    pollux.PolluxIO.write_alm(fname.replace('.npz', '.fits'), alm)
    assert builder._path_alms(1).endswith('.fits')
    builder.compute_tidal_tensor_maps(1, smooth_Mpc=10.)
    assert np.array_equal(builder.field_map(1, smooth_Mpc=10.), density)


def test_loader(tmp_path):
    # products of both formats read by the loader as the same HealSparse maps
    for map_format in ('healsparse', 'healpix'):
        run(tmp_path / map_format, map_format, map_format)
    maps = []
    for map_format in ('healsparse', 'healpix'):
        loader = pollux.io.loader.PolluxLoader(str(tmp_path / map_format), seed=0, nz='snapshots')
        maps.append([loader.read_shell(1), loader.read_tidal(1, smooth_Mpc=10.)[0], loader.read_kappa(2), loader.read_shear(2)[1]])
        assert np.array_equal(loader.read_shell_alms(1), loader.read_shell_alms(1))
    for a, b in zip(*maps):
        assert np.array_equal(a.valid_pixels, b.valid_pixels) and np.array_equal(a[a.valid_pixels], b[b.valid_pixels])


def test_bad_format(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, map_format='parquet')
    shutil.rmtree(tmp_path)
