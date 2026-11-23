"""Catalog files: HDF5 (laid out for TXPipe's binned catalogs) and FITS, on a small synthetic run."""
import os

import numpy as np
import pytest

import pollux
from pollux.catalog import CatalogAugmentorShear

h5py = pytest.importorskip('h5py')

NSIDE = 32
NBINS = 2


@pytest.fixture(scope='module')
def lensing(tmp_path_factory):
    """Shells, and the convergence and shear maps per shell and of two n(z) bins."""
    root = tmp_path_factory.mktemp('catalogs')
    reader = pollux.SyntheticReader(boxsize=200., zsnapshots=(0.02, 0.04, 0.06), nparticles_per_sub=20000)
    footprint = pollux.Footprint(NSIDE, pollux.PolluxIO(str(root)))
    builder = pollux.shells.SnapshotShellBuilder(reader, footprint, seed=0, apply_grid_isometry=True)
    builder.construct_maps()
    snap = builder.snapshot_lensing_builder()
    snap.construct_maps()
    snap.compute_alms()
    snap.compute_shear_maps()
    zz = np.linspace(0., 0.1, 100)
    nzs = [(zz, np.exp(-((zz - z0) / 0.01)**2)) for z0 in (0.03, 0.05)]
    born = pollux.lensing.BornLensingBuilder(builder, nzs, tag='test')
    born.construct_maps()
    return born


def make_catalog(lensing, catalog_format):
    return pollux.catalog.CatalogClusteredPoisson(
        lensing,
        tag=catalog_format,
        nbar_input=2e-3,
        bias=1.5,
        interp_fields=('kappa', 'gamma1', 'gamma2', 'density'),
        smooth_fields=(),
        augmentors=[CatalogAugmentorShear(compute_reduced_shear=True, generate_intrinsic=True, compute_observed=True)],
        catalog_format=catalog_format)


def test_formats(lensing):
    cats = {}
    for catalog_format in ('hdf5', 'fits'):
        cat = make_catalog(lensing, catalog_format)
        cat.generate()
        assert cat.path(0).endswith(f'.{catalog_format}')
        cats[catalog_format] = cat.read_catalogs()
    for ibin in range(NBINS):
        hdf5, fits = cats['hdf5'][ibin], cats['fits'][ibin]
        assert list(hdf5) == list(fits) == make_catalog(lensing, 'hdf5').columns
        assert hdf5['ra'].size > 1000
        for k in hdf5:
            assert np.array_equal(hdf5[k], fits[k]), k


def test_txpipe_layout(lensing):
    cat = make_catalog(lensing, 'hdf5')
    cat.generate()
    columns = cat.read_catalogs()
    # each bin is a bin of TXPipe's binned catalogs, linking its columns
    with h5py.File(cat.path(1), 'r') as f:
        for kind in ('shear', 'lens'):
            assert dict(f[kind].attrs) == dict(nbin=1, bin_0=1)
        view = f['shear/bin_1']
        assert np.array_equal(view['g1'][:], columns[1]['e1_obs']) and np.array_equal(view['g2'][:], columns[1]['e2_obs'])
        assert np.all(view['weight'][:] == 1.) and view['weight'].id.get_storage_size() == 0  # unit weights, not stored
        assert np.array_equal(f['lens/bin_1/comoving_distance'][:], columns[1]['chi'])
    # the file of the bins, read as TXPipe does (BinnedCatalog.get_bins)
    with h5py.File(cat.path_txpipe, 'r') as f:
        for kind in ('shear', 'lens'):
            attrs = dict(f[kind].attrs)
            bins = [f'bin_{attrs[f"bin_{i}"]}' for i in range(attrs['nbin'])]
            assert bins == [f'bin_{i}' for i in range(NBINS)]
            for ibin, name in enumerate(bins):
                assert np.array_equal(f[kind][name]['ra'][:], columns[ibin]['ra'])
    assert os.path.dirname(cat.path_txpipe) == os.path.dirname(cat.path(0))


def test_shells_merged(lensing):
    # catalogs made shell by shell, then merged: the same as all at once, in both formats
    for catalog_format in ('hdf5', 'fits'):
        whole = make_catalog(lensing, catalog_format)
        whole.generate()
        expected = whole.read_catalogs()
        for ishell in range(whole.nshells):
            whole.generate(ishell=ishell)
        whole.merge_shell_catalogs()
        merged = whole.read_catalogs()
        for ibin in range(NBINS):
            for k in expected[ibin]:
                assert np.array_equal(merged[ibin][k], expected[ibin][k]), (catalog_format, k)


def test_subsample(lensing):
    sub = {}
    for catalog_format in ('hdf5', 'fits'):
        cat = make_catalog(lensing, catalog_format)
        cat.generate(0)
        metadata, sub[catalog_format] = cat.read_catalogs(0, return_metadata=True, subsample=0.1, columns=['ra', 'g1'], remove_mean=['g1'])
        assert metadata['POLLUXV'] and metadata['NBAR_0'] == 2e-3
        assert cat._ngal_used(0, 0.1) == sub[catalog_format]['ra'].size
        assert cat.io.read_columns_info(cat.path(0))[1] == cat.columns
    for k in ('ra', 'g1'):
        assert np.array_equal(sub['hdf5'][k], sub['fits'][k])
    assert abs(sub['hdf5']['g1'].mean()) < 1e-6


def test_txpipe_nz(lensing):
    # the n(z) of the bins in qp's format, as TXPipe reads it (QPNOfZFile: groups meta and data, last row all bins)
    cat = make_catalog(lensing, 'hdf5')
    cat.generate()
    z, nzs = cat.nzs
    assert z[-1] == lensing.shell_builder.shell_edges_redshift[-1] and np.allclose(np.trapz(nzs, z, axis=1), 1.)
    with h5py.File(cat.path_nzs, 'r') as f:
        assert f['meta/pdf_name'][0].decode() == 'interp' and f['meta/pdf_version'][0] == 0
        xvals, yvals = f['meta/xvals'][:], f['data/yvals'][:]
    assert xvals.shape == (1, len(z)) and yvals.shape == (NBINS + 1, len(z))
    assert np.allclose(yvals[:NBINS], nzs) and np.allclose(yvals[-1], cat.Ngal @ nzs / cat.Ngal.sum())
    # the galaxies follow it, up to the mean density of each shell over the footprint (clustered catalogs)
    columns = cat.read_catalogs(columns=['z'])
    for ibin in range(NBINS):
        cdf = np.concatenate([[0.], np.cumsum(0.5 * np.diff(z) * (nzs[ibin][1:] + nzs[ibin][:-1]))])
        zgal = np.sort(columns[ibin]['z'])
        assert np.max(np.abs(np.interp(zgal, z, cdf) - np.arange(1, zgal.size + 1) / zgal.size)) < 0.05


def test_txpipe_nz_qp(lensing):
    # read by qp itself, where it is installed (it needs numpy 2)
    qp = pytest.importorskip('qp')
    cat = make_catalog(lensing, 'hdf5')
    cat.generate()
    z, nzs = cat.nzs
    ens = qp.read(cat.path_nzs)
    assert ens.npdf == NBINS + 1 and np.allclose(ens.pdf(z)[:NBINS], nzs)
