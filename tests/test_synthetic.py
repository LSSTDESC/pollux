"""Shells of a synthetic simulation (:class:`pollux.reader.SyntheticReader`) on small maps: the reader, the shell
builder and its products, and the checks of the configurations when a product directory is used again."""
import numpy as np
import pytest

import pollux

NSIDE = 32


def make_reader(nparticles_per_sub=20000):
    """Small box near the observer: a few shells, a few replicas."""
    return pollux.SyntheticReader(boxsize=200., zsnapshots=(0.02, 0.04, 0.06), nparticles_per_sub=nparticles_per_sub)


def make_builder(root, reader=None, **kwargs):
    footprint = pollux.Footprint(NSIDE, pollux.PolluxIO(str(root)))
    return pollux.shells.SnapshotShellBuilder(reader or make_reader(), footprint, seed=0, **kwargs)


def test_reader():
    reader = pollux.SyntheticReader(nsplit=2, nparticles_per_sub=1000)
    assert reader.nsubs == 8 and reader.nparticles == 8000 and np.isfinite(reader.mass_particles)
    xyz, edges = reader.read_sub(1, 5)
    assert xyz.shape == (1000, 3)
    assert np.all((xyz >= edges.min(axis=0)) & (xyz <= edges.max(axis=0)))
    assert np.array_equal(xyz, reader.read_sub(1, 5)[0])
    assert not np.array_equal(xyz, reader.read_sub(2, 5)[0])
    assert not np.array_equal(xyz, pollux.SyntheticReader(nsplit=2, nparticles_per_sub=1000, seed=1).read_sub(1, 5)[0])
    empty, edges_only = reader.read_sub(1, 5, read_box_edges_only=True)
    assert len(empty) == 0 and np.array_equal(edges_only, edges)
    with pytest.raises(ValueError):
        reader.read_sub(0, 0, extra_variables=['id'])


def test_shells(tmp_path):
    builder = make_builder(tmp_path)
    assert builder.nmaps == 3
    builder.construct_maps()
    maps = [builder.read_maps(ishell) for ishell in range(builder.nmaps)]
    for m in maps:
        values = m[m.valid_pixels]
        assert m.valid_pixels.size > 0 and np.all(np.isfinite(values)) and np.all(values >= 0)

    # the same products, read by another builder of the same options
    again = make_builder(tmp_path)
    for ishell, m in enumerate(maps):
        assert np.array_equal(again.read_maps(ishell)[m.valid_pixels], m[m.valid_pixels])

    # other options on the same products: refused
    with pytest.raises(ValueError):
        make_builder(tmp_path, apply_grid_isometry=True)


def test_without_particles(tmp_path):
    # a reader without particles (e.g. for predictions only): its configuration compares equal to itself
    reader = make_reader(nparticles_per_sub=0)
    assert np.isnan(reader.mass_particles)
    make_builder(tmp_path, reader)
    make_builder(tmp_path, reader)


def test_footprint_config(tmp_path):
    io = pollux.PolluxIO(str(tmp_path))
    pollux.Footprint(NSIDE, io, mask='octant', padding_deg=10.)
    pollux.Footprint(NSIDE, io, mask='octant', padding_deg=10.)
    with pytest.raises(ValueError):
        pollux.Footprint(NSIDE, io, mask='octant', padding_deg=5.)


def test_octant_corner_at_observer(tmp_path):
    # without isometry, a corner of the box is at the observer: its replicas are kept for every pixel (no NaN direction)
    reader = make_reader()
    footprint = pollux.Footprint(NSIDE, pollux.PolluxIO(str(tmp_path)), mask='octant', padding_deg=10.)
    builder = pollux.shells.SnapshotShellBuilder(reader, footprint, seed=0)
    builder.construct_maps()
    m = builder.read_maps(0)
    assert m.valid_pixels.size > 0 and np.all(m[m.valid_pixels] >= 0)
    assert pollux.shells._query_replica_ipix(np.array([[0., 0., 0.], *np.random.default_rng(0).uniform(1., 2., (7, 3))]), NSIDE) is None


def test_inter_diff():
    # particles of two snapshots matched by id (SnapshotInterpShellBuilder), in any order
    from pollux.shells import _inter_diff
    rng = np.random.default_rng(0)
    ids = rng.permutation(10**6)[:2000]
    ar1, ar2 = rng.permutation(ids[:1500]), rng.permutation(ids[500:])
    i1, i2, o1, o2 = _inter_diff(ar1, ar2)
    assert np.array_equal(ar1[i1], ar2[i2]) and set(ar1[i1]) == set(ids[500:1500])
    assert set(ar1[o1]) == set(ids[:500]) and set(ar2[o2]) == set(ids[1500:])
