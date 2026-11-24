"""Figures of the documentation, from the products of the HACC-HOS-Y1 shared space at NERSC (the PNG files are committed:
Read the Docs can't reach the data).

    python make_figures.py [name ...]   (default: all; names: workflow, workflow_combined, lightcone, zoom, lensing, baryons, nz, cone, tracers)

Run with the environment of the shared space (``source .../HACC-HOS-Y1/scripts/setup.sh``), on a login node.
"""
import os
import runpy
import sys
import tempfile

import healpy as hp
import healsparse as hsp
import matplotlib
import numpy as np
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = '/global/cfs/cdirs/desc-wl/projects/hos-tt/HACC-HOS-Y1'
PRODUCTS = f'{SHARED}/products'
FULLSKY = f'{PRODUCTS}/HACC150_3/nside8192/seed0/dmo'
CATALOG_TAG = 'poisson_b2.0_nbar1.5_lognormal_bias10pc'
DPI = 150  # figures are 9 inches wide at most, the width of the docs (so their fonts look alike)

plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.dpi': DPI})


CACHE = os.environ.get('POLLUX_FIGURES_CACHE', os.path.join(HERE, '.cache'))


def cached(name, compute):
    """The arrays of a figure: from ``{CACHE}/{name}.npz`` if there, else from ``compute()`` (a dict), then saved there."""
    fname = os.path.join(CACHE, f'{name}.npz')
    if os.path.exists(fname):
        with np.load(fname) as f:
            return {k: f[k] for k in f.files}
    data = compute()
    os.makedirs(CACHE, exist_ok=True)
    np.savez(fname, **data)
    return {k: np.asarray(v) for k, v in data.items()}


def save(fig, name):
    """Save a figure to ``{name}.png`` next to this script, and close it."""
    fname = os.path.join(HERE, f'{name}.png')
    fig.savefig(fname)
    plt.close(fig)
    print(fname, flush=True)


# ======================================================================================================================
# region Style

# lean figures: no frames or ticks on maps (scale bars instead), small labels, thin colour bars; the font of the report
# (Fira Sans, from the TeX Live of NERSC) when it is there
INK, MUTED, LINE = '#1c2430', '#5b6678', '#c9d2df'
FIRA = '/global/common/software/nersc9/texlive/2024-17c650b50c05ee823c6093081103ed0ab76a21df/texmf-dist/fonts/opentype/public/fira'
if os.path.isdir(FIRA):
    from matplotlib import font_manager
    for _style in ('Regular', 'Italic', 'Medium', 'Bold'):
        if os.path.exists(os.path.join(FIRA, f'FiraSans-{_style}.otf')):
            font_manager.fontManager.addfont(os.path.join(FIRA, f'FiraSans-{_style}.otf'))
    plt.rcParams.update({'font.family': 'Fira Sans', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Fira Sans', 'mathtext.it': 'Fira Sans:italic'})
plt.rcParams.update({
    'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': LINE,
    'axes.linewidth': 0.6, 'axes.titlesize': 10, 'axes.titlelocation': 'left', 'axes.titlecolor': INK, 'axes.titlepad': 5,
    'legend.frameon': False, 'savefig.facecolor': 'white'})


def clean(ax):
    """No ticks or frame (maps)."""
    ax.set(xticks=[], yticks=[])
    for side in ax.spines.values():
        side.set_visible(False)


def scalebar(ax, length, label, size, color='white'):
    """A scale bar of ``length`` (the unit of the axes) at the bottom left of a square map of ``size``."""
    x0, y0 = -size / 2 + 0.06 * size, -size / 2 + 0.07 * size
    sign = -1 if ax.xaxis_inverted() else 1  # east to the left
    ax.plot([sign * x0, sign * (x0 + length)], [y0, y0], color=color, lw=2, solid_capstyle='butt')
    ax.text(sign * (x0 + length / 2), y0 + 0.025 * size, label, color=color, ha='center', va='bottom', fontsize=9)


def mollweide(ax, m, nest=False, **kwargs):
    """A full-sky HEALPix map on a Mollweide axes, east to the left (as healpy's mollview), without grid or ticks."""
    lon = np.linspace(-180, 180, 1600)
    lat = np.linspace(-90, 90, 800)
    lon2, lat2 = np.meshgrid(lon, lat)
    values = m[hp.ang2pix(hp.npix2nside(len(m)), -lon2, lat2, lonlat=True, nest=nest)]
    im = ax.pcolormesh(np.radians(lon), np.radians(lat), values, shading='auto', rasterized=True, **kwargs)
    ax.set(xticks=[], yticks=[])
    ax.grid(False)
    ax.spines['geo'].set_edgecolor(LINE)
    return im


def outline(ax, ra0, dec0, size, color='white', mollweide=False):
    """The outline of a square patch: on a Mollweide axes, or on a map in gnomonic coordinates (degrees)."""
    h = size / 2
    corners = np.array([(-h, -h), (h, -h), (h, h), (-h, h), (-h, -h)])
    if mollweide:
        lon = ra0 + corners[:, 0] / np.cos(np.radians(dec0))
        ax.plot(np.radians(-lon), np.radians(dec0 + corners[:, 1]), color=color, lw=1.2)
    else:
        ax.plot(corners[:, 0], corners[:, 1], color=color, lw=1.)


def colorbar(fig, mappable, axes, label, **kwargs):
    """A thin horizontal colour bar under the panels."""
    cbar = fig.colorbar(mappable, ax=axes, orientation='horizontal', fraction=0.05, aspect=60, pad=0.04, shrink=0.6, **kwargs)
    cbar.set_label(label, color=INK)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, color=MUTED, labelsize=9)
    return cbar


# endregion

# ======================================================================================================================
# region Helpers


def run_globals(script, config):
    """Module-level objects of a run script of the shared space (its dictionaries of tasks), for its configuration
    with the products of the shared space as ``save_dir_root``, without running it."""
    folder = f'{SHARED}/scripts' if os.path.exists(f'{SHARED}/scripts/{script}') else f'{SHARED}/scripts/test_runs'
    with open(f'{folder}/configs/{config}') as f:
        options = yaml.safe_load(f)
    options['save_dir_root'] = PRODUCTS
    argv = sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, config)
        with open(path, 'w') as f:
            yaml.safe_dump(options, f)
        sys.argv = [f'{folder}/{script}', path]
        try:
            return runpy.run_path(sys.argv[0], run_name='figures')
        finally:
            sys.argv = argv


def patch_grid(ra0, dec0, size, npix):
    """RA and Dec of a gnomonic grid of ``npix`` x ``npix`` points, ``size`` degrees wide, centered on (ra0, dec0), and its
    extent in degrees (x to the east, i.e. to the left on the sky)."""
    x = np.linspace(-size / 2, size / 2, npix)
    X, Y = np.meshgrid(x, x)
    # gnomonic projection: tangent plane coordinates (radians) to sky coordinates
    xi, eta = np.radians(-X), np.radians(Y)
    d0, a0 = np.radians(dec0), np.radians(ra0)
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    with np.errstate(invalid='ignore', divide='ignore'):
        dec = np.arcsin(np.cos(c) * np.sin(d0) + np.where(rho > 0, eta * np.sin(c) * np.cos(d0) / rho, 0.))
        ra = a0 + np.arctan2(xi * np.sin(c), rho * np.cos(d0) * np.cos(c) - eta * np.sin(d0) * np.sin(c))
    return np.degrees(ra) % 360., np.degrees(dec), (size / 2, -size / 2, -size / 2, size / 2)


def to_plane(ra, dec, ra0, dec0):
    """Gnomonic coordinates (degrees, x to the east) of sky positions, around (ra0, dec0)."""
    a, d, a0, d0 = np.radians(ra), np.radians(dec), np.radians(ra0), np.radians(dec0)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    x = np.cos(d) * np.sin(a - a0) / cosc
    y = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return np.degrees(x), np.degrees(y)


# endregion

# ======================================================================================================================
# region Figures


WORKFLOW_STYLE = """
.pollux-workflow { --wf-bg: #ffffff; --wf-card: #ffffff; --wf-line: #c9d2df; --wf-edge: #8a97ab; --wf-ink: #1c2430; --wf-muted: #5b6678;
  --wf-band: #f3f6fa; --wf-default: #2f6fdf; --wf-heavy: #d9822b; --wf-highmem: #2a9d78; --wf-input: #8a97ab;
  font-family: Inter, "Helvetica Neue", Arial, sans-serif; }
.pollux-workflow .bg { fill: var(--wf-bg); }
.pollux-workflow .band { fill: var(--wf-band); }
.pollux-workflow .card { fill: var(--wf-card); stroke: var(--wf-line); stroke-width: 1; }
.pollux-workflow .card.input { fill: var(--wf-band); stroke-dasharray: 4 3; }
.pollux-workflow .name { font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace; font-size: 12px; font-weight: 600;
  fill: var(--wf-ink); }
.pollux-workflow .desc { font-size: 11.5px; fill: var(--wf-muted); }
.pollux-workflow .count { font-size: 11px; font-weight: 600; fill: var(--wf-ink); font-variant-numeric: tabular-nums; }
.pollux-workflow .edge { fill: none; stroke: var(--wf-edge); stroke-width: 1.4; }
.pollux-workflow .edge.feed { stroke-dasharray: 5 4; }
.pollux-workflow .edge-label { font-size: 10.5px; fill: var(--wf-muted); }
.pollux-workflow .head { font-family: "Space Grotesk", Inter, Arial, sans-serif; font-size: 14px; font-weight: 600; fill: var(--wf-ink); }
.pollux-workflow .sub { font-size: 11.5px; fill: var(--wf-muted); }
.pollux-workflow .legend { font-size: 11.5px; fill: var(--wf-muted); }
.pollux-workflow .legend-head { font-size: 11.5px; font-weight: 600; fill: var(--wf-ink); }
.pollux-workflow .dot-default { fill: var(--wf-default); }
.pollux-workflow .dot-heavy { fill: var(--wf-heavy); }
.pollux-workflow .dot-highmem { fill: var(--wf-highmem); }
.pollux-workflow .dot-input { fill: var(--wf-input); }
.pollux-workflow .arrowhead { fill: var(--wf-edge); }
"""


def rasterize(name):
    """Write ``{name}.png`` from ``{name}.svg``, for the PDF of the documentation (LaTeX does not take SVG files), with
    the values of the CSS variables of the SVG, which cairosvg does not resolve. Needs cairosvg."""
    import re
    try:
        import cairosvg
    except ImportError:
        print(f'{name}.png not written: cairosvg is not installed', flush=True)
        return
    with open(os.path.join(HERE, f'{name}.svg')) as fh:
        svg = fh.read()
    values = dict(re.findall(r'(--wf-[\w-]+):\s*([^;]+);', svg))
    svg = re.sub(r'var\((--wf-[\w-]+)\)', lambda m: values[m.group(1)].strip(), svg)
    fname = os.path.join(HERE, f'{name}.png')
    cairosvg.svg2png(bytestring=svg.encode(), write_to=fname, output_width=2 * int(re.search(r'viewBox="0 0 (\d+)', svg).group(1)), background_color='white')
    print(fname, flush=True)


def workflow():
    """Tasks of a realization (pollux.pipeline.tasks) as an SVG diagram, about 840 pixels wide: cards in three columns,
    coloured by executor, with their numbers for HACC091_1024 at nside 2048 (dark matter only and baryonified, one catalog
    and its 2pt functions). Colours are CSS variables with light defaults, so that a page can restyle them."""
    W, H, PAD, GAP, MARGIN = 232, 64, 24, 36, 30
    col = [PAD + i * (W + GAP) for i in range(3)]
    row = dict(A=84, B=188, C=292, D=396, E=500, F=656, G=760)
    nodes = {
        'particles': ('A', 0, 'particles', 'GenericIO snapshots', None, 'input'),
        'nz': ('B', 0, 'n(z)', 'redshift distributions', None, 'input'),
        'sub': ('A', 1, 'construct_shell_sub', 'bin each sub-volume', '3,264', 'default'),
        'combine': ('A', 2, 'combine_shell_subs', 'sum them into shells', '51', 'default'),
        'shell_alms': ('B', 1, 'compute_shell_alms', 'harmonics, then Cls', '51', 'heavy'),
        'kappa': ('B', 2, 'construct_kappa_map', 'Born convergence per shell', '51', 'default'),
        'tidal': ('C', 1, 'compute_tidal_maps', 'tidal field, smoothed δ', '51', 'heavy'),
        'kappa_alms': ('C', 2, 'compute_kappa_alms', 'harmonics of κ, then Cls', '51', 'heavy'),
        'cat': ('D', 1, 'construct_catalog_shell', 'galaxies of each shell', '51', 'highmem'),
        'shear': ('D', 2, 'compute_shear_maps', 'γ₁, γ₂ from κ', '51', 'heavy'),
        'merge': ('E', 0, 'merge_catalog', 'one catalog per bin', '5', 'highmem'),
        'pairs': ('E', 1, 'compute_catalog_2pt_sub', 'TreeCorr, per bin pair', '175', 'heavy'),
        '2pt': ('E', 2, 'compute_catalog_2pt', 'SACC files and plots', '1', 'default'),
        'halos': ('F', 0, 'halo properties', 'GenericIO', None, 'input'),
        'halocat': ('F', 1, 'construct_halo_catalog', 'halos of each shell', '51', 'highmem'),
        'bary': ('F', 2, 'construct_baryonified_shell', 'BaryonForge, in parallel', '51', 'highmem'),
    }
    edges = [('particles', 'sub'), ('sub', 'combine'), ('combine', 'shell_alms'), ('combine', 'kappa'), ('shell_alms', 'tidal'), ('kappa', 'kappa_alms'),
             ('tidal', 'cat'), ('kappa_alms', 'shear'), ('shear', 'cat'), ('cat', 'merge'), ('merge', 'pairs'), ('pairs', '2pt'), ('halos', 'halocat'),
             ('halocat', 'bary')]

    def box(name):
        return col[nodes[name][1]], row[nodes[name][0]]

    def path(a, b):
        (xa, ya), (xb, yb) = box(a), box(b)
        if ya == yb:  # same row: straight, either way
            if xb > xa:
                return f'M{xa + W},{ya + H / 2:.0f} L{xb - 1},{yb + H / 2:.0f}'
            return f'M{xa},{ya + H / 2:.0f} L{xb + W + 1},{yb + H / 2:.0f}'
        x1, y1, x2, y2 = xa + W / 2, ya + H, xb + W / 2, yb  # down: a smooth curve, ending straight into the card
        y3 = y2 - 12
        dy = (y3 - y1) * 0.55
        return f'M{x1:.0f},{y1:.0f} C{x1:.0f},{y1 + dy:.0f} {x2:.0f},{y3 - dy:.0f} {x2:.0f},{y3:.0f} L{x2:.0f},{y2 - 1:.0f}'

    width, height = col[2] + W + MARGIN, row['G'] + H + 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" class="pollux-workflow" role="img" '
           f'aria-label="Tasks of a pollux realization">', f'<style>{WORKFLOW_STYLE}</style>', f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
           '<defs><marker id="wf-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto">'
           '<path class="arrowhead" d="M0,1 L10,5 L0,9 z"/></marker></defs>']
    # bands and their headings
    top = row['A'] - 48
    out.append(f'<rect class="band" x="8" y="{top}" width="{width - 16}" height="{row["E"] + H + 20 - top}" rx="14"/>')
    out.append(f'<text class="head" x="{PAD}" y="{row["A"] - 20}">Dark matter only</text>')
    out.append(f'<text class="sub" x="{PAD + 128}" y="{row["A"] - 20}">dmo/</text>')
    out.append(f'<rect class="band" x="8" y="{row["F"] - 48}" width="{width - 16}" height="{row["G"] + H + 20 - (row["F"] - 48)}" rx="14"/>')
    out.append(f'<text class="head" x="{PAD}" y="{row["F"] - 20}">Baryonified</text>')
    out.append(f'<text class="sub" x="{PAD + 96}" y="{row["F"] - 20}">dmb/{{tag}}/ · halos move the matter of the dmo shells</text>')
    # legend of the executors, in the empty part of the first column
    lx, ly = col[0] + 6, row['C'] + 30
    out.append(f'<text class="legend-head" x="{lx}" y="{ly}">Executors (tasks per node)</text>')
    for k, (kind, label) in enumerate((('default', 'default: 32, light tasks'), ('heavy', 'heavy: 1, all the cores'),
                                       ('highmem', 'highmem: 4, large memory'), ('input', 'input files'))):
        out.append(f'<circle class="dot-{kind}" cx="{lx + 5}" cy="{ly + 22 + 22 * k}" r="5"/>')
        out.append(f'<text class="legend" x="{lx + 16}" y="{ly + 26 + 22 * k}">{label}</text>')
    out.append(f'<text class="legend" x="{lx}" y="{ly + 128}">×n: tasks per realization</text>')
    for a, b in edges:
        out.append(f'<path class="edge" d="{path(a, b)}" marker-end="url(#wf-arrow)"/>')
    # the n(z) gives the redshifts of the galaxies
    (xn, yn), (xt, yt) = box('nz'), box('cat')
    gx1, ym, yt2 = col[1] - GAP / 2, yn + H / 2, yt + H / 2
    out.append(f'<path class="edge feed" d="M{xn + W},{ym:.0f} L{gx1 - 10:.0f},{ym:.0f} Q{gx1:.0f},{ym:.0f} {gx1:.0f},{ym + 10:.0f} L{gx1:.0f},{yt2 - 10:.0f} '
               f'Q{gx1:.0f},{yt2:.0f} {gx1 + 10:.0f},{yt2:.0f} L{xt - 1},{yt2:.0f}" marker-end="url(#wf-arrow)"/>')
    # the dmo shells feed the baryonification: down the right margin
    (xc, yc), (xb, yb) = box('combine'), box('bary')
    gx = xc + W + MARGIN / 2
    ys, ye = yc + H / 2, yb + H / 2
    out.append(f'<path class="edge feed" d="M{xc + W},{ys:.0f} L{gx - 10:.0f},{ys:.0f} Q{gx:.0f},{ys:.0f} {gx:.0f},{ys + 10:.0f} L{gx:.0f},{ye - 10:.0f} '
               f'Q{gx:.0f},{ye:.0f} {gx - 10:.0f},{ye:.0f} L{xb + W + 1},{ye:.0f}" marker-end="url(#wf-arrow)"/>')
    out.append(f'<text class="edge-label" x="{gx - 5:.0f}" y="{row["F"] - 56}" text-anchor="end">dmo shells</text>')
    # the baryonified shells go through the same chain
    y = row['G']
    out.append(f'<path class="edge" d="M{xb + W / 2:.0f},{yb + H} L{xb + W / 2:.0f},{y - 1}" marker-end="url(#wf-arrow)"/>')
    out.append(f'<rect class="card input" x="{col[0]}" y="{y}" width="{col[2] + W - col[0]}" height="{H}" rx="10"/>')
    out.append(f'<text class="name" x="{col[0] + 16}" y="{y + 27}">the same tasks, from the baryonified shells</text>')
    out.append(f'<text class="desc" x="{col[0] + 16}" y="{y + 46}">harmonics, convergence and shear, catalogs and 2pt functions with baryons</text>')
    for name, (r, c, title, desc, count, kind) in nodes.items():
        x, y = col[c], row[r]
        out.append(f'<rect class="card{" input" if kind == "input" else ""}" x="{x}" y="{y}" width="{W}" height="{H}" rx="10"/>')
        out.append(f'<text class="name" x="{x + 14}" y="{y + 25}">{title}</text>')
        out.append(f'<text class="desc" x="{x + 14}" y="{y + 45}">{desc}</text>')
        if count:
            out.append(f'<circle class="dot-{kind}" cx="{x + W - 17}" cy="{y + 41}" r="5"/>')
            out.append(f'<text class="count" x="{x + W - 28}" y="{y + 45}" text-anchor="end">×{count}</text>')
    out.append(f'<text class="sub" x="{PAD}" y="{height - 14}">One realization of HACC091_1024 at nside 2048 (pollux.pipeline.tasks). '
               'A run resumes where it stopped: tasks skip what is written.</text>')
    out.append('</svg>')
    fname = os.path.join(HERE, 'workflow.svg')
    with open(fname, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    print(fname, flush=True)
    rasterize('workflow')


def workflow_combined():
    """Tasks of a realization (pollux.pipeline.tasks) as one chain, with the baryonification as an optional step between
    the shells and the rest (``workflow_combined.svg``, about 840 pixels wide; ``workflow`` draws the two matter models
    apart). Numbers of HACC091_1024 at nside 2048."""
    W, H, PAD, GAP, MARGIN = 232, 64, 24, 36, 30
    col = [PAD + i * (W + GAP) for i in range(3)]
    row = dict(A=60, B=164, C=268, D=372, E=476, F=580)
    nodes = {
        'particles': ('A', 0, 'particles', 'GenericIO snapshots', None, 'input'),
        'sub': ('A', 1, 'construct_shell_sub', 'bin each sub-volume', '3,264', 'default'),
        'combine': ('A', 2, 'combine_shell_subs', 'sum them into shells', '51', 'default'),
        'halos': ('B', 0, 'halo properties', 'GenericIO', None, 'input'),
        'halocat': ('B', 1, 'construct_halo_catalog', 'halos of each shell', '51', 'highmem'),
        'bary': ('B', 2, 'construct_baryonified_shell', 'optional: BaryonForge', '51', 'highmem'),
        'shell_alms': ('C', 1, 'compute_shell_alms', 'harmonics, then Cls', '51', 'heavy'),
        'kappa': ('C', 2, 'construct_kappa_map', 'Born convergence per shell', '51', 'default'),
        'tidal': ('D', 1, 'compute_tidal_maps', 'tidal field, smoothed δ', '51', 'heavy'),
        'kappa_alms': ('D', 2, 'compute_kappa_alms', 'harmonics of κ, then Cls', '51', 'heavy'),
        'nz': ('E', 0, 'n(z)', 'redshift distributions', None, 'input'),
        'cat': ('E', 1, 'construct_catalog_shell', 'galaxies of each shell', '51', 'highmem'),
        'shear': ('E', 2, 'compute_shear_maps', 'γ₁, γ₂ from κ', '51', 'heavy'),
        'merge': ('F', 0, 'merge_catalog', 'one catalog per bin', '5', 'highmem'),
        'pairs': ('F', 1, 'compute_catalog_2pt_sub', 'TreeCorr, per bin pair', '175', 'heavy'),
        '2pt': ('F', 2, 'compute_catalog_2pt', 'SACC files and plots', '1', 'default'),
    }
    edges = [('particles', 'sub'), ('sub', 'combine'), ('combine', 'bary'), ('halos', 'halocat'), ('halocat', 'bary'), ('bary', 'shell_alms'), ('bary', 'kappa'),
             ('shell_alms', 'tidal'), ('kappa', 'kappa_alms'), ('tidal', 'cat'), ('kappa_alms', 'shear'), ('shear', 'cat'), ('nz', 'cat'), ('cat', 'merge'),
             ('merge', 'pairs'), ('pairs', '2pt')]

    def box(name):
        return col[nodes[name][1]], row[nodes[name][0]]

    def path(a, b):
        (xa, ya), (xb, yb) = box(a), box(b)
        if ya == yb:  # same row: straight, either way
            if xb > xa:
                return f'M{xa + W},{ya + H / 2:.0f} L{xb - 1},{yb + H / 2:.0f}'
            return f'M{xa},{ya + H / 2:.0f} L{xb + W + 1},{yb + H / 2:.0f}'
        x1, y1, x2, y2 = xa + W / 2, ya + H, xb + W / 2, yb  # down: a smooth curve, ending straight into the card
        y3 = y2 - 12
        dy = (y3 - y1) * 0.55
        return f'M{x1:.0f},{y1:.0f} C{x1:.0f},{y1 + dy:.0f} {x2:.0f},{y3 - dy:.0f} {x2:.0f},{y3:.0f} L{x2:.0f},{y2 - 1:.0f}'

    width, height = col[2] + W + MARGIN, row['F'] + H + 112
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" class="pollux-workflow" role="img" '
           f'aria-label="Tasks of a pollux realization">', f'<style>{WORKFLOW_STYLE}</style>', f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
           '<defs><marker id="wf-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto">'
           '<path class="arrowhead" d="M0,1 L10,5 L0,9 z"/></marker></defs>']
    top = row['A'] - 40
    out.append(f'<rect class="band" x="8" y="{top}" width="{width - 16}" height="{row["F"] + H + 20 - top}" rx="14"/>')
    for a, b in edges:
        out.append(f'<path class="edge" d="{path(a, b)}" marker-end="url(#wf-arrow)"/>')
    # without baryons, the dmo shells go on directly: down the right margin, into the convergence
    (xc, yc), (xk, yk) = box('combine'), box('kappa')
    gx = xc + W + MARGIN / 2
    ys, ye = yc + H / 2, yk + H / 2
    out.append(f'<path class="edge feed" d="M{xc + W},{ys:.0f} L{gx - 10:.0f},{ys:.0f} Q{gx:.0f},{ys:.0f} {gx:.0f},{ys + 10:.0f} L{gx:.0f},{ye - 10:.0f} '
               f'Q{gx:.0f},{ye:.0f} {gx - 10:.0f},{ye:.0f} L{xk + W + 1},{ye:.0f}" marker-end="url(#wf-arrow)"/>')
    out.append(f'<text class="edge-label" x="{gx - 5:.0f}" y="{row["B"] - 12}" text-anchor="end">dmo shells</text>')
    for name, (r, c, title, desc, count, kind) in nodes.items():
        x, y = col[c], row[r]
        dash = ' stroke-dasharray="5 3"' if name == 'bary' else ''  # optional
        out.append(f'<rect class="card{" input" if kind == "input" else ""}" x="{x}" y="{y}" width="{W}" height="{H}" rx="10"{dash}/>')
        out.append(f'<text class="name" x="{x + 14}" y="{y + 25}">{title}</text>')
        out.append(f'<text class="desc" x="{x + 14}" y="{y + 45}">{desc}</text>')
        if count:
            out.append(f'<circle class="dot-{kind}" cx="{x + W - 17}" cy="{y + 41}" r="5"/>')
            out.append(f'<text class="count" x="{x + W - 28}" y="{y + 45}" text-anchor="end">×{count}</text>')
    # legend and notes, under the band
    ly = row['F'] + H + 48
    x = PAD
    for kind, label in (('default', 'default: 32 tasks per node'), ('heavy', 'heavy: 1 per node, all cores'), ('highmem', 'highmem: 4 per node'),
                        ('input', 'input files')):
        out.append(f'<circle class="dot-{kind}" cx="{x + 5}" cy="{ly - 4}" r="5"/>')
        out.append(f'<text class="legend" x="{x + 15}" y="{ly}">{label}</text>')
        x += 15 + 7.0 * len(label) + 24
    out.append(f'<text class="sub" x="{PAD}" y="{ly + 26}">With baryons, the tasks after the shells run twice: on the dmo shells (dmo/) and on the '
               'baryonified ones (dmb/{tag}/).</text>')
    out.append(f'<text class="sub" x="{PAD}" y="{ly + 46}">×n: tasks of one realization of HACC091_1024 at nside 2048. A run resumes where it stopped: '
               'tasks skip what is written.</text>')
    out.append('</svg>')
    fname = os.path.join(HERE, 'workflow_combined.svg')
    with open(fname, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    print(fname, flush=True)
    rasterize('workflow_combined')


def lightcone():
    """A slice through the observer of a lightcone made of replicas of one snapshot box of HACC091_1024, without (left)
    and with (right) the random isometries of the shells."""

    def compute():
        from pollux.pipeline import tasks as T
        g = run_globals('run_validation.py', 'run_validation_HACC091_1024.yaml')
        builder = T._make_shell_builder(g['reader_dict'], g['shell_dicts'][0], None)
        reader, boxsize, edges = builder.reader, builder.reader.boxsize, builder.shell_edges_comoving_distance
        # density mesh of one snapshot, from all its particles
        nmesh, isnap = 512, 8
        mesh = np.zeros((nmesh,) * 3, dtype=np.float32)
        for isub in reader.isubs_for_snapshot(isnap):
            xyz = reader.read_sub(isnap, isub)[0]
            idx = np.floor(np.mod(xyz, boxsize) / boxsize * nmesh).astype(np.int64) % nmesh
            mesh += np.bincount(np.ravel_multi_index(idx.T, mesh.shape), minlength=mesh.size).reshape(mesh.shape).astype(np.float32)
        mesh /= mesh.mean()
        # the plane of the observer, sampled in a slab of 20 Mpc
        rmax, npix = 1.6 * boxsize, 900
        u = np.linspace(-rmax, rmax, npix)
        U, V = np.meshgrid(u, u)
        r = np.hypot(U, V)
        ishell = np.clip(np.searchsorted(edges, r) - 1, 0, len(edges) - 2)

        def slice_(isometries):
            image = np.zeros_like(U)
            for w in np.linspace(-10., 10., 6):
                p = np.stack([U, V, np.full_like(U, w)], axis=-1)
                X = np.empty_like(p)
                for i in np.unique(ishell):
                    sel = ishell == i
                    A, b = isometries[i]
                    X[sel] = p[sel] @ A.T - b  # p = (x + b) A for a particle at x in its box (and replica)
                k = (np.floor(np.mod(X, boxsize) / boxsize * nmesh).astype(np.int64)) % nmesh
                image += mesh[k[..., 0], k[..., 1], k[..., 2]]
            return np.where(r < rmax, image / 6, np.nan)

        identity = [(np.eye(3), np.zeros(3))] * (len(edges) - 1)
        rotated = [tuple(np.asarray(x, dtype=float) for x in Ab) for Ab in builder.Abs]
        return dict(replicas=slice_(identity), rotated=slice_(rotated), boxsize=boxsize, edges=edges, rmax=rmax)

    d = cached('lightcone', compute)
    boxsize, edges, rmax = float(d['boxsize']), d['edges'], float(d['rmax'])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.9), layout='constrained')
    for ax, image, label in zip(axes, (d['replicas'], d['rotated']), ('replicas', 'with isometries')):
        with np.errstate(divide='ignore', invalid='ignore'):
            im = ax.imshow(np.log10(np.maximum(image, 0.05)), origin='lower', extent=(-rmax, rmax, -rmax, rmax), cmap='magma', vmin=-0.8, vmax=1.2)
        disk = matplotlib.patches.Circle((0, 0), rmax, transform=ax.transData)
        for e in edges[edges < rmax]:
            ax.add_patch(matplotlib.patches.Circle((0, 0), e, fill=False, color='white', lw=0.5, alpha=0.3))
        ax.plot(0, 0, '*', color='white', ms=9)
        ax.set(xlim=(-rmax, rmax), ylim=(-rmax, rmax), aspect='equal', title=label)
        clean(ax)
        if ax is axes[0]:  # the faces of the (unrotated) replicas, inside the disk, and one box size
            for k in range(-2, 3):
                for line in (ax.plot([k * boxsize] * 2, [-rmax, rmax]), ax.plot([-rmax, rmax], [k * boxsize] * 2)):
                    line[0].set(color='white', lw=0.8, ls='--', alpha=0.8, clip_path=disk)
            ax.annotate('', xy=(boxsize, -rmax * 1.04), xytext=(0, -rmax * 1.04), arrowprops=dict(arrowstyle='<->', color=INK, lw=1.), annotation_clip=False)
            ax.text(boxsize / 2, -rmax * 1.1, f'{boxsize:.0f} Mpc', color=INK, ha='center', va='top', fontsize=9)
    colorbar(fig, im, axes, r'$\log_{10}(1 + \delta)$')
    save(fig, 'lightcone')


def zoom():
    """A shell of HACC150_3 at nside 8192: the full sky, then patches of 10 and 1 degrees at full resolution."""
    from matplotlib.colors import LogNorm
    ishell, ra0, dec0 = 18, 55., 20.
    sizes = ((10., 900), (1., 700))

    def compute():
        m = hsp.HealSparseMap.read(f'{FULLSKY}/shells/shell_{ishell:03d}.fits')
        mean = np.mean(m[m.valid_pixels])
        full = m.degrade(512).generate_healpix_map(nest=True)
        out = dict(full=np.where(full > hp.UNSEEN, full / mean, np.nan))
        for size, npix in sizes:
            ra, dec, _ = patch_grid(ra0, dec0, size, npix)
            v = m.get_values_pos(ra, dec, lonlat=True)
            out[f'patch{size:g}'] = np.where(v > hp.UNSEEN, v, 0.) / mean
        return out

    d = cached('zoom', compute)
    norm = LogNorm(0.3, 10.)
    fig = plt.figure(figsize=(9, 3.), layout='constrained')
    grid = fig.add_gridspec(1, 3, width_ratios=(2, 1, 1))
    ax = fig.add_subplot(grid[0], projection='mollweide')
    im = mollweide(ax, np.clip(d['full'], 0.3, None), nest=True, cmap='magma', norm=norm)
    outline(ax, ra0, dec0, 10., mollweide=True)
    for k, ((size, _), bar, label, inner) in enumerate(zip(sizes, (2., 10 / 60), ('2°', "10'"), (1., None))):
        ax = fig.add_subplot(grid[k + 1])
        ax.imshow(np.clip(d[f'patch{size:g}'], 0.3, None), origin='lower', extent=(size / 2, -size / 2, -size / 2, size / 2), cmap='magma', norm=norm,
                  interpolation='nearest')
        ax.set(xlim=(size / 2, -size / 2), ylim=(-size / 2, size / 2), aspect='equal')
        clean(ax)
        scalebar(ax, bar, label, size)
        if inner:
            outline(ax, 0., 0., inner)
    colorbar(fig, im, fig.axes, r'$1 + \delta$')
    save(fig, 'zoom')


def lensing():
    """Convergence and shear of the last tomographic bin of HACC150_3 on a patch of 3 degrees."""
    from scipy.ndimage import gaussian_filter
    ibin, ra0, dec0, size, npix = 4, 40., 10., 3., 900

    def compute():
        maps = {k: hsp.HealSparseMap.read(f'{FULLSKY}/kappa/SRDv1_nz/{k}_bin{ibin}.fits') for k in ('kappa', 'gamma1', 'gamma2')}
        ra, dec, _ = patch_grid(ra0, dec0, size, npix)
        return {name: maps[name].get_values_pos(ra, dec, lonlat=True) for name in maps}

    d = cached('lensing', compute)
    k, g1, g2 = d['kappa'], d['gamma1'], d['gamma2']
    pixel = size * 60 / npix  # arcmin
    ks = gaussian_filter(k, 1. / pixel)  # 1 arcmin, for display
    # plot coordinates of the image: x to the east (left), as on the sky
    x = np.linspace(size / 2, -size / 2, npix)
    y = np.linspace(-size / 2, size / 2, npix)
    # shear averaged over cells of the image
    nw = 45
    cell = npix // nw
    s1, s2 = (g[:cell * nw, :cell * nw].reshape(nw, cell, nw, cell).mean(axis=(1, 3)) for g in (g1, g2))
    xw = x[:cell * nw].reshape(nw, cell).mean(axis=1)[None, :].repeat(nw, 0)
    yw = y[:cell * nw].reshape(nw, cell).mean(axis=1)[:, None].repeat(nw, 1)
    # orientation on the plot: the convention for which the shear is tangential around the main convergence peak
    iy, ix = np.unravel_index(np.argmax(gaussian_filter(k, 5. / pixel)), k.shape)
    xp, yp = x[ix], y[iy]
    psi = np.arctan2(yw - yp, xw - xp)
    weight = 1. / np.hypot(np.hypot(xw - xp, yw - yp), 0.1)
    signs = max(((a, b) for a in (1, -1) for b in (1, -1)), key=lambda c: np.sum(np.cos(2 * (0.5 * np.arctan2(c[1] * s2, c[0] * s1) - psi - np.pi / 2)) * weight))
    beta = 0.5 * np.arctan2(signs[1] * s2, signs[0] * s1)
    amp = np.hypot(s1, s2)
    length = 0.95 * size / nw * np.minimum(amp / np.percentile(amp, 95), 1.2)
    dx, dy = length * np.cos(beta), length * np.sin(beta)
    fig, ax = plt.subplots(figsize=(5.4, 5.9), layout='constrained')
    im = ax.imshow(ks, origin='lower', extent=(size / 2, -size / 2, -size / 2, size / 2), cmap='RdBu_r', vmin=-0.05, vmax=0.05)
    segments = np.stack([np.c_[xw.ravel() - dx.ravel() / 2, yw.ravel() - dy.ravel() / 2], np.c_[xw.ravel() + dx.ravel() / 2, yw.ravel() + dy.ravel() / 2]], axis=1)
    ax.add_collection(matplotlib.collections.LineCollection(segments, colors=INK, linewidths=0.9, capstyle='round'))
    ax.set(xlim=(size / 2, -size / 2), ylim=(-size / 2, size / 2), aspect='equal')
    clean(ax)
    scalebar(ax, 0.5, "30'", size, color=INK)
    colorbar(fig, im, ax, r'$\kappa$')
    save(fig, 'lensing')


def baryons():
    """Baryonification around the most massive halo of a shell of HACC091_1024 (nside 2048): the shell, the change of
    its density, and the displacement of its pixels by this halo (arrow lengths on a log scale)."""
    ishell, ng = 8, 21

    def compute():
        import pollux.baryons as B
        import pyccl
        from pollux.pipeline import tasks as T
        g = run_globals('run_validation.py', 'run_validation_HACC091_1024.yaml')
        dmb = T._make_shell_builder(g['reader_dict'], g['shell_dicts'][0], g['baryon_dict'])
        dmb._setup_interpolator(ishell)
        halos = dmb.halo_builder.read_catalogs(ishell, variables=['ra', 'dec', 'z', dmb.mass_name])
        M = halos[dmb.mass_name] / dmb.cosmo['h']
        j = int(np.argmax(M))
        ra0, dec0, z0, M0 = float(halos['ra'][j]), float(halos['dec'][j]), float(halos['z'][j]), float(M[j])
        # displacement of the pixels by this halo alone (the computation of BaryonifiedShellBuilder._baryonify)
        a = np.array([1 / (1 + z0)])
        mass_def = pyccl.halos.massdef.MassDef(*dmb.mass_def)
        R = (np.array([M0]) / (4. / 3. * np.pi * mass_def.get_Delta(dmb.cosmo_linear, a) * dmb.cosmo_linear.rho_x(a, mass_def.rho_type)))**(1. / 3.)
        D = pyccl.angular_diameter_distance(dmb.cosmo_linear, a)
        B._set_state(dict(halos=dict(vec=np.atleast_2d(hp.ang2vec(ra0, dec0, lonlat=True)), M=np.array([M0]), a=a, R=R, D=D, ra=np.array([ra0]),
                                     dec=np.array([dec0])), nside=dmb.nside, model=dmb.model, epsilon_max=dmb.epsilon_max))
        pix, offsets = B._displace_pixels(np.array([0]))
        B._set_state(None)
        radius = np.degrees(R[0] * dmb.epsilon_max / D[0])
        size = 2.3 * radius  # the extent of the displacements, and a margin
        # the shell without and with baryons on the patch
        ra, dec, _ = patch_grid(ra0, dec0, size, 500)
        v = [mm.get_values_pos(ra, dec, lonlat=True) for mm in (dmb.shell_builder_ref.read_maps(ishell), dmb.read_maps(ishell))]
        mean = np.mean(v[0][v[0] > hp.UNSEEN])
        # displacements on a regular grid of the patch: of the pixel at each grid point
        rg, dg, _ = patch_grid(ra0, dec0, size * (1 - 1 / ng), ng)
        ipix = hp.ang2pix(dmb.nside, rg.ravel(), dg.ravel(), lonlat=True)
        k = np.searchsorted(pix, ipix)
        found = (k < pix.size) & (pix[np.minimum(k, pix.size - 1)] == ipix)
        vec = np.stack(hp.pix2vec(dmb.nside, ipix), axis=1)
        off = np.where(found[:, None], offsets[np.minimum(k, pix.size - 1)], 0.)
        lon, lat = hp.vec2ang(vec, lonlat=True)
        lon2, lat2 = hp.vec2ang(vec + off, lonlat=True)
        x, y = to_plane(lon, lat, ra0, dec0)
        x2, y2 = to_plane(lon2, lat2, ra0, dec0)
        return dict(dmo=v[0] / mean, dmb=v[1] / mean, x=x, y=y, x2=x2, y2=y2, radius=radius, size=size)

    d = cached('baryons', compute)
    x, y, radius, size = d['x'], d['y'], float(d['radius']), float(d['size'])
    extent = (size / 2, -size / 2, -size / 2, size / 2)
    u, v = d['x2'] - x, d['y2'] - y
    disp = np.hypot(u, v) * 60  # arcmin
    # arrow lengths on a log scale: from 1/1000 of the largest displacement to the largest, 0.9 of the spacing
    dmax = max(disp.max(), 1e-12)
    length = 0.9 * size / ng * np.clip(np.log10(np.maximum(disp, 1e-30) / dmax) + 3., 0., 3.) / 3.
    with np.errstate(invalid='ignore', divide='ignore'):
        ux, uy = np.where(disp > 0, u / np.hypot(u, v), 0.), np.where(disp > 0, v / np.hypot(u, v), 0.)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), layout='constrained')
    im0 = axes[0].imshow(np.maximum(d['dmo'], 0.5), origin='lower', extent=extent, cmap='magma', norm=matplotlib.colors.LogNorm(0.5, 40.),
                         interpolation='nearest')
    diff = d['dmb'] - d['dmo']
    lim = np.percentile(np.abs(diff), 99.5)
    im1 = axes[1].imshow(diff, origin='lower', extent=extent, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
    show = disp > dmax * 1e-3
    q = axes[2].quiver(x[show], y[show], (ux * length)[show], (uy * length)[show], disp[show], cmap='viridis',
                       norm=matplotlib.colors.LogNorm(dmax * 1e-3, dmax), angles='xy', scale_units='xy', scale=1., width=0.006)
    for ax, label, mappable, clabel in zip(axes, ('dark matter', 'with baryons − dark matter', 'displacements'), (im0, im1, q),
                                           (r'$1 + \delta$', r'$\Delta\delta$', 'arcmin')):
        ax.add_patch(matplotlib.patches.Circle((0, 0), radius, fill=False, ls='--', color=MUTED, lw=1.))
        ax.set(xlim=(size / 2, -size / 2), ylim=(-size / 2, size / 2), aspect='equal', title=label)
        clean(ax)
        colorbar(fig, mappable, ax, clabel)
    bar = min((0.5, 1., 2., 5.), key=lambda b: abs(b - size / 6))
    scalebar(axes[0], bar, f'{bar:g}°', size)
    save(fig, 'baryons')


def nz():
    """Redshift distributions of the tomographic bins of the SRD (v1) and v2 (Y1 lenses), over the shells of HACC150_3."""

    def compute():
        from astropy.io import fits
        shells = fits.open(f'{FULLSKY}/shells/shells.fits')[1].data
        return dict(edges=np.append(shells['zmin'], shells['zmax'][-1]), sources=np.loadtxt(f'{SHARED}/data/SRDv1_nz.txt'),
                    lenses=np.loadtxt(f'{SHARED}/data/SRDv2_Y1_lens_nz.txt'))

    d = cached('nz', compute)
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.), sharex=True, layout='constrained')
    for ax, key, label, cmap in zip(axes, ('sources', 'lenses'), ('sources', 'lenses'), ('viridis', 'magma')):
        n = d[key]
        nb = n.shape[1] - 1
        for i in range(nb):
            color = plt.get_cmap(cmap)(0.15 + 0.7 * i / max(nb - 1, 1))
            ax.fill_between(n[:, 0], n[:, i + 1], color=color, alpha=0.3, lw=0)
            ax.plot(n[:, 0], n[:, i + 1], color=color, lw=1.3)
        for e in d['edges']:
            ax.axvline(e, color='0.86', lw=0.5, zorder=0)
        ax.set(ylim=(0, None), yticks=[])
        ax.text(0.99, 0.85, label, transform=ax.transAxes, ha='right', va='top', color=INK)
        for side in ('left', 'right', 'top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='x', length=2)
    axes[-1].set(xlabel='z', xlim=(0, 3.))
    save(fig, 'nz')


def catalog_samples():
    """Galaxies of the full-sky catalogs of HACC150_3 near the equator (for :func:`cone`) and around the patch of
    :func:`tracers`: position, comoving distance and bin, from one pass over the catalogs."""

    def compute():
        import fitsio
        out = {k: [] for k in ('cone_ra', 'cone_chi', 'cone_bin', 'patch_ra', 'patch_dec', 'patch_chi')}
        for ibin in range(5):
            with fitsio.FITS(f'{FULLSKY}/catalogs/SRDv1_nz/{CATALOG_TAG}/cat_bin{ibin}.fits') as f:
                n, step = f[1].get_nrows(), 20_000_000
                for start in range(0, n, step):
                    g = f[1][['ra', 'dec', 'chi']][start:min(start + step, n)]
                    near = np.abs(g['dec']) < CONE_DEC
                    out['cone_ra'].append(g['ra'][near].astype(np.float32))
                    out['cone_chi'].append(g['chi'][near].astype(np.float32))
                    out['cone_bin'].append(np.full(near.sum(), ibin, dtype=np.int8))
                    near = (np.abs(g['dec'] - TRACER_PATCH[1]) < TRACER_PATCH[2]) & (np.abs(g['ra'] - TRACER_PATCH[0]) < 2 * TRACER_PATCH[2])
                    for k in ('ra', 'dec', 'chi'):
                        out[f'patch_{k}'].append(g[k][near].astype(np.float64))
            print(f'bin {ibin}', flush=True)
        return {k: np.concatenate(v) for k, v in out.items()}

    return cached('catalog_samples', compute)


CONE_DEC = 0.15  # half thickness of the slice of the cone, degrees
TRACER_PATCH = (55., 20., 3.)  # center and size of the patch of tracers (that of zoom), degrees


def cone():
    """The galaxies of the five source bins of HACC150_3 in a slice through the equator, by comoving distance and right
    ascension, with the edges of the shells."""
    from astropy.io import fits
    d = catalog_samples()
    shells = fits.open(f'{FULLSKY}/shells/shells.fits')[1].data
    zmax, ramax = 2.5, 60.
    chimax = np.interp(zmax, shells['zmax'], shells['chimax'])
    sel = (d['cone_ra'] < ramax) & (d['cone_chi'] < chimax)
    rng = np.random.default_rng(0)
    order = rng.permutation(sel.sum())  # no bin on top of the others
    ra, chi, b = (d[k][sel][order] for k in ('cone_ra', 'cone_chi', 'cone_bin'))
    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(projection='polar')
    ax.set(thetamin=0, thetamax=ramax, rlim=(0, chimax))
    ax.set_theta_zero_location('E')
    colors = plt.get_cmap('viridis')(0.15 + 0.7 * np.arange(5) / 4)
    ax.scatter(np.radians(ra), chi, c=colors[b], s=0.25, lw=0, alpha=0.5, rasterized=True)
    theta = np.radians(np.linspace(0, ramax, 200))
    for e in shells['chimax'][shells['chimax'] < chimax]:
        ax.plot(theta, np.full_like(theta, e), color='0.86', lw=0.4, zorder=0)
    ax.grid(False)
    ax.set_xticks([])
    zt = np.array([0.5, 1., 1.5, 2.])
    ax.set_yticks(np.interp(zt, shells['zmax'], shells['chimax']), [f'z = {z:g}' for z in zt], color=MUTED, fontsize=9)
    ax.tick_params(axis='y', pad=2)
    ax.spines['polar'].set_visible(False)
    ax.spines['start'].set_color(LINE)
    ax.spines['end'].set_visible(False)
    ax.spines['inner'].set_visible(False)
    for i, c in enumerate(colors):
        ax.plot([], [], 'o', color=c, ms=5, label=f'bin {i + 1}')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.98), handletextpad=0.2, fontsize=9, labelcolor=INK)
    save(fig, 'cone')


def tracers():
    """A shell of HACC150_3 on a patch of 3 degrees (the patch of zoom), and the galaxies of the catalogs in it."""
    from astropy.io import fits
    from matplotlib.colors import LogNorm
    ishell = 18
    ra0, dec0, size = TRACER_PATCH
    shell = fits.open(f'{FULLSKY}/shells/shells.fits')[1].data[ishell]

    def compute():
        m = hsp.HealSparseMap.read(f'{FULLSKY}/shells/shell_{ishell:03d}.fits')
        mean = np.mean(m[m.valid_pixels])
        ra, dec, _ = patch_grid(ra0, dec0, size, 900)
        v = m.get_values_pos(ra, dec, lonlat=True)
        return dict(density=np.where(v > hp.UNSEEN, v, 0.) / mean)

    density = cached('tracers', compute)['density']
    d = catalog_samples()
    inshell = (d['patch_chi'] >= shell['chimin']) & (d['patch_chi'] < shell['chimax'])
    x, y = to_plane(d['patch_ra'][inshell], d['patch_dec'][inshell], ra0, dec0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.9), layout='constrained')
    extent = (size / 2, -size / 2, -size / 2, size / 2)
    im = axes[0].imshow(np.clip(density, 0.3, None), origin='lower', extent=extent, cmap='magma', norm=LogNorm(0.3, 10.), interpolation='nearest')
    axes[1].scatter(x, y, s=1.2, lw=0, color=INK)
    for ax, label in zip(axes, ('matter', 'galaxies')):
        ax.set(xlim=(size / 2, -size / 2), ylim=(-size / 2, size / 2), aspect='equal', title=label)
        clean(ax)
    axes[1].spines[:].set_visible(True)
    axes[1].spines[:].set_color(LINE)
    scalebar(axes[0], 0.5, "30'", size)
    scalebar(axes[1], 0.5, "30'", size, color=INK)
    colorbar(fig, im, axes[0], r'$1 + \delta$')
    save(fig, 'tracers')


# endregion

FIGURES = dict(workflow=workflow, workflow_combined=workflow_combined, lightcone=lightcone, zoom=zoom, lensing=lensing, baryons=baryons, nz=nz, cone=cone, tracers=tracers)

if __name__ == '__main__':
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
