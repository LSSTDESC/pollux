from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

MODES = ('values', 'ratio', 'diff_over_error', 'ratio_to_reference')

_DEFAULT_YLABELS = {
    'values': {
        0: r'$C_\ell$',
        1: r'$\ell C_\ell$',
        2: r'$\ell(\ell+1)C_\ell/2\pi$'
    },
    'ratio': r'data / theory $- 1$',
    'diff_over_error': r'(data $-$ theory) / $\sigma$',
    'ratio_to_reference': r'ratio to reference $- 1$',
}


def ell_factor_of(x: np.ndarray, ell_factor: int) -> float | np.ndarray:
    """Return the factor multiplying spectra or correlation functions in plots.

    Parameters
    ----------
    x : ndarray
        Multipoles or angular separations.
    ell_factor : {0, 1, 2}
        0: 1, 1: ``x``, 2: ``x (x + 1) / 2 pi``.

    Returns
    -------
    float or ndarray
    """
    if ell_factor == 0:
        return 1.
    if ell_factor == 1:
        return x
    if ell_factor == 2:
        return x * (x + 1.) / (2. * np.pi)
    raise ValueError(f"ell_factor must be 0, 1 or 2, not {ell_factor!r}")


def plot_pairs(
        x: np.ndarray | dict,
        data: dict | None = None,
        theory: dict | None = None,
        data_errors: dict | None = None,
        theory_errors: dict | None = None,
        data_ref: dict | None = None,
        theory_ref: dict | None = None,
        mode: str = 'values',
        nbins: int | None = None,
        auto_only: bool = False,
        ell_factor: int = 0,
        scales: dict | None = None,
        color: str = 'dodgerblue',
        color_theory: str | None = None,
        logx: bool = True,
        logy: bool = False,
        ylim: tuple | None = None,
        xlabel: str = r'$\ell$',
        ylabel: str | None = None,
        title: str = 'bins {i} {j}',
        sharey: bool | str = 'row',
        figaxes: tuple | None = None,
        fname: str | None = None) -> tuple:
    """Plot two-point statistics of bin pairs, one panel per pair.

    Panels are laid out on a ``nbins x nbins`` grid, with pair ``(i, j)`` in row ``i``
    and column ``j`` (or, with ``auto_only``, auto-pairs in order on a near-square
    grid); panels without any pair are hidden. Data are plotted as points and theory as lines.

    Parameters
    ----------
    x : ndarray or dict
        Multipoles or angular separations, shared by all pairs or keyed by ``(i, j)``.
    data, theory : dict or None, optional
        Measured and predicted statistics keyed by ``(i, j)``.
    data_errors, theory_errors : dict or None, optional
        Error bars on the data (plotted as error bars) and on the theory (plotted as bands).
    data_ref, theory_ref : dict or None, optional
        Reference data and theory for ``mode='ratio_to_reference'``.
    mode : {'values', 'ratio', 'diff_over_error', 'ratio_to_reference'}, optional
        - ``'values'`` (default): data and theory, multiplied by the ``ell_factor``;
        - ``'ratio'``: ``data / theory - 1`` (with the relative theory errors as a band);
        - ``'diff_over_error'``: ``(data - theory) / sigma``, with ``sigma`` the theory
          errors if given, else the data errors;
        - ``'ratio_to_reference'``: ``data / data_ref - 1`` and ``theory / theory_ref - 1``.
    nbins : int or None, optional
        Number of bins. Default is one more than the largest bin index.
    auto_only : bool, optional
        If True, only plot auto-pairs ``(i, i)``.
    ell_factor : {0, 1, 2}, optional
        Factor of the values in ``'values'`` mode, see :func:`ell_factor_of`.
    scales : dict or None, optional
        Reference scales (e.g. :meth:`pollux.map_builder.MapBuilder.reference_scales`),
        per bin: ``ell_smooth``/``ell_box`` (or ``theta_smooth``/``theta_box`` if ``xlabel``
        does not contain ``ell``), marked at the geometric mean of the two bins with
        dotted and dashed lines.
    color, color_theory : str, optional
        Colors of the data and theory. ``color_theory`` defaults to ``color``.
    logx, logy : bool, optional
        Logarithmic axes. ``logy`` only applies to ``'values'`` mode.
    ylim : tuple or None, optional
        Limits of the y axis.
    xlabel, ylabel : str or None, optional
        Axis labels. ``ylabel`` defaults to a label of the mode (of ``C_ell`` and the
        ``ell_factor`` in ``'values'`` mode).
    title : str, optional
        Panel titles, formatted with ``i`` and ``j``.
    sharey : bool or str, optional
        Passed to :func:`matplotlib.pyplot.subplots`. Default is ``'row'``.
    figaxes : tuple or None, optional
        Existing ``(fig, axes)`` from a previous call to draw into (e.g. another statistic
        in another color).
    fname : str or None, optional
        If given, save the figure to this file and close it.

    Returns
    -------
    fig, axes
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")

    # pairs to plot, and the panel grid (or that of a previous call)
    color_theory = color_theory or color
    pairs = sorted(set(data or {}) | set(theory or {}))
    if auto_only:
        pairs = [(i, j) for i, j in pairs if i == j]
    nbins = nbins or max(max(p) for p in pairs) + 1
    ncols = int(np.ceil(np.sqrt(nbins))) if auto_only else nbins
    if figaxes is None:
        nrows = int(np.ceil(nbins / ncols)) if auto_only else nbins
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows), sharex=True, sharey=sharey, layout='tight', squeeze=False)
        xrange = [np.inf, -np.inf]
    else:
        fig, axes = figaxes
        xrange = list(axes[0, 0].get_xlim())
    harmonic = 'ell' in xlabel

    def get(d: dict | None, key: tuple) -> np.ndarray | None:
        # value of a pair, if any
        return d.get(key) if d is not None else None

    # one panel per pair: data as points, theory as lines, errors as bars or bands
    for i, j in pairs:
        ax = axes[i // ncols, i % ncols] if auto_only else axes[i, j]
        xx = x[i, j] if isinstance(x, dict) else x
        d, th, de, te = get(data, (i, j)), get(theory, (i, j)), get(data_errors, (i, j)), get(theory_errors, (i, j))
        # values, ratios or residuals of the data (and theory)
        if mode == 'values':
            f = ell_factor_of(xx, ell_factor)
            if th is not None:
                ax.plot(xx, f * th, color=color_theory, ls='--')
                if te is not None:
                    ax.fill_between(xx, f * (th - te), f * (th + te), color=color_theory, alpha=0.2, lw=0, zorder=-99)
            if d is not None:
                ax.errorbar(xx, f * d, None if de is None else f * de, ls=' ', marker='o', markersize=3, color=color)
            if logy:
                ax.set_yscale('log')
        elif mode == 'ratio':
            if d is None or th is None:
                continue
            ax.plot(xx, d / th - 1., ls=' ', marker='o', markersize=3, color=color)
            if te is not None:
                ax.fill_between(xx, -te / np.abs(th), te / np.abs(th), color=color_theory, alpha=0.2, lw=0, zorder=-99)
        elif mode == 'diff_over_error':
            sigma = te if te is not None else de
            if d is None or th is None or sigma is None:
                continue
            ax.plot(xx, (d - th) / sigma, ls=' ', marker='o', markersize=3, color=color)
            ax.axhspan(-1., 1., color=color_theory, alpha=0.2, lw=0, zorder=-99)
        else:
            dr, tr = get(data_ref, (i, j)), get(theory_ref, (i, j))
            if d is not None and dr is not None:
                ax.plot(xx, d / dr - 1., ls=' ', marker='o', markersize=3, color=color)
            if th is not None and tr is not None:
                ax.plot(xx, th / tr - 1., ls='--', color=color_theory)
        if mode != 'values':
            ax.axhline(0., color='k', ls='--', lw=1)
        # reference scales of the pair
        if scales is not None:
            prefix = 'ell' if harmonic else 'theta'
            for key, ls in (('smooth', ':'), ('box', '--')):
                values = scales.get(f'{prefix}_{key}')
                if values is not None:
                    ax.axvline(np.sqrt(values[i] * values[j]), color='0.2', ls=ls, lw=1)
        if logx:
            ax.set_xscale('log')
        ax.set_title(title.format(i=i, j=j))
        xrange = [min(xrange[0], np.min(xx) / 1.1), max(xrange[1], np.max(xx) * 1.1)]

    # x range of the data only (scale lines may lie outside); hide panels without pairs
    used = np.array([[ax.has_data() for ax in row] for row in axes])
    for ax, u in zip(axes.flat, used.flat):
        ax.axis('on' if u else 'off')
        ax.set_xlim(*xrange)
        if ylim is not None:
            ax.set_ylim(*ylim)
    # labels on the first used panel of each row and the last used panel of each column
    for k, row in enumerate(used):
        if row.any():
            default = _DEFAULT_YLABELS[mode][ell_factor] if mode == 'values' else _DEFAULT_YLABELS[mode]
            axes[k, np.argmax(row)].set_ylabel(ylabel or default)
    for k, col in enumerate(used.T):
        if col.any():
            ax = axes[len(col) - 1 - np.argmax(col[::-1]), k]
            ax.set_xlabel(xlabel)
            ax.xaxis.set_tick_params(labelbottom=True)
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.debug("Saved to file %s.", fname)
    return fig, axes
