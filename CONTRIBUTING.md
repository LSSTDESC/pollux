# Contributing to pollux

Conventions for the code of pollux: formatting, documentation, naming, organisation, plots and pipeline tasks. New code should read like the code around it.

## Formatting and checks
- Format with [yapf](https://github.com/google/yapf), using the repository's `.style.yapf` (Google style, 160 columns): `yapf -i pollux/*.py pollux/pipeline/*.py`.
- Keep the code clean under `python -m pyflakes`.
- Python 3.10 or later, with `from __future__ import annotations` and type hints on public functions and methods.

## Docstrings
- [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html), complete and kept up to date: `Parameters`, `Returns` (or `Yields`), and `Notes` where needed (e.g. file names or layouts).
- A one-line summary first, in the imperative for functions ("Return ...", "Compute ...").
- Constructor parameters go in the class docstring, not in `__init__`.
- Give units in names or descriptions (`smooth_Mpc`, angles in arcmin or radians).
- Cross-reference with `:meth:`, `:class:` and `:func:`.

## Comments
- Short inline comments before each major step of a computation, with a blank line between steps, so that the steps read as an outline.
- Say what a step does, or what problem a line fixes, in a few words, without explaining at length (e.g. `# CCL's FFTLog fails on spectra smoothed to vanishing values at high ell, or zero everywhere`).
- No commented-out code.

## Naming
- Methods start with a verb when they act: `compute_`, `construct_`, `predict_`, `generate_`, `load_`, `plot_`. Drop `get_` when the noun is enough (`field_map`, not `get_field_map`).
- Prefix with `_` everything used only internally, including module-level helpers and numba kernels.
- Use the same names for the same things across modules: `imap`/`ishell`/`ikappa` for indices, `smooth_Mpc` for comoving smoothing scales, `seed`, `nside`, `ell_factor`, `hspmap` for HealSparse maps.
- Class-level constants in upper case (`FIELDS`, `SMOOTHING_SCALES`).
- Optional flags of public methods are keyword-only (after `*`).

## Organisation
- Order the methods of a class along its workflow:
  1. construction and iteration;
  2. making products;
  3. I/O (`get_fname`, save, load);
  4. footprint and masks;
  5. transforms (alms, smoothing);
  6. tracers and theory;
  7. statistics, in the order compute, save, load, predict, then plot.
- Put helpers next to the methods that use them, and group related methods (e.g. the smoothing methods).
- Define a rule used in several places once, and use it everywhere. For example, `Tracer.effective_smooth_Mpc` gives the smoothing actually applied, both to the maps and to their predictions.
- Avoid small single-use helpers that clutter a module, and keep public APIs small.
- Use Python features when they make the code clearer, shorter or safer (`functools.cached_property`, `functools.cache`), not for their own sake. Dataclasses don't suit the builders, whose attributes are set and changed after construction.

## Plots
Statistics are plotted with `plotting.plot_pairs`, one panel per bin pair:
- Each figure is saved in `{save_dir}/plots` (`{catalog_dir}/plots` for catalogs), one figure per mode (values, ratio, ...).
- x limits come from the binning (Cls) or the measured range (2pt).
- Reference scales are vertical lines when within the x range (box size, pixel size, smoothing), with their legend below the figure.
- Ratio panels don't share their y axis, and values beyond ±1 are drawn at ±1 as arrows.
- Panels have a 4:3 aspect ratio.
- Full-sky maps are drawn at a lower resolution (`plot_maps(nside_plot=...)`).

## Pipeline tasks (`pollux.pipeline.tasks`)
- One product per task. Plots get their own tasks (`plot_shell_maps`, `plot_kappa_maps`, `plot_catalog_2pt`, `plot_catalog_cls`), separate from the tasks computing what they plot.
- Tasks skip outputs already on disk, so that a run can be resumed. A true rerun needs a new output directory or tag.
- End every memory-heavy task with `_free_memory()`: Parsl workers live across tasks, and would otherwise keep a task's peak memory for the next tasks.
- Run each task on the executor matching its memory (`pollux.pipeline.config`).
- Run scripts and their configurations live outside the repository, with the data. Each run records its code and environment (`code_version/`), and refuses to run with uncommitted changes unless allowed.
- Don't update a shared clone while runs use it: workers started later would import the new code.

## Commits
Commit related changes together, with a message summarising what changed and why. Keep branches pushed once others run from them.
