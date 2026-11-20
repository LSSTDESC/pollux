# Contributing to pollux

Conventions for the code of pollux: formatting, documentation, naming, organisation, plots and pipeline tasks. New code should read like the code around it.

## Formatting and checks
- Format with [yapf](https://github.com/google/yapf), using the repository's `.style.yapf` (Google style, 160 columns): `yapf -i -r pollux`.
- Keep the code clean under `python -m pyflakes`.
- Python 3.10 or later, with `from __future__ import annotations` and type hints on public functions and methods.

## Docstrings
- Every module (and package `__init__`) starts with a docstring: a one-line summary of what it provides, followed if useful by a short paragraph naming its main classes or functions, or its command line.
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
- Use the same names for the same things across modules: `imap`/`ishell`/`ikappa`/`ibin` (maps, shells, convergence maps, catalog bins) for indices, `smooth_Mpc` for comoving smoothing scales, `seed`, `nside`, `ell_factor`, `hspmap` for HealSparse maps.
- Class-level constants in upper case (`FIELDS`, `SMOOTHING_SCALES`).
- Optional flags of public methods are keyword-only (after `*`).

## Organisation
- Order the members of a class as:
  1. class attributes and constants;
  2. `__init__`, then the other special methods (`__getstate__`, ...);
  3. `config()`, for the objects that define products: what goes into their `config.yaml` (`product`, `parameters`,
     `inputs` for builders and catalogs; their entry in the catalog's parameters for augmentors), extended by
     subclasses through `super().config()`;
  4. properties (`property`, `cached_property`);
  5. the other methods along the workflow (making products; I/O; footprint and masks; transforms; tracers and theory;
     statistics, in the order compute, save, load, predict, then plot), each public method followed by the private
     helpers it is the first to use.
- In a module, public classes and functions come first, and private helpers (module-level functions, numba kernels)
  at the end.
- A helper used by one module stays in it, private (`_name`). A helper independent of pollux's objects and useful
  beyond one module goes to `pollux.utils`, by theme (numerics, sky, binning, FITS tables), public.
- Statistics are mixins of `pollux.statistics` (`MapStatistics`, `CatalogStatistics`, on a common
  `TwoPointStatistics` for SACC files and bandpower windows), inherited by the builders and catalogs.
- Directories and file names come from `pollux.io.layout`, and product configurations (`config.yaml`) from
  `pollux.io.config`, for the builders, catalogs and `PolluxLoader` alike: never build a product path elsewhere.
- Objects declare what other objects need from them rather than being searched for: e.g. augmentors have a `kind`
  (`'shear'`, `'ia'`), from which the catalog sets `shear_augmentor` and `ia_augmentor` once.
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
