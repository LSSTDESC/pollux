"""Layout of the output products: directories, intermediate files and product configurations.

All products of one realization (simulation, nside, seed and footprint padding) live under
``{root_dir}/nside{nside}/seed{seed}[/pad{padding}]``::

    halos/                          halo catalogs, shared by all baryon models
    dmo/                            dark-matter-only products
        shells/                     shell_{iii}.fits, cls.fits, info.npy
        smoothed/{s}Mpc/            density_{iii}, t11_{iii}, t22_{iii}, t12_{iii}
        kappa/snapshots/            kappa_{iii}, gamma1_{iii}, gamma2_{iii} (one source plane per shell)
        kappa/{nz}/                 kappa_bin{b}, gamma1_bin{b}, gamma2_bin{b}, nzs.npy
        catalogs/{nz}/{tag}/        cat_bin{b}.fits, cat_2pt.fits, cat_cls.fits
    dmb/{tag}/                      baryonified products, with the same subtree as dmo/
    intermediate/                   files needed only to make or resume the products (sub-volume maps,
                                    a_lm, partial catalogs, workspaces), at the same relative paths
    code_version/{date}/            code and environment of each run

Each product directory (shells, halos, kappa, catalogs) holds a ``config.yaml`` with its parameters,
inputs and code version, so that it can be shared (e.g. registered) on its own. Footprint products
(NaMaster workspaces, padded masks) are shared by all realizations at ``{root_dir}/nside{nside}``.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import yaml

from .utils import code_version

INTERMEDIATE = 'intermediate'


def realization_dir(root_dir: str, nside: int, seed: int, padding_deg: float | None = None) -> str:
    """Return the directory of the products of one realization.

    Parameters
    ----------
    root_dir : str
        Root directory of the simulation's products.
    nside : int
        Healpix resolution of the maps.
    seed : int
        Seed of the random isometries.
    padding_deg : float or None, optional
        Padding of the footprint (masked footprints only).

    Returns
    -------
    str
        ``{root_dir}/nside{nside}/seed{seed}[/pad{padding_deg}]``.
    """
    path = os.path.join(root_dir, f'nside{nside}', f'seed{seed}')
    return os.path.join(path, f'pad{padding_deg:g}') if padding_deg else path


def matter_dir(realization: str, baryon_tag: str | None = None) -> str:
    """Return the directory of the dark-matter-only (``dmo``) or baryonified (``dmb/{baryon_tag}``) products.

    Parameters
    ----------
    realization : str
        Directory of the realization (:func:`realization_dir`).
    baryon_tag : str or None, optional
        Tag of the baryon model, None for dark matter only.

    Returns
    -------
    str
    """
    return os.path.join(realization, 'dmo') if baryon_tag is None else os.path.join(realization, 'dmb', baryon_tag)


def intermediate_dir(product_dir: str, realization: str | None) -> str:
    """Return the directory of the intermediate files of a product, at its path relative to the realization.

    Parameters
    ----------
    product_dir : str
        Directory of the product.
    realization : str or None
        Directory of the realization. If None (product outside the layout), the intermediate
        files go to ``{product_dir}/intermediate``.

    Returns
    -------
    str
    """
    if realization is None:
        return os.path.join(product_dir, INTERMEDIATE)
    return os.path.join(realization, INTERMEDIATE, os.path.relpath(product_dir, realization))


def _plain(value: Any) -> Any:
    # parameters as plain YAML types (numpy scalars and arrays, tuples, dtypes)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.dtype, type)):
        return np.dtype(value).name
    return value


def write_config(product_dir: str, product: str, parameters: dict, inputs: list[str] | None = None, realization: str | None = None) -> None:
    """Write the configuration of a product, or check it against the existing one.

    The configuration (``{product_dir}/config.yaml``) records the kind of product, its parameters,
    its input products and the code version that first wrote it. Products are made or resumed in
    place, so an existing configuration with other parameters means that the directory holds a
    different product: this raises instead of mixing them.

    Parameters
    ----------
    product_dir : str
        Directory of the product.
    product : str
        Kind of product (e.g. ``'shells'``, ``'kappa'``, ``'catalogs'``).
    parameters : dict
        Parameters defining the product.
    inputs : list of str or None, optional
        Directories of the input products.
    realization : str or None, optional
        Directory of the realization, relative to which inputs are recorded.

    Raises
    ------
    ValueError
        If the directory already holds a product with other parameters.
    """
    config = dict(product=product, parameters=_plain(parameters))
    fname = os.path.join(product_dir, 'config.yaml')

    # existing product: same kind and parameters
    if os.path.exists(fname):
        with open(fname) as f:
            existing = yaml.safe_load(f)
        if {k: existing.get(k) for k in config} != config:
            raise ValueError(f"{product_dir} holds another product (see its config.yaml): use another tag or directory.")
        return

    # new product, written atomically (several workers may create it at once)
    config['inputs'] = [os.path.relpath(p, realization) if realization else p for p in (inputs or [])]
    config['code_version'] = dict(code_version())
    os.makedirs(product_dir, exist_ok=True)
    tmp = f'{fname}.{os.getpid()}'
    with open(tmp, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    os.replace(tmp, fname)
