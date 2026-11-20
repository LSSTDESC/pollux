"""Configurations of the products: the ``config.yaml`` of each product directory (:func:`write_config`, :func:`read_config`).

A configuration records the kind of product, its parameters, its input products and the code version that first
wrote it, so that a product can be shared (e.g. registered) on its own, and that a directory never mixes products of
different parameters.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import yaml

from .code_version import code_version
from .layout import atomic_write

CONFIG_FNAME = 'config.yaml'


def write_config(product_dir: str, config: dict, realization: str | None = None) -> None:
    """Write the configuration of a product, or check it against the existing one.

    The configuration (``{product_dir}/config.yaml``) records the kind of product, its parameters,
    its input products and the code version that first wrote it. Products are made or resumed in
    place, so an existing configuration with other parameters means that the directory holds a
    different product: this raises instead of mixing them.

    Parameters
    ----------
    product_dir : str
        Directory of the product.
    config : dict
        ``product`` (kind of product, e.g. ``'shells'``, ``'kappa'``, ``'catalogs'``), ``parameters``
        (defining the product) and ``inputs`` (directories of the input products), as given by the
        ``config()`` method of the builders and catalogs.
    realization : str or None, optional
        Directory of the realization, relative to which inputs are recorded.

    Raises
    ------
    ValueError
        If the directory already holds a product with other parameters.
    """
    inputs = config.get('inputs') or []
    config = dict(product=config['product'], parameters=_plain(config['parameters']))
    fname = os.path.join(product_dir, CONFIG_FNAME)

    # existing product: same kind and parameters
    if os.path.exists(fname):
        with open(fname) as f:
            existing = yaml.safe_load(f)
        if {k: existing.get(k) for k in config} != config:
            raise ValueError(f"{product_dir} holds another product (see its config.yaml): use another tag or directory.")
        return

    # new product, written atomically (several workers may create it at once)
    config['inputs'] = [os.path.relpath(p, realization) if realization else p for p in inputs]
    config['code_version'] = dict(code_version())
    os.makedirs(product_dir, exist_ok=True)
    with atomic_write(fname) as tmp, open(tmp, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)


def read_config(product_dir: str) -> dict:
    """Return the configuration of a product (its ``config.yaml``).

    Parameters
    ----------
    product_dir : str
        Directory of the product.

    Returns
    -------
    dict
        ``product``, ``parameters``, ``inputs`` and ``code_version``.
    """
    with open(os.path.join(product_dir, CONFIG_FNAME)) as f:
        return yaml.safe_load(f)


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
