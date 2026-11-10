"""Index of the products under a root directory, from their ``config.yaml`` (see :mod:`pollux.layout`).

Usage::

    python -m pollux.index ROOT [--output products.yaml]

lists every product (shells, halos, convergence maps, catalogs) under ``ROOT``, with its kind,
parameters, inputs and code version, skipping intermediate files. The index is the starting
point to share the products, e.g. to register them in the DESC data registry.
"""
from __future__ import annotations

import argparse
import os

import yaml

from .layout import INTERMEDIATE


def product_index(root_dir: str) -> list[dict]:
    """Return the products under a root directory.

    Parameters
    ----------
    root_dir : str
        Root directory to search.

    Returns
    -------
    list of dict
        Per product, sorted by path: ``path`` (relative to ``root_dir``), and the ``product``,
        ``parameters``, ``inputs`` (relative to the realization) and ``code_version`` of its
        ``config.yaml``.
    """
    products = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = sorted(d for d in dirnames if d != INTERMEDIATE)
        if 'config.yaml' in filenames:
            with open(os.path.join(dirpath, 'config.yaml')) as f:
                products.append(dict(path=os.path.relpath(dirpath, root_dir), **yaml.safe_load(f)))
    return sorted(products, key=lambda p: p['path'])


def main() -> None:
    parser = argparse.ArgumentParser(description="Index of the pollux products under a root directory.")
    parser.add_argument('root_dir', help="root directory to search")
    parser.add_argument('--output', help="YAML file to write the index to (otherwise, a summary is printed)")
    args = parser.parse_args()

    products = product_index(args.root_dir)
    if args.output:
        with open(args.output, 'w') as f:
            yaml.safe_dump(products, f, sort_keys=False)
        print(f"{len(products)} products written to {args.output}")
        return
    for p in products:
        commit = (p.get('code_version') or {}).get('commit') or ''
        print(f"{p['product']:10s} {commit[:8]:8s} {p['path']}")


if __name__ == '__main__':
    main()
