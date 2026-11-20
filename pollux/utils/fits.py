"""FITS tables: reading chosen rows of large tables (:func:`read_fits_rows`)."""
from __future__ import annotations

import fitsio
import numpy as np


def read_fits_rows(fname: str, rows: np.ndarray, columns: list[str] | None = None, block: int = 5_000_000) -> np.ndarray:
    """Read given rows of the table of a FITS file, by contiguous blocks of rows.

    Reading scattered rows directly makes one small read per row; reading blocks and keeping
    the rows of each is several times faster for subsamples of large catalogs (e.g. 1 min
    instead of 5 for 0.5% of 2e8 rows on Lustre), and gives the same rows.

    Parameters
    ----------
    fname : str
        FITS file, with the table in HDU 1.
    rows : ndarray of int
        Sorted row indices.
    columns : list of str or None, optional
        Columns to read. If None, all columns.
    block : int, optional
        Number of rows per block.

    Returns
    -------
    numpy.ndarray
        Structured array of the rows.
    """
    with fitsio.FITS(fname) as f:
        hdu = f[1] if columns is None else f[1][columns]
        parts = []
        for start in range(0, int(rows[-1]) + 1 if len(rows) else 0, block):
            i0, i1 = np.searchsorted(rows, [start, start + block])
            if i1 > i0:
                parts.append(hdu[start:min(start + block, int(rows[-1]) + 1)][rows[i0:i1] - start])
        return np.concatenate(parts) if parts else f[1].read(columns=columns, rows=rows)
