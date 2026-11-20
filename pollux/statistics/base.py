"""Base of the two-point statistics of the maps and catalogs (:class:`TwoPointStatistics`): their SACC files and bandpower windows."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import sacc

from ..io.layout import PolluxIO

logger = logging.getLogger(__name__)


class TwoPointStatistics(ABC):
    """Base of :class:`~pollux.statistics.maps.MapStatistics` and :class:`~pollux.statistics.catalogs.CatalogStatistics`.

    Statistics are saved as SACC files of ``{data_type: {(i, j): values}}``, the legs ``i`` and ``j``
    of each data type having a role (e.g. ``'density'`` or ``'shear'``) whose SACC tracers are given by
    :meth:`_sacc_tracers`. Bandpower windows are saved apart from their NaMaster workspace.
    """

    def _save_sacc(
            self,
            fname: str,
            values: dict[str, dict[tuple[int, int], np.ndarray]],
            coords: np.ndarray | dict[str, dict[tuple[int, int], np.ndarray]],
            legs: dict[str, tuple[str, str]],
            *,
            coord: str = 'ell',
            errors: dict[str, dict[tuple[int, int], np.ndarray]] | None = None,
            metadata: dict | None = None) -> str:
        """Save statistics to a SACC file, atomically.

        Parameters
        ----------
        fname : str
            Output file.
        values : dict
            ``{data_type: {(i, j): values}}``.
        coords : ndarray or dict
            Multipoles or angles: one array for all, or keyed like ``values``.
        legs : dict
            Roles of the legs ``(i, j)`` of each data type (keys of :meth:`_sacc_tracers`).
        coord : {'ell', 'theta'}, optional
            Tag of the coordinates.
        errors : dict or None, optional
            Standard deviations keyed like ``values``, saved as a diagonal covariance.
        metadata : dict or None, optional
            Metadata of the SACC file (e.g. ``options_id`` of the pipeline).

        Returns
        -------
        str
            ``fname``.
        """
        s = sacc.Sacc()
        s.metadata.update(metadata or {})
        tracers = self._sacc_tracers()
        for role_tracers in tracers.values():
            for t in role_tracers:
                if t.name not in s.tracers:
                    s.add_tracer_object(t)
        for data_type, pairs in values.items():
            role_i, role_j = legs[data_type]
            for (i, j), v in pairs.items():
                x = coords[data_type][i, j] if isinstance(coords, dict) else coords
                for x_k, v_k in zip(x, v, strict=True):
                    s.add_data_point(data_type, [tracers[role_i][i].name, tracers[role_j][j].name], v_k, **{coord: x_k})
        if errors is not None:
            # same order as the data points added above
            s.add_covariance(np.diag(np.concatenate([errors[data_type][ij]**2 for data_type, pairs in values.items() for ij in pairs])))
        PolluxIO.write_sacc(fname, s)
        return fname

    @abstractmethod
    def _sacc_tracers(self) -> dict[str, list]:
        """Return the SACC tracers of each leg role, indexed by map or bin.

        Returns
        -------
        dict
            ``{role: [sacc tracer, ...]}``, all added to the SACC files, in this order.
        """

    @staticmethod
    def _load_sacc(fname: str, return_dict: bool = False, return_errors: bool = False) -> sacc.Sacc | tuple:
        """Load statistics saved by :meth:`_save_sacc`.

        Parameters
        ----------
        fname : str
            SACC file.
        return_dict : bool, optional
            If True, return dictionaries, else the SACC object.
        return_errors : bool, optional
            With ``return_dict``, also return the standard deviations of the covariance (empty
            dictionaries without covariance).

        Returns
        -------
        sacc.Sacc or tuple
            The SACC object, or ``(coords, values[, errors])``, each ``{data_type: {(i, j): array}}``,
            with ``(i, j)`` the indices of the tracer names.
        """
        s = PolluxIO.read_sacc(fname)
        logger.debug("Found and loaded file at %s.", fname)
        if not return_dict:
            return s
        sigma = np.sqrt(np.diag(s.covariance.covmat)) if return_errors and s.covariance is not None else None
        coords, values, errors = {}, {}, {}
        for data_type in s.get_data_types():
            coords[data_type], values[data_type], errors[data_type] = {}, {}, {}
            for names in s.get_tracer_combinations(data_type):
                key = tuple(_tracer_index(n) for n in names)
                indices = s.indices(data_type, names)
                tag = 'ell' if 'ell' in s.data[indices[0]].tags else 'theta'
                coords[data_type][key] = np.array([s.data[k].get_tag(tag) for k in indices])
                values[data_type][key] = s.mean[indices]
                if sigma is not None:
                    errors[data_type][key] = sigma[indices]
        return (coords, values, errors) if return_errors else (coords, values)

    @staticmethod
    def _saved_bpws(wsp_fname: str, wsp: Callable[[], Any]) -> np.ndarray:
        """Return the bandpower windows of a workspace, saved apart from it.

        Predictions then don't load the coupling matrix (several GB at high resolution), only once
        to save them.

        Parameters
        ----------
        wsp_fname : str
            File of the workspace; the windows are at ``{wsp_fname without .wsp}_bpws.npy``.
        wsp : callable
            Returns the workspace, called only if the windows are not saved.

        Returns
        -------
        ndarray
            Shape ``(n_cl, n_bands, n_cl, lmax + 1)``.
        """
        fname = wsp_fname.replace('.wsp', '_bpws.npy')
        if not os.path.exists(fname):
            PolluxIO.write_npy(fname, wsp().get_bandpower_windows())
        return PolluxIO.read_npy(fname)


def _tracer_index(name: str) -> int:
    # bin or map index of a tracer name (e.g. 'kappa_3', 'density_012')
    return int(name.rsplit('_', 1)[-1])
