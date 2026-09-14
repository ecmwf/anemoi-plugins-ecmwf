# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

import numpy as np

from . import backend_registry
from .base import CalculationBackend

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ctrans4py backend
# ---------------------------------------------------------------------------


@backend_registry.register("ctrans4py")
class ctrans4py(CalculationBackend):
    """Backend using ctrans4py (C spectral-transform library via transi ABI).

    ``vordiv_to_uv`` uses the spectral-only path (no Legendre/FFT).
    ``uv_to_vordiv`` uses a full direct transform.

    The Transform is initialised with nlev=1 (cheapest Legendre setup) and
    reused for any number of fields — ctrans4py supports this.
    """

    # Process-wide cache: (trunc, grid_str) -> Transform instance
    _transform_cache: dict[tuple[int, str], object] = {}

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            import ctrans4py  # type: ignore[reportMissingImports]  # noqa: F401

            return True, "ctrans4py is available"
        except Exception as e:
            return False, f"ctrans4py is not available: {e}"

    def __init__(self, kloen: np.ndarray, trunc: int, grid: str | None = None):
        super().__init__(kloen, trunc, grid)
        import os

        import ctrans4py  # type: ignore[reportMissingImports]

        self._lib = ctrans4py
        cores = len(os.sched_getaffinity(0))
        self._lib.init_env(omp_num_threads=cores)
        self._grid = self._lib.detect_grid(np.asarray(self.kloen, dtype=np.int32))

    def _get_transform(self):
        """Get or create a cached Transform (nlev=1, cheapest init)."""
        key = (self.trunc, self._grid)
        if key not in ctrans4py._transform_cache:
            LOG.debug("ctrans4py: creating Transform cache for T%d %s", self.trunc, self._grid)
            ctrans4py._transform_cache[key] = self._lib.Transform(self.trunc, grid=self._grid, nlev=1)
        return ctrans4py._transform_cache[key]

    def vordiv_to_uv(self, vorticity: np.ndarray, divergence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        squeezed = vorticity.ndim == 1
        vor = np.atleast_2d(np.ascontiguousarray(vorticity, dtype=np.float64))
        div = np.atleast_2d(np.ascontiguousarray(divergence, dtype=np.float64))

        tr = self._get_transform()
        u, v = tr.vordiv_to_uv(vor, div)

        if squeezed:
            return u.squeeze(0), v.squeeze(0)
        return u, v

    def uv_to_vordiv(
        self, u_component_of_wind: np.ndarray, v_component_of_wind: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        squeezed = u_component_of_wind.ndim == 1
        u = np.atleast_2d(np.ascontiguousarray(u_component_of_wind, dtype=np.float64))
        v = np.atleast_2d(np.ascontiguousarray(v_component_of_wind, dtype=np.float64))

        tr = self._get_transform()
        result = tr.uv_to_vordiv(u, v)

        if squeezed:
            return result[0].squeeze(0), result[1].squeeze(0)
        return result

    def sh_to_gg(self, scalars: np.ndarray) -> np.ndarray:
        squeezed = scalars.ndim == 1
        q = np.atleast_2d(np.ascontiguousarray(scalars, dtype=np.float64))

        tr = self._get_transform()
        out = tr.sh_to_gg(q)

        if squeezed:
            return out.squeeze(0)
        return out
