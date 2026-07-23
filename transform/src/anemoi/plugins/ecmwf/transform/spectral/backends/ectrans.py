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

from .base import CalculationBackend

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ectrans4py backend
# ---------------------------------------------------------------------------


class _EctransRuntime:
    """Process-wide singleton managing ectrans4py library state.

    Handles the one-time global initialisation (environment, MPL, processor
    grid) and per-resolution setup.  Thread-safe by virtue of the GIL;
    multiple ``ectrans4py`` backend instances with different truncations
    share this runtime.
    """

    _instance: _EctransRuntime | None = None

    def __new__(cls) -> _EctransRuntime:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialised"):
            self._initialised = False
            self._resolutions: dict[tuple[int, tuple[int, ...]], dict] = {}

    @property
    def lib(self):
        import ectrans4py as _et  # type: ignore[reportMissingImports]

        return _et

    def _ensure_init(self) -> None:
        """One-time: env vars, MPL init, processor grid."""
        if self._initialised:
            return

        import os
        import platform

        system = platform.system()
        if system not in ("Linux", "Darwin"):
            raise NotImplementedError("ectrans4py does not support this platform")

        cores = len(os.sched_getaffinity(0))
        self.lib.init_env(
            unlimited_stack=(system == "Linux"),
            omp_num_threads=cores,
        )
        self.lib.mpl_init4py()
        self.lib.setup_trans0_4py(1, 1, 1, True, len(self._resolutions) + 1, True)

        self._initialised = True

    def resolution(self, trunc: int, kloen: np.ndarray) -> dict:
        """Return cached {kresol, kspec2, kgptot} for a (trunc, kloen) pair."""
        key = (trunc, tuple(kloen.tolist()))
        if key in self._resolutions:
            return self._resolutions[key]

        self._ensure_init()

        kdgl = kloen.size
        kresol = self.lib.setup_trans_4py(trunc, kdgl, kdgl, kloen, True, True)

        result = self.lib.trans_inq4py(kresol, kdgl, trunc, kdgl, kloen, 1)

        info = {
            "kresol": kresol,
            "kgptot": result[0],
            "kspec": result[1],
            "kspec2": result[2],
        }
        self._resolutions[key] = info
        return info


class ectrans4py(CalculationBackend):
    """Backend using ectrans4py (serial via mpi_serial stub)."""

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            import ectrans4py  # type: ignore[reportMissingImports]  # noqa: F401

            return True, "ectrans4py is available"
        except Exception as e:
            return False, f"ectrans4py is not available: {e}"

    def __init__(self, kloen: np.ndarray, trunc: int, grid: str | None = None):
        super().__init__(kloen, trunc, grid)
        self._rt = _EctransRuntime()
        self._info = self._rt.resolution(self.trunc, self.kloen)

    def vordiv_to_uv(self, vorticity: np.ndarray, divergence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kspec2 = self._info["kspec2"]
        kgptot = self._info["kgptot"]
        squeezed = vorticity.ndim == 1

        vor = np.atleast_2d(np.ascontiguousarray(vorticity, dtype=np.float64))
        div = np.atleast_2d(np.ascontiguousarray(divergence, dtype=np.float64))

        if vor.shape[-1] != kspec2:
            raise ValueError(f"Spectral dimension mismatch: got {vor.shape[-1]}, expected {kspec2} for T{self.trunc}")

        u, v = self._rt.lib.inv_trans_uv_dist4py(
            kspec2,
            kgptot,
            vor.shape[0],
            vor,
            div,
        )

        if squeezed:
            return u.squeeze(0), v.squeeze(0)
        return u, v

    def uv_to_vordiv(
        self, u_component_of_wind: np.ndarray, v_component_of_wind: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        kspec2 = self._info["kspec2"]
        kgptot = self._info["kgptot"]
        squeezed = u_component_of_wind.ndim == 1

        u = np.atleast_2d(np.ascontiguousarray(u_component_of_wind, dtype=np.float64))
        v = np.atleast_2d(np.ascontiguousarray(v_component_of_wind, dtype=np.float64))

        vor, div = self._rt.lib.dir_trans_uv_dist4py(
            kspec2,
            kgptot,
            u.shape[0],
            u,
            v,
        )

        if squeezed:
            return vor.squeeze(0), div.squeeze(0)
        return vor, div

    def sh_to_gg(self, scalars: np.ndarray) -> np.ndarray:
        kspec2 = self._info["kspec2"]
        kgptot = self._info["kgptot"]
        squeezed = scalars.ndim == 1

        q = np.atleast_2d(np.ascontiguousarray(scalars, dtype=np.float64))

        if q.shape[-1] != kspec2:
            raise ValueError(f"Spectral dimension mismatch: got {q.shape[-1]}, expected {kspec2} for T{self.trunc}")

        out = self._rt.lib.inv_trans_scalar_dist4py(
            kspec2,
            kgptot,
            q.shape[0],
            q,
        )

        if squeezed:
            return out.squeeze(0)
        return out
