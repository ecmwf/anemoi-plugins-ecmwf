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
from abc import ABC
from abc import abstractmethod

import numpy as np

LOG = logging.getLogger(__name__)


class CalculationBackend(ABC):
    """Abstract base class for spectral transform backends.

    Subclasses must implement ``available``, ``vordiv_to_uv`` and
    ``uv_to_vordiv``.  Initialisation (library init, resolution setup)
    is handled lazily on first use -- callers just construct and call.
    """

    def __init__(self, kloen: np.ndarray, trunc: int):
        self.kloen = np.asarray(kloen, dtype=np.int64)
        self.trunc = trunc

    @classmethod
    @abstractmethod
    def available(cls) -> tuple[bool, str]:
        """Check if the backend library can be imported."""
        ...

    @abstractmethod
    def vordiv_to_uv(self, vorticity: np.ndarray, divergence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Spectral vorticity/divergence to grid-point u/v."""
        ...

    @abstractmethod
    def uv_to_vordiv(
        self, u_component_of_wind: np.ndarray, v_component_of_wind: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grid-point u/v to spectral vorticity/divergence."""
        ...


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
        self.lib.setup_trans0_4py(1, 1, 1, True, len(self._resolutions) + 1)

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

            return True, "ectrans4py is available"
        except Exception as e:
            return False, f"ectrans4py is not available: {e}"

    def __init__(self, kloen: np.ndarray, trunc: int):
        super().__init__(kloen, trunc)
        self._rt = _EctransRuntime()
        self._info = self._rt.resolution(self.trunc, self.kloen)

    def vordiv_to_uv(self, vorticity: np.ndarray, divergence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kspec2 = self._info["kspec2"]
        kgptot = self._info["kgptot"]
        squeezed = vorticity.ndim == 1

        vor = np.atleast_2d(np.ascontiguousarray(vorticity))
        div = np.atleast_2d(np.ascontiguousarray(divergence))

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

        u = np.atleast_2d(np.ascontiguousarray(u_component_of_wind))
        v = np.atleast_2d(np.ascontiguousarray(v_component_of_wind))

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


# ---------------------------------------------------------------------------
# ctrans4py backend
# ---------------------------------------------------------------------------


class ctrans4py(CalculationBackend):
    """Backend using ctrans4py (C spectral-transform library via transi ABI).

    ``vordiv_to_uv`` uses the spectral-only path (no Legendre/FFT).
    ``uv_to_vordiv`` uses a full direct transform.
    """

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            return True, "ctrans4py is available"
        except Exception as e:
            return False, f"ctrans4py is not available: {e}"

    def __init__(self, kloen: np.ndarray, trunc: int):
        super().__init__(kloen, trunc)
        import os

        import ctrans4py  # type: ignore[reportMissingImports]

        self._lib = ctrans4py
        cores = len(os.sched_getaffinity(0))
        self._lib.init_env(omp_num_threads=cores)

    def vordiv_to_uv(self, vorticity: np.ndarray, divergence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grid = self._lib.detect_grid(np.asarray(self.kloen, dtype=np.int32))
        squeezed = vorticity.ndim == 1
        vor = np.atleast_2d(np.ascontiguousarray(vorticity, dtype=np.float64))
        div = np.atleast_2d(np.ascontiguousarray(divergence, dtype=np.float64))

        tr = self._lib.Transform(self.trunc, grid=grid, nlev=vor.shape[0])
        try:
            u, v = tr.vordiv_to_uv(vor, div)
        finally:
            tr.close()

        if squeezed:
            return u.squeeze(0), v.squeeze(0)
        return u, v

    def uv_to_vordiv(
        self, u_component_of_wind: np.ndarray, v_component_of_wind: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        grid = self._lib.detect_grid(np.asarray(self.kloen, dtype=np.int32))
        squeezed = u_component_of_wind.ndim == 1
        u = np.atleast_2d(np.ascontiguousarray(u_component_of_wind, dtype=np.float64))
        v = np.atleast_2d(np.ascontiguousarray(v_component_of_wind, dtype=np.float64))

        tr = self._lib.Transform(self.trunc, grid=grid, nlev=u.shape[0])
        try:
            result = tr.uv_to_vordiv(u, v)
        finally:
            tr.close()

        if squeezed:
            return result[0].squeeze(0), result[1].squeeze(0)
        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BACKENDS: dict[str, type[CalculationBackend]] = {
    "ctrans4py": ctrans4py,
    "ectrans4py": ectrans4py,
}


def get_backend(order: list[str] | None = None) -> type[CalculationBackend]:
    """Return the first available backend class.

    Parameters
    ----------
    order : list of str, optional
        Backend names to try, in preference order.  Defaults to all registered.

    Raises
    ------
    RuntimeError
        If no backend is available.

    Returns
    -------
    type[CalculationBackend]
        The first available backend class.
    """
    if order is None:
        order = list(BACKENDS.keys())

    error_messages = []
    for name in order:
        cls = BACKENDS.get(name)
        if cls is None:
            error_messages.append(f"Backend {name} not found")
            continue

        ok, msg = cls.available()
        if ok:
            LOG.info("vordiv <-> uv: Using backend: %s", name)
            return cls
        error_messages.append(f"{name}: {msg}")

    raise RuntimeError("No available backend found:\n" + "\n".join(error_messages))
