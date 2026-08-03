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

    Subclasses must implement ``available``, ``vordiv_to_uv``,
    ``uv_to_vordiv`` and ``sh_to_gg``.  Initialisation (library init, resolution setup)
    is handled lazily on first use -- callers just construct and call.
    """

    def __init__(self, kloen: np.ndarray, trunc: int, grid: str | None = None):
        self.kloen = np.asarray(kloen, dtype=np.int64)
        self.trunc = trunc
        self.grid = grid if grid is not None else f"O{trunc + 1}"

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

    @abstractmethod
    def sh_to_gg(self, scalars: np.ndarray) -> np.ndarray:
        """Spectral scalar field(s) to grid-point."""
        ...
