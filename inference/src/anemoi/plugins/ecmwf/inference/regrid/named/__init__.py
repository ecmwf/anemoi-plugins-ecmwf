# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib.resources
from dataclasses import dataclass
from typing import Literal

import numpy as np

_NAMED_PKG = "anemoi.plugins.ecmwf.inference.regrid.named"

KNOWN_GRIDS = [
    f.name
    for f in importlib.resources.files(_NAMED_PKG).iterdir()
    if f.is_dir() and (f / "latitudes.npz").is_file() and (f / "longitudes.npz").is_file()
]


@dataclass
class NamedRegrid:
    name: str

    def __post_init__(self):
        self.name = self.name.lower()
        if self.name not in KNOWN_GRIDS:
            raise ValueError(f"Unknown grid name: {self.name}")

    def _open_coord(self, name: Literal["latitudes", "longitudes"]) -> np.ndarray:
        """Open the specified coordinate array for this grid name."""
        ref = importlib.resources.files(_NAMED_PKG) / self.name / f"{name}.npz"
        with importlib.resources.as_file(ref) as path:
            with np.load(path) as f:
                return f["data"]

    @property
    def latitudes(self) -> list[float]:
        return self._open_coord("latitudes").tolist()

    @property
    def longitudes(self) -> list[float]:
        return self._open_coord("longitudes").tolist()

    @property
    def gridspec(self) -> dict[str, dict[str, list[float]]]:
        return {"grid": {"latitudes": self.latitudes, "longitudes": self.longitudes}}
