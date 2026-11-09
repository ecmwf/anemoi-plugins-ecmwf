# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Target grids and helpers for the regridding tests.

A new target grid that only needs the generic checks (one field out, grid type,
metadata, point count, constant values, paramId) is a single ``RegridCase``
entry in ``CASES``, run by ``test_mir.py``.

This module must not import ``mir`` at module level.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

from anemoi.plugins.ecmwf.transform.regrid.backend import GridSpec
from anemoi.plugins.ecmwf.transform.regrid.backend import mir_regrid
from anemoi.plugins.ecmwf.transform.regrid.named import NamedRegrid

if TYPE_CHECKING:
    import earthkit.data as ekd

#: Grid of the synthetic source fields.
SOURCE_GRID = "O32"
#: Constant value of the first synthetic source field.
BASE_VALUE = 300.0
#: Absolute tolerance for a constant field staying constant after interpolation.
ATOL = 1.0
#: Area ``[north, west, south, east]`` straddling the Greenwich meridian.
O1280_AREA = [40, -25, -40, 55]
#: Small unstructured target grid of explicit coordinates.
UNSTRUCTURED_GRID = {
    "latitudes": [0.0, 10.0, 20.0, 30.0],
    "longitudes": [0.0, 10.0, 20.0, 30.0],
}


def regrid(
    grib_fieldlist: Callable[..., ekd.FieldList],
    grid: GridSpec,
    *,
    param_id: int = 130,
    nfields: int = 1,
    base_value: float = BASE_VALUE,
    **regrid_kwargs: Any,
) -> ekd.FieldList:
    """Create synthetic source fields and regrid them with ``mir_regrid``.

    Parameters
    ----------
    grib_fieldlist : Callable[..., ekd.FieldList]
        The ``grib_fieldlist`` factory fixture.
    grid : GridSpec
        The target grid, passed to ``mir_regrid``.
    param_id : int
        GRIB paramId of the source fields.
    nfields : int
        Number of source fields; field i gets ``base_value + i * 10.0``.
    base_value : float
        Constant value of the first source field.
    **regrid_kwargs : Any
        Further keyword arguments passed to ``mir_regrid`` (``area``, ...).

    Returns
    -------
    ekd.FieldList
        The regridded fields.
    """
    fields = grib_fieldlist(grid=SOURCE_GRID, nfields=nfields, base_value=base_value, param_id=param_id)
    return mir_regrid(fields, grid, **regrid_kwargs)


def _meps_grid() -> GridSpec:
    """Coordinate dict of the named ``meps`` grid (``mir_regrid`` does not resolve names)."""
    return NamedRegrid("meps").gridspec["grid"]


@dataclass(frozen=True)
class RegridCase:
    """A target grid exercised by the generic tests.

    Parameters
    ----------
    id : str
        Test id.
    grid : GridSpec | Callable[[], GridSpec]
        Target grid passed to ``mir_regrid``, or a factory building it lazily.
    area : list[float] | None
        Target area ``[north, west, south, east]``.
    grid_type : str
        Expected GRIB ``gridType`` of the output.
    expected_metadata : dict[str, Any]
        Further GRIB keys with their expected values.
    expected_npoints : int | None
        Expected number of output points, when known exactly. For a coordinate
        dict grid the number of coordinate pairs is expected regardless.
    """

    id: str
    grid: GridSpec | Callable[[], GridSpec]
    grid_type: str
    area: list[float] | None = None
    expected_metadata: dict[str, Any] = field(default_factory=dict)
    expected_npoints: int | None = None

    def make_grid(self) -> GridSpec:
        """Return the target grid, building it if it is given as a factory."""
        return self.grid() if callable(self.grid) else self.grid

    def regrid_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments for ``regrid`` (grid and area)."""
        return {"grid": self.make_grid(), "area": self.area}

    def npoints(self) -> int | None:
        """Return the expected number of output points, or None when unknown."""
        grid = self.make_grid()
        if isinstance(grid, dict) and "latitudes" in grid:
            return len(grid["latitudes"])
        return self.expected_npoints


CASES = [
    RegridCase("O16", "O16", "reduced_gg"),
    RegridCase("O16-area", "O16", "reduced_gg", area=[90, 0, 0, 180]),
    RegridCase("latlon-1deg", [1.0, 1.0], "regular_ll"),
    RegridCase("unstructured", UNSTRUCTURED_GRID, "unstructured_grid"),
    RegridCase("meps", _meps_grid, "unstructured_grid"),
    RegridCase(
        "O1280-area",
        "O1280",
        "reduced_gg",
        area=O1280_AREA,
        expected_metadata={"N": 1280, "isOctahedral": 1, "gridName": "O1280"},
    ),
]
