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

import functools
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pyproj
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

#: The MEPS Lambert grid (2.5 km, 849 x 969 points) as a dict of GRIB keys: Lambert conformal, tangent
#: at 63.3 N, on a sphere of radius 6371000 m (``shapeOfTheEarth=1``), scanning from the south-west corner.
MEPS_LAMBERT_GRID: dict[str, Any] = {
    "gridType": "lambert",
    "Nx": 849,
    "Ny": 969,
    "latitudeOfFirstGridPointInDegrees": 51.633816,
    "longitudeOfFirstGridPointInDegrees": 1.60491,
    "LaDInDegrees": 63.3,
    "LoVInDegrees": 15.0,
    "Latin1InDegrees": 63.3,
    "Latin2InDegrees": 63.3,
    "DxInMetres": 2500,
    "DyInMetres": 2500,
    "scanningMode": 64,
    "shapeOfTheEarth": 1,
    "scaleFactorOfRadiusOfSphericalEarth": 0,
    "scaledValueOfRadiusOfSphericalEarth": 6371000,
    "projectionCentreFlag": 0,
    "resolutionAndComponentFlags": 48,
    "latitudeOfSouthernPoleInDegrees": -90.0,
    "longitudeOfSouthernPoleInDegrees": 0.0,
}
MEPS_NPOINTS = 849 * 969
#: A small, fast Lambert grid with the geometry of MEPS: the same first point and projection, 25 km steps.
LAMBERT_NX, LAMBERT_NY = 60, 70
LAMBERT_GRID: dict[str, Any] = {
    **MEPS_LAMBERT_GRID,
    "Nx": LAMBERT_NX,
    "Ny": LAMBERT_NY,
    "DxInMetres": 25000,
    "DyInMetres": 25000,
}

#: Radius in metres of the sphere of the ``shapeOfTheEarth`` values with a fixed radius.
SHAPE_RADII = {0: 6367470.0, 6: 6371229.0, 8: 6371200.0}
#: Earth variants of a Lambert grid: label, the keys that make the earth, and its radius in metres.
EARTHS = {
    "shape-0": ({"shapeOfTheEarth": 0}, 6367470.0),
    "shape-6": ({"shapeOfTheEarth": 6}, 6371229.0),
    "shape-8": ({"shapeOfTheEarth": 8}, 6371200.0),
    "shape-1-6371000": (
        {
            "shapeOfTheEarth": 1,
            "scaleFactorOfRadiusOfSphericalEarth": 0,
            "scaledValueOfRadiusOfSphericalEarth": 6371000,
        },
        6371000.0,
    ),
    "shape-1-scaled": (
        {
            "shapeOfTheEarth": 1,
            "scaleFactorOfRadiusOfSphericalEarth": 1,
            "scaledValueOfRadiusOfSphericalEarth": 63700000,
        },
        6370000.0,
    ),
}
#: Largest distance in metres between the position a value says it is at (source values = latitude and
#: longitude, ieee packing) and the point it is at, by source grid: the interpolation error, measured
#: (O320 on MEPS 138 m, O640 36 m) and rounded up.
VALUE_TOLERANCE_M = {"O320": 150.0, "O640": 50.0}
#: Largest distance in metres between a decoded grid point and the exact point (pyproj), 1e-4 degrees is 11 m.
POSITION_ATOL_DEGREES = 1e-4

#: Keys of the header of a Lambert field that a test compares with the grid.
HEADER_KEYS = (*MEPS_LAMBERT_GRID, "edition", "jScansPositively", "numberOfDataPoints", "packingType")


def lambert_grid(**changes: Any) -> dict[str, Any]:
    """The small Lambert grid with keys changed.

    The radius keys of the base grid (``shapeOfTheEarth=1``) are dropped if ``changes`` has another shape,
    unless it gives them.
    """
    grid = {**LAMBERT_GRID, **changes}
    if grid["shapeOfTheEarth"] != 1:
        for key in ("scaleFactorOfRadiusOfSphericalEarth", "scaledValueOfRadiusOfSphericalEarth"):
            if key not in changes:
                del grid[key]
    return grid


def signed_longitude(longitude: Any) -> Any:
    """Longitudes in [-180, 180)."""
    return (np.asarray(longitude, dtype=float) + 180.0) % 360.0 - 180.0


def distance_m(latitude1: Any, longitude1: Any, latitude2: Any, longitude2: Any, radius: float = 6371000.0) -> Any:
    """Great-circle distance in metres between points in degrees (the haversine form: exact for tiny distances)."""
    la1, lo1, la2, lo2 = (
        np.radians(np.asarray(a, dtype=float)) for a in (latitude1, longitude1, latitude2, longitude2)
    )
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def grid_radius(grid: dict[str, Any]) -> float:
    """Radius of the earth of a Lambert grid, from the keys (independent of the package)."""
    if grid.get("shapeOfTheEarth", 6) == 1:
        return grid["scaledValueOfRadiusOfSphericalEarth"] / 10.0 ** grid["scaleFactorOfRadiusOfSphericalEarth"]
    return SHAPE_RADII[grid.get("shapeOfTheEarth", 6)]


def pyproj_points(grid: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Exact points of a tangent Lambert grid by pyproj (an independent projection), flat, south-west corner first.

    The columns run east and the rows north. Built from a PROJ string, with the radius of the grid read
    from its keys, so it does not share the construction of ``lambert_projection``.
    """
    lad = grid["LaDInDegrees"]
    lcc = pyproj.Proj(
        f"+proj=lcc +lat_0={lad} +lat_1={lad} +lat_2={lad} +lon_0={grid['LoVInDegrees']} +R={grid_radius(grid)}"
    )
    x0, y0 = lcc(grid["longitudeOfFirstGridPointInDegrees"], grid["latitudeOfFirstGridPointInDegrees"])
    x = x0 + grid["DxInMetres"] * np.arange(grid["Nx"])[None, :] + np.zeros((grid["Ny"], 1))
    y = y0 + grid["DyInMetres"] * np.arange(grid["Ny"])[:, None] + np.zeros((1, grid["Nx"]))
    longitudes, latitudes = lcc(x.ravel(), y.ravel(), inverse=True)
    return np.asarray(latitudes), np.asarray(longitudes)


#: Run in a fresh interpreter with ``ECCODES_ECKIT_GEO=0`` (ecCodes reads it when it initialises) by ``probe``.
#: The only argument is a JSON object (see ``probe``). Prints a JSON report line, and saves the arrays.
_PROBE_CODE = """
import json, sys
import earthkit.data as ekd
import eccodes
import numpy as np
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid

args = json.loads(sys.argv[1])
source = args["source"]
sample = f"reduced_gg_pl_{source.split('-')[0][1:]}_grib{1 if source.endswith('-grib1') else 2}"
handle = eccodes.codes_grib_new_from_samples(sample)
if args["input_shape"] is not None and not source.endswith("-grib1"):
    eccodes.codes_set(handle, "shapeOfTheEarth", args["input_shape"])
lat = eccodes.codes_get_array(handle, "latitudes")
lon = eccodes.codes_get_array(handle, "longitudes")
values = {
    "constant": np.full(lat.size, 300.0),
    "latitude": lat,
    "longitude": (lon + 180.0) % 360.0 - 180.0,
}[args["quantity"]]
if source.endswith("-grib1"):
    # GRIB1 has no ieee packing: 24 bits are about a metre for a coordinate
    eccodes.codes_set(handle, "bitsPerValue", 24)
else:
    eccodes.codes_set(handle, "packingType", "grid_ieee")
eccodes.codes_set_values(handle, values)
fields = ekd.from_source("memory", eccodes.codes_get_message(handle))
eccodes.codes_release(handle)

field = MIRRegrid(grid=args["grid"], packing=args["packing"], accuracy=args["accuracy"]).forward(fields)[0]
glat, glon = field.grid_points()
np.savez(args["output"], values=field.values, latitudes=glat, longitudes=glon)
print(json.dumps({
    "header": {key: field.metadata(key, default=None) for key in args["keys"]},
    "source_edition": fields[0].metadata("edition"),
    "mars_grid": field.metadata().geography.mars_grid(),
    "proj": field.projection().to_proj_string() if field.metadata("gridType") == "lambert" else None,
}))
"""


@functools.cache
def _probe(arguments: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "out.npz"
        env = {**os.environ, "ECCODES_ECKIT_GEO": "0"}
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE, json.dumps({**json.loads(arguments), "output": str(output)})],
            capture_output=True,
            env=env,
            check=False,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        with np.load(output) as arrays:
            report.update({key: arrays[key] for key in arrays.files})
    return report


def probe(
    grid: GridSpec,
    source: str = "O32",
    quantity: str = "constant",
    input_shape: int | None = None,
    packing: str = "ieee",
) -> dict[str, Any]:
    """Regrid a Gaussian field to ``grid`` with ``MIRRegrid`` in a fresh interpreter with ``ECCODES_ECKIT_GEO=0``.

    Parameters
    ----------
    grid : GridSpec
        The target grid (JSON serialisable).
    source : str
        The source: ``O<N>`` (the ecCodes sample ``reduced_gg_pl_<N>_grib2``), or ``O<N>-grib1`` (GRIB1).
    quantity : str
        The values of the source: ``constant``, ``latitude`` or ``longitude`` (in [-180, 180)) of its points.
    input_shape : int, optional
        The ``shapeOfTheEarth`` of a GRIB2 source, the sample's if None.
    packing : str
        The packing of the output (``accuracy`` is 64 for ``ieee``, 16 otherwise).

    Returns
    -------
    dict
        ``header`` (the keys of ``HEADER_KEYS`` of the output), ``source_edition``, ``mars_grid``,
        ``proj`` (the projection of a Lambert output, as ecCodes decodes it), and the arrays ``values``,
        ``latitudes`` and ``longitudes`` (the points ecCodes reports for the output, ``grid_points``).
    """
    arguments = {
        "grid": grid,
        "source": source,
        "quantity": quantity,
        "input_shape": input_shape,
        "packing": packing,
        "accuracy": 64 if packing == "ieee" else 16,
        "keys": list(HEADER_KEYS),
    }
    return _probe(json.dumps(arguments, sort_keys=True))


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


def _lambert_metadata(grid: dict[str, Any]) -> dict[str, Any]:
    """The keys of a Lambert grid as expected in the header of the output (``gridType`` is checked apart)."""
    return {key: value for key, value in grid.items() if key != "gridType"}


CASES = [
    RegridCase("O16", "O16", "reduced_gg"),
    RegridCase("O16-area", "O16", "reduced_gg", area=[90, 0, 0, 180]),
    RegridCase("latlon-1deg", [1.0, 1.0], "regular_ll"),
    RegridCase("unstructured", UNSTRUCTURED_GRID, "unstructured_grid"),
    RegridCase("meps", _meps_grid, "unstructured_grid"),
    RegridCase(
        "lambert",
        LAMBERT_GRID,
        "lambert",
        expected_metadata=_lambert_metadata(LAMBERT_GRID),
        expected_npoints=LAMBERT_NX * LAMBERT_NY,
    ),
    RegridCase(
        "meps-lambert",
        MEPS_LAMBERT_GRID,
        "lambert",
        expected_metadata=_lambert_metadata(MEPS_LAMBERT_GRID),
        expected_npoints=MEPS_NPOINTS,
    ),
    RegridCase(
        "O1280-area",
        "O1280",
        "reduced_gg",
        area=O1280_AREA,
        expected_metadata={"N": 1280, "isOctahedral": 1, "gridName": "O1280"},
    ),
]
