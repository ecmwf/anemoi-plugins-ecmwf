# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""MIRRegrid as an anemoi-inference pre-processor to a Lambert grid, and the mir template for that grid.

The steps (``MIRRegrid.forward``, ``EkdInput.set_private_attributes``, the template
lookup with the real ``TemplateManager._grid``, ``MirTemplatesProvider.template``)
are run by ``preprocessor_lambert_script.py`` in a fresh interpreter with ``ECCODES_ECKIT_GEO=0``:
ecCodes reads that variable when it initialises, and with ``ECCODES_ECKIT_GEO=1``
(set by the regrid package unless it is already set) it cannot build the geography of a Lambert grid.
"""

import functools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import pytest
from anemoi.plugins.ecmwf.transform.regrid.backend import canonical_lambert_grid
from anemoi.plugins.ecmwf.transform.regrid.backend import lambert_points
from cases import LAMBERT_GRID
from cases import LAMBERT_NX
from cases import LAMBERT_NY
from cases import MEPS_LAMBERT_GRID
from cases import MEPS_NPOINTS

SCRIPT = Path(__file__).with_name("preprocessor_lambert_script.py")
#: The regional area of the checkpoint stub of the script, which the template manager falls back to.
CHECKPOINT_AREA = [70.0, -20.0, 30.0, 40.0]
#: Largest distance in metres between the position a value says it is at (source values = latitude and longitude,
#: ieee packing) and its point: the interpolation error, measured (O320 on MEPS 138 m, O640 36 m), rounded up.
VALUE_TOLERANCE_M = {"O320": 150.0, "O640": 50.0}
#: Largest difference in degrees between a decoded point and the exact point (11 m).
POSITION_ATOL_DEGREES = 1e-4

#: Earths of a Lambert grid: the keys that make it, and the radius in metres.
EARTHS = {
    "shape-0": ({"shapeOfTheEarth": 0}, 6367470.0),
    "shape-8": ({"shapeOfTheEarth": 8}, 6371200.0),
    "shape-1": (
        {
            "shapeOfTheEarth": 1,
            "scaleFactorOfRadiusOfSphericalEarth": 1,
            "scaledValueOfRadiusOfSphericalEarth": 63700000,
        },
        6370000.0,
    ),
}

pytest.importorskip("mir", reason="MIR not available")


def with_earth(grid: dict[str, Any], earth: dict[str, Any]) -> dict[str, Any]:
    """``grid`` on another earth: the radius keys it has are dropped unless the earth has them."""
    new = {**grid, **earth}
    if new["shapeOfTheEarth"] != 1:
        new = {k: v for k, v in new.items() if "RadiusOfSphericalEarth" not in k}
    return new


def distance_m(latitude1: Any, longitude1: Any, latitude2: Any, longitude2: Any, radius: float = 6371000.0) -> Any:
    """Great-circle distance in metres between points in degrees (the haversine form: exact for tiny distances)."""
    la1, lo1, la2, lo2 = (
        np.radians(np.asarray(a, dtype=float)) for a in (latitude1, longitude1, latitude2, longitude2)
    )
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def signed(longitude: Any) -> Any:
    """Longitudes in [-180, 180)."""
    return (np.asarray(longitude, dtype=float) + 180.0) % 360.0 - 180.0


def run_script(arguments: dict[str, Any], eckit_geo: str = "0") -> subprocess.CompletedProcess:
    """Run the script with the arguments (JSON), the arrays it computes are saved to ``arguments["output"]``."""
    env = {**os.environ, "ECCODES_ECKIT_GEO": eckit_geo}
    return subprocess.run(
        [sys.executable, str(SCRIPT), json.dumps(arguments)],
        capture_output=True,
        env=env,
        check=False,
        text=True,
    )


@functools.cache
def _run(arguments: str) -> dict[str, Any]:
    """Run the script and return its report, with the arrays it saved."""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "out.npz"
        proc = run_script({**json.loads(arguments), "output": str(output)})
        assert proc.returncode == 0, proc.stderr
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        with np.load(output) as arrays:
            report.update({key: arrays[key] for key in arrays.files})
    return report


def run(grid: dict[str, Any], source: str = "O32", quantity: str = "constant", input_shape: int | None = None):
    return _run(json.dumps({"grid": grid, "source": source, "quantity": quantity, "input_shape": input_shape}))


def expected_header(grid: dict[str, Any]) -> dict[str, Any]:
    """The keys of the canonical grid, scanning from south to north, as GRIB2."""
    keys = {key: value for key, value in canonical_lambert_grid(grid).items() if key != "gridType"}
    return {**keys, "jScansPositively": 1, "edition": 2, "numberOfDataPoints": grid["Nx"] * grid["Ny"]}


def assert_header(header: dict[str, Any], grid: dict[str, Any]) -> None:
    assert header["gridType"] == "lambert"
    for key, value in expected_header(grid).items():
        assert header[key] == pytest.approx(value, abs=1e-9), key


def assert_template_is_the_field(result: dict[str, Any]) -> None:
    """The template has the header and the points of the regridded field."""
    assert result["template"] == result["field"]
    np.testing.assert_allclose(result["template_latitudes"], result["field_latitudes"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        signed(result["template_longitudes"] - result["field_longitudes"]), 0.0, atol=1e-9, rtol=0
    )


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return run(LAMBERT_GRID)


def test_state_geography_is_the_lambert_grid(result: dict[str, Any]):
    """anemoi-inference records the canonical Lambert grid for the two fields, which got the keys in another order.

    ``set_private_attributes`` compares ``str`` of the grid of each field: it records none if they differ.
    """
    assert result["nfields"] == 2
    assert result["geography"] == {"grid": canonical_lambert_grid(LAMBERT_GRID)}
    assert result["field"] == result["second_field"]


def test_lookup_grid_is_the_lambert_grid_and_area_the_checkpoint_one(result: dict[str, Any]):
    """The real ``TemplateManager.load_template`` makes a lookup with the Lambert dict, not the checkpoint grid.

    The regridded fields report no area, so the manager falls back to the (regional) area of the checkpoint metadata:
    the provider is given it and must not pass it to MIR.
    """
    assert result["geography"] == {"grid": canonical_lambert_grid(LAMBERT_GRID)}
    assert "area" not in result["geography"]
    assert result["lookup"] == {"levtype": "sfc", "grid": canonical_lambert_grid(LAMBERT_GRID), "area": CHECKPOINT_AREA}
    assert result["lookup"]["area"]  # non-empty, regional


def test_regridded_field_is_the_lambert_grid(result: dict[str, Any]):
    assert_header(result["field"], LAMBERT_GRID)
    assert result["field"]["numberOfDataPoints"] == LAMBERT_NX * LAMBERT_NY


def test_template_is_the_grid_of_the_field(result: dict[str, Any]):
    """The template has the same header (every key) and the same points as the regridded field."""
    assert_template_is_the_field(result)
    assert len(result["template_latitudes"]) == LAMBERT_NX * LAMBERT_NY


def test_template_reports_the_lambert_grid(result: dict[str, Any]):
    assert result["template_mars_grid"] == canonical_lambert_grid(LAMBERT_GRID)


def test_the_template_scans_from_south_to_north(result: dict[str, Any]):
    """The template and the field are written scanning south to north, with the south-west corner first."""
    assert result["field"]["jScansPositively"] == result["template"]["jScansPositively"] == 1
    assert result["template_latitudes"][0] < result["template_latitudes"][-1]
    assert result["template_latitudes"][0] == pytest.approx(LAMBERT_GRID["latitudeOfFirstGridPointInDegrees"], abs=1e-6)
    assert signed(result["template_longitudes"][0]) == pytest.approx(
        LAMBERT_GRID["longitudeOfFirstGridPointInDegrees"], abs=1e-6
    )


def test_the_meps_grid_end_to_end():
    """MEPS: the field and the template have the header of the patch and the exact (LCC) points, within 1e-4 degrees.

    The values (O640 source, values = latitude and longitude) are at the points to the interpolation error.
    """
    by_latitude = run(MEPS_LAMBERT_GRID, "O640", "latitude")
    by_longitude = run(MEPS_LAMBERT_GRID, "O640", "longitude")

    assert by_latitude["geography"] == {"grid": canonical_lambert_grid(MEPS_LAMBERT_GRID)}
    assert by_latitude["lookup"]["grid"] == canonical_lambert_grid(MEPS_LAMBERT_GRID)
    assert_header(by_latitude["field"], MEPS_LAMBERT_GRID)
    for key, value in MEPS_LAMBERT_GRID.items():
        assert by_latitude["field"][key] == value, key
    assert_template_is_the_field(by_latitude)
    assert len(by_latitude["template_latitudes"]) == MEPS_NPOINTS

    latitudes, longitudes = lambert_points(MEPS_LAMBERT_GRID)
    np.testing.assert_allclose(by_latitude["template_latitudes"], latitudes, atol=POSITION_ATOL_DEGREES, rtol=0)
    np.testing.assert_allclose(
        signed(by_latitude["template_longitudes"] - longitudes), 0.0, atol=POSITION_ATOL_DEGREES, rtol=0
    )
    error = distance_m(
        by_latitude["field_values"],
        signed(by_longitude["field_values"]),
        by_latitude["template_latitudes"],
        signed(by_latitude["template_longitudes"]),
    )
    assert error.max() < VALUE_TOLERANCE_M["O640"]


def test_the_meps_points_are_those_of_pyproj():
    """The points of the MEPS template are those of pyproj (an independent PROJ string)."""
    by_latitude = run(MEPS_LAMBERT_GRID, "O640", "latitude")
    lcc = pyproj.Proj("+proj=lcc +lat_0=63.3 +lat_1=63.3 +lat_2=63.3 +lon_0=15 +R=6371000")
    x0, y0 = lcc(1.60491, 51.633816)
    x = x0 + 2500 * np.arange(849)[None, :] + np.zeros((969, 1))
    y = y0 + 2500 * np.arange(969)[:, None] + np.zeros((1, 849))
    longitudes, latitudes = lcc(x.ravel(), y.ravel(), inverse=True)

    np.testing.assert_allclose(by_latitude["template_latitudes"], latitudes, atol=POSITION_ATOL_DEGREES, rtol=0)
    np.testing.assert_allclose(
        signed(by_latitude["template_longitudes"] - longitudes), 0.0, atol=POSITION_ATOL_DEGREES, rtol=0
    )


@pytest.mark.parametrize("quantity", ["latitude", "longitude"])
def test_the_values_are_the_coordinates_of_the_template_points(quantity: str):
    """Source values = latitude (or longitude): each value of the field is the coordinate of its template point."""
    result = run(LAMBERT_GRID, "O320", quantity)

    key = "latitudes" if quantity == "latitude" else "longitudes"
    points = np.array(result[f"template_{key}"])
    if quantity == "longitude":
        points = signed(points)
    # 150 m is 0.0014 degrees of latitude, 0.003 of longitude at 63 N
    np.testing.assert_allclose(result["field_values"], points, atol=0.004, rtol=0)
    assert len(points) == LAMBERT_NX * LAMBERT_NY


@pytest.mark.parametrize("label", EARTHS)
@pytest.mark.parametrize("input_shape", [6, 1])
def test_the_earth_of_the_grid_is_not_that_of_the_input(label: str, input_shape: int):
    """The field and the template have the earth of the grid, and the exact points of the grid on it.

    The base templates have another earth than the data of an IFS run, so the points would
    differ by up to a kilometre otherwise.
    """
    earth, _ = EARTHS[label]
    grid = with_earth(LAMBERT_GRID, earth)

    result = run(grid, input_shape=input_shape)

    assert result["field"]["shapeOfTheEarth"] == result["template"]["shapeOfTheEarth"] == earth["shapeOfTheEarth"]
    assert_header(result["field"], grid)
    assert_template_is_the_field(result)
    latitudes, longitudes = lambert_points(grid)
    np.testing.assert_allclose(result["template_latitudes"], latitudes, atol=POSITION_ATOL_DEGREES, rtol=0)
    np.testing.assert_allclose(
        signed(result["template_longitudes"] - longitudes), 0.0, atol=POSITION_ATOL_DEGREES, rtol=0
    )


def test_grib1_input():
    """GRIB1 input: the field and the template have the header and the points of the grid."""
    result = run(LAMBERT_GRID, "O320-grib1", "latitude")

    assert_header(result["field"], LAMBERT_GRID)
    assert_template_is_the_field(result)
    latitudes, _ = lambert_points(LAMBERT_GRID)
    np.testing.assert_allclose(result["template_latitudes"], latitudes, atol=POSITION_ATOL_DEGREES, rtol=0)
    # the values are the latitudes of the points (0.004 degrees is 440 m, the 24 bit packing of the source is 1 m)
    np.testing.assert_allclose(result["field_values"], latitudes, atol=0.0015, rtol=0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param({"shapeOfTheEarth": 2}, "shapeOfTheEarth=2 is not supported", id="ellipsoid"),
        pytest.param({"shapeOfTheEarth": 3}, "shapeOfTheEarth=3 is not supported", id="shape-3"),
        pytest.param(
            {"Latin1InDegrees": 50.0, "Latin2InDegrees": 58.0}, "only tangent grids are supported", id="secant"
        ),
        pytest.param({"scanningMode": 0}, "only scanningMode=64", id="scanning-mode"),
        pytest.param({"Bogus": 1}, "unknown key", id="unknown-key"),
        pytest.param({"projectionCentreFlag": 128}, "projectionCentreFlag=0", id="projection-centre"),
    ],
)
def test_unsupported_grids_are_an_error(changes: dict[str, Any], message: str):
    """Earths, secant grids, scanning modes and keys that are not supported are refused, not silently mis-encoded."""
    proc = run_script(
        {
            "grid": {**LAMBERT_GRID, **changes},
            "source": "O32",
            "quantity": "constant",
            "input_shape": None,
            "output": "x",
        }
    )

    assert proc.returncode != 0
    assert "ValueError" in proc.stderr
    assert message in proc.stderr


def test_a_missing_key_is_an_error():
    grid = {key: value for key, value in LAMBERT_GRID.items() if key != "DxInMetres"}

    proc = run_script({"grid": grid, "source": "O32", "quantity": "constant", "input_shape": None, "output": "x"})

    assert proc.returncode != 0
    assert "missing required key(s) ['DxInMetres']" in proc.stderr


def test_eckit_geo_on_warns_and_fails_reading_the_points():
    """With ``ECCODES_ECKIT_GEO=1`` the regrid warns, and ecCodes cannot give the points of a Lambert grid."""
    proc = run_script(
        {"grid": LAMBERT_GRID, "source": "O32", "quantity": "constant", "input_shape": None, "output": "x"},
        eckit_geo="1",
    )

    assert proc.returncode != 0
    assert "ECCODES_ECKIT_GEO=0" in proc.stderr
    assert "GeocalculusError" in proc.stderr
