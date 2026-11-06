# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests that run real MIR through ``mir_regrid`` (``regrid.backend``).

Target grids that need only the generic checks are listed in ``cases.CASES``:
adding a new target grid is a one-entry change there. Everything else is tested
in the case-specific classes and functions below. ``mir_regrid`` returns fields decoded from
GRIB bytes in memory, so no further GRIB round trip is needed.

The coordinates of a Lambert field are what ecCodes decodes with ``ECCODES_ECKIT_GEO=0``
(it reads the variable when it initialises), so they are read in a fresh interpreter
(``cases.probe``) and compared with pyproj, an independent projection.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from typing import Any

import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.plugins.ecmwf.transform.regrid.backend import canonical_lambert_grid
from anemoi.plugins.ecmwf.transform.regrid.backend import lambert_points
from anemoi.plugins.ecmwf.transform.regrid.backend import mir_regrid
from earthkit.data.core.metadata import WrappedMetadata

from . import requires_mir
from .cases import ATOL
from .cases import BASE_VALUE
from .cases import CASES
from .cases import EARTHS
from .cases import HEADER_KEYS
from .cases import LAMBERT_GRID
from .cases import LAMBERT_NX
from .cases import LAMBERT_NY
from .cases import MEPS_LAMBERT_GRID
from .cases import MEPS_NPOINTS
from .cases import O1280_AREA
from .cases import POSITION_ATOL_DEGREES
from .cases import SOURCE_GRID
from .cases import UNSTRUCTURED_GRID
from .cases import VALUE_TOLERANCE_M
from .cases import distance_m
from .cases import lambert_grid
from .cases import probe
from .cases import pyproj_points
from .cases import regrid
from .cases import signed_longitude

pytestmark = requires_mir


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
class TestMirRegridCases:
    """Generic ``mir_regrid`` checks run against every target grid in ``CASES``."""

    def test_single_field_out(self, grib_fieldlist, case):
        """One field in gives one field out."""
        assert len(regrid(grib_fieldlist, **case.regrid_kwargs())) == 1

    def test_grid_type(self, grib_fieldlist, case):
        """The output has the expected grid type and metadata."""
        out = regrid(grib_fieldlist, **case.regrid_kwargs())[0]

        assert out.metadata("gridType") == case.grid_type
        for key, value in case.expected_metadata.items():
            assert out.metadata(key) == pytest.approx(value), key

    def test_npoints(self, grib_fieldlist, case):
        """The number of values matches the GRIB header (and the expected count, if known)."""
        out = regrid(grib_fieldlist, **case.regrid_kwargs())[0]

        assert len(out.values) == out.metadata("numberOfDataPoints")
        if case.npoints() is not None:
            assert len(out.values) == case.npoints()

    def test_constant_values_preserved(self, grib_fieldlist, case):
        """A constant field stays finite and constant."""
        values = regrid(grib_fieldlist, **case.regrid_kwargs())[0].values

        assert np.isfinite(values).all()
        np.testing.assert_allclose(values, BASE_VALUE, atol=ATOL)

    def test_param_id_preserved(self, grib_fieldlist, case):
        """The paramId is kept."""
        out = regrid(grib_fieldlist, param_id=130, **case.regrid_kwargs())[0]

        assert out.metadata("paramId") == 130


class TestMirRegridStructured:
    """``mir_regrid`` onto structured target grids (Gaussian / regular lat-lon)."""

    def test_empty_fieldlist_returns_early(self):
        """Empty fieldlists are returned immediately without calling MIR."""
        result = mir_regrid(ekd.SimpleFieldList(), "O16")

        assert len(result) == 0

    def test_grid_normalised_before_regrid(self, grib_fieldlist):
        """List grid specs are normalised (same result as string equivalent)."""
        result_list = regrid(grib_fieldlist, [1.0, 1.0])
        result_str = regrid(grib_fieldlist, "1.0/1.0")

        np.testing.assert_array_equal(result_list[0].values, result_str[0].values)

    def test_multiple_fields(self, grib_fieldlist):
        """Multiple fields are regridded, each keeping its own constant value."""
        result = regrid(grib_fieldlist, "O16", nfields=3, base_value=250.0)

        assert len(result) == 3
        for i, field in enumerate(result):
            assert np.isfinite(field.values).all()
            np.testing.assert_allclose(field.values, 250.0 + i * 10.0, atol=ATOL)

    def test_coarser_grid_has_fewer_points(self, grib_fieldlist):
        """O16 has fewer points than the O32 source."""
        source = grib_fieldlist(grid=SOURCE_GRID)
        result = regrid(grib_fieldlist, "O16")

        assert len(result[0].values) < len(source[0].values)

    def test_area_has_fewer_points(self, grib_fieldlist):
        """An area constraint produces fewer points than the global grid."""
        result_global = regrid(grib_fieldlist, "O16")
        result_area = regrid(grib_fieldlist, "O16", area=[90, 0, 0, 180])

        assert len(result_area[0].values) < len(result_global[0].values)

    def test_other_grids_keep_their_geography(self, grib_fieldlist):
        """The geography of other grids is not wrapped: the grid and area ecCodes reports are kept."""
        out = regrid(grib_fieldlist, "O16", area=[90, 0, 0, 180])[0]

        assert not isinstance(out._metadata, WrappedMetadata)
        assert out.metadata().geography.mars_grid() == "O16"


class TestMirRegridUnstructured:
    """``mir_regrid`` onto the unstructured lat/lon dict path.

    When the grid is a dict of ``latitudes``/``longitudes`` lists, MIR
    interpolates onto an unstructured grid of arbitrary points (one output
    value per coordinate pair).
    """

    LATITUDES = UNSTRUCTURED_GRID["latitudes"]
    LONGITUDES = UNSTRUCTURED_GRID["longitudes"]

    def test_multiple_fields(self, grib_fieldlist):
        """Multiple fields are regridded onto the same unstructured grid."""
        result = regrid(grib_fieldlist, UNSTRUCTURED_GRID, nfields=3, base_value=250.0)

        assert len(result) == 3
        for i, field in enumerate(result):
            assert len(field.values) == len(self.LATITUDES)
            assert np.isfinite(field.values).all()
            np.testing.assert_allclose(field.values, 250.0 + i * 10.0, atol=ATOL)

    def test_to_numpy(self, grib_fieldlist):
        """Values of the unstructured output are retrievable via ``to_numpy``."""
        out = regrid(grib_fieldlist, UNSTRUCTURED_GRID)[0]

        array = out.to_numpy()
        assert array.shape == (len(self.LATITUDES),)
        assert np.isfinite(array).all()
        np.testing.assert_allclose(array, BASE_VALUE, atol=ATOL)

        flat = out.to_numpy(flatten=True)
        assert flat.shape == (len(self.LATITUDES),)
        np.testing.assert_array_equal(flat, out.values)

    def test_to_latlon(self, grib_fieldlist):
        """Lat/lon coordinates are retrievable from the unstructured output."""
        out = regrid(grib_fieldlist, UNSTRUCTURED_GRID)[0]

        ll = out.to_latlon(flatten=True)
        np.testing.assert_allclose(ll["lat"], self.LATITUDES)
        np.testing.assert_allclose(ll["lon"], self.LONGITUDES)

    def test_grid_points(self, grib_fieldlist):
        """Grid points are retrievable from the unstructured output."""
        out = regrid(grib_fieldlist, UNSTRUCTURED_GRID)[0]

        lat, lon = out.grid_points()
        np.testing.assert_allclose(lat, self.LATITUDES)
        np.testing.assert_allclose(lon, self.LONGITUDES)


class TestMirRegridO1280Area:
    """Tests for ``grid: O1280`` with ``area: [40, -25, -40, 55]``.

    The area is given as ``[north, west, south, east]`` and straddles the
    Greenwich meridian, so output longitudes wrap from 335 to 55 degrees.
    The generic checks for this grid are in ``CASES``.
    """

    GRID = "O1280"
    AREA = O1280_AREA
    NORTH, WEST, SOUTH, EAST = AREA
    #: Number of points on the global O1280 grid (4 * N * (N + 9)).
    GLOBAL_O1280_POINTS = 4 * 1280 * (1280 + 9)

    def test_points_within_area(self, grib_fieldlist):
        """All output points lie inside the requested area."""
        result = regrid(grib_fieldlist, self.GRID, area=self.AREA)
        lat, lon = result[0].grid_points()
        # Normalise longitudes to [-180, 180) so the area across 0E is contiguous.
        lon = (np.asarray(lon) + 180.0) % 360.0 - 180.0

        assert lat.max() <= self.NORTH
        assert lat.min() >= self.SOUTH
        assert lon.min() >= self.WEST
        assert lon.max() <= self.EAST
        # The area is filled, not just a sliver of it.
        assert lat.max() > self.NORTH - 0.1
        assert lat.min() < self.SOUTH + 0.1
        assert lon.min() < self.WEST + 0.1
        assert lon.max() > self.EAST - 0.1

    def test_area_corners_in_metadata(self, grib_fieldlist):
        """GRIB first/last grid points match the area, wrapping 25W to 335E."""
        out = regrid(grib_fieldlist, self.GRID, area=self.AREA)[0]

        np.testing.assert_allclose(out.metadata("latitudeOfFirstGridPointInDegrees"), self.NORTH, atol=0.1)
        np.testing.assert_allclose(out.metadata("latitudeOfLastGridPointInDegrees"), self.SOUTH, atol=0.1)
        np.testing.assert_allclose(out.metadata("longitudeOfFirstGridPointInDegrees"), self.WEST % 360)
        np.testing.assert_allclose(out.metadata("longitudeOfLastGridPointInDegrees"), self.EAST)

    def test_fewer_points_than_global(self, grib_fieldlist):
        """The cropped area has far fewer points than the global O1280 grid."""
        npoints = len(regrid(grib_fieldlist, self.GRID, area=self.AREA)[0].values)

        # Area covers 80 of 180 degrees latitude and 80 of 360 degrees longitude.
        assert 0.05 * self.GLOBAL_O1280_POINTS < npoints < 0.2 * self.GLOBAL_O1280_POINTS

    @pytest.mark.xfail(
        strict=True,
        # grid_points() silently returns N1280 coordinates; to_numpy() fails to reshape.
        raises=(AssertionError, ValueError),
        reason=(
            "ecCodes gridSpec names cropped octahedral grids 'N1280' instead of 'O1280', so "
            "earthkit-data (with ECCODES_ECKIT_GEO=1) derives the wrong shape for the field"
        ),
    )
    def test_to_numpy(self, grib_fieldlist):
        """The output converts with to_numpy() and has matching coordinates."""
        result = regrid(grib_fieldlist, self.GRID, area=self.AREA)
        out = result[0]
        npoints = out.metadata("numberOfDataPoints")

        lat, lon = out.grid_points()
        assert lat.shape == lon.shape == (npoints,)

        array = out.to_numpy()
        assert isinstance(array, np.ndarray)
        assert array.shape == (npoints,)
        assert np.isfinite(array).all()
        np.testing.assert_allclose(array, BASE_VALUE, atol=ATOL)
        np.testing.assert_array_equal(array, out.values)

        stacked = result.to_numpy()
        assert stacked.shape == (1, npoints)
        np.testing.assert_array_equal(stacked[0], array)


def assert_header_is_the_grid(header: dict[str, Any], grid: dict[str, Any]) -> None:
    """Every key of the Lambert grid is in the header of the output with its value (and the output is GRIB2)."""
    assert header["edition"] == 2
    assert header["jScansPositively"] == 1
    for key, value in canonical_lambert_grid(grid).items():
        assert header[key] == pytest.approx(value, abs=1e-9), key


def assert_points_are_exact(result: dict[str, Any], grid: dict[str, Any]) -> None:
    """The points ecCodes decodes for the output are the exact (pyproj) points of the grid, 1e-4 degrees."""
    latitudes, longitudes = pyproj_points(grid)

    assert len(result["latitudes"]) == len(latitudes)
    np.testing.assert_allclose(result["latitudes"], latitudes, atol=POSITION_ATOL_DEGREES, rtol=0)
    np.testing.assert_allclose(
        signed_longitude(result["longitudes"] - longitudes), 0.0, atol=POSITION_ATOL_DEGREES, rtol=0
    )


def value_positions(grid: dict[str, Any], source: str, **kwargs: Any) -> np.ndarray:
    """The distance in metres between the position the values say a point is at, and the point ecCodes decodes.

    The source of the first field has the latitude of its points as values, that of the second the longitude.
    """
    by_latitude = probe(grid, source, "latitude", **kwargs)
    by_longitude = probe(grid, source, "longitude", **kwargs)
    return distance_m(
        by_latitude["values"],
        signed_longitude(by_longitude["values"]),
        by_latitude["latitudes"],
        signed_longitude(by_latitude["longitudes"]),
    )


class TestMirRegridLambert:
    """``mir_regrid`` onto a Lambert Conformal Conic grid given as a dict of GRIB keys.

    The generic checks for the small grid and for MEPS are in ``CASES``.
    """

    def test_meps_header_equals_the_patch(self):
        """The header of the output has every key of the MEPS patch with its value (sphere of radius 6371000 m)."""
        header = probe(MEPS_LAMBERT_GRID, "O640", "latitude")["header"]

        assert header["gridType"] == "lambert"
        assert header["numberOfDataPoints"] == MEPS_NPOINTS
        assert header["packingType"] == "grid_ieee"
        for key, value in MEPS_LAMBERT_GRID.items():
            assert header[key] == value, key
        assert header["edition"] == 2
        assert header["jScansPositively"] == 1

    @pytest.mark.parametrize("label", EARTHS)
    def test_header_is_the_grid(self, label):
        """Header and shape of the earth are those of the grid, whatever the earth of the input."""
        grid = lambert_grid(**EARTHS[label][0])

        header = probe(grid, "O32", input_shape=0)["header"]

        assert_header_is_the_grid(header, grid)

    @pytest.mark.parametrize("label", EARTHS)
    def test_decoded_projection_has_the_radius_of_the_earth(self, label):
        """The projection ecCodes decodes from the output is the sphere of the grid."""
        earth, radius = EARTHS[label]

        result = probe(lambert_grid(**earth), "O32", input_shape=6)

        assert result["proj"] == (
            f"+proj=lcc +lon_0=15.000000 +lat_0=63.300000 +lat_1=63.300000 +lat_2=63.300000 +R={radius:.6f}"
        )

    @pytest.mark.parametrize("shape", [0, 1, 5, 6, 8])
    def test_the_earth_of_the_input_does_not_matter(self, shape):
        """The input earth does not change the points: MIR places them on the same sphere for every one."""
        reference = probe(LAMBERT_GRID, "O32", "latitude", input_shape=6)
        result = probe(LAMBERT_GRID, "O32", "latitude", input_shape=shape)

        np.testing.assert_array_equal(result["values"], reference["values"])
        np.testing.assert_array_equal(result["latitudes"], reference["latitudes"])

    @pytest.mark.parametrize("label", EARTHS)
    def test_decoded_points_are_exact(self, label):
        """The points ecCodes decodes are the exact points of the grid on its earth (not those on MIR's sphere)."""
        grid = lambert_grid(**EARTHS[label][0])

        assert_points_are_exact(probe(grid, "O32", input_shape=6 if label == "shape-0" else 0), grid)

    def test_decoded_points_are_those_of_lambert_points(self):
        """``lambert_points`` and an independent pyproj PROJ string agree on the points ecCodes decodes."""
        result = probe(LAMBERT_GRID, "O32")

        latitudes, longitudes = lambert_points(LAMBERT_GRID)

        np.testing.assert_allclose(result["latitudes"], latitudes, atol=POSITION_ATOL_DEGREES, rtol=0)
        np.testing.assert_allclose(
            signed_longitude(result["longitudes"] - longitudes), 0.0, atol=POSITION_ATOL_DEGREES, rtol=0
        )

    @pytest.mark.parametrize("label", EARTHS)
    def test_values_are_the_coordinates_of_the_points(self, label):
        """Source values = latitude (and longitude): the values of the output are at the points ecCodes decodes."""
        grid = lambert_grid(**EARTHS[label][0])

        error = value_positions(grid, "O320", input_shape=0)

        assert error.max() < VALUE_TOLERANCE_M["O320"]

    def test_the_output_scans_from_south_to_north(self):
        """The first point is the south-west corner as given, and the latitudes of the points rise."""
        result = probe(LAMBERT_GRID, "O32")

        assert result["header"]["jScansPositively"] == 1
        assert result["latitudes"][0] == pytest.approx(LAMBERT_GRID["latitudeOfFirstGridPointInDegrees"], abs=1e-6)
        assert result["longitudes"][0] == pytest.approx(LAMBERT_GRID["longitudeOfFirstGridPointInDegrees"], abs=1e-6)
        assert result["latitudes"][0] < result["latitudes"][-1]

    def test_meps_points_are_exact_and_values_follow_them(self):
        """MEPS: decoded points equal the exact points within 1e-4 degrees, the values (O640 source) are at them.

        The value error is the interpolation error of a 16 km source onto 2.5 km: 36 m at the most.
        """
        result = probe(MEPS_LAMBERT_GRID, "O640", "latitude")

        assert_points_are_exact(result, MEPS_LAMBERT_GRID)
        assert value_positions(MEPS_LAMBERT_GRID, "O640").max() < VALUE_TOLERANCE_M["O640"]

    def test_grib1_input(self):
        """GRIB1 input: the header, the points and the values are those of GRIB2 input (MIR is given no LaD key)."""
        header = probe(LAMBERT_GRID, "O320-grib1", "latitude")

        assert header["source_edition"] == 1
        assert_header_is_the_grid(header["header"], LAMBERT_GRID)
        assert_points_are_exact(header, LAMBERT_GRID)
        # The 24 bit packing of the GRIB1 source adds about a metre
        assert value_positions(LAMBERT_GRID, "O320-grib1").max() < VALUE_TOLERANCE_M["O320"] + 5.0

    def test_default_packing_values(self, grib_fieldlist):
        """With the default packing (CCSDS, 16 bits) a constant field stays constant."""
        out = regrid(grib_fieldlist, LAMBERT_GRID)[0]

        assert out.metadata("packingType") == "grid_ccsds"
        np.testing.assert_allclose(out.values, BASE_VALUE, atol=ATOL)

    def test_mars_grid_and_area_overridden(self, grib_fieldlist):
        """``mars_grid()`` is the canonical Lambert grid and ``mars_area()`` is None without an area.

        ecCodes would give ``[None, None]`` and a partial area for a Lambert field (or fail with
        ``ECCODES_ECKIT_GEO=1``); the override does not read the geography for these two.
        """
        fields = regrid(grib_fieldlist, LAMBERT_GRID, nfields=2)

        for field in fields:
            assert field.metadata().geography.mars_grid() == canonical_lambert_grid(LAMBERT_GRID)
            assert field.metadata().geography.mars_area() is None

    @pytest.mark.parametrize("label", EARTHS)
    def test_input_fields_are_not_modified(self, grib_fieldlist, label):
        """Setting the earth works on a copy: the message and metadata of the input stay as they are."""
        source = grib_fieldlist(grid=SOURCE_GRID)[0].clone(metadata={"shapeOfTheEarth": 5})
        fields = ekd.FieldList.from_fields([source])
        message, values = fields[0].message(), fields[0].values.copy()

        mir_regrid(fields, lambert_grid(**EARTHS[label][0]))

        assert fields[0].message() == message
        assert fields[0].metadata("shapeOfTheEarth") == 5
        np.testing.assert_array_equal(fields[0].values, values)

    def test_unsupported_shape_raises_before_mir(self, grib_fieldlist):
        """Shapes that are ellipsoids or need axes are refused."""
        with pytest.raises(ValueError, match="shapeOfTheEarth=2 is not supported"):
            regrid(grib_fieldlist, lambert_grid(shapeOfTheEarth=2))

    def test_geography_in_a_process_with_eckit_geo_off(self, grib_fieldlist):
        """Reading and writing the regridded fields works with ``ECCODES_ECKIT_GEO=0``.

        ecCodes reads the variable when it initialises, hence a fresh interpreter.
        The fields report the Lambert grid and no area, and their values, other metadata,
        grid points, clones and GRIB messages are those of the Lambert field.
        """
        source = grib_fieldlist(grid=SOURCE_GRID, nfields=2, base_value=BASE_VALUE)
        buffer = io.BytesIO()
        source.to_target("file", buffer)

        code = f"""
import io, json, sys
import earthkit.data as ekd
import numpy as np
from anemoi.plugins.ecmwf.transform.regrid.backend import mir_regrid

source = ekd.from_source("memory", sys.stdin.buffer.read())
out = mir_regrid(source, json.loads({json.dumps(LAMBERT_GRID)!r}))
field = out[0]
lat, lon = field.grid_points()
clone = field.clone(values=field.values * 2)
reread = ekd.from_source("memory", field.message())[0]
print(json.dumps({{
    "mars_grid": field.metadata().geography.mars_grid(),
    "mars_area": field.metadata().geography.mars_area(),
    "gridType": field.metadata("gridType"),
    "paramId": field.metadata("paramId"),
    "npoints": field.metadata("numberOfDataPoints"),
    "nvalues": len(field.values),
    "nlat": len(lat),
    "lat_range": [float(lat.min()), float(lat.max())],
    "reread_lat": float(reread.grid_points()[0].min()),
    "clone_mars_grid": clone.metadata().geography.mars_grid(),
    "clone_doubled": bool(np.allclose(clone.values, 2 * field.values)),
    "shape": list(field.to_numpy().shape),
    "latlon_keys": sorted(field.to_latlon()),
}}))
"""
        env = {**os.environ, "ECCODES_ECKIT_GEO": "0"}
        proc = subprocess.run(
            [sys.executable, "-c", code], input=buffer.getvalue(), capture_output=True, env=env, check=False
        )

        assert proc.returncode == 0, proc.stderr.decode()
        result = json.loads(proc.stdout.decode())
        assert result["mars_grid"] == canonical_lambert_grid(LAMBERT_GRID)
        assert result["mars_area"] is None
        assert result["clone_mars_grid"] == canonical_lambert_grid(LAMBERT_GRID)
        assert result["clone_doubled"]
        assert result["gridType"] == "lambert"
        assert result["paramId"] == 130
        assert result["npoints"] == result["nvalues"] == result["nlat"] == LAMBERT_NX * LAMBERT_NY
        assert result["lat_range"][0] == pytest.approx(lambert_points(LAMBERT_GRID)[0].min(), abs=1e-5)
        assert result["reread_lat"] == pytest.approx(result["lat_range"][0])
        assert result["shape"] == [LAMBERT_NY, LAMBERT_NX]
        assert result["latlon_keys"] == ["lat", "lon"]


def test_header_keys_are_the_patch():
    """``HEADER_KEYS`` has every key of the MEPS patch (so the header tests compare all of them)."""
    assert set(MEPS_LAMBERT_GRID) <= set(HEADER_KEYS)


def test_control_values_match_the_ecc_grid_points_on_a_gaussian_grid():
    """Control of the Lambert value tests: O320 to O16, values = latitude, match the reported latitudes (to 0.03 deg)."""
    result = probe("O16", "O320", "latitude")

    np.testing.assert_allclose(result["values"], result["latitudes"], atol=0.03, rtol=0)
