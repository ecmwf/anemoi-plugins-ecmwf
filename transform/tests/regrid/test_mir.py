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
GRIB bytes in memory; ``TestMirRegridO1280Area.test_read_back`` also checks that those fields
can be written with ``to_target("file")`` and decoded again unchanged.
"""

from __future__ import annotations

import io

import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.plugins.ecmwf.transform.regrid.backend import mir_regrid

from . import requires_mir
from .cases import ATOL
from .cases import BASE_VALUE
from .cases import CASES
from .cases import O1280_AREA
from .cases import SOURCE_GRID
from .cases import UNSTRUCTURED_GRID
from .cases import regrid

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

    def test_read_back(self, grib_fieldlist):
        """The output written with ``to_target("file")`` decodes to the same grid and values."""
        result = regrid(grib_fieldlist, self.GRID, area=self.AREA)
        buffer = io.BytesIO()
        result.to_target("file", buffer)
        read_back = ekd.from_source("memory", buffer.getvalue())

        assert len(read_back) == 1
        original, out = result[0], read_back[0]
        for key in ("gridType", "gridName", "N", "isOctahedral", "numberOfDataPoints", "paramId", "edition"):
            assert out.metadata(key) == original.metadata(key), key
        assert out.metadata("gridName") == self.GRID
        np.testing.assert_array_equal(out.values, original.values)
        np.testing.assert_allclose(out.values, BASE_VALUE, atol=ATOL)

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
