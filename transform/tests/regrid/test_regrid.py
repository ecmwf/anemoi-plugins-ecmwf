# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``MIRRegrid`` filter (``regrid.regrid``).

Covers the public filter: construction (incl. named-grid resolution),
``forward()`` integration with MIR, ``__repr__`` and import paths.
"""

from __future__ import annotations

import numpy as np
import pytest
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid
from anemoi.plugins.ecmwf.transform.regrid.named import KNOWN_GRIDS
from anemoi.plugins.ecmwf.transform.regrid.named import NamedRegrid

from . import requires_mir


@requires_mir
class TestMIRRegridForward:
    """Integration tests for MIRRegrid.forward() running MIR properly."""

    def test_forward_regrids_to_target_grid(self, grib_fieldlist):
        """forward() actually regrids fields to the target grid."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        r = MIRRegrid(grid="O16")
        result = r.forward(fields)

        assert len(result) == 1
        # O16 has fewer points than O32
        assert len(result[0].values) < len(fields[0].values)
        # Constant field should stay constant
        np.testing.assert_allclose(result[0].values, 300.0, atol=1.0)

    def test_forward_multiple_fields(self, grib_fieldlist):
        """forward() regrids multiple fields correctly."""
        fields = grib_fieldlist(grid="O32", nfields=3, base_value=250.0)
        r = MIRRegrid(grid="O16")
        result = r.forward(fields)

        assert len(result) == 3
        for i, field in enumerate(result):
            expected = 250.0 + i * 10.0
            np.testing.assert_allclose(field.values, expected, atol=1.0)

    def test_forward_with_area(self, grib_fieldlist):
        """forward() with area constraint produces fewer points."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        r_global = MIRRegrid(grid="O16")
        r_area = MIRRegrid(grid="O16", area=[90, 0, 0, 180])

        result_global = r_global.forward(fields)
        result_area = r_area.forward(fields)

        assert len(result_area[0].values) < len(result_global[0].values)

    def test_forward_latlon_grid(self, grib_fieldlist):
        """forward() works with lat-lon grid specification."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        r = MIRRegrid(grid=[1.0, 1.0])
        result = r.forward(fields)

        assert len(result) == 1
        values = result[0].values
        assert np.isfinite(values).all()
        np.testing.assert_allclose(values, 300.0, atol=1.0)

    def test_forward_preserves_param_id(self, grib_fieldlist):
        """forward() preserves paramId metadata after regridding."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0, param_id=130)
        r = MIRRegrid(grid="O16")
        result = r.forward(fields)

        assert result[0].metadata("paramId") == 130

    def test_forward_empty_fieldlist(self):
        """forward() with empty fields returns empty."""
        import earthkit.data as ekd

        empty = ekd.SimpleFieldList()
        r = MIRRegrid(grid="O16")
        result = r.forward(empty)
        assert len(result) == 0


class TestMIRRegridNamedGrid:
    """Tests for regridding to named grids (e.g. "meps").

    Named grids are resolved via ``NamedRegrid`` into a dict of explicit
    ``latitudes``/``longitudes`` lists, which drives MIR's unstructured path.
    The resolution tests need no MIR; ``forward()`` tests do.
    """

    def test_named_grid_resolved_to_coord_dict(self):
        """A known grid name is resolved to a lat/lon coordinate dict."""
        r = MIRRegrid(grid="meps")

        assert isinstance(r.grid, dict)
        assert set(r.grid) == {"latitudes", "longitudes"}
        assert len(r.grid["latitudes"]) == len(r.grid["longitudes"])
        assert len(r.grid["latitudes"]) > 0

    def test_named_grid_case_insensitive(self):
        """Named grids are matched case-insensitively."""
        r_lower = MIRRegrid(grid="meps")
        r_upper = MIRRegrid(grid="MEPS")

        assert r_upper.grid["latitudes"] == r_lower.grid["latitudes"]
        assert r_upper.grid["longitudes"] == r_lower.grid["longitudes"]

    def test_named_grid_matches_namedregrid(self):
        """MIRRegrid resolves the same coordinates as NamedRegrid directly."""
        r = MIRRegrid(grid="meps")
        named = NamedRegrid("meps")

        assert r.grid["latitudes"] == named.latitudes
        assert r.grid["longitudes"] == named.longitudes

    def test_unknown_named_grid_passed_through(self):
        """An unknown string grid is not treated as a named grid."""
        # "O16" is a valid MIR grid string, not a named grid.
        assert "o16" not in KNOWN_GRIDS
        r = MIRRegrid(grid="O16")
        assert r.grid == "O16"

    @requires_mir
    def test_forward_to_named_grid(self, grib_fieldlist):
        """forward() regrids onto a named grid's unstructured point set."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        r = MIRRegrid(grid="meps")
        result = r.forward(fields)

        assert len(result) == 1
        values = result[0].values
        # One output value per named-grid coordinate pair
        assert len(values) == len(r.grid["latitudes"])
        assert np.isfinite(values).all()
        # Constant field stays constant after interpolation
        np.testing.assert_allclose(values, 300.0, atol=1.0)

    @requires_mir
    def test_forward_to_named_grid_is_unstructured(self, grib_fieldlist):
        """Regridding to a named grid produces an unstructured grid."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        r = MIRRegrid(grid="meps")
        result = r.forward(fields)

        assert result[0].metadata("gridType") == "unstructured_grid"


class TestMIRRegridO1280Area:
    """Tests for the config ``grid: O1280`` with ``area: [40, -25, -40, 55]``.

    The area is given as ``[north, west, south, east]`` and straddles the
    Greenwich meridian, so output longitudes wrap from 335 to 55 degrees.
    """

    GRID = "O1280"
    AREA = [40, -25, -40, 55]
    NORTH, WEST, SOUTH, EAST = AREA
    #: Number of points on the global O1280 grid (4 * N * (N + 9)).
    GLOBAL_O1280_POINTS = 4 * 1280 * (1280 + 9)

    def _regrid(self, grib_fieldlist, **kwargs):
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0, **kwargs)
        return MIRRegrid(grid=self.GRID, area=self.AREA).forward(fields)

    def test_config_stored(self):
        """The grid and area from the config are kept unchanged."""
        r = MIRRegrid(grid=self.GRID, area=self.AREA)

        assert r.grid == "O1280"
        assert r._area == [40, -25, -40, 55]

    def test_config_repr(self):
        """repr shows the O1280 grid and the full area."""
        r = MIRRegrid(grid=self.GRID, area=self.AREA)

        assert repr(r) == "MIRRegrid(grid='O1280', area=[40, -25, -40, 55])"

    @requires_mir
    def test_forward_output_is_octahedral_o1280(self, grib_fieldlist):
        """The output is an octahedral reduced Gaussian grid with N=1280."""
        result = self._regrid(grib_fieldlist)

        assert len(result) == 1
        field = result[0]
        assert field.metadata("gridType") == "reduced_gg"
        assert field.metadata("N") == 1280
        assert field.metadata("isOctahedral") == 1

    @requires_mir
    def test_forward_points_within_area(self, grib_fieldlist):
        """All output points lie inside the requested area."""
        result = self._regrid(grib_fieldlist)
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

    @requires_mir
    def test_forward_area_corners_in_metadata(self, grib_fieldlist):
        """GRIB first/last grid points match the area, wrapping 25W to 335E."""
        field = self._regrid(grib_fieldlist)[0]

        np.testing.assert_allclose(field.metadata("latitudeOfFirstGridPointInDegrees"), self.NORTH, atol=0.1)
        np.testing.assert_allclose(field.metadata("latitudeOfLastGridPointInDegrees"), self.SOUTH, atol=0.1)
        np.testing.assert_allclose(field.metadata("longitudeOfFirstGridPointInDegrees"), self.WEST % 360)
        np.testing.assert_allclose(field.metadata("longitudeOfLastGridPointInDegrees"), self.EAST)

    @requires_mir
    def test_forward_fewer_points_than_global(self, grib_fieldlist):
        """The cropped area has far fewer points than the global O1280 grid."""
        field = self._regrid(grib_fieldlist)[0]
        npoints = len(field.values)

        assert npoints == field.metadata("numberOfDataPoints")
        # Area covers 80 of 180 degrees latitude and 80 of 360 degrees longitude.
        assert 0.05 * self.GLOBAL_O1280_POINTS < npoints < 0.2 * self.GLOBAL_O1280_POINTS

    @requires_mir
    def test_forward_values_and_metadata_preserved(self, grib_fieldlist):
        """A constant field stays constant and keeps its paramId."""
        field = self._regrid(grib_fieldlist, param_id=130)[0]

        assert np.isfinite(field.values).all()
        np.testing.assert_allclose(field.values, 300.0, atol=1.0)
        assert field.metadata("paramId") == 130

    def _regrid_and_read_back(self, grib_fieldlist):
        """Regrid with MIR, encode the result as GRIB in memory and decode it again."""
        import io

        import earthkit.data as ekd

        result = self._regrid(grib_fieldlist)
        buffer = io.BytesIO()
        result.to_target("file", buffer)
        return result, ekd.from_source("memory", buffer.getvalue())

    @requires_mir
    def test_read_back(self, grib_fieldlist):
        """MIR output re-decoded from GRIB bytes has the same grid and values."""
        result, read_back = self._regrid_and_read_back(grib_fieldlist)

        assert len(read_back) == 1
        original, field = result[0], read_back[0]
        for key in ("gridType", "gridName", "N", "isOctahedral", "numberOfDataPoints", "paramId", "edition"):
            assert field.metadata(key) == original.metadata(key), key
        assert field.metadata("gridName") == "O1280"
        np.testing.assert_array_equal(field.values, original.values)
        np.testing.assert_allclose(field.values, 300.0, atol=1.0)

    @requires_mir
    @pytest.mark.xfail(
        strict=True,
        # grid_points() silently returns N1280 coordinates; to_numpy() fails to reshape.
        raises=(AssertionError, ValueError),
        reason=(
            "ecCodes gridSpec names cropped octahedral grids 'N1280' instead of 'O1280', so "
            "earthkit-data (with ECCODES_ECKIT_GEO=1) derives the wrong shape for the field"
        ),
    )
    def test_read_back_to_numpy(self, grib_fieldlist):
        """The read-back field converts with to_numpy() and has matching coordinates."""
        _, read_back = self._regrid_and_read_back(grib_fieldlist)
        field = read_back[0]
        npoints = field.metadata("numberOfDataPoints")

        lat, lon = field.grid_points()
        assert lat.shape == lon.shape == (npoints,)

        array = field.to_numpy()
        assert isinstance(array, np.ndarray)
        assert array.shape == (npoints,)
        assert np.isfinite(array).all()
        np.testing.assert_allclose(array, 300.0, atol=1.0)
        np.testing.assert_array_equal(array, field.values)

        stacked = read_back.to_numpy()
        assert stacked.shape == (1, npoints)
        np.testing.assert_array_equal(stacked[0], array)


class TestMIRRegridRepr:
    """Tests for MIRRegrid.__repr__()."""

    def test_repr_string_grid(self):
        """repr shows grid and area."""
        r = MIRRegrid(grid="O32", area=[90, 0, -90, 360])
        assert "O32" in repr(r)
        assert "90" in repr(r)

    def test_repr_no_area(self):
        """repr shows None area."""
        r = MIRRegrid(grid="N320")
        assert "N320" in repr(r)
        assert "None" in repr(r)
