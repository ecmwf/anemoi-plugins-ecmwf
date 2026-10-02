# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``anemoi.plugins.ecmwf.transform.regrid.backend``.

Covers the low-level ``mir_regrid`` driver (structured and unstructured
target grids), the ``grid_repr`` logging helper, and the ``GridSpec`` type.
"""

from __future__ import annotations

import numpy as np
import pytest
from anemoi.plugins.ecmwf.transform.regrid.backend import GridSpec
from anemoi.plugins.ecmwf.transform.regrid.backend import grid_repr
from anemoi.plugins.ecmwf.transform.regrid.backend import mir_regrid

from . import requires_mir

# ``grid_repr`` and ``GridSpec`` are pure helpers that need no extra deps and
# are always exercised. Tests that actually run MIR are guarded (per class)
# with ``requires_mir`` and skipped when the stack is unavailable.


@requires_mir
class TestMirRegridStructured:
    """``mir_regrid`` onto structured target grids (Gaussian / regular lat-lon)."""

    def test_empty_fieldlist_returns_early(self):
        """Empty fieldlists are returned immediately without calling MIR."""
        import earthkit.data as ekd

        empty = ekd.SimpleFieldList()
        result = mir_regrid(empty, "O32")
        assert len(result) == 0

    def test_regrid_single_field(self, grib_fieldlist):
        """Regrid a single field."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        result = mir_regrid(fields, "O16")

        assert len(result) == 1
        values = result[0].values
        assert np.isfinite(values).all()
        # Constant field should remain constant after regridding
        np.testing.assert_allclose(values, 300.0, atol=1.0)

    def test_regrid_preserves_metadata(self, grib_fieldlist):
        """Regridded field preserves key GRIB metadata."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0, param_id=130)
        result = mir_regrid(fields, "O16")

        assert result[0].metadata("paramId") == 130

    def test_regrid_changes_grid(self, grib_fieldlist):
        """Regridded field has a different number of points than input."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        result = mir_regrid(fields, "O16")

        input_npoints = len(fields[0].values)
        output_npoints = len(result[0].values)
        assert output_npoints != input_npoints
        assert output_npoints < input_npoints  # O16 < O32

    def test_regrid_multiple_fields(self, grib_fieldlist):
        """Regrid multiple fields."""
        fields = grib_fieldlist(grid="O32", nfields=3, base_value=250.0)
        result = mir_regrid(fields, "O16")

        assert len(result) == 3
        for i, field in enumerate(result):
            values = field.values
            assert np.isfinite(values).all()
            np.testing.assert_allclose(values, 250.0 + i * 10.0, atol=1.0)

    def test_regrid_with_area(self, grib_fieldlist):
        """Regridding with an area constraint produces fewer points."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        result_global = mir_regrid(fields, "O16")
        result_area = mir_regrid(fields, "O16", area=[90, 0, 0, 180])

        # Area-limited output should have fewer points
        assert len(result_area[0].values) < len(result_global[0].values)

    def test_regrid_to_latlon_grid(self, grib_fieldlist):
        """Regrid to a regular lat-lon grid."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        result = mir_regrid(fields, [1.0, 1.0])

        assert len(result) == 1
        values = result[0].values
        assert np.isfinite(values).all()
        np.testing.assert_allclose(values, 300.0, atol=1.0)

    def test_grid_normalised_before_regrid(self, grib_fieldlist):
        """List grid specs are normalised (same result as string equivalent)."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        result_list = mir_regrid(fields, [1.0, 1.0])
        result_str = mir_regrid(fields, "1.0/1.0")

        np.testing.assert_array_equal(result_list[0].values, result_str[0].values)


@requires_mir
class TestMirRegridUnstructured:
    """``mir_regrid`` onto the unstructured lat/lon dict path.

    When the grid is a dict of ``latitudes``/``longitudes`` lists, MIR
    interpolates onto an unstructured grid of arbitrary points (one output
    value per coordinate pair).
    """

    def test_regrid_to_unstructured_grid(self, grib_fieldlist):
        """Regrid to an unstructured grid defined by explicit lat/lon lists."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        latitudes = [0.0, 10.0, 20.0, 30.0]
        longitudes = [0.0, 10.0, 20.0, 30.0]
        result = mir_regrid(fields, {"latitudes": latitudes, "longitudes": longitudes})

        assert len(result) == 1
        values = result[0].values
        # One output value per requested coordinate pair
        assert len(values) == len(latitudes)
        assert np.isfinite(values).all()
        # Constant field stays constant after interpolation
        np.testing.assert_allclose(values, 300.0, atol=1.0)

    def test_regrid_unstructured_grid_type(self, grib_fieldlist):
        """Output of the unstructured path is an unstructured grid."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        latitudes = [0.0, 10.0, 20.0]
        longitudes = [0.0, 10.0, 20.0]
        result = mir_regrid(fields, {"latitudes": latitudes, "longitudes": longitudes})

        assert result[0].metadata("gridType") == "unstructured_grid"

    def test_regrid_unstructured_point_count_matches_coords(self, grib_fieldlist):
        """Number of output points equals the number of coordinate pairs."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        latitudes = [10.0, 20.0, 30.0, 40.0, 50.0]
        longitudes = [5.0, 15.0, 25.0, 35.0, 45.0]
        result = mir_regrid(fields, {"latitudes": latitudes, "longitudes": longitudes})

        assert len(result[0].values) == 5

    def test_regrid_unstructured_multiple_fields(self, grib_fieldlist):
        """Regrid multiple fields onto the same unstructured grid."""
        fields = grib_fieldlist(grid="O32", nfields=3, base_value=250.0)
        latitudes = [0.0, 10.0, 20.0]
        longitudes = [0.0, 10.0, 20.0]
        result = mir_regrid(fields, {"latitudes": latitudes, "longitudes": longitudes})

        assert len(result) == 3
        for i, field in enumerate(result):
            values = field.values
            assert len(values) == len(latitudes)
            assert np.isfinite(values).all()
            np.testing.assert_allclose(values, 250.0 + i * 10.0, atol=1.0)

    def test_regrid_unstructured_preserves_metadata(self, grib_fieldlist):
        """Unstructured regridding preserves key GRIB metadata."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0, param_id=130)
        result = mir_regrid(fields, {"latitudes": [0.0, 10.0], "longitudes": [0.0, 10.0]})

        assert result[0].metadata("paramId") == 130

    def test_regrid_unstructured_to_numpy(self, grib_fieldlist):
        """Values of the unstructured output are retrievable via ``to_numpy``."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        latitudes = [0.0, 10.0, 20.0, 30.0]
        longitudes = [0.0, 10.0, 20.0, 30.0]
        result = mir_regrid(fields, {"latitudes": latitudes, "longitudes": longitudes})

        arr = result[0].to_numpy()
        assert arr.shape == (len(latitudes),)
        assert np.isfinite(arr).all()
        np.testing.assert_allclose(arr, 300.0, atol=1.0)

        flat = result[0].to_numpy(flatten=True)
        assert flat.shape == (len(latitudes),)
        np.testing.assert_array_equal(flat, result[0].values)

    @pytest.mark.xfail(
        reason=(
            "MIR leaves uuidOfHGrid all-zero on unstructured lat/lon output, so the "
            "grid geometry cannot be decoded. To be fixed upstream; xpass once fixed."
        ),
        strict=True,
    )
    def test_regrid_unstructured_to_latlon(self, grib_fieldlist):
        """Lat/lon coordinates are retrievable from the unstructured output."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        latitudes = [0.0, 10.0, 20.0, 30.0]
        longitudes = [0.0, 10.0, 20.0, 30.0]
        result = mir_regrid(fields, {"latitudes": latitudes, "longitudes": longitudes})

        ll = result[0].to_latlon(flatten=True)
        np.testing.assert_allclose(ll["lat"], latitudes)
        np.testing.assert_allclose(ll["lon"], longitudes)

    @pytest.mark.xfail(
        reason=(
            "MIR leaves uuidOfHGrid all-zero on unstructured lat/lon output, so the "
            "grid geometry cannot be decoded. To be fixed upstream; xpass once fixed."
        ),
        strict=True,
    )
    def test_regrid_unstructured_grid_points(self, grib_fieldlist):
        """Grid points are retrievable from the unstructured output."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        latitudes = [0.0, 10.0, 20.0, 30.0]
        longitudes = [0.0, 10.0, 20.0, 30.0]
        result = mir_regrid(fields, {"latitudes": latitudes, "longitudes": longitudes})

        lat, lon = result[0].grid_points()
        np.testing.assert_allclose(lat, latitudes)
        np.testing.assert_allclose(lon, longitudes)


class TestGridRepr:
    """Tests for ``grid_repr`` which renders grid specs for logging."""

    def test_string_grid_uppercased(self):
        """A string grid specification is uppercased."""
        assert grid_repr("n320") == "N320"
        assert grid_repr("O32") == "O32"

    def test_short_list_grid_shown_in_full(self):
        """A short list grid is shown as-is."""
        assert grid_repr([0.25, 0.25]) == "[0.25, 0.25]"

    def test_long_list_grid_truncated(self):
        """A long list grid is truncated with an ellipsis."""
        result = grid_repr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        assert "..." in result

    def test_dict_grid_summarised(self):
        """A dict grid is summarised by coordinate-list lengths."""
        result = grid_repr({"latitudes": [1.0, 2.0], "longitudes": [3.0, 4.0]})
        assert "latitudes" in result
        assert "list of len 2" in result


class TestGridSpecType:
    """Tests for the GridSpec type alias."""

    def test_string_is_valid(self):
        """Strings are valid GridSpec values."""
        grid: GridSpec = "O32"
        assert isinstance(grid, str)

    def test_list_is_valid(self):
        """Lists of floats are valid GridSpec values."""
        grid: GridSpec = [0.25, 0.25]
        assert isinstance(grid, list)

    def test_dict_is_valid(self):
        """Dicts with list values are valid GridSpec values."""
        grid: GridSpec = {"latitudes": [1.0], "longitudes": [2.0]}
        assert isinstance(grid, dict)
