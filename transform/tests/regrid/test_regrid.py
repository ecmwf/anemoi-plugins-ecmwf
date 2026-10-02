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
