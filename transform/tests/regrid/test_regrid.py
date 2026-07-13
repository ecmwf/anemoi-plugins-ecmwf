# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import numpy as np
import pytest
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid
from anemoi.plugins.ecmwf.transform.regrid.regrid import MIRRegrid as MIRRegridDirect

# Check dependencies for integration tests
mir = pytest.importorskip("mir", reason="MIR not available")
pytest.importorskip("eccodes", reason="eccodes not available")
ekd = pytest.importorskip("earthkit.data", reason="earthkit.data not available")


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

    def test_forward_array_method(self, grib_fieldlist):
        """forward() with method='array' produces correct results."""
        fields = grib_fieldlist(grid="O32", nfields=2, base_value=200.0)
        r = MIRRegrid(grid="O16", method="array")
        result = r.forward(fields)

        assert len(result) == 2
        for i, field in enumerate(result):
            expected = 200.0 + i * 10.0
            np.testing.assert_allclose(field.values, expected, atol=1.0)

    def test_forward_preserves_param_id(self, grib_fieldlist):
        """forward() preserves paramId metadata after regridding."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0, param_id=130)
        r = MIRRegrid(grid="O16")
        result = r.forward(fields)

        assert result[0].metadata("paramId") == 130

    def test_forward_empty_fieldlist(self):
        """forward() with empty fields returns empty."""
        empty = ekd.SimpleFieldList()
        r = MIRRegrid(grid="O16")
        result = r.forward(empty)
        assert len(result) == 0

    @pytest.mark.slow
    def test_forward_grib_and_array_agree(self, grib_fieldlist):
        """GRIB and array methods produce consistent results."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        r_grib = MIRRegrid(grid="O16", method="grib")
        r_array = MIRRegrid(grid="O16", method="array")

        result_grib = r_grib.forward(fields)
        result_array = r_array.forward(fields)

        np.testing.assert_allclose(result_grib[0].values, result_array[0].values, rtol=1e-5, atol=1e-3)


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


class TestMIRRegridImport:
    """Tests for MIRRegrid import paths."""

    def test_importable_from_package(self):
        """MIRRegrid is importable from the regrid package."""
        assert MIRRegrid is MIRRegridDirect
