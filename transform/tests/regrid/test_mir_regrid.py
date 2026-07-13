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
from anemoi.plugins.ecmwf.transform.regrid.backend import TEMPLATE_OVERRIDE_KEYS
from anemoi.plugins.ecmwf.transform.regrid.backend import GridSpec
from anemoi.plugins.ecmwf.transform.regrid.backend import mir_regrid

# Check if MIR and dependencies are available
pytest.importorskip("mir", reason="MIR not available")
pytest.importorskip("eccodes", reason="eccodes not available")
ekd = pytest.importorskip("earthkit.data", reason="earthkit.data not available")


class TestMirRegrid:
    """Tests for the mir_regrid function running MIR properly."""

    def test_empty_fieldlist_returns_early(self):
        """Empty fieldlists are returned immediately without calling MIR."""
        empty = ekd.SimpleFieldList()
        result = mir_regrid(empty, "O32")
        assert len(result) == 0

    def test_grib_regrid_single_field(self, grib_fieldlist):
        """Regrid a single field via the GRIB method."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        result = mir_regrid(fields, "O16", method="grib")

        assert len(result) == 1
        values = result[0].values
        assert np.isfinite(values).all()
        # Constant field should remain constant after regridding
        np.testing.assert_allclose(values, 300.0, atol=1.0)

    def test_grib_regrid_preserves_metadata(self, grib_fieldlist):
        """Regridded field preserves key GRIB metadata."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0, param_id=130)
        result = mir_regrid(fields, "O16", method="grib")

        assert result[0].metadata("paramId") == 130

    def test_grib_regrid_changes_grid(self, grib_fieldlist):
        """Regridded field has a different number of points than input."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        result = mir_regrid(fields, "O16", method="grib")

        input_npoints = len(fields[0].values)
        output_npoints = len(result[0].values)
        assert output_npoints != input_npoints
        assert output_npoints < input_npoints  # O16 < O32

    def test_grib_regrid_multiple_fields(self, grib_fieldlist):
        """Regrid multiple fields via the GRIB method."""
        fields = grib_fieldlist(grid="O32", nfields=3, base_value=250.0)
        result = mir_regrid(fields, "O16", method="grib")

        assert len(result) == 3
        for i, field in enumerate(result):
            values = field.values
            assert np.isfinite(values).all()
            np.testing.assert_allclose(values, 250.0 + i * 10.0, atol=1.0)

    def test_array_regrid_single_field(self, grib_fieldlist):
        """Regrid a single field via the array method."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        result = mir_regrid(fields, "O16", method="array")

        assert len(result) == 1
        values = result[0].values
        assert np.isfinite(values).all()
        np.testing.assert_allclose(values, 300.0, atol=1.0)

    def test_array_regrid_multiple_fields(self, grib_fieldlist):
        """Regrid multiple fields via the array method."""
        fields = grib_fieldlist(grid="O32", nfields=3, base_value=250.0)
        result = mir_regrid(fields, "O16", method="array")

        assert len(result) == 3
        for i, field in enumerate(result):
            values = field.values
            assert np.isfinite(values).all()
            np.testing.assert_allclose(values, 250.0 + i * 10.0, atol=1.0)

    @pytest.mark.slow
    def test_grib_and_array_methods_agree(self, grib_fieldlist):
        """GRIB and array regridding methods produce consistent results."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)

        grib_result = mir_regrid(fields, "O16", method="grib")
        array_result = mir_regrid(fields, "O16", method="array")

        np.testing.assert_allclose(grib_result[0].values, array_result[0].values, rtol=1e-5, atol=1e-3)

    def test_regrid_with_area(self, grib_fieldlist):
        """Regridding with an area constraint produces fewer points."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        result_global = mir_regrid(fields, "O16", method="grib")
        result_area = mir_regrid(fields, "O16", area=[90, 0, 0, 180], method="grib")

        # Area-limited output should have fewer points
        assert len(result_area[0].values) < len(result_global[0].values)

    def test_regrid_to_latlon_grid(self, grib_fieldlist):
        """Regrid to a regular lat-lon grid."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        result = mir_regrid(fields, [1.0, 1.0], method="grib")

        assert len(result) == 1
        values = result[0].values
        assert np.isfinite(values).all()
        np.testing.assert_allclose(values, 300.0, atol=1.0)

    def test_grid_normalised_before_regrid(self, grib_fieldlist):
        """List grid specs are normalised (same result as string equivalent)."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)
        result_list = mir_regrid(fields, [1.0, 1.0], method="grib")
        result_str = mir_regrid(fields, "1.0/1.0", method="grib")

        np.testing.assert_array_equal(result_list[0].values, result_str[0].values)


class TestTemplateOverrideKeys:
    """Tests for TEMPLATE_OVERRIDE_KEYS constant."""

    def test_is_tuple(self):
        """TEMPLATE_OVERRIDE_KEYS is a tuple."""
        assert isinstance(TEMPLATE_OVERRIDE_KEYS, tuple)

    def test_contains_essential_keys(self):
        """Essential GRIB keys are present."""
        assert "paramId" in TEMPLATE_OVERRIDE_KEYS
        assert "dataDate" in TEMPLATE_OVERRIDE_KEYS
        assert "typeOfLevel" in TEMPLATE_OVERRIDE_KEYS
        assert "level" in TEMPLATE_OVERRIDE_KEYS

    def test_paramid_is_last(self):
        """paramId must be last (as per the module docstring)."""
        assert TEMPLATE_OVERRIDE_KEYS[-1] == "paramId"

    def test_type_of_level_before_level(self):
        """typeOfLevel must come before level."""
        tol_idx = TEMPLATE_OVERRIDE_KEYS.index("typeOfLevel")
        level_idx = TEMPLATE_OVERRIDE_KEYS.index("level")
        assert tol_idx < level_idx


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
