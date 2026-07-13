# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import os

import numpy as np
import pytest
from anemoi.plugins.ecmwf.transform.utils import fields_to_numpy_parallel

# Check dependencies
pytest.importorskip("eccodes", reason="eccodes not available")
ekd = pytest.importorskip("earthkit.data", reason="earthkit.data not available")


class TestFieldsToNumpyParallel:
    """Test parallel GRIB decoding with real earthkit fields."""

    def test_empty_fieldlist(self):
        """Empty fieldlist returns empty array."""
        empty = ekd.FieldList()
        result = fields_to_numpy_parallel(empty)
        assert result.shape == (0, 0)

    def test_single_field(self, grib_fieldlist):
        """Single field decoded correctly."""
        fields = grib_fieldlist(grid="O32", nfields=1, base_value=300.0)

        result = fields_to_numpy_parallel(fields)

        assert result.shape[0] == 1
        assert result.shape[1] > 0  # Should have grid points
        # Default dtype is float32
        assert result.dtype == np.float32
        # Values should be close to 300.0 (GRIB packing introduces small errors)
        np.testing.assert_allclose(result[0], 300.0, atol=0.1)

    def test_multiple_fields(self, grib_fieldlist):
        """Multiple fields decoded in parallel with correct values."""
        nfields = 5
        fields = grib_fieldlist(grid="O32", nfields=nfields, base_value=250.0)

        result = fields_to_numpy_parallel(fields)

        assert result.shape[0] == nfields
        assert result.shape[1] > 0
        for i in range(nfields):
            expected = 250.0 + i * 10.0
            np.testing.assert_allclose(result[i], expected, atol=0.1)

    def test_dtype_float64(self, grib_fieldlist):
        """Custom dtype=float64 is respected."""
        fields = grib_fieldlist(grid="O32", nfields=2, base_value=100.0)

        result = fields_to_numpy_parallel(fields, dtype=np.float64)

        assert result.dtype == np.float64
        np.testing.assert_allclose(result[0], 100.0, atol=0.01)
        np.testing.assert_allclose(result[1], 110.0, atol=0.01)

    def test_max_workers_parameter(self, grib_fieldlist):
        """max_workers parameter is accepted and produces correct results."""
        nfields = 4
        fields = grib_fieldlist(grid="O32", nfields=nfields, base_value=200.0)

        result = fields_to_numpy_parallel(fields, max_workers=2)

        assert result.shape[0] == nfields
        for i in range(nfields):
            expected = 200.0 + i * 10.0
            np.testing.assert_allclose(result[i], expected, atol=0.1)

    def test_env_var_max_workers(self, grib_fieldlist):
        """ANEMOI_GRIB_DECODE_THREADS environment variable is respected."""
        nfields = 4
        fields = grib_fieldlist(grid="O32", nfields=nfields, base_value=200.0)

        original = os.environ.get("ANEMOI_GRIB_DECODE_THREADS")
        try:
            os.environ["ANEMOI_GRIB_DECODE_THREADS"] = "2"
            result = fields_to_numpy_parallel(fields)
            assert result.shape == (nfields, result.shape[1])
            for i in range(nfields):
                expected = 200.0 + i * 10.0
                np.testing.assert_allclose(result[i], expected, atol=0.1)
        finally:
            if original is None:
                os.environ.pop("ANEMOI_GRIB_DECODE_THREADS", None)
            else:
                os.environ["ANEMOI_GRIB_DECODE_THREADS"] = original

    def test_invalid_env_var_ignored(self, grib_fieldlist):
        """Invalid ANEMOI_GRIB_DECODE_THREADS value is handled gracefully."""
        nfields = 3
        fields = grib_fieldlist(grid="O32", nfields=nfields, base_value=100.0)

        original = os.environ.get("ANEMOI_GRIB_DECODE_THREADS")
        try:
            os.environ["ANEMOI_GRIB_DECODE_THREADS"] = "not_a_number"
            result = fields_to_numpy_parallel(fields)
            assert result.shape[0] == nfields
        finally:
            if original is None:
                os.environ.pop("ANEMOI_GRIB_DECODE_THREADS", None)
            else:
                os.environ["ANEMOI_GRIB_DECODE_THREADS"] = original

    def test_varying_field_values(self, grib_fieldlist):
        """Fields with spatially varying data are decoded correctly."""

        def ramp_values(i, npoints):
            return np.linspace(i * 100.0, i * 100.0 + 50.0, npoints, dtype=np.float64)

        fields = grib_fieldlist(grid="O32", nfields=3, value_fn=ramp_values)

        result = fields_to_numpy_parallel(fields, dtype=np.float64)

        assert result.shape[0] == 3
        # Check that min/max are approximately correct for each field
        for i in range(3):
            assert result[i].min() >= i * 100.0 - 1.0
            assert result[i].max() <= i * 100.0 + 51.0

    def test_all_fields_same_npoints(self, grib_fieldlist):
        """All decoded fields have the same number of grid points."""
        fields = grib_fieldlist(grid="O32", nfields=4, base_value=0.0)

        result = fields_to_numpy_parallel(fields)

        # Result is a contiguous 2D array - all rows same length
        assert result.ndim == 2
        assert result.shape[0] == 4
