# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the NaN fill / missing value logic of the multio output.

ecCodes marks missing points by exact equality with ``missingValue`` (a double), so the
value written into the field and ``misc-missingValue`` must be the same number and exactly
representable in the dtype of the field.
"""

import numpy as np
import pytest

pytest.importorskip("multio", reason="multio not installed")

from anemoi.plugins.ecmwf.inference.multio.multio_output import MISSING_VALUE  # noqa: E402
from anemoi.plugins.ecmwf.inference.multio.multio_output import _fill_missing  # noqa: E402
from anemoi.plugins.ecmwf.inference.multio.multio_output import _missing_value_for  # noqa: E402

DTYPES = [np.float32, np.float64]


@pytest.mark.parametrize("dtype", DTYPES)
def test_missing_value_is_exact_in_dtype(dtype) -> None:
    missing_value = _missing_value_for(np.dtype(dtype))

    assert isinstance(missing_value, float)
    assert float(dtype(missing_value)) == missing_value
    assert missing_value == pytest.approx(MISSING_VALUE, rel=1e-6)


def test_float32_missing_value_differs_from_the_double_sentinel() -> None:
    """Guards the reason for the helper: float32(-9999.99) != -9999.99."""
    assert float(np.float32(MISSING_VALUE)) != MISSING_VALUE
    assert _missing_value_for(np.dtype(np.float32)) == float(np.float32(MISSING_VALUE))


@pytest.mark.parametrize("dtype", DTYPES)
def test_fill_with_nans(dtype) -> None:
    field = np.array([1.5, np.nan, 3.0, np.nan, -2.0], dtype=dtype)

    filled, keys = _fill_missing(field)

    missing_value = keys["misc-missingValue"]
    assert filled.dtype == dtype
    assert not np.isnan(filled).any()
    # What ecCodes sees: values cast to double, compared exactly with missingValue
    assert np.array_equal(filled.astype(np.float64) == missing_value, np.isnan(field))
    assert keys["missingValue"] == missing_value
    assert keys["misc-bitmapPresent"] is True
    np.testing.assert_array_equal(filled[~np.isnan(field)], field[~np.isnan(field)])


@pytest.mark.parametrize("dtype", DTYPES)
def test_fill_without_nans_adds_no_keys(dtype) -> None:
    field = np.array([1.5, 2.0, 3.0], dtype=dtype)

    filled, keys = _fill_missing(field)

    assert keys == {}
    np.testing.assert_array_equal(filled, field)
    assert filled.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_fill_all_nans(dtype) -> None:
    filled, keys = _fill_missing(np.full(4, np.nan, dtype=dtype))

    assert (filled.astype(np.float64) == keys["misc-missingValue"]).all()
