# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Integration tests: NaNs written through real multio become a real GRIB bitmap.

ecCodes marks a point as missing by exact equality with ``missingValue`` (a double).
The plugin fills NaNs in the field dtype, so the fill value must be exactly
representable in that dtype, otherwise no bitmap is built and the sentinel is packed
as data (ruining the packing range).

The field is encoded with the real multio ``encode-mtg2`` plan into a file sink, then
decoded with eccodes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from .schema import Case
from .test_encoding import _run_worker
from .test_encoding import eccodes
from .test_encoding import multio  # noqa: F401 - importorskip gate, see test_encoding

pytestmark = pytest.mark.integration

DECODED_MISSING = 1.0e30
"""``missingValue`` set on the decoded handle, far from any data or from -9999."""

NAN_FRACTION = 0.3


def _case(dtype: str, nan_fraction: float) -> Case:
    return Case.model_validate(
        {
            "name": f"missing-{dtype}-{nan_fraction}",
            "grid": "O48",
            "user": {"class": "ai", "type": "fc", "stream": "oper", "expver": "0001", "model": "aifs-single"},
            "reference": {"date": 20260923, "time": 0},
            "step": 0,
            "variable": {"name": "2t", "mars": {"param": "2t", "levtype": "sfc"}},
            "field": {"fill": [250.0, 310.0], "nan_fraction": nan_fraction, "dtype": dtype},
        }
    )


def _decode(grib_path: Path) -> dict:
    """Decode the single message of ``grib_path``; missing points are set to ``DECODED_MISSING``."""
    with open(grib_path, "rb") as handle:
        gid = eccodes.codes_grib_new_from_file(handle)
        assert gid is not None, "no GRIB message written"
        try:
            eccodes.codes_set(gid, "missingValue", DECODED_MISSING)
            return {
                "bitmapPresent": eccodes.codes_get(gid, "bitmapPresent"),
                "numberOfMissing": eccodes.codes_get(gid, "numberOfMissing"),
                "bitsPerValue": eccodes.codes_get(gid, "bitsPerValue"),
                "values": eccodes.codes_get_values(gid),
            }
        finally:
            eccodes.codes_release(gid)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nans_become_bitmap(dtype: str, tmp_path: Path) -> None:
    """NaNs are encoded as bitmap-missing points and the other values keep their precision."""
    from ._worker import _make_field
    from ._worker import _npoints

    case = _case(dtype, NAN_FRACTION)
    expected = _make_field(case.to_worker_dict(), _npoints(case.grid))
    assert expected.dtype == dtype
    is_nan = np.isnan(expected)
    assert is_nan.any()

    result, out_grib = _run_worker(case, tmp_path)
    assert result["ok"], f"multio encode failed:\n{result['error'][:1000]}"

    decoded = _decode(out_grib)
    values = decoded["values"]

    assert decoded["bitmapPresent"] == 1
    assert decoded["numberOfMissing"] == int(is_nan.sum())
    assert np.array_equal(values == DECODED_MISSING, is_nan)
    assert not np.isclose(values, -9999.0, atol=1.0).any(), "missing value sentinel survived as data"

    # Packing error is at most one quantisation step of the (NaN-free) data range
    valid = expected[~is_nan].astype(np.float64)
    step = (valid.max() - valid.min()) / (2 ** decoded["bitsPerValue"] - 1)
    np.testing.assert_allclose(values[~is_nan], valid, rtol=0, atol=step)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_no_nans_no_bitmap(dtype: str, tmp_path: Path) -> None:
    """Fields without NaNs are encoded without a bitmap."""
    case = _case(dtype, 0.0)

    result, out_grib = _run_worker(case, tmp_path)
    assert result["ok"], f"multio encode failed:\n{result['error'][:1000]}"

    decoded = _decode(out_grib)
    assert decoded["bitmapPresent"] == 0
    assert decoded["numberOfMissing"] == 0
