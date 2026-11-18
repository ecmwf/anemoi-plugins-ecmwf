# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests of the comparison helpers used by the multio -> GRIB integration suite.

These need neither multio nor eccodes, so they run (and are not skipped) everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from .checks import DECODED_MISSING
from .checks import check_message
from .checks import check_roundtrip
from .checks import within_bounds
from .schema import Roundtrip


@pytest.mark.parametrize(
    ("got", "bounds", "expected"),
    [
        (5.0, {"gte": 5.0, "lte": 10.0}, True),  # inclusive lower endpoint
        (10.0, {"gte": 5.0, "lte": 10.0}, True),  # inclusive upper endpoint
        (4.999, {"gte": 5.0, "lte": 10.0}, False),
        (10.001, {"gte": 5.0, "lte": 10.0}, False),
        (-1e9, {"lte": 0.0}, True),  # single-sided
        (0.1, {"lte": 0.0}, False),
        (1e9, {"gte": 0.0}, True),
        (-0.1, {"gte": 0.0}, False),
        (1.0, {}, True),
    ],
)
def test_within_bounds(got, bounds, expected) -> None:
    assert within_bounds(got, bounds) is expected


def test_within_bounds_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="gt"):
        within_bounds(1.0, {"gt": 0.0})


def test_check_message_exact_matching_unchanged() -> None:
    header = {"paramId": 167, "shortName": "2t", "stepRange": "0-6", "numberOfMissing": 3}

    assert check_message(header, {"paramId": 167, "shortName": "2t", "stepRange": "0-6"}) == []
    # YAML ints vs eccodes strings and the reverse are normalised as before
    assert check_message({"stepRange": 6}, {"stepRange": "6"}) == []
    assert len(check_message(header, {"paramId": 168})) == 1
    assert len(check_message(header, {"missingKey": 1})) == 1


def test_check_message_bounds() -> None:
    header = {"min": 250.0, "max": 310.0, "shortName": "2t", "absent": None}

    assert check_message(header, {"min": {"gte": 250.0, "lte": 260.0}, "max": {"lte": 310.0}}) == []
    assert len(check_message(header, {"max": {"lte": 309.9}})) == 1
    # Non-numeric or absent keys cannot satisfy bounds
    assert len(check_message(header, {"shortName": {"gte": 0}})) == 1
    assert len(check_message(header, {"absent": {"gte": 0}})) == 1
    with pytest.raises(ValueError):
        check_message(header, {"min": {"above": 0}})


def _field(rng_seed: int = 0, n: int = 1000, nans: int = 250) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    field = rng.uniform(250.0, 310.0, size=n)
    field[rng.choice(n, size=nans, replace=False)] = np.nan
    return field


def _quantise(field: np.ndarray, bits: int) -> np.ndarray:
    """Mimic GRIB simple packing of the valid values, missing points set to DECODED_MISSING."""
    valid = ~np.isnan(field)
    low, high = field[valid].min(), field[valid].max()
    step = (high - low) / (2**bits - 1)
    out = np.full(field.shape, DECODED_MISSING)
    out[valid] = low + np.round((field[valid] - low) / step) * step
    return out


def test_roundtrip_ok() -> None:
    field = _field()
    assert check_roundtrip(Roundtrip(), field, _quantise(field, 16), 16) == []


def test_roundtrip_no_nans_requires_nothing_missing() -> None:
    field = _field(nans=0)
    spec = Roundtrip()

    assert check_roundtrip(spec, field, _quantise(field, 16), 16) == []

    decoded = _quantise(field, 16)
    decoded[3] = DECODED_MISSING
    assert any("missing mask" in m for m in check_roundtrip(spec, field, decoded, 16))


def test_roundtrip_detects_moved_missing_points() -> None:
    field = _field()
    decoded = _quantise(field, 16)
    nan_idx = np.flatnonzero(np.isnan(field))[0]
    valid_idx = np.flatnonzero(~np.isnan(field))[0]
    decoded[nan_idx], decoded[valid_idx] = decoded[valid_idx], DECODED_MISSING

    mismatches = check_roundtrip(Roundtrip(), field, decoded, 16)

    assert any("missing mask" in m and "2 point(s)" in m for m in mismatches)


def test_roundtrip_mask_check_can_be_disabled() -> None:
    field = _field()
    decoded = _quantise(field, 16)
    decoded[np.flatnonzero(np.isnan(field))[0]] = 280.0

    assert check_roundtrip(Roundtrip(missing_mask=False), field, decoded, 16) == []


def test_roundtrip_detects_values_collapsed_to_few_levels() -> None:
    """Valid values packed into 4 distinct levels (e.g. a ruined packing range) must fail."""
    field = _field()
    decoded = _quantise(field, 2)  # 4 levels

    mismatches = check_roundtrip(Roundtrip(), field, decoded, 16)

    assert any("max |decoded - input|" in m for m in mismatches)
    assert not any("missing mask" in m for m in mismatches)


def test_roundtrip_detects_non_finite_values() -> None:
    """A NaN or Inf decoded at a valid point must fail rather than slip past the error check."""
    field = _field()
    decoded = np.where(np.isnan(field), DECODED_MISSING, field)
    valid_index = np.flatnonzero(~np.isnan(field))[0]
    decoded[valid_index] = np.nan

    mismatches = check_roundtrip(Roundtrip(), field, decoded, 16)

    assert any("non-finite" in m for m in mismatches)


def test_roundtrip_packing_limit_follows_bits_per_value() -> None:
    field = _field()
    decoded = _quantise(field, 8)

    assert check_roundtrip(Roundtrip(), field, decoded, 8) == []
    assert check_roundtrip(Roundtrip(), field, decoded, 16) != []


def test_roundtrip_explicit_max_error() -> None:
    field = _field()
    decoded = _quantise(field, 8)

    assert check_roundtrip(Roundtrip(max_error=1.0), field, decoded, 16) == []
    assert check_roundtrip(Roundtrip(max_error=1e-6), field, decoded, 16) != []


def test_roundtrip_shape_mismatch() -> None:
    field = _field()

    assert check_roundtrip(Roundtrip(), field, _quantise(field, 16)[:-1], 16) != []
