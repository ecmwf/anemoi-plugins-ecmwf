# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pure comparison helpers of the multio -> GRIB integration suite.

They need neither multio nor eccodes (``test_encoding`` reads the GRIB and hands the
data over), so they are unit tested in ``test_checks.py`` even when multio is absent.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .schema import Roundtrip

DECODED_MISSING = 1.0e30
"""``missingValue`` to set on the decoded GRIB handle: far from any data or from -9999."""


def within_bounds(got: float, bounds: dict[str, float]) -> bool:
    """Check ``got`` against inclusive ``gte`` / ``lte`` bounds (either may be omitted)."""
    unknown = set(bounds) - {"gte", "lte"}
    if unknown:
        raise ValueError(f"Unknown bound(s) {sorted(unknown)}; use 'gte' and/or 'lte'")
    return bounds.get("gte", -math.inf) <= got <= bounds.get("lte", math.inf)


def check_message(header: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    """Return a list of human-readable mismatches (empty == all good)."""
    mismatches = []
    for key, want in expect.items():
        got = header.get(key)
        if isinstance(want, dict):
            # Bounds for numeric keys (min/max of the decoded values, ...)
            if not isinstance(got, (int, float)) or not within_bounds(got, want):
                mismatches.append(f"  {key}: expected within {want!r}, got {got!r}")
            continue
        # eccodes returns numbers as int/float and strings as str; normalise the
        # comparison so YAML ints match eccodes ints and YAML strings match keys
        # like stepRange ("0-6").
        if isinstance(want, str) and not isinstance(got, str):
            got_cmp: Any = str(got)
        else:
            got_cmp = got
        if got_cmp != want:
            mismatches.append(f"  {key}: expected {want!r}, got {got!r}")
    return mismatches


def check_roundtrip(spec: Roundtrip, expected: np.ndarray, decoded: np.ndarray, bits_per_value: int) -> list[str]:
    """Compare the decoded GRIB values with the generated input field.

    Parameters
    ----------
    spec : Roundtrip
        Which checks to run.
    expected : np.ndarray
        The input field, NaN where the value is missing.
    decoded : np.ndarray
        Decoded GRIB values, ``DECODED_MISSING`` where the GRIB marks a point missing.
    bits_per_value : int
        ``bitsPerValue`` of the GRIB message, defining the quantisation step.

    Returns
    -------
    list[str]
        Human-readable mismatches (empty == all good).
    """
    if decoded.shape != expected.shape:
        return [f"  roundtrip: expected {expected.size} values, decoded {decoded.size}"]

    mismatches = []
    expected_missing = np.isnan(expected)
    decoded_missing = decoded == DECODED_MISSING

    if spec.missing_mask and not np.array_equal(decoded_missing, expected_missing):
        moved = int((decoded_missing != expected_missing).sum())
        mismatches.append(
            f"  roundtrip missing mask: {moved} point(s) differ "
            f"(expected {int(expected_missing.sum())} missing, decoded {int(decoded_missing.sum())})"
        )

    valid = ~(expected_missing | decoded_missing)
    if not valid.any():
        return mismatches

    want = expected[valid].astype(np.float64)
    if spec.max_error == "packing":
        step = (want.max() - want.min()) / max(2**bits_per_value - 1, 1)
        limit = step * (1 + 1e-6) + 1e-9
    else:
        limit = spec.max_error

    got = decoded[valid]
    non_finite = int((~np.isfinite(got)).sum())
    if non_finite:
        mismatches.append(f"  roundtrip: {non_finite} non-finite decoded value(s) at valid points")
        return mismatches

    error = float(np.abs(got - want).max())
    if error > limit:
        mismatches.append(
            f"  roundtrip max |decoded - input| {error:.6g} exceeds {limit:.6g} (max_error={spec.max_error})"
        )
    return mismatches
