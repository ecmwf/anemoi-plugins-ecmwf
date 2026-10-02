# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared helpers for the regrid test suite."""

import pytest


def _mir_stack_available() -> bool:
    """Return True if MIR and its GRIB/earthkit dependencies are importable."""
    try:
        import earthkit.data  # noqa: F401
        import eccodes  # noqa: F401
        import mir  # noqa: F401

        return True
    except ImportError:
        return False


#: Skip marker for tests that actually run MIR (as opposed to pure helpers).
requires_mir = pytest.mark.skipif(
    not _mir_stack_available(),
    reason="MIR / eccodes / earthkit.data not available",
)
