# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared test fixtures for creating synthetic GRIB fields.

Uses eccodes samples to create GRIB messages in memory, then loads them
via earthkit-data's ``from_source("memory", ...)``. This avoids needing
MARS access or pre-made test data files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import earthkit.data as ekd


def _grib_available() -> bool:
    """Check if eccodes and earthkit.data are available."""
    try:
        import earthkit.data  # noqa: F401
        import eccodes  # noqa: F401

        return True
    except ImportError:
        return False


def make_grib_fieldlist(
    grid: str = "O32",
    nfields: int = 1,
    base_value: float = 300.0,
    param_id: int = 130,
    value_fn=None,
) -> "ekd.FieldList":
    """Create a FieldList with synthetic GRIB fields on a reduced Gaussian grid.

    Parameters
    ----------
    grid : str
        Grid name (e.g. "O32", "O16"). Currently only octahedral grids are
        supported for sample creation.
    nfields : int
        Number of fields to create.
    base_value : float
        Base constant value; field i gets ``base_value + i * 10.0``.
    param_id : int
        GRIB paramId for all fields.
    value_fn : callable, optional
        If provided, called as ``value_fn(i, npoints)`` to produce field values.
        Overrides base_value.

    Returns
    -------
    earthkit.data.FieldList
        Sample grib ekd fieldlist
    """
    import earthkit.data as ekd
    import eccodes

    # Parse the grid string to extract N (Gaussian number)
    prefix = grid[0].upper()
    n = int(grid[1:])

    if prefix == "O":
        sample_name = f"reduced_gg_pl_{n}_grib2"
    else:
        sample_name = "reduced_gg_pl_640_grib2"

    sample_id = eccodes.codes_grib_new_from_samples(sample_name)
    try:
        eccodes.codes_set(sample_id, "paramId", param_id)
        eccodes.codes_set(sample_id, "dataDate", 20240101)
        eccodes.codes_set(sample_id, "dataTime", 0)

        npoints = eccodes.codes_get_size(sample_id, "values")
        messages = []

        for i in range(nfields):
            clone_id = eccodes.codes_clone(sample_id)
            try:
                if value_fn is not None:
                    values = value_fn(i, npoints)
                else:
                    values = np.full(npoints, base_value + i * 10.0, dtype=np.float64)
                eccodes.codes_set_values(clone_id, values)
                messages.append(eccodes.codes_get_message(clone_id))
            finally:
                eccodes.codes_release(clone_id)
    finally:
        eccodes.codes_release(sample_id)

    return ekd.from_source("memory", b"".join(messages))


@pytest.fixture
def grib_fieldlist():
    """Factory fixture for creating synthetic GRIB fieldlists."""
    pytest.importorskip("eccodes", reason="eccodes not available")
    pytest.importorskip("earthkit.data", reason="earthkit.data not available")
    return make_grib_fieldlist
