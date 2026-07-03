# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from datetime import datetime
from datetime import timedelta
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import State
from anemoi.plugins.ecmwf.inference.compute_tv.compute_tv import ComputeTvPlugin

NPOINTS = 50


@pytest.fixture
def mock_state() -> State:
    """State with temperature/humidity on levels plus an unrelated field."""
    return {
        "latitudes": np.random.uniform(-90, 90, size=NPOINTS),
        "longitudes": np.random.uniform(-180, 180, size=NPOINTS),
        "fields": {
            "t_850": np.random.uniform(250, 310, size=NPOINTS),
            "q_850": np.random.uniform(0, 0.02, size=NPOINTS),
            "t_500": np.random.uniform(230, 290, size=NPOINTS),
            "q_500": np.random.uniform(0, 0.02, size=NPOINTS),
            "t_100": np.random.uniform(200, 250, size=NPOINTS),  # no matching q
            "msl": np.random.uniform(500, 1500, size=NPOINTS),
        },
        "date": datetime(2020, 1, 1, 0, 0),
        "step": timedelta(hours=6),
    }


def _make_processor() -> ComputeTvPlugin:
    context = cast(Context, MagicMock())
    metadata = cast(Metadata, MagicMock())
    return ComputeTvPlugin(context=context, metadata=metadata)


def test_compute_tv_replaces_matched_levels(mock_state: State):
    original = {k: v.copy() for k, v in mock_state["fields"].items()}
    processor = _make_processor()

    new_state = processor.process(mock_state)

    for lev in ("850", "500"):
        expected = original[f"t_{lev}"] * (1.0 + 0.61 * original[f"q_{lev}"])
        np.testing.assert_allclose(new_state["fields"][f"t_{lev}"], expected)


def test_compute_tv_leaves_unmatched_fields_untouched(mock_state: State):
    original = {k: v.copy() for k, v in mock_state["fields"].items()}
    processor = _make_processor()

    new_state = processor.process(mock_state)

    # temperature without a matching humidity field is unchanged
    np.testing.assert_array_equal(new_state["fields"]["t_100"], original["t_100"])
    # unrelated fields and humidity fields are unchanged
    np.testing.assert_array_equal(new_state["fields"]["msl"], original["msl"])
    np.testing.assert_array_equal(new_state["fields"]["q_850"], original["q_850"])


def test_compute_tv_does_not_mutate_input(mock_state: State):
    original = {k: v.copy() for k, v in mock_state["fields"].items()}
    processor = _make_processor()

    processor.process(mock_state)

    # the original state's fields dict must be untouched
    for key, value in original.items():
        np.testing.assert_array_equal(mock_state["fields"][key], value)
